use std::sync::Arc;

use blueprint_sdk::crypto::sp_core::SpSr25519;
use blueprint_sdk::crypto::tangle_pair_signer::sp_core;
use blueprint_sdk::crypto::tangle_pair_signer::TanglePairSigner;
use blueprint_sdk::keystore::backends::Backend;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::consumer::TangleConsumer;
use blueprint_sdk::tangle::producer::TangleProducer;
use blueprint_sdk::tangle_subxt::subxt::{OnlineClient, PolkadotConfig};

use vllm_inference::config::OperatorConfig;
use vllm_inference::health;
use vllm_inference::InferenceServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    // Check GPU availability (non-fatal)
    match health::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(
                    name = %gpu.name,
                    vram_mib = gpu.memory_total_mib,
                    "GPU"
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed — running in CPU mode");
        }
    }

    // Load operator config
    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    // Load blueprint environment
    let env = BlueprintEnvironment::load()?;
    let keystore = env.keystore();
    let ws_url = env.ws_rpc_endpoint.to_string();

    // Connect to Tangle via WebSocket
    let client = OnlineClient::<PolkadotConfig>::from_insecure_url(&ws_url)
        .await
        .map_err(|e| blueprint_sdk::Error::Other(format!("Tangle connection failed: {e}")))?;

    // Producer: subscribes to finalized blocks and extracts JobCalled events
    let tangle_producer = TangleProducer::finalized_blocks(client.clone())
        .await
        .map_err(|e| blueprint_sdk::Error::Other(format!("TangleProducer failed: {e}")))?;

    // Consumer: submits job results via sr25519 signer
    let sr25519_key = keystore
        .first_local::<SpSr25519>()
        .map_err(|e| blueprint_sdk::Error::Other(format!("no sr25519 key: {e}")))?;
    let sr25519_pair = keystore
        .get_secret::<SpSr25519>(&sr25519_key)
        .map_err(|e| blueprint_sdk::Error::Other(format!("sr25519 secret load failed: {e}")))?;
    let signer = TanglePairSigner::<sp_core::sr25519::Pair>::new(sr25519_pair.0);
    let tangle_consumer = TangleConsumer::new(client, signer);

    // Background service: vLLM subprocess + HTTP server
    let inference_server = InferenceServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(vllm_inference::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(inference_server)
        .run()
        .await?;

    Ok(())
}
