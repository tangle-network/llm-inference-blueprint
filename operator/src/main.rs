use std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use vllm_inference::config::OperatorConfig;
use vllm_inference::health;
use vllm_inference::InferenceServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for InferenceBSM.onRegister.
/// Format: abi.encode(string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.vllm.model.clone(),
        gpu_count,
        total_vram,
        gpu_model,
        endpoint,
    )
        .abi_encode()
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    // Load operator config
    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    // Load blueprint environment
    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit registration inputs and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            model = %config.vllm.model,
            "Registration payload saved"
        );
        return Ok(());
    }

    // Check GPU availability (non-fatal)
    match health::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed — running in CPU mode");
        }
    }

    // Get Tangle client
    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    // Get service ID
    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    // Producer + Consumer
    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

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
