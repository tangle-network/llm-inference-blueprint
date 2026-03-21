use std::sync::Arc;

use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;

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

    // Load operator config from file or env
    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    // Load blueprint environment (CLI args, keystore, RPC endpoints)
    let env = BlueprintEnvironment::load()?;

    // Create the inference background service (vLLM subprocess + HTTP server)
    let inference_server = InferenceServer {
        config: config.clone(),
    };

    // Build and run
    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(vllm_inference::router())
        .background_service(inference_server)
        .run()
        .await?;

    Ok(())
}
