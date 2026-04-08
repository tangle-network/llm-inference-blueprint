pub mod config;
pub mod qos;
pub mod server;
pub mod vllm;

// Re-export shared infrastructure so downstream crates can `use llm_inference::*`.
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerTokenCostModel, RequestGuard, SpendAuthPayload,
};
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing,
    validate_spend_auth,
};
// Re-export metrics module for tests/downstream use.
pub use tangle_inference_core::metrics;
// Alias billing module path for downstream callers that imported
// `llm_inference::billing::BillingClient` before the refactor.
pub use tangle_inference_core::billing;

use blueprint_sdk::std::sync::{Arc, OnceLock};
use blueprint_sdk::std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::vllm::VllmProcess;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call.
    struct InferenceRequest {
        string prompt;
        uint32 maxTokens;
        /// Fixed-point temperature: 1000 = 1.0, 700 = 0.7, etc.
        uint64 temperature;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct InferenceResult {
        string text;
        uint32 promptTokens;
        uint32 completionTokens;
    }
}

// --- Job IDs ---

pub const INFERENCE_JOB: u8 = 0;

// --- Shared state for the on-chain job handler ---

/// vLLM connection config set by InferenceServer::start(), read by run_inference.
static VLLM_ENDPOINT: OnceLock<VllmEndpoint> = OnceLock::new();

struct VllmEndpoint {
    url: String,
    model: String,
    client: reqwest::Client,
}

/// Called by InferenceServer to register the vLLM endpoint for on-chain job handlers.
#[allow(clippy::result_large_err)]
fn register_vllm_endpoint(config: &OperatorConfig) -> Result<(), RunnerError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| RunnerError::Other(format!("failed to build HTTP client: {e}").into()))?;
    let endpoint = VllmEndpoint {
        url: format!(
            "http://{}:{}/v1/chat/completions",
            config.vllm.host, config.vllm.port
        ),
        model: config.vllm.model.clone(),
        client,
    };
    let _ = VLLM_ENDPOINT.set(endpoint);
    Ok(())
}

/// Initialize the vLLM endpoint for testing. Call this before submitting jobs
/// when running without the full InferenceServer background service (e.g. in
/// BlueprintHarness tests with a mock vLLM endpoint).
pub fn init_for_testing(base_url: &str, model: &str) {
    let endpoint = VllmEndpoint {
        url: format!("{base_url}/v1/chat/completions"),
        model: model.to_string(),
        client: reqwest::Client::new(),
    };
    let _ = VLLM_ENDPOINT.set(endpoint);
}

// --- Router ---

pub fn router() -> Router {
    Router::new().route(
        INFERENCE_JOB,
        run_inference.layer(TangleLayer),
    )
}

/// Direct inference call — same logic as run_inference but without TangleArg.
/// Used for testing without the Tangle context.
pub async fn run_inference_direct(request: &InferenceRequest) -> Result<InferenceResult, RunnerError> {
    let endpoint = VLLM_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other("vLLM endpoint not registered".into())
    })?;

    let temperature = request.temperature as f32 / 1000.0;
    let vllm_body = serde_json::json!({
        "model": endpoint.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.maxTokens,
        "temperature": temperature,
        "stream": false,
    });

    let resp = endpoint.client.post(&endpoint.url).json(&vllm_body).send().await
        .map_err(|e| RunnerError::Other(format!("vLLM request failed: {e}").into()))?;
    let body: serde_json::Value = resp.json().await
        .map_err(|e| RunnerError::Other(format!("vLLM parse failed: {e}").into()))?;

    Ok(InferenceResult {
        text: body["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
        promptTokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completionTokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
    })
}

// --- Job handler ---

/// Handle an inference job submitted on-chain.
///
/// Uses the vLLM endpoint registered by [`InferenceServer`] rather than
/// hardcoded values. The shared reqwest::Client is reused across calls.
#[debug_job]
pub async fn run_inference(
    TangleArg(request): TangleArg<InferenceRequest>,
) -> Result<TangleResult<InferenceResult>, RunnerError> {
    let endpoint = VLLM_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other("vLLM endpoint not registered — InferenceServer not started".into())
    })?;

    let temperature = request.temperature as f32 / 1000.0;
    let max_tokens = request.maxTokens;

    let vllm_body = serde_json::json!({
        "model": endpoint.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false,
    });

    let resp = endpoint
        .client
        .post(&endpoint.url)
        .json(&vllm_body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "vLLM request failed");
            RunnerError::Other(format!("vLLM request failed: {e}").into())
        })?;

    let body: serde_json::Value = resp.json().await.map_err(|e| {
        tracing::error!(error = %e, "vLLM response parse failed");
        RunnerError::Other(format!("vLLM response parse failed: {e}").into())
    })?;

    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

    Ok(TangleResult(InferenceResult {
        text,
        promptTokens: prompt_tokens,
        completionTokens: completion_tokens,
    }))
}

// --- Background service: HTTP server + vLLM subprocess ---

/// Runs the vLLM subprocess and the OpenAI-compatible HTTP proxy as a
/// [`BackgroundService`]. This starts before the BlueprintRunner begins
/// polling for on-chain jobs.
///
/// Includes a watchdog loop that monitors the vLLM process and respawns
/// it if it exits unexpectedly.
#[derive(Clone)]
pub struct InferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for InferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Start the vLLM subprocess
            let vllm_handle = match VllmProcess::spawn(config.clone()).await {
                Ok(h) => Arc::new(h),
                Err(e) => {
                    tracing::error!(error = %e, "failed to spawn vLLM");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            tracing::info!("vLLM process started, waiting for readiness");
            if let Err(e) = vllm_handle.wait_ready().await {
                tracing::error!(error = %e, "vLLM failed to become ready");
                let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                return;
            }
            tracing::info!("vLLM is ready");

            // Register the vLLM endpoint for on-chain job handlers
            if let Err(e) = register_vllm_endpoint(&config) {
                tracing::error!(error = %e, "failed to register vLLM endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // 2. Build billing client
            let billing_client = match BillingClient::new(&config.tangle, &config.billing) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 3. Create shutdown channel for graceful shutdown
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 4. Build the HTTP server state via the shared AppStateBuilder,
            //    attaching the vLLM process as the backend extension.
            let operator_address = billing_client.operator_address();
            let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
            let backend = server::VllmBackend::new(config.clone(), vllm_handle.clone());

            let state = match AppStateBuilder::new()
                .billing(billing_client)
                .nonce_store(nonce_store)
                .server_config(Arc::new(config.server.clone()))
                .billing_config(Arc::new(config.billing.clone()))
                .tangle_config(Arc::new(config.tangle.clone()))
                .operator_address(operator_address)
                .backend(backend)
                .build()
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started — background service ready");
                    // Signal readiness to the BlueprintRunner.
                    // Ok(()) means "started successfully". The runner treats this
                    // as the service finishing; since we keep running in the
                    // watchdog loop below, this is the correct signal.
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // 6. Watchdog loop: monitor vLLM process and respawn on crash.
            //    _live_handle keeps the most recently spawned VllmProcess alive
            //    so its Drop impl doesn't kill the child process.
            //    Loop exits on SIGINT/SIGTERM for graceful shutdown.
            let mut _live_handle: Option<VllmProcess> = None;
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("received shutdown signal, stopping watchdog");
                        let _ = shutdown_tx.send(true);
                        vllm_handle.shutdown().await;
                        if let Some(ref h) = _live_handle {
                            h.shutdown().await;
                        }
                        return;
                    }
                }

                if !vllm_handle.is_healthy().await {
                    tracing::error!("vLLM health check failed — attempting respawn");

                    // Shut down the old process
                    vllm_handle.shutdown().await;

                    // Attempt respawn with backoff
                    let mut respawn_delay = Duration::from_secs(5);
                    loop {
                        tracing::info!(delay_secs = respawn_delay.as_secs(), "respawning vLLM");
                        match VllmProcess::spawn(config.clone()).await {
                            Ok(new_handle) => {
                                match new_handle.wait_ready().await {
                                    Ok(()) => {
                                        tracing::info!("vLLM respawned successfully");
                                        // Store the new handle to prevent Drop from killing it.
                                        // The HTTP server proxies via host:port so requests
                                        // reach the new process automatically.
                                        _live_handle = Some(new_handle);
                                        break;
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, "respawned vLLM failed readiness");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "failed to respawn vLLM");
                            }
                        }
                        tokio::time::sleep(respawn_delay).await;
                        respawn_delay = (respawn_delay * 2).min(Duration::from_secs(120));
                    }
                }
            }
        });

        Ok(rx)
    }
}
