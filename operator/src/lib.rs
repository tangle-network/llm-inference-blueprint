pub mod billing;
pub mod config;
pub mod health;
pub mod vllm;

mod server;

use std::sync::{Arc, OnceLock};

use alloy::primitives::Address;
use alloy::signers::local::PrivateKeySigner;
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

/// vLLM base URL set once during startup, read by the on-chain job handler.
static VLLM_BASE_URL: OnceLock<String> = OnceLock::new();

/// Model name set once during startup, read by the on-chain job handler.
static VLLM_MODEL_NAME: OnceLock<String> = OnceLock::new();

/// Shared HTTP client for on-chain job handler, initialized once during startup.
static VLLM_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

// ─── ABI types for on-chain job encoding ─────────────────────────────────

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

// ─── Job IDs ─────────────────────────────────────────────────────────────

pub const INFERENCE_JOB: u8 = 0;

// ─── Router ──────────────────────────────────────────────────────────────

pub fn router() -> Router {
    Router::new().route(INFERENCE_JOB, run_inference.layer(TangleLayer))
}

// ─── Job handler ─────────────────────────────────────────────────────────

/// Handle an inference job submitted on-chain.
///
/// The vLLM subprocess must already be running (started by the
/// [`InferenceServer`] background service). This handler calls the local
/// vLLM OpenAI-compatible endpoint and returns the result on-chain.
#[debug_job]
pub async fn run_inference(
    TangleArg(request): TangleArg<InferenceRequest>,
) -> TangleResult<InferenceResult> {
    let temperature = request.temperature as f32 / 1000.0;
    let max_tokens = request.maxTokens;

    let base_url = match VLLM_BASE_URL.get() {
        Some(url) => url,
        None => {
            tracing::error!("VLLM_BASE_URL not initialized — InferenceServer must start first");
            return TangleResult(InferenceResult {
                text: "error: vLLM not initialized".to_string(),
                promptTokens: 0,
                completionTokens: 0,
            });
        }
    };

    let client = match VLLM_CLIENT.get() {
        Some(c) => c,
        None => {
            tracing::error!("VLLM_CLIENT not initialized — InferenceServer must start first");
            return TangleResult(InferenceResult {
                text: "error: HTTP client not initialized".to_string(),
                promptTokens: 0,
                completionTokens: 0,
            });
        }
    };
    let model_name = VLLM_MODEL_NAME
        .get()
        .map(|s| s.as_str())
        .unwrap_or("default");

    let vllm_body = serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false,
    });

    let resp = match client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&vllm_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "vLLM request failed");
            return TangleResult(InferenceResult {
                text: format!("error: vLLM request failed: {e}"),
                promptTokens: 0,
                completionTokens: 0,
            });
        }
    };

    let body: serde_json::Value = match resp.json().await {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "vLLM response parse failed");
            return TangleResult(InferenceResult {
                text: format!("error: vLLM response parse failed: {e}"),
                promptTokens: 0,
                completionTokens: 0,
            });
        }
    };

    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

    TangleResult(InferenceResult {
        text,
        promptTokens: prompt_tokens,
        completionTokens: completion_tokens,
    })
}

// ─── Background service: HTTP server + vLLM subprocess ───────────────────

/// Runs the vLLM subprocess and the OpenAI-compatible HTTP proxy as a
/// [`BackgroundService`]. This starts before the BlueprintRunner begins
/// polling for on-chain jobs.
#[derive(Clone)]
pub struct InferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for InferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        // Set the vLLM base URL, model name, and shared HTTP client for the on-chain job handler
        let vllm_url = format!("http://{}:{}", config.vllm.host, config.vllm.port);
        let _ = VLLM_BASE_URL.set(vllm_url);
        let _ = VLLM_MODEL_NAME.set(config.vllm.model.clone());
        let _ = VLLM_CLIENT.set(reqwest::Client::new());

        tokio::spawn(async move {
            // 0. Derive operator address from private key
            let operator_address: Address =
                match config.tangle.operator_key.parse::<PrivateKeySigner>() {
                    Ok(signer) => signer.address(),
                    Err(e) => {
                        tracing::error!(error = %e, "failed to parse operator key");
                        let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                        return;
                    }
                };
            tracing::info!(address = %operator_address, "operator address derived");

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

            // 2. Build billing client
            let billing_client = match billing::BillingClient::new(config.clone()).await {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 3. Start the HTTP server
            let state = server::AppState {
                config: config.clone(),
                vllm: vllm_handle.clone(),
                billing: billing_client,
                concurrency: Arc::new(tokio::sync::Semaphore::new(
                    config.server.max_concurrent_requests,
                )),
                operator_address,
            };

            match server::start(state).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started");
                    // Don't send on tx yet — server runs until shutdown.
                    // The oneshot stays open, signaling "still alive" to the runner.
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                }
            }
        });

        Ok(rx)
    }
}
