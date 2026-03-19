pub mod billing;
pub mod config;
pub mod health;
pub mod metrics;
pub mod vllm;

pub mod server;

use std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::{oneshot, Semaphore};

use crate::config::OperatorConfig;
use crate::vllm::VllmProcess;

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
) -> Result<TangleResult<InferenceResult>, RunnerError> {
    let temperature = request.temperature as f32 / 1000.0;
    let max_tokens = request.maxTokens;

    let client = reqwest::Client::new();
    let vllm_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false,
    });

    let resp = client
        .post("http://127.0.0.1:8000/v1/chat/completions")
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

            // 2. Build billing client
            let billing_client = match billing::BillingClient::new(config.clone()).await {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 3. Build semaphore from config (0 = unlimited)
            let max_concurrent = config.server.max_concurrent_requests;
            let semaphore = Arc::new(if max_concurrent == 0 {
                Semaphore::new(Semaphore::MAX_PERMITS)
            } else {
                Semaphore::new(max_concurrent)
            });

            // 4. Start the HTTP server
            let state = server::AppState {
                config: config.clone(),
                vllm: vllm_handle.clone(),
                billing: billing_client,
                semaphore,
            };

            match server::start(state).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started");
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
