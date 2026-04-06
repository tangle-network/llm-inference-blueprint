//! OpenAI-compatible HTTP server for the vLLM operator.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! only contains:
//!
//! * `VllmBackend` — the backend attached to `AppState` via `AppStateBuilder`.
//! * Request/response types for the OpenAI chat/completions wire format.
//! * HTTP handlers that glue the shared billing flow to the vLLM subprocess.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
use tangle_inference_core::{
    detect_gpus, AppState, CostModel, CostParams, GpuInfo, PerTokenCostModel, RequestGuard,
    SpendAuthPayload,
};

use crate::config::OperatorConfig;
use crate::vllm::VllmProcess;

/// Backend attached to `AppState` via `AppStateBuilder::backend`. Handlers
/// retrieve it via `state.backend::<VllmBackend>().unwrap()`.
///
/// Owns the vLLM subprocess handle, a reference to the full operator config
/// (for vllm-specific knobs only — all shared knobs are already on AppState),
/// and a pre-built per-token cost model.
pub struct VllmBackend {
    pub config: Arc<OperatorConfig>,
    pub vllm: Arc<VllmProcess>,
    pub cost_model: PerTokenCostModel,
}

impl VllmBackend {
    pub fn new(config: Arc<OperatorConfig>, vllm: Arc<VllmProcess>) -> Self {
        let cost_model = PerTokenCostModel {
            price_per_input_token: config.vllm.price_per_input_token,
            price_per_output_token: config.vllm.price_per_output_token,
        };
        Self {
            config,
            vllm,
            cost_model,
        }
    }

    /// Calculate the cost for a request given token counts.
    pub fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        self.cost_model.calculate_cost(&CostParams {
            prompt_tokens,
            completion_tokens,
            ..Default::default()
        })
    }
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<VllmBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a VllmBackend"))?;
    let max_request_body_bytes = state.server_config.max_request_body_bytes;
    let stream_timeout_secs = state.server_config.stream_timeout_secs;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let _ = backend; // ensure we validated before spawning

    let app = HttpRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(stream_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal, draining connections");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request / Response types (OpenAI-compatible) ---

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// ShieldedCredits spend authorization (required when billing_required is true).
    /// Can also be provided via x402 headers (X-Payment-Signature).
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

// --- Handlers ---

fn backend_from(state: &AppState) -> &VllmBackend {
    state
        .backend::<VllmBackend>()
        .expect("AppState backend is VllmBackend (checked in server::start)")
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.as_deref().unwrap_or(&backend.config.vllm.model);
    let metrics_guard = RequestGuard::new(model_name);

    // 1. Acquire semaphore permit
    let permit: OwnedSemaphorePermit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server at capacity".to_string(),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // 2. x402 flow: if no spend_auth in body, check X-Payment-Signature header
    if req.spend_auth.is_none() {
        if let Some(x402_auth) = extract_x402_spend_auth(&headers) {
            req.spend_auth = Some(x402_auth);
        }
    }

    // 3. Enforce billing requirement — return 402 Payment Required if missing
    if state.billing_config.billing_required && req.spend_auth.is_none() {
        // Estimate for a typical 1000-input/512-output request
        let estimated = backend.calculate_cost(1000, 512);
        return payment_required(
            &state.billing_config,
            &state.tangle_config,
            state.operator_address,
            estimated,
        );
    }

    // 4. Validate SpendAuth (signature, account info, nonce replay, etc).
    let preauth_amount: Option<u64> = if let Some(ref spend_auth) = req.spend_auth {
        match validate_spend_auth(&state, spend_auth).await {
            Ok(amt) => Some(amt),
            Err(resp) => return resp,
        }
    } else {
        None
    };

    // 5. Cost sanity check: pre-auth amount cannot exceed 1.5x estimated max cost.
    if let (Some(_), Some(preauth)) = (&req.spend_auth, preauth_amount) {
        let estimated_prompt_tokens: u32 = req
            .messages
            .iter()
            .map(|m| (m.content.len() as u32) / 4 + 1)
            .sum();
        let estimated_max_cost = backend.calculate_cost(estimated_prompt_tokens, req.max_tokens);
        let preauth_ceiling = estimated_max_cost.saturating_mul(3) / 2;
        if estimated_max_cost > 0 && preauth > preauth_ceiling {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "pre-auth amount ({preauth}) exceeds 1.5x estimated max cost ({estimated_max_cost}) — \
                     contract settles full pre-auth, reduce amount to avoid overcharging"
                ),
                "billing_error",
                "excessive_preauth",
            );
        }

        // 5a. Per-account concurrency limit (checked via shared AppState map).
        let max_per_account = state.server_config.max_per_account_requests;
        if max_per_account > 0 {
            let spend_auth = req.spend_auth.as_ref().unwrap();
            let mut map = state.active_per_account.write().await;
            let count = map.entry(spend_auth.commitment.clone()).or_insert(0);
            if *count >= max_per_account {
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("account has {count} active requests (limit: {max_per_account})"),
                    "rate_limit_error",
                    "per_account_limit",
                );
            }
            *count += 1;
        }

        // 5b. Check vLLM health before committing gas
        if !backend.vllm.is_healthy().await {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference backend is unavailable — billing not initiated".to_string(),
                "upstream_error",
                "vllm_unhealthy",
            );
        }

        if let Err(e) = state
            .billing
            .authorize_spend(req.spend_auth.as_ref().unwrap())
            .await
        {
            tracing::error!(error = %e, "authorizeSpend failed");
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                format!("billing authorization failed: {e}"),
                "billing_error",
                "authorization_failed",
            );
        }

        // Record the nonce as used AFTER successful on-chain authorization
        let spend_auth = req.spend_auth.as_ref().unwrap();
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        state
            .nonce_store
            .insert(
                nonce_key,
                spend_auth.expiry,
                state.billing_config.clock_skew_tolerance_secs,
            )
            .await;
    }

    // 6. Dispatch to streaming or non-streaming path
    if req.stream {
        handle_streaming(state, req, preauth_amount, metrics_guard, permit).await
    } else {
        handle_non_streaming(state, req, preauth_amount, metrics_guard, permit).await
    }
}

async fn handle_non_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    preauth_amount: Option<u64>,
    mut metrics_guard: RequestGuard,
    _permit: OwnedSemaphorePermit,
) -> Response {
    let backend = backend_from(&state);
    let vllm_response = match backend.vllm.chat_completion(&req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "vLLM request failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream vLLM error: {e}"),
                "upstream_error",
                "vllm_error",
            );
        }
    };

    metrics_guard.set_tokens(
        vllm_response.usage.prompt_tokens,
        vllm_response.usage.completion_tokens,
    );
    metrics_guard.set_success();

    // Post-response settlement
    if let (Some(ref spend_auth), Some(preauth)) = (&req.spend_auth, preauth_amount) {
        let actual_cost = backend.calculate_cost(
            vllm_response.usage.prompt_tokens,
            vllm_response.usage.completion_tokens,
        );
        if let Err(e) = settle_billing(&state.billing, spend_auth, preauth, actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    Json(vllm_response).into_response()
}

async fn handle_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    preauth_amount: Option<u64>,
    mut metrics_guard: RequestGuard,
    permit: OwnedSemaphorePermit,
) -> Response {
    let backend = backend_from(&state);
    // Get the raw upstream SSE response as a byte stream
    let upstream = match backend.vllm.chat_completion_stream(&req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "vLLM streaming request failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream vLLM error: {e}"),
                "upstream_error",
                "vllm_error",
            );
        }
    };

    let byte_stream = upstream.bytes_stream();

    let spend_auth_for_settlement = req.spend_auth;
    let billing_for_settlement = state.billing.clone();
    // Clone needed for cost calculation in the background settlement task.
    let state_for_settlement = state.clone();

    let (usage_tx, usage_rx) = tokio::sync::oneshot::channel::<(u32, u32)>();

    let idle_timeout = Duration::from_secs(state.server_config.idle_chunk_timeout_secs);
    let max_line_buf = state.server_config.max_line_buf_bytes;

    // Wrap the byte stream with a per-chunk idle timeout
    let timed_stream = tokio_stream::StreamExt::timeout(
        tokio_stream::wrappers::ReceiverStream::new({
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio::spawn(async move {
                tokio::pin!(byte_stream);
                while let Some(chunk) = byte_stream.next().await {
                    if tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            });
            rx
        }),
        idle_timeout,
    );

    let proxied_stream = {
        let mut usage_sender = Some(usage_tx);
        let mut line_buf = String::new();

        timed_stream.map(move |item| {
            match item {
                Ok(Ok(bytes)) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        line_buf.push_str(text);

                        // Cap line_buf to prevent unbounded memory growth
                        if line_buf.len() > max_line_buf {
                            tracing::warn!(
                                size = line_buf.len(),
                                max = max_line_buf,
                                "line_buf exceeded max size, clearing"
                            );
                            line_buf.clear();
                        }

                        // Only process complete lines (terminated by \n).
                        while let Some(newline_pos) = line_buf.find('\n') {
                            {
                                let complete_line = &line_buf[..newline_pos];
                                if let Some(json_str) = complete_line.strip_prefix("data: ") {
                                    let json_str = json_str.trim();
                                    if json_str != "[DONE]" {
                                        if let Ok(val) =
                                            serde_json::from_str::<serde_json::Value>(json_str)
                                        {
                                            if let Some(usage) = val.get("usage") {
                                                if !usage.is_null() {
                                                    let pt = usage
                                                        .get("prompt_tokens")
                                                        .and_then(|v| v.as_u64())
                                                        .unwrap_or(0)
                                                        as u32;
                                                    let ct = usage
                                                        .get("completion_tokens")
                                                        .and_then(|v| v.as_u64())
                                                        .unwrap_or(0)
                                                        as u32;
                                                    if let Some(sender) = usage_sender.take() {
                                                        let _ = sender.send((pt, ct));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // In-place removal instead of allocating a new String
                            line_buf.replace_range(..newline_pos + 1, "");
                        }
                    }
                    Ok::<_, std::io::Error>(bytes)
                }
                Ok(Err(e)) => Err(std::io::Error::other(e)),
                Err(_elapsed) => {
                    tracing::warn!("stream idle timeout exceeded");
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "stream idle chunk timeout",
                    ))
                }
            }
        })
    };

    let body = Body::from_stream(proxied_stream);

    // Background task: waits for the stream to complete, settles billing,
    // records metrics, releases the semaphore permit on drop.
    //
    // The JoinHandle is stored in `_settlement_handle` so that if the runtime
    // shuts down before completion, the task's drop is visible in logs via
    // the tracing guard inside the future.
    let max_tokens_for_fallback = req.max_tokens;
    let _settlement_handle = tokio::spawn(async move {
        // Guard that logs if this future is cancelled mid-flight (e.g. on shutdown).
        struct SettlementDropGuard(bool);
        impl Drop for SettlementDropGuard {
            fn drop(&mut self) {
                if !self.0 {
                    tracing::warn!(
                        "streaming settlement task dropped before completion — \
                         on-chain settlement may be lost, manual recovery required"
                    );
                }
            }
        }
        let mut drop_guard = SettlementDropGuard(false);

        let backend = state_for_settlement
            .backend::<VllmBackend>()
            .expect("backend");
        match usage_rx.await {
            Ok((prompt_tokens, completion_tokens)) => {
                metrics_guard.set_tokens(prompt_tokens, completion_tokens);
                metrics_guard.set_success();

                if let (Some(ref spend_auth), Some(preauth)) =
                    (&spend_auth_for_settlement, preauth_amount)
                {
                    let actual_cost = backend.calculate_cost(prompt_tokens, completion_tokens);
                    if let Err(e) = settle_billing(&billing_for_settlement, spend_auth, preauth, actual_cost).await {
                        tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                    }
                }
            }
            Err(_) => {
                tracing::warn!(
                    "streaming response ended without usage data — settling with max_tokens fallback"
                );

                if let (Some(ref spend_auth), Some(preauth)) =
                    (&spend_auth_for_settlement, preauth_amount)
                {
                    let actual_cost = backend.calculate_cost(0, max_tokens_for_fallback);
                    if let Err(e) = settle_billing(&billing_for_settlement, spend_auth, preauth, actual_cost).await {
                        tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                    }
                }
            }
        }

        // Disarm the drop guard — settlement completed successfully.
        drop_guard.0 = true;
        drop(permit);
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build SSE response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let backend = backend_from(&state);
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: backend.config.vllm.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

/// Operator info endpoint for discovery. Returns model, pricing, GPU caps, endpoint.
async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.vllm.model,
        "pricing": {
            "price_per_input_token": backend.config.vllm.price_per_input_token,
            "price_per_output_token": backend.config.vllm.price_per_output_token,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": backend.config.gpu.expected_gpu_count,
            "min_vram_mib": backend.config.gpu.min_vram_mib,
            "model": backend.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.server_config.max_concurrent_requests,
            "max_context_length": backend.config.vllm.max_model_len,
        },
        "billing_required": state.billing_config.billing_required,
        "payment_token": state.billing_config.payment_token_address,
    }))
}

async fn health_check(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend = backend_from(&state);
    let vllm_healthy = backend.vllm.is_healthy().await;

    if vllm_healthy {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": backend.config.vllm.model,
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn gpu_health() -> Result<Json<Vec<GpuInfo>>, (StatusCode, String)> {
    match detect_gpus().await {
        Ok(gpus) => Ok(Json(gpus)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn metrics_handler() -> Response {
    let body = tangle_inference_core::metrics::gather();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(body))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build metrics response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}
