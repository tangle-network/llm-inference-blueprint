use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::billing::{self, BillingClient};
use crate::config::OperatorConfig;
use crate::health;
use crate::metrics::{self, RequestGuard};
use crate::vllm::VllmProcess;

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub vllm: Arc<VllmProcess>,
    pub billing: Arc<BillingClient>,
    pub semaphore: Arc<Semaphore>,
}

/// Start the HTTP server, returns a join handle.
pub async fn start(state: AppState) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// ─── Request / Response types (OpenAI-compatible) ────────────────────────

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

    /// ShieldedCredits spend authorization (required for billing)
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
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

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

fn error_response(status: StatusCode, message: String, error_type: &str, code: &str) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message,
            r#type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

// ─── Handlers ────────────────────────────────────────────────────────────

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let metrics_guard = RequestGuard::new();

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

    // 2. Verify SpendAuth signature off-chain
    if let Some(ref spend_auth) = req.spend_auth {
        let valid = billing::verify_spend_auth_signature(
            spend_auth,
            &state.config.tangle.shielded_credits,
            state.config.tangle.chain_id,
        );
        if !valid {
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                "invalid SpendAuth signature".to_string(),
                "billing_error",
                "invalid_spend_auth",
            );
        }

        // 2a. Enforce max_spend_per_request policy
        let max_spend = state.config.billing.max_spend_per_request;
        if max_spend > 0 {
            let requested: u64 = spend_auth.amount.parse().unwrap_or(0);
            if requested > max_spend {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "spend authorization amount ({requested}) exceeds max_spend_per_request ({max_spend})"
                    ),
                    "billing_error",
                    "exceeds_max_spend",
                );
            }
        }

        // 2b. Enforce min_credit_balance policy
        let min_balance = state.config.billing.min_credit_balance;
        if min_balance > 0 {
            match state
                .billing
                .get_account_balance(&spend_auth.commitment)
                .await
            {
                Ok(balance) => {
                    if balance < alloy::primitives::U256::from(min_balance) {
                        return error_response(
                            StatusCode::PAYMENT_REQUIRED,
                            format!(
                                "credit balance ({balance}) is below minimum required ({min_balance})"
                            ),
                            "billing_error",
                            "insufficient_balance",
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to check credit balance");
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to verify credit balance".to_string(),
                        "billing_error",
                        "balance_check_failed",
                    );
                }
            }
        }
    }

    // 3. Pre-authorize billing on-chain BEFORE sending upstream request
    if let Some(ref spend_auth) = req.spend_auth {
        if let Err(e) = state.billing.authorize_spend(spend_auth).await {
            tracing::error!(error = %e, "authorizeSpend failed");
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                format!("billing authorization failed: {e}"),
                "billing_error",
                "authorization_failed",
            );
        }
    }

    // 4. Dispatch to streaming or non-streaming path
    if req.stream {
        handle_streaming(state, req, metrics_guard, permit).await
    } else {
        handle_non_streaming(state, req, metrics_guard, permit).await
    }
}

async fn handle_non_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    mut metrics_guard: RequestGuard,
    _permit: OwnedSemaphorePermit,
) -> Response {
    let vllm_response = match state.vllm.chat_completion(&req).await {
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

    // Post-response settlement (claim payment) — distinct from pre-auth.
    // Charge the actual metered cost, capped by the pre-authorized ceiling.
    if let Some(ref spend_auth) = req.spend_auth {
        let actual_cost = state.billing.calculate_cost(
            vllm_response.usage.prompt_tokens,
            vllm_response.usage.completion_tokens,
        );
        let preauth_amount = spend_auth.amount.parse::<u64>().unwrap_or(0);
        let charge_amount = actual_cost.min(preauth_amount);
        if charge_amount > 0 {
            if let Err(e) = state.billing.claim_payment(spend_auth, charge_amount).await {
                tracing::warn!(error = %e, charge_amount, "billing claim failed");
            }
        }
    }

    // _permit is dropped here, releasing the semaphore slot
    Json(vllm_response).into_response()
}

async fn handle_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    mut metrics_guard: RequestGuard,
    permit: OwnedSemaphorePermit,
) -> Response {
    // Get the raw upstream SSE response as a byte stream
    let upstream = match state.vllm.chat_completion_stream(&req).await {
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

    let (usage_tx, usage_rx) = tokio::sync::oneshot::channel::<(u32, u32)>();

    let proxied_stream = {
        let mut usage_sender = Some(usage_tx);
        let mut accumulated = String::new();

        byte_stream.map(move |chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    // Scan this chunk for usage data
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        accumulated.push_str(text);
                        // Check for usage in SSE data lines
                        for line in accumulated.lines() {
                            if let Some(json_str) = line.strip_prefix("data: ") {
                                if json_str.trim() == "[DONE]" {
                                    continue;
                                }
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str)
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
                        // Only keep the last incomplete line for next chunk
                        if let Some(last_newline) = accumulated.rfind('\n') {
                            accumulated = accumulated[last_newline + 1..].to_string();
                        }
                    }
                    Ok::<_, std::io::Error>(bytes)
                }
                Err(e) => Err(std::io::Error::other(e)),
            }
        })
    };

    let body = Body::from_stream(proxied_stream);

    // Background task: waits for the stream to complete, then settles billing,
    // records metrics, and releases the semaphore permit.
    tokio::spawn(async move {
        // usage_rx resolves when the stream closure sends usage data, or errors
        // if the stream was dropped (client disconnect / upstream failure).
        match usage_rx.await {
            Ok((prompt_tokens, completion_tokens)) => {
                metrics_guard.set_tokens(prompt_tokens, completion_tokens);
                metrics_guard.set_success();

                // Post-stream settlement (claim payment)
                if let Some(ref spend_auth) = spend_auth_for_settlement {
                    let actual_cost =
                        billing_for_settlement.calculate_cost(prompt_tokens, completion_tokens);
                    let preauth_amount = spend_auth.amount.parse::<u64>().unwrap_or(0);
                    let charge_amount = actual_cost.min(preauth_amount);
                    if charge_amount > 0 {
                        if let Err(e) = billing_for_settlement
                            .claim_payment(spend_auth, charge_amount)
                            .await
                        {
                            tracing::warn!(error = %e, charge_amount, "billing claim failed after stream");
                        }
                    }
                }
            }
            Err(_) => {
                // Stream was dropped before completion — metrics_guard drops as error
                tracing::warn!("streaming response ended before completion");
            }
        }

        // permit is held until here, covering the full stream lifetime
        drop(permit);
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.vllm.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let vllm_healthy = state.vllm.is_healthy().await;

    if vllm_healthy {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": state.config.vllm.model,
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn gpu_health() -> Result<Json<Vec<health::GpuInfo>>, (StatusCode, String)> {
    match health::detect_gpus().await {
        Ok(gpus) => Ok(Json(gpus)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn metrics_handler() -> Response {
    let body = metrics::gather();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(body))
        .unwrap()
}
