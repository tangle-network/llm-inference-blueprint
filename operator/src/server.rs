use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::billing::{self, BillingClient};
use crate::config::OperatorConfig;
use crate::health;
use crate::vllm::VllmProcess;

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub(crate) struct AppState {
    pub config: Arc<OperatorConfig>,
    pub vllm: Arc<VllmProcess>,
    pub billing: Arc<BillingClient>,
}

/// Start the HTTP server, returns a join handle.
pub(crate) async fn start(state: AppState) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
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
pub(crate) struct ChatCompletionRequest {
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
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpendAuthPayload {
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
pub(crate) struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Usage {
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
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

// ─── Handlers ────────────────────────────────────────────────────────────

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    // 1. Verify SpendAuth if billing is enabled
    if let Some(ref spend_auth) = req.spend_auth {
        let valid = billing::verify_spend_auth_signature(
            spend_auth,
            &state.config.tangle.shielded_credits,
            state.config.tangle.chain_id,
        );
        if !valid {
            return Err((
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "invalid SpendAuth signature".to_string(),
                        r#type: "billing_error".to_string(),
                        code: "invalid_spend_auth".to_string(),
                    },
                }),
            ));
        }
    }

    // 2. Proxy to vLLM
    let vllm_response = state
        .vllm
        .chat_completion(&req)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "vLLM request failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("upstream vLLM error: {e}"),
                        r#type: "upstream_error".to_string(),
                        code: "vllm_error".to_string(),
                    },
                }),
            )
        })?;

    // 3. Post-serve billing: authorize spend + claim payment on-chain
    if let Some(spend_auth) = req.spend_auth {
        let actual_cost = state.billing.calculate_cost(
            vllm_response.usage.prompt_tokens,
            vllm_response.usage.completion_tokens,
        );
        let charge_amount = actual_cost.min(
            spend_auth
                .amount
                .parse::<u64>()
                .unwrap_or(0),
        );

        if charge_amount > 0 {
            if let Err(e) = state
                .billing
                .authorize_and_claim(&spend_auth, charge_amount)
                .await
            {
                tracing::warn!(error = %e, "billing claim failed (inference already served)");
            }
        }
    }

    Ok(Json(vllm_response))
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

async fn health_check(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
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
