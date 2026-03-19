use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use alloy::primitives::Address;

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
    pub concurrency: Arc<Semaphore>,
    pub operator_address: Address,
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

#[derive(Debug, Clone, Deserialize)]
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
    // 0. Enforce concurrency limit
    let _permit = state.concurrency.try_acquire().map_err(|_| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "too many concurrent requests".to_string(),
                    r#type: "rate_limit_error".to_string(),
                    code: "too_many_requests".to_string(),
                },
            }),
        )
    })?;

    // 0b. Reject streaming requests (not yet supported)
    if req.stream {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "streaming is not supported; set stream=false".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: "unsupported_stream".to_string(),
                },
            }),
        ));
    }

    // 1. Require and verify SpendAuth when billing is enabled
    if state.config.billing.required && req.spend_auth.is_none() {
        return Err((
            StatusCode::PAYMENT_REQUIRED,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "spend_auth is required".to_string(),
                    r#type: "billing_error".to_string(),
                    code: "missing_spend_auth".to_string(),
                },
            }),
        ));
    }

    if let Some(ref spend_auth) = req.spend_auth {
        let recovered = billing::verify_spend_auth_signature(
            spend_auth,
            &state.config.tangle.shielded_credits,
            state.config.tangle.chain_id,
            &state.operator_address,
        );
        let recovered_addr = match recovered {
            Some(addr) => addr,
            None => {
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
        };

        // Validate recovered signer against the on-chain spending key
        let spending_key = state
            .billing
            .get_spending_key(&spend_auth.commitment)
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "failed to look up spending key on-chain");
                (
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "failed to verify spending key on-chain".to_string(),
                            r#type: "billing_error".to_string(),
                            code: "spending_key_lookup_failed".to_string(),
                        },
                    }),
                )
            })?;

        if recovered_addr != spending_key {
            tracing::warn!(
                recovered = %recovered_addr,
                expected = %spending_key,
                "SpendAuth signer does not match on-chain spending key"
            );
            return Err((
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "SpendAuth signer does not match on-chain spending key"
                            .to_string(),
                        r#type: "billing_error".to_string(),
                        code: "spending_key_mismatch".to_string(),
                    },
                }),
            ));
        }

        // Enforce billing limits on the authorized amount
        let authorized: alloy::primitives::U256 = spend_auth.amount.parse().map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "invalid spend_auth amount".to_string(),
                        r#type: "billing_error".to_string(),
                        code: "invalid_amount".to_string(),
                    },
                }),
            )
        })?;

        let max_spend = alloy::primitives::U256::from(state.config.billing.max_spend_per_request);
        if authorized > max_spend {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!(
                            "authorized amount exceeds max_spend_per_request ({})",
                            state.config.billing.max_spend_per_request
                        ),
                        r#type: "billing_error".to_string(),
                        code: "amount_too_large".to_string(),
                    },
                }),
            ));
        }

        let min_balance = alloy::primitives::U256::from(state.config.billing.min_credit_balance);
        if authorized < min_balance {
            return Err((
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!(
                            "authorized amount below min_credit_balance ({})",
                            state.config.billing.min_credit_balance
                        ),
                        r#type: "billing_error".to_string(),
                        code: "insufficient_credit".to_string(),
                    },
                }),
            ));
        }
    }

    // 2. Proxy to vLLM
    let vllm_response = state.vllm.chat_completion(&req).await.map_err(|e| {
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

        if actual_cost > 0 {
            if let Err(e) = state
                .billing
                .authorize_and_claim(&spend_auth, actual_cost)
                .await
            {
                // Log at error with full audit context — the operator served inference
                // but failed to collect payment. This requires manual investigation.
                tracing::error!(
                    error = %e,
                    commitment = %spend_auth.commitment,
                    authorized_amount = %spend_auth.amount,
                    actual_cost = actual_cost,
                    nonce = spend_auth.nonce,
                    "billing claim failed after serving inference — operator revenue lost"
                );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_chat_request_minimal() {
        let json = r#"{
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.model.is_none());
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hello");
        assert_eq!(req.max_tokens, 512); // default
        assert!((req.temperature - 0.7).abs() < f32::EPSILON); // default
        assert!(!req.stream);
        assert!(req.spend_auth.is_none());
    }

    #[test]
    fn test_deserialize_chat_request_full() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}
            ],
            "max_tokens": 1024,
            "temperature": 0.5,
            "stream": true,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["END"],
            "spend_auth": {
                "commitment": "0x0000000000000000000000000000000000000000000000000000000000000001",
                "service_id": 1,
                "job_index": 0,
                "amount": "1000",
                "operator": "0x0000000000000000000000000000000000000001",
                "nonce": 42,
                "expiry": 9999999999,
                "signature": "0xdeadbeef"
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("gpt-4"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, 1024);
        assert!((req.temperature - 0.5).abs() < f32::EPSILON);
        assert!(req.stream);
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(req.stop, Some(vec!["END".to_string()]));
        let auth = req.spend_auth.unwrap();
        assert_eq!(auth.service_id, 1);
        assert_eq!(auth.nonce, 42);
    }

    #[test]
    fn test_deserialize_chat_request_missing_messages_fails() {
        let json = r#"{"model": "test"}"#;
        let result = serde_json::from_str::<ChatCompletionRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_chat_request_empty_messages() {
        let json = r#"{"messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
    }

    #[test]
    fn test_chat_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "chatcmpl-123");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_chat_response_roundtrip() {
        let resp = ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp2.usage.prompt_tokens, 1);
        assert_eq!(resp2.usage.completion_tokens, 2);
    }

    #[test]
    fn test_default_values() {
        assert_eq!(default_max_tokens(), 512);
        assert!((default_temperature() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_semaphore_try_acquire_exhaustion() {
        // Verify the concurrency pattern works as expected
        let sem = Arc::new(Semaphore::new(2));
        let _p1 = sem.try_acquire().unwrap();
        let _p2 = sem.try_acquire().unwrap();
        assert!(sem.try_acquire().is_err(), "third acquire should fail");
        drop(_p1);
        assert!(sem.try_acquire().is_ok(), "should succeed after drop");
    }
}
