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

use alloy::primitives::{Address, Bytes, U256};

use crate::billing::{self, BillingClient, RLNProof};
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
        .route("/v1/payment_methods", get(payment_methods))
        .route("/v1/relay", post(relay_transaction))
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

    /// ShieldedCredits spend authorization (Credit Mode billing)
    pub spend_auth: Option<SpendAuthPayload>,

    /// RLN Mode payment proof (alternative to spend_auth)
    pub rln_proof: Option<RLNProof>,
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

// ─── Relay types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct RelayRequest {
    /// "credits" or "rln" — which gateway function to call
    pub mode: String,
    /// Hex-encoded proof data for the shielded withdrawal
    pub proof_data: String,
    /// Public inputs for the withdrawal proof
    pub public_inputs: Vec<String>,
    /// Target contract address (ShieldedGateway)
    pub gateway_address: String,
    /// Credit commitment (bytes32, required for credits mode)
    pub commitment: Option<String>,
    /// Spending key address (required for credits mode)
    pub spending_key: Option<String>,
    /// Fee taken by the relayer from the withdrawal amount (in token base units)
    pub fee: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RelayResponse {
    pub tx_hash: String,
    pub status: String,
}

// ─── Payment methods ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PaymentMethod {
    r#type: String,
    description: String,
}

#[derive(Debug, Serialize)]
struct PaymentMethodsResponse {
    payment_methods: Vec<PaymentMethod>,
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

    // 1. Require payment when billing is enabled (either SpendAuth or RLN proof)
    if state.config.billing.required && req.spend_auth.is_none() && req.rln_proof.is_none() {
        return Err((
            StatusCode::PAYMENT_REQUIRED,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "spend_auth or rln_proof is required".to_string(),
                    r#type: "billing_error".to_string(),
                    code: "missing_payment".to_string(),
                },
            }),
        ));
    }

    // 1b. Verify RLN proof if provided
    if let Some(ref rln_proof) = req.rln_proof {
        let result = state.billing.verify_rln_proof(rln_proof).await.map_err(|e| {
            tracing::warn!(error = %e, "RLN proof verification failed");
            (
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("RLN proof verification failed: {e}"),
                        r#type: "billing_error".to_string(),
                        code: "invalid_rln_proof".to_string(),
                    },
                }),
            )
        })?;

        if !result.is_fresh {
            return Err((
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "RLN nullifier already used".to_string(),
                        r#type: "billing_error".to_string(),
                        code: "nullifier_used".to_string(),
                    },
                }),
            ));
        }
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
    } else if let Some(rln_proof) = req.rln_proof {
        // RLN Mode: record the claim for batch settlement
        state.billing.record_rln_claim(&rln_proof).await;
        tracing::info!(
            epoch = rln_proof.epoch,
            amount = rln_proof.amount,
            "RLN claim recorded for batch settlement"
        );
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

async fn payment_methods(State(state): State<AppState>) -> Json<PaymentMethodsResponse> {
    let mut methods = vec![PaymentMethod {
        r#type: "credit_mode".to_string(),
        description: "EIP-712 SpendAuth".to_string(),
    }];

    if state.config.rln.is_some() {
        methods.push(PaymentMethod {
            r#type: "rln_mode".to_string(),
            description: "RLN ZK proof".to_string(),
        });
    }

    Json(PaymentMethodsResponse {
        payment_methods: methods,
    })
}

async fn relay_transaction(
    State(state): State<AppState>,
    Json(payload): Json<RelayRequest>,
) -> Result<Json<RelayResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate mode
    if payload.mode != "credits" && payload.mode != "rln" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "mode must be 'credits' or 'rln'".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: "invalid_relay_mode".to_string(),
                },
            }),
        ));
    }

    // Validate gateway address
    let gateway: Address = payload.gateway_address.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "invalid gateway_address".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: "invalid_address".to_string(),
                },
            }),
        )
    })?;

    // Validate proof data is valid hex
    let proof_bytes = hex::decode(
        payload
            .proof_data
            .strip_prefix("0x")
            .unwrap_or(&payload.proof_data),
    )
    .map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "invalid proof_data hex".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: "invalid_proof_data".to_string(),
                },
            }),
        )
    })?;

    // Credit Mode relay: shieldedFundCredits(proof, commitment, spendingKey)
    if payload.mode == "credits" {
        let commitment_str = payload.commitment.as_deref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "commitment is required for credits mode".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        code: "missing_commitment".to_string(),
                    },
                }),
            )
        })?;

        let spending_key_str = payload.spending_key.as_deref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "spending_key is required for credits mode".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        code: "missing_spending_key".to_string(),
                    },
                }),
            )
        })?;

        let commitment: alloy::primitives::FixedBytes<32> =
            commitment_str.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "invalid commitment bytes32".to_string(),
                            r#type: "invalid_request_error".to_string(),
                            code: "invalid_commitment".to_string(),
                        },
                    }),
                )
            })?;

        let spending_key: Address = spending_key_str.parse().map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "invalid spending_key address".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        code: "invalid_spending_key".to_string(),
                    },
                }),
            )
        })?;

        let tx_hash = state
            .billing
            .relay_shielded_fund_credits(
                gateway,
                Bytes::from(proof_bytes),
                commitment,
                spending_key,
            )
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "relay transaction submission failed");
                (
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: format!("relay submission failed: {e}"),
                            r#type: "relay_error".to_string(),
                            code: "relay_failed".to_string(),
                        },
                    }),
                )
            })?;

        tracing::info!(
            tx_hash = %tx_hash,
            mode = "credits",
            gateway = %gateway,
            "relay transaction submitted"
        );

        return Ok(Json(RelayResponse {
            tx_hash,
            status: "submitted".to_string(),
        }));
    }

    // RLN Mode relay: same pattern, different target function (not yet wired)
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: "RLN relay not yet implemented; use credits mode".to_string(),
                r#type: "not_implemented".to_string(),
                code: "rln_relay_pending".to_string(),
            },
        }),
    ))
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
        assert!(req.rln_proof.is_none());
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
