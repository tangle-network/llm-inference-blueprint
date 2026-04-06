use blueprint_sdk::std::collections::HashMap;
use blueprint_sdk::std::path::PathBuf;
use blueprint_sdk::std::sync::{Arc, RwLock};
use blueprint_sdk::std::time::Duration;

use alloy::primitives::Address;
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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::billing::{self, BillingClient};
use crate::config::OperatorConfig;
use crate::health;
use crate::metrics::{self, RequestGuard};
use crate::vllm::VllmProcess;

/// Nonce key: (commitment, nonce) pair. Prevents replay of SpendAuth signatures.
type NonceKey = (String, u64);

// --- x402 constants ---

const X402_PAYMENT_REQUIRED: &str = "X-Payment-Required";
const X402_PAYMENT_TOKEN: &str = "X-Payment-Token";
const X402_PAYMENT_RECIPIENT: &str = "X-Payment-Recipient";
const X402_PAYMENT_NETWORK: &str = "X-Payment-Network";
const X402_PAYMENT_SIGNATURE: &str = "X-Payment-Signature";

// --- Persistent Nonce Store ---

#[derive(Serialize, Deserialize)]
struct NonceRecord {
    commitment: String,
    nonce: u64,
    expiry: u64,
}

/// Replay-protection nonce store with optional file persistence.
///
/// Without persistence (`nonce_store_path` unset), operator restarts clear all
/// nonces, allowing replay of any unexpired SpendAuth signatures.
pub struct NonceStore {
    nonces: RwLock<HashMap<NonceKey, u64>>,
    path: Option<PathBuf>,
}

impl NonceStore {
    /// Create a new nonce store, loading persisted nonces from disk if path is set.
    pub fn load(path: Option<PathBuf>) -> Self {
        let nonces: HashMap<NonceKey, u64> = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|data| serde_json::from_str::<Vec<NonceRecord>>(&data).ok())
            .map(|records| {
                records
                    .into_iter()
                    .map(|r| ((r.commitment, r.nonce), r.expiry))
                    .collect()
            })
            .unwrap_or_default();

        if path.is_some() {
            tracing::info!(count = nonces.len(), "loaded persisted nonces");
        } else {
            tracing::warn!(
                "nonce_store_path not configured — nonces are in-memory only. \
                 Operator restart will allow replay of unexpired SpendAuth signatures."
            );
        }

        Self {
            nonces: RwLock::new(nonces),
            path,
        }
    }

    /// Evict expired nonces and check if the key is already used.
    /// Returns true if replay detected (nonce already seen).
    pub fn check_replay(&self, key: &NonceKey, tolerance: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut nonces = self.nonces.write().unwrap_or_else(|e| e.into_inner());
        nonces.retain(|_, expiry| now <= expiry.saturating_add(tolerance));
        nonces.contains_key(key)
    }

    /// Record a nonce as used, evict expired entries, and persist to disk.
    pub fn insert(&self, key: NonceKey, expiry: u64, tolerance: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut nonces = self.nonces.write().unwrap_or_else(|e| e.into_inner());
        nonces.retain(|_, exp| now <= exp.saturating_add(tolerance));
        nonces.insert(key, expiry);
        self.persist(&nonces);
    }

    fn persist(&self, nonces: &HashMap<NonceKey, u64>) {
        let Some(ref path) = self.path else { return };
        let records: Vec<NonceRecord> = nonces
            .iter()
            .map(|((commitment, nonce), expiry)| NonceRecord {
                commitment: commitment.clone(),
                nonce: *nonce,
                expiry: *expiry,
            })
            .collect();
        let Ok(data) = serde_json::to_string(&records) else {
            tracing::error!("failed to serialize nonce store");
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() {
            if let Err(e) = std::fs::rename(&tmp, path) {
                tracing::warn!(error = %e, "failed to persist nonce store");
            }
        }
    }
}

// --- Per-Account Concurrency Guard ---

/// RAII guard that decrements the per-account active request count on drop.
struct AccountGuard {
    commitment: String,
    active: Arc<RwLock<HashMap<String, usize>>>,
}

impl Drop for AccountGuard {
    fn drop(&mut self) {
        let mut map = self.active.write().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = map.get_mut(&self.commitment) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                map.remove(&self.commitment);
            }
        }
    }
}

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub vllm: Arc<VllmProcess>,
    pub billing: Arc<BillingClient>,
    pub semaphore: Arc<Semaphore>,
    /// Persistent nonce replay-protection store.
    pub nonce_store: Arc<NonceStore>,
    /// Tracks active requests per credit account for per-account rate limiting.
    pub active_per_account: Arc<RwLock<HashMap<String, usize>>>,
    /// This operator's address, derived from operator_key at startup.
    pub operator_address: Address,
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(
            state.config.server.max_request_body_bytes,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.stream_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
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

/// Deserialize a u64 that may be JSON number or string (JS bigints serialize as strings).
fn de_u64_flex<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    use serde::Deserialize as _;
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum N { Num(u64), Str(String) }
    match N::deserialize(deserializer)? {
        N::Num(n) => Ok(n),
        N::Str(s) => s.parse::<u64>().map_err(serde::de::Error::custom),
    }
}

fn de_u8_flex<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u8, D::Error> {
    use serde::Deserialize as _;
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum N { Num(u8), Str(String) }
    match N::deserialize(deserializer)? {
        N::Num(n) => Ok(n),
        N::Str(s) => s.parse::<u8>().map_err(serde::de::Error::custom),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpendAuthPayload {
    pub commitment: String,
    #[serde(deserialize_with = "de_u64_flex")]
    pub service_id: u64,
    #[serde(deserialize_with = "de_u8_flex")]
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    #[serde(deserialize_with = "de_u64_flex")]
    pub nonce: u64,
    #[serde(deserialize_with = "de_u64_flex")]
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

// --- x402 Payment Required response ---

/// Build a 402 Payment Required response with x402 headers.
/// This tells the client exactly how to pay for the request.
fn x402_payment_required(state: &AppState) -> Response {
    let estimated_amount = state.config.billing.min_charge_amount.max(
        // Default to 1000 input + 512 output tokens at configured rates
        state.billing.calculate_cost(1000, 512),
    );

    let body = serde_json::json!({
        "error": "payment_required",
        "amount": estimated_amount.to_string(),
        "token": state.config.billing.payment_token_address.as_deref().unwrap_or("0x0000000000000000000000000000000000000000"),
        "recipient": format!("{}", state.operator_address),
        "network": state.config.tangle.chain_id.to_string(),
        "accepts": ["spend_auth"],
        "description": "ShieldedCredits SpendAuth required. Include spend_auth in request body or X-Payment-Signature header."
    });

    Response::builder()
        .status(StatusCode::PAYMENT_REQUIRED)
        .header(header::CONTENT_TYPE, "application/json")
        .header(X402_PAYMENT_REQUIRED, estimated_amount.to_string())
        .header(
            X402_PAYMENT_TOKEN,
            state
                .config
                .billing
                .payment_token_address
                .as_deref()
                .unwrap_or("0x0000000000000000000000000000000000000000"),
        )
        .header(
            X402_PAYMENT_RECIPIENT,
            format!("{}", state.operator_address),
        )
        .header(
            X402_PAYMENT_NETWORK,
            state.config.tangle.chain_id.to_string(),
        )
        .body(Body::from(serde_json::to_string(&body).unwrap_or_default()))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build 402 response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}

/// Try to extract a SpendAuth from x402 headers (X-Payment-Signature).
/// The header value is a base64-encoded JSON SpendAuthPayload.
fn extract_x402_spend_auth(headers: &HeaderMap) -> Option<SpendAuthPayload> {
    let header_val = headers.get(X402_PAYMENT_SIGNATURE)?.to_str().ok()?;

    // Try direct JSON first (URL-safe), then base64-encoded JSON
    if let Ok(payload) = serde_json::from_str::<SpendAuthPayload>(header_val) {
        return Some(payload);
    }

    // Try base64-encoded JSON
    use alloy::primitives::hex;
    // The header might be hex-encoded JSON or base64
    if let Ok(decoded) = hex::decode(header_val.strip_prefix("0x").unwrap_or(header_val)) {
        if let Ok(payload) = serde_json::from_slice::<SpendAuthPayload>(&decoded) {
            return Some(payload);
        }
    }

    None
}

// --- Billing settlement helper ---

/// Settle billing after successful inference. Calculates the actual cost,
/// caps at pre-authorized amount, and claims payment on-chain with retries.
///
/// IMPORTANT: The on-chain `claimPayment(bytes32, address)` settles the full
/// pre-authorized amount -- there is no partial settlement in the current
/// ShieldedCredits contract. The `actual_amount` is logged for auditing and
/// will be forwarded on-chain when the contract supports partial claims.
///
/// `preauth_amount` is the pre-parsed u64 from step 3a -- avoids re-parsing.
async fn settle_billing(
    billing: &BillingClient,
    spend_auth: &SpendAuthPayload,
    preauth_amount: u64,
    prompt_tokens: u32,
    completion_tokens: u32,
) {
    let actual_cost = billing.calculate_cost(prompt_tokens, completion_tokens);
    let charge_amount = actual_cost.min(preauth_amount);

    tracing::info!(
        actual_cost,
        preauth_amount,
        charge_amount,
        prompt_tokens,
        completion_tokens,
        "settling billing (contract settles full pre-auth)"
    );

    if charge_amount > 0 {
        if let Err(e) = billing.claim_payment(spend_auth, charge_amount).await {
            tracing::error!(
                error = %e,
                charge_amount,
                "billing settlement failed — revenue lost"
            );
        }
    }
}

// --- Handlers ---

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Response {
    let model_name = req
        .model
        .as_deref()
        .unwrap_or(&state.config.vllm.model);
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

    // 3. Enforce billing requirement -- return x402 402 if missing
    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return x402_payment_required(&state);
    }

    // 4. Verify SpendAuth signature off-chain and check signer identity
    if let Some(ref spend_auth) = req.spend_auth {
        // 4a. Parse and validate the amount field early
        let requested_amount: u64 = match spend_auth.amount.parse() {
            Ok(v) => v,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid spend_auth amount: must be a valid u64 integer".to_string(),
                    "billing_error",
                    "invalid_amount",
                );
            }
        };

        // 4b. Enforce min_charge_amount (gas cost protection)
        let min_charge = state.config.billing.min_charge_amount;
        if min_charge > 0 && requested_amount < min_charge {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend authorization amount ({requested_amount}) is below minimum charge ({min_charge})"
                ),
                "billing_error",
                "below_min_charge",
            );
        }

        // 4c. Enforce max_spend_per_request policy
        let max_spend = state.config.billing.max_spend_per_request;
        if max_spend > 0 && requested_amount > max_spend {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend authorization amount ({requested_amount}) exceeds max_spend_per_request ({max_spend})"
                ),
                "billing_error",
                "exceeds_max_spend",
            );
        }

        // 4d. Validate the operator field matches THIS operator's address
        let spend_operator: Address = match spend_auth.operator.parse() {
            Ok(addr) => addr,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid operator address in spend_auth".to_string(),
                    "billing_error",
                    "invalid_operator",
                );
            }
        };
        if spend_operator != state.operator_address {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend_auth operator ({spend_operator}) does not match this operator ({})",
                    state.operator_address
                ),
                "billing_error",
                "operator_mismatch",
            );
        }

        // 4e. Validate service_id matches this operator's configured service
        if let Some(expected_service_id) = state.config.tangle.service_id {
            if spend_auth.service_id != expected_service_id {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "spend_auth service_id ({}) does not match operator service ({expected_service_id})",
                        spend_auth.service_id
                    ),
                    "billing_error",
                    "service_id_mismatch",
                );
            }
        }

        // 4f. Nonce replay protection
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        if state
            .nonce_store
            .check_replay(&nonce_key, state.config.billing.clock_skew_tolerance_secs)
        {
            return error_response(
                StatusCode::BAD_REQUEST,
                "spend_auth nonce already used (replay detected)".to_string(),
                "billing_error",
                "nonce_replay",
            );
        }

        // 4g. Recover signer from EIP-712 signature
        let recovered_address = match billing::recover_spend_auth_signer(
            spend_auth,
            &state.config.tangle.shielded_credits,
            state.config.tangle.chain_id,
            state.config.billing.clock_skew_tolerance_secs,
        ) {
            Ok(addr) => addr,
            Err(reason) => {
                return error_response(
                    StatusCode::PAYMENT_REQUIRED,
                    format!("invalid SpendAuth signature: {reason}"),
                    "billing_error",
                    "invalid_spend_auth",
                );
            }
        };

        // 4h. Verify the recovered signer matches the account's on-chain spending key
        match state.billing.get_account_info(&spend_auth.commitment).await {
            Ok(account_info) => {
                if recovered_address != account_info.spending_key {
                    return error_response(
                        StatusCode::PAYMENT_REQUIRED,
                        "SpendAuth signer does not match account spending key".to_string(),
                        "billing_error",
                        "signer_mismatch",
                    );
                }

                // 4i. Enforce min_credit_balance policy
                let min_balance = state.config.billing.min_credit_balance;
                if min_balance > 0
                    && account_info.balance < alloy::primitives::U256::from(min_balance)
                {
                    return error_response(
                        StatusCode::PAYMENT_REQUIRED,
                        format!(
                            "credit balance ({}) is below minimum required ({min_balance})",
                            account_info.balance
                        ),
                        "billing_error",
                        "insufficient_balance",
                    );
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to check account info");
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to verify account info".to_string(),
                    "billing_error",
                    "account_check_failed",
                );
            }
        }
    }

    // 5. Pre-authorize billing on-chain BEFORE sending upstream request
    let mut account_guard: Option<AccountGuard> = None;
    if let Some(ref spend_auth) = req.spend_auth {
        // Estimate max cost
        let estimated_prompt_tokens: u32 = req
            .messages
            .iter()
            .map(|m| (m.content.len() as u32) / 4 + 1)
            .sum();
        let estimated_max_cost = state
            .billing
            .calculate_cost(estimated_prompt_tokens, req.max_tokens);
        let requested_amount: u64 = spend_auth.amount.parse().unwrap_or(0);
        // Cap at 1.5x estimated cost
        let preauth_ceiling = estimated_max_cost.saturating_mul(3) / 2;
        if estimated_max_cost > 0 && requested_amount > preauth_ceiling {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "pre-auth amount ({requested_amount}) exceeds 1.5x estimated max cost ({estimated_max_cost}) — \
                     contract settles full pre-auth, reduce amount to avoid overcharging"
                ),
                "billing_error",
                "excessive_preauth",
            );
        }

        // 5a. Per-account concurrency limit
        let max_per_account = state.config.server.max_per_account_requests;
        if max_per_account > 0 {
            let mut map = state
                .active_per_account
                .write()
                .unwrap_or_else(|e| e.into_inner());
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
            account_guard = Some(AccountGuard {
                commitment: spend_auth.commitment.clone(),
                active: Arc::clone(&state.active_per_account),
            });
        }

        // 5b. Check vLLM health before committing gas
        if !state.vllm.is_healthy().await {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference backend is unavailable — billing not initiated".to_string(),
                "upstream_error",
                "vllm_unhealthy",
            );
        }

        if let Err(e) = state.billing.authorize_spend(spend_auth).await {
            tracing::error!(error = %e, "authorizeSpend failed");
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                format!("billing authorization failed: {e}"),
                "billing_error",
                "authorization_failed",
            );
        }

        // Record the nonce as used AFTER successful on-chain authorization
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        state.nonce_store.insert(
            nonce_key,
            spend_auth.expiry,
            state.config.billing.clock_skew_tolerance_secs,
        );
    }

    // 6. Capture pre-parsed amount for settlement
    let preauth_amount: Option<u64> = req
        .spend_auth
        .as_ref()
        .map(|sa| sa.amount.parse::<u64>().unwrap_or(0));

    // 7. Dispatch to streaming or non-streaming path
    if req.stream {
        handle_streaming(
            state,
            req,
            preauth_amount,
            metrics_guard,
            permit,
            account_guard,
        )
        .await
    } else {
        handle_non_streaming(
            state,
            req,
            preauth_amount,
            metrics_guard,
            permit,
            account_guard,
        )
        .await
    }
}

async fn handle_non_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    preauth_amount: Option<u64>,
    mut metrics_guard: RequestGuard,
    _permit: OwnedSemaphorePermit,
    _account_guard: Option<AccountGuard>,
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

    // Post-response settlement
    if let (Some(ref spend_auth), Some(preauth)) = (&req.spend_auth, preauth_amount) {
        settle_billing(
            &state.billing,
            spend_auth,
            preauth,
            vllm_response.usage.prompt_tokens,
            vllm_response.usage.completion_tokens,
        )
        .await;
    }

    Json(vllm_response).into_response()
}

async fn handle_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    preauth_amount: Option<u64>,
    mut metrics_guard: RequestGuard,
    permit: OwnedSemaphorePermit,
    account_guard: Option<AccountGuard>,
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

    let idle_timeout = Duration::from_secs(state.config.server.idle_chunk_timeout_secs);
    let max_line_buf = state.config.server.max_line_buf_bytes;

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

    // Background task: waits for the stream to complete, then settles billing,
    // records metrics, and releases the semaphore permit.
    let max_tokens_for_fallback = req.max_tokens;
    tokio::spawn(async move {
        match usage_rx.await {
            Ok((prompt_tokens, completion_tokens)) => {
                metrics_guard.set_tokens(prompt_tokens, completion_tokens);
                metrics_guard.set_success();

                if let (Some(ref spend_auth), Some(preauth)) =
                    (&spend_auth_for_settlement, preauth_amount)
                {
                    settle_billing(
                        &billing_for_settlement,
                        spend_auth,
                        preauth,
                        prompt_tokens,
                        completion_tokens,
                    )
                    .await;
                }
            }
            Err(_) => {
                tracing::warn!(
                    "streaming response ended without usage data — settling with max_tokens fallback"
                );

                if let (Some(ref spend_auth), Some(preauth)) =
                    (&spend_auth_for_settlement, preauth_amount)
                {
                    settle_billing(
                        &billing_for_settlement,
                        spend_auth,
                        preauth,
                        0,
                        max_tokens_for_fallback,
                    )
                    .await;
                }
            }
        }

        // permit and account_guard held until here, covering the full stream lifetime
        drop(permit);
        drop(account_guard);
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
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.vllm.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

/// Operator info endpoint for discovery. Returns model, pricing, GPU caps, and endpoint.
/// Frontends query this to build operator comparison grids.
async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let gpu_info = health::detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": state.config.vllm.model,
        "pricing": {
            "price_per_input_token": state.config.billing.price_per_input_token,
            "price_per_output_token": state.config.billing.price_per_output_token,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": state.config.gpu.expected_gpu_count,
            "min_vram_mib": state.config.gpu.min_vram_mib,
            "model": state.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.config.server.max_concurrent_requests,
            "max_context_length": state.config.vllm.max_model_len,
        },
        "billing_required": state.config.billing.billing_required,
        "payment_token": state.config.billing.payment_token_address,
    }))
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
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build metrics response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}
