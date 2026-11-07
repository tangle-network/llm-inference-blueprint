//! OpenAI HTTP adapter for the core serving engine and billing state.
use crate::chat::ChatClient;
use crate::config::OperatorConfig;
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tangle_inference_core::payment::PaymentProof;
use tangle_inference_core::server::{
    authorize_request, backend_unavailable, error_response, metrics_handler, payment_required,
    resolve_payment_proof, settle_request, AuthorizedPayment,
};
use tangle_inference_core::{
    detect_gpus, AppState, CostModel, CostParams, GpuInfo, PerTokenCostModel, RequestGuard,
    SpendAuthPayload,
};
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use unicode_normalization::UnicodeNormalization;
/// Model configuration and request accounting attached to core application state.
pub struct VllmBackend {
    pub config: Arc<OperatorConfig>,
    pub vllm: Arc<ChatClient>,
    pub cost_model: PerTokenCostModel,
    usage: UsageLedger,
}
/// Totals behind `GET /v1/usage`. The per-response `usage` object is unchanged.
struct UsageLedger {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}
impl UsageLedger {
    const fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        }
    }
    fn record(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
    }
}
impl VllmBackend {
    pub fn new(config: Arc<OperatorConfig>, vllm: Arc<ChatClient>) -> Self {
        let cost_model = PerTokenCostModel {
            price_per_input_token: config.vllm.price_per_input_token,
            price_per_output_token: config.vllm.price_per_output_token,
        };
        Self {
            config,
            vllm,
            cost_model,
            usage: UsageLedger::new(),
        }
    }
    /// Count one finished chat, including a stream whose usage chunk arrived.
    pub fn record_usage(&self, prompt_tokens: u32, completion_tokens: u32) {
        self.usage
            .record(u64::from(prompt_tokens), u64::from(completion_tokens));
    }
    pub fn usage_snapshot(&self) -> (u64, u64, u64) {
        (
            self.usage.requests.load(Ordering::Relaxed),
            self.usage.prompt_tokens.load(Ordering::Relaxed),
            self.usage.completion_tokens.load(Ordering::Relaxed),
        )
    }
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
        .route("/v1/usage", get(usage_report))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(stream_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            tangle_inference_core::server::authenticate_endpoint,
        ))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");
    let state_for_drain = state;
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
        let _ = tokio::time::timeout(Duration::from_secs(30), state_for_drain.drain_settlements())
            .await;
    });
    Ok(handle)
}
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
    /// Core payment proof; accepted rails come from the billing configuration.
    #[serde(default)]
    pub payment: Option<PaymentProof>,
    #[serde(flatten)]
    pub options: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(flatten)]
    pub extensions: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(flatten)]
    pub extensions: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
    #[serde(flatten)]
    pub extensions: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(flatten)]
    pub details: serde_json::Map<String, serde_json::Value>,
}
#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
    context_length: u32,
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
fn backend_from(state: &AppState) -> &VllmBackend {
    state
        .backend::<VllmBackend>()
        .expect("AppState backend is VllmBackend (checked in server::start)")
}
fn blocked_minor_content(messages: &[ChatMessage]) -> bool {
    use regex::Regex;
    use std::sync::LazyLock;
    static CHILD: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
        r"\b(?:minor|underage|preteen|prepubescent|child|children|toddler|infant|loli|shota)\b|\b(?:[0-9]|1[0-7])[ -]*(?:years?[ -]old|yo\b|y/o\b)"
    ).expect("minor policy regex")
    });
    static SEXUAL: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
        r"\b(?:porn(?:ography|ographic)?|erotica|erotic|fetish|masturbat\w*)\b|(?:explicit|graphic)\s+(?:sex|sexual)|(?:sex|sexual|nude|naked)\s+(?:scene|story|image|photo|video|roleplay|fantasy|description|depiction)s?\b"
    ).expect("sexual content policy regex")
    });
    static CSAM: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b(?:csam|child\s+porn\w*)\b").expect("CSAM policy regex"));
    let text = serde_json::to_string(
        &messages
            .iter()
            .filter(|message| message.role == "user")
            .map(|message| &message.content)
            .collect::<Vec<_>>(),
    )
    .expect("JSON message values serialize")
    .nfkc()
    .flat_map(char::to_lowercase)
    .filter(|character| !matches!(character, '\u{200b}'..='\u{200f}' | '\u{feff}'))
    .collect::<String>();
    CSAM.is_match(&text) || CHILD.is_match(&text) && SEXUAL.is_match(&text)
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Response {
    if blocked_minor_content(&req.messages) {
        return error_response(
            StatusCode::FORBIDDEN,
            "Sexual content involving minors is prohibited.".into(),
            "invalid_request_error",
            "content_policy_violation",
        );
    }
    let backend = backend_from(&state);
    let model_name = req.model.as_deref().unwrap_or(&backend.config.vllm.model);
    if model_name != backend.config.vllm.model
        && backend.config.vllm.provider_for(model_name).is_none()
    {
        return error_response(
            StatusCode::NOT_FOUND,
            "Unknown model.".into(),
            "invalid_request_error",
            "model_not_found",
        );
    }
    let metrics_guard = RequestGuard::new(model_name);
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
    let cold_deadline = tokio::time::Instant::now() + Duration::from_secs(12);
    let backend_ready = async {
        // A configured remote provider owns its own health API; its HTTP response is checked below.
        backend.config.vllm.provider_for(model_name).is_some()
            || tokio::time::timeout_at(cold_deadline, backend.vllm.is_healthy())
                .await
                .unwrap_or(false)
    };
    // Resolve the payment proof (body payment | spend_auth | x402 header), then
    // gate -> authorize -> (after serving) settle through the shared, rail-
    // agnostic billing flow. Shielded and direct differ only inside those core
    // functions; this handler is identical for both.
    let proof = resolve_payment_proof(&headers, req.spend_auth.take(), req.payment.take());
    if state.billing_config.billing_required && proof.is_none() {
        let estimated = backend.calculate_cost(1000, 512);
        return payment_required(
            &state.billing_config,
            &state.tangle_config,
            state.operator_address,
            estimated,
        );
    }
    let authorized: Option<AuthorizedPayment> = if let Some(proof) = proof {
        // Estimated max cost feeds the shielded pre-auth ceiling. authorize_request
        // validates first, then checks backend health, then commits — so a bad
        // request gets a precise 4xx rather than a health 503.
        let estimated_prompt_tokens: u32 = req
            .messages
            .iter()
            .map(|m| (m.content.to_string().len() as u32) / 4 + 1)
            .sum();
        let estimated_max_cost = backend.calculate_cost(estimated_prompt_tokens, req.max_tokens);
        match authorize_request(&state, proof, estimated_max_cost, backend_ready).await {
            Ok(a) => Some(a),
            Err(resp) => return resp,
        }
    } else {
        if !backend_ready.await {
            return backend_unavailable();
        }
        None
    };
    if req.stream {
        handle_streaming(state, req, authorized, metrics_guard, permit, cold_deadline).await
    } else {
        handle_non_streaming(state, req, authorized, metrics_guard, permit).await
    }
}
fn upstream_failure(error: &anyhow::Error) -> Response {
    if let Some(error) = error.downcast_ref::<reqwest::Error>() {
        if error.is_timeout()
            || error.is_connect()
            || error
                .status()
                .is_some_and(|status| matches!(status.as_u16(), 408 | 429 | 502 | 503 | 504))
        {
            return backend_unavailable();
        }
    }
    error_response(
        StatusCode::BAD_GATEWAY,
        "Backend rejected the request.".into(),
        "upstream_error",
        "vllm_error",
    )
}

async fn handle_non_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    authorized: Option<AuthorizedPayment>,
    mut metrics_guard: RequestGuard,
    _permit: OwnedSemaphorePermit,
) -> Response {
    let backend = backend_from(&state);
    let vllm_response = match backend.vllm.chat_completion(&req).await {
        Ok(r) => r,
        Err(error) => return upstream_failure(&error),
    };
    metrics_guard.set_tokens(
        vllm_response.usage.prompt_tokens,
        vllm_response.usage.completion_tokens,
    );
    metrics_guard.set_success();
    backend.record_usage(
        vllm_response.usage.prompt_tokens,
        vllm_response.usage.completion_tokens,
    );
    // Keep the account guard until asynchronous metered settlement finishes.
    if let Some(auth) = authorized {
        let actual_cost = backend.calculate_cost(
            vllm_response.usage.prompt_tokens,
            vllm_response.usage.completion_tokens,
        );
        let settling = state.clone();
        state.spawn_settlement(
            async move { settle_request(&settling, &auth, Some(actual_cost)).await },
        );
    }
    Json(vllm_response).into_response()
}
async fn handle_streaming(
    state: AppState,
    req: ChatCompletionRequest,
    authorized: Option<AuthorizedPayment>,
    mut metrics_guard: RequestGuard,
    permit: OwnedSemaphorePermit,
    cold_deadline: tokio::time::Instant,
) -> Response {
    let backend = backend_from(&state);
    let expected_model = req.model.as_deref().unwrap_or(&backend.config.vllm.model);
    let opening = tokio::time::timeout_at(cold_deadline, async {
        let upstream = backend
            .vllm
            .chat_completion_stream(&req)
            .await
            .map_err(|error| upstream_failure(&error))?;
        if !upstream
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains("text/event-stream"))
        {
            return Err(error_response(
                StatusCode::BAD_GATEWAY,
                "Backend returned no event stream.".into(),
                "upstream_error",
                "invalid_backend_response",
            ));
        }
        let mut stream = Box::pin(upstream.bytes_stream());
        let mut prefix = Vec::new();
        let mut cursor = 0;
        let mut finished = false;
        let mut usage = false;
        loop {
            let chunk = stream
                .next()
                .await
                .ok_or_else(backend_unavailable)?
                .map_err(|_| backend_unavailable())?;
            if prefix.len() + chunk.len() > state.server_config.max_line_buf_bytes {
                return Err(backend_unavailable());
            }
            prefix.extend_from_slice(&chunk);
            while let Some(end) = prefix[cursor..].iter().position(|byte| *byte == b'\n') {
                let line = &prefix[cursor..cursor + end];
                cursor += end + 1;
                let Some(raw) = line.strip_prefix(b"data:") else {
                    continue;
                };
                if raw.trim_ascii() == b"[DONE]" {
                    continue;
                }
                let value: serde_json::Value =
                    serde_json::from_slice(raw).map_err(|_| backend_unavailable())?;
                if value
                    .get("model")
                    .and_then(|value| value.as_str())
                    .is_some_and(|model| model != expected_model)
                {
                    return Err(error_response(
                        StatusCode::BAD_GATEWAY,
                        "Backend model identity changed.".into(),
                        "upstream_error",
                        "invalid_backend_response",
                    ));
                }
                usage |= ["prompt_tokens", "completion_tokens"]
                    .iter()
                    .all(|key| value["usage"][*key].as_u64().is_some())
                    && value["usage"]["prompt_tokens_details"]["cached_tokens"]
                        .as_u64()
                        .is_some();
                let semantic = value["choices"].as_array().is_some_and(|choices| {
                    choices.iter().any(|choice| {
                        finished |= choice["finish_reason"]
                            .as_str()
                            .is_some_and(|reason| !reason.is_empty());
                        let delta = &choice["delta"];
                        ["content", "reasoning_content", "reasoning"]
                            .iter()
                            .any(|key| delta[*key].as_str().is_some_and(|text| !text.is_empty()))
                            || delta["tool_calls"]
                                .as_array()
                                .is_some_and(|calls| !calls.is_empty())
                    })
                });
                if semantic || finished && usage {
                    return Ok((bytes::Bytes::from(prefix), stream));
                }
            }
        }
    });
    let (prefix, stream) = match opening.await {
        Ok(Ok(stream)) => stream,
        Ok(Err(response)) => return response,
        Err(_) => return backend_unavailable(),
    };
    let byte_stream = futures_util::stream::once(async move { Ok(prefix) }).chain(stream);
    // The authorization (with its per-account guard) moves into the settlement
    // task, so the account slot is held across the whole stream + settlement.
    let authorized_for_settlement = authorized;
    let state_for_settlement = state.clone();
    let (usage_tx, usage_rx) = tokio::sync::oneshot::channel::<(u32, u32)>();
    let idle_timeout = Duration::from_secs(state.server_config.idle_chunk_timeout_secs);
    let max_line_buf = state.server_config.max_line_buf_bytes;
    let timed_stream = tokio_stream::StreamExt::timeout(byte_stream, idle_timeout);
    let proxied_stream = {
        let mut usage_sender = Some(usage_tx);
        let mut line_buf = Vec::new();
        timed_stream.map(move |item| match item {
            Ok(Ok(bytes)) => {
                line_buf.extend_from_slice(&bytes);
                if line_buf.len() > max_line_buf {
                    return Err(std::io::Error::other("oversized SSE line"));
                }
                while let Some(end) = line_buf.iter().position(|byte| *byte == b'\n') {
                    if let Some(raw) = line_buf[..end].strip_prefix(b"data:") {
                        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(raw) {
                            if let (Some(prompt), Some(completion)) = (
                                value["usage"]["prompt_tokens"].as_u64(),
                                value["usage"]["completion_tokens"].as_u64(),
                            ) {
                                if let Some(sender) = usage_sender.take() {
                                    let _ = sender.send((prompt as u32, completion as u32));
                                }
                            }
                        }
                    }
                    line_buf.drain(..=end);
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
        })
    };
    let body = Body::from_stream(proxied_stream);
    state.spawn_settlement(async move {
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
                backend.record_usage(prompt_tokens, completion_tokens);
                if let Some(ref auth) = authorized_for_settlement {
                    let actual_cost = backend.calculate_cost(prompt_tokens, completion_tokens);
                    settle_request(&state_for_settlement, auth, Some(actual_cost)).await;
                }
            }
            Err(_) => {
                // Usage metering is unavailable (the backend ended the stream
                // without a usage chunk): settle with `None` so the shared gate
                // falls back to `claimPayment` of the full pre-auth — the
                // original settlement behavior.
                tracing::warn!(
                    "streaming response ended without usage data — \
                     metering unavailable, settling full pre-auth via claimPayment"
                );
                if let Some(ref auth) = authorized_for_settlement {
                    settle_request(&state_for_settlement, auth, None).await;
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
async fn usage_report(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let (requests, prompt_tokens, completion_tokens) = backend.usage_snapshot();
    Json(serde_json::json!({
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }))
}
async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let backend = backend_from(&state);
    let mut data = vec![ModelInfo {
        id: backend.config.vllm.model.clone(),
        object: "model".to_string(),
        owned_by: "operator".to_string(),
        context_length: backend.config.vllm.max_model_len,
    }];
    for provider in &backend.config.vllm.providers {
        for model in &provider.models {
            if data.iter().any(|info| info.id == *model) {
                continue;
            }
            data.push(ModelInfo {
                id: model.clone(),
                object: "model".to_string(),
                owned_by: provider.name.clone(),
                context_length: backend.config.vllm.max_model_len,
            });
        }
    }
    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}
async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    let runtime = backend.vllm.runtime_info().await.ok();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.vllm.model,
        "tee": tangle_inference_core::tee_status(),
        "runtime": runtime,
        "providers": backend.config.vllm.providers.iter().map(|provider| &provider.name).collect::<Vec<_>>(),
        "pricing": {
            "price_per_input_token": backend.config.vllm.price_per_input_token,
            "price_per_output_token": backend.config.vllm.price_per_output_token,
            "currency": "payment_token",
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
        // Payment surface, so clients can self-configure the ShieldedCredits
        // EIP-712 domain (verifyingContract + chainId) with no hardcoding.
        "shielded_credits": state.tangle_config.shielded_credits,
        "chain_id": state.tangle_config.chain_id,
    }))
}
async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
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
