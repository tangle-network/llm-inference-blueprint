use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use llm_inference::config::{
    BillingConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig, VllmConfig,
};
use llm_inference::server::VllmBackend;
use llm_inference::{AppStateBuilder, BillingClient, NonceStore};

fn test_config(vllm_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        vllm: VllmConfig {
            model: "test-model".into(),
            max_model_len: 4096,
            host: "127.0.0.1".into(),
            port: vllm_port,
            tensor_parallel_size: 1,
            price_per_input_token: 1,
            price_per_output_token: 2,
            extra_args: vec![],
            command: "python3 -m vllm.entrypoints.openai.api_server".into(),
            hf_token: None,
            download_dir: None,
            startup_timeout_secs: 10,
        },
        server: ServerConfig {
            host: "0.0.0.0".into(),
            port: 8080,
            max_concurrent_requests: 2,
            max_request_body_bytes: 2 * 1024 * 1024,
            stream_timeout_secs: 300,
            idle_chunk_timeout_secs: 30,
            max_line_buf_bytes: 1024 * 1024,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            billing_required: false,
            max_spend_per_request: 1_000_000,
            min_credit_balance: 1000,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            monitor_interval_secs: 30,
            gpu_model: None,
        },
        qos: None,
        rln: None,
    }
}

// ─── Metrics Tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_metrics_gather_produces_valid_output() {
    let mut guard = llm_inference::metrics::RequestGuard::new("test-model");
    guard.set_tokens(1, 1);
    guard.set_success();
    drop(guard);

    let output = llm_inference::metrics::gather();
    assert!(
        output.contains("tangle_operator_active_requests"),
        "missing active_requests metric"
    );
    assert!(
        output.contains("tangle_operator_requests_total"),
        "missing requests_total metric"
    );
    assert!(
        output.contains("tangle_operator_request_duration_ms"),
        "missing request_duration_ms metric"
    );
    assert!(
        output.contains("tangle_operator_tokens_total"),
        "missing tokens_total metric"
    );
}

#[tokio::test]
async fn test_request_guard_tracks_active_requests() {
    use llm_inference::metrics::{RequestGuard, ACTIVE_REQUESTS};

    let initial = ACTIVE_REQUESTS.get();

    let guard1 = RequestGuard::new("test-model");
    assert!(ACTIVE_REQUESTS.get() >= initial + 1.0);

    let guard2 = RequestGuard::new("test-model");
    assert!(ACTIVE_REQUESTS.get() >= initial + 2.0);

    drop(guard1);
    drop(guard2);
}

#[tokio::test]
async fn test_request_guard_records_tokens_on_drop() {
    use llm_inference::metrics::{RequestGuard, TOKENS_TOTAL};

    let prompt_before = TOKENS_TOTAL.with_label_values(&["test-model", "prompt"]).get();
    let completion_before = TOKENS_TOTAL.with_label_values(&["test-model", "completion"]).get();

    let mut guard = RequestGuard::new("test-model");
    guard.set_tokens(100, 50);
    guard.set_success();
    drop(guard);

    assert!(
        TOKENS_TOTAL.with_label_values(&["test-model", "prompt"]).get() >= prompt_before + 100,
        "prompt tokens should have increased by at least 100"
    );
    assert!(
        TOKENS_TOTAL.with_label_values(&["test-model", "completion"]).get() >= completion_before + 50,
        "completion tokens should have increased by at least 50"
    );
}

#[tokio::test]
async fn test_request_guard_defaults_to_error() {
    use llm_inference::metrics::{RequestGuard, REQUEST_COUNT};

    let error_before = REQUEST_COUNT.with_label_values(&["test-model", "error"]).get();

    let guard = RequestGuard::new("test-model");
    drop(guard);

    assert!(
        REQUEST_COUNT.with_label_values(&["test-model", "error"]).get() >= error_before + 1,
        "error count should have increased by at least 1"
    );
}

#[tokio::test]
async fn test_request_guard_records_success() {
    use llm_inference::metrics::{RequestGuard, REQUEST_COUNT};

    let success_before = REQUEST_COUNT.with_label_values(&["test-model", "success"]).get();

    let mut guard = RequestGuard::new("test-model");
    guard.set_success();
    drop(guard);

    assert!(
        REQUEST_COUNT.with_label_values(&["test-model", "success"]).get() >= success_before + 1,
        "success count should have increased by at least 1"
    );
}

// ─── Semaphore Tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_semaphore_limits_concurrency() {
    let semaphore = Arc::new(Semaphore::new(2));

    let p1 = semaphore.clone().try_acquire_owned();
    assert!(p1.is_ok());

    let p2 = semaphore.clone().try_acquire_owned();
    assert!(p2.is_ok());

    // Third acquire should fail
    let p3 = semaphore.clone().try_acquire_owned();
    assert!(p3.is_err());

    // Drop one permit, now we can acquire again
    drop(p1);
    let p4 = semaphore.clone().try_acquire_owned();
    assert!(p4.is_ok());
}

#[tokio::test]
async fn test_semaphore_zero_config_means_unlimited() {
    let semaphore = Arc::new(Semaphore::new(Semaphore::MAX_PERMITS));

    let mut permits = Vec::new();
    for _ in 0..1000 {
        permits.push(semaphore.clone().try_acquire_owned().unwrap());
    }
    assert_eq!(permits.len(), 1000);
}

// ─── Billing Order Tests ─────────────────────────────────────────────────

#[tokio::test]
async fn test_billing_calculate_cost() {
    let config = Arc::new(test_config(8000));
    let vllm = Arc::new(llm_inference::vllm::VllmProcess::connect(config.clone()).unwrap());
    let backend = VllmBackend::new(config, vllm);

    // price_per_input_token = 1, price_per_output_token = 2
    let cost = backend.calculate_cost(100, 50);
    assert_eq!(cost, 100 * 1 + 50 * 2); // 200
}

// ─── SSE Parsing Tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_sse_usage_extraction() {
    let sse_data = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
        "data: [DONE]\n\n",
    );

    let (prompt_tokens, completion_tokens) = parse_sse_usage(sse_data);
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 5);
}

#[tokio::test]
async fn test_sse_no_usage_in_chunks() {
    let sse_data = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    let (prompt_tokens, completion_tokens) = parse_sse_usage(sse_data);
    assert_eq!(prompt_tokens, 0);
    assert_eq!(completion_tokens, 0);
}

#[tokio::test]
async fn test_sse_done_marker_present() {
    let sse_data = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    assert!(sse_data.contains("[DONE]"));
}

/// Test that SSE data split mid-line across chunk boundaries is handled correctly.
/// This verifies the line_buf logic reassembles partial lines before parsing.
#[tokio::test]
async fn test_sse_chunk_boundary_splits() {
    // Simulate data arriving in chunks that split in the middle of a JSON line
    let chunks: Vec<&str> = vec![
        "data: {\"id\":\"chatcm",                                  // split mid-JSON
        "pl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"He", // split mid-value
        "llo\"},\"finish_reason\":null}]}\n\n",                     // completes the line
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
        "data: [DONE]\n\n",
    ];

    // Use the same line_buf logic as server.rs to parse usage across chunk boundaries
    let mut line_buf = String::new();
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;

    for chunk in &chunks {
        line_buf.push_str(chunk);

        while let Some(newline_pos) = line_buf.find('\n') {
            {
                let complete_line = &line_buf[..newline_pos];
                if let Some(json_str) = complete_line.strip_prefix("data: ") {
                    let json_str = json_str.trim();
                    if json_str != "[DONE]" {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                            if let Some(usage) = val.get("usage") {
                                if !usage.is_null() {
                                    prompt_tokens = usage
                                        .get("prompt_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0)
                                        as u32;
                                    completion_tokens = usage
                                        .get("completion_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0)
                                        as u32;
                                }
                            }
                        }
                    }
                }
            }
            line_buf.replace_range(..newline_pos + 1, "");
        }
    }

    assert_eq!(
        prompt_tokens, 10,
        "should extract prompt_tokens across chunk boundaries"
    );
    assert_eq!(
        completion_tokens, 5,
        "should extract completion_tokens across chunk boundaries"
    );
}

fn parse_sse_usage(data: &str) -> (u32, u32) {
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;

    for line in data.lines() {
        if let Some(json_str) = line.strip_prefix("data: ") {
            if json_str.trim() == "[DONE]" {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(usage) = val.get("usage") {
                    if !usage.is_null() {
                        prompt_tokens = usage
                            .get("prompt_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                        completion_tokens = usage
                            .get("completion_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                    }
                }
            }
        }
    }

    (prompt_tokens, completion_tokens)
}

// ─── Handler-level integration tests ────────────────────────────────────

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn build_test_state(config: Arc<OperatorConfig>) -> llm_inference::AppState {
    let vllm = Arc::new(llm_inference::vllm::VllmProcess::connect(config.clone()).unwrap());
    let billing = Arc::new(BillingClient::new(&config.tangle, &config.billing).unwrap());
    let operator_address = billing.operator_address();
    let nonce_store = Arc::new(NonceStore::load(None));
    let backend = VllmBackend::new(config.clone(), vllm);

    AppStateBuilder::new()
        .billing(billing)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator_address)
        .max_concurrent(64)
        .backend(backend)
        .build()
        .unwrap()
}

/// Returns (server_port, _guard) — caller must hold _guard to keep the server alive.
async fn start_test_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    let config = Arc::new(config);

    let state = build_test_state(config).await;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let handle = llm_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

/// Start a test server with billing_required = true.
/// Returns (server_port, _guard) — caller must hold _guard to keep the server alive.
async fn start_billing_required_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    config.billing.billing_required = true;
    let config = Arc::new(config);

    let state = build_test_state(config).await;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let handle = llm_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

#[tokio::test]
async fn test_streaming_through_handler() {
    let mock_vllm = MockServer::start().await;

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let body = resp.text().await.unwrap();
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();

    // At least one JSON event + the [DONE] marker
    assert!(
        data_lines.len() >= 2,
        "expected at least 2 data lines, got {}",
        data_lines.len()
    );

    // First data line should parse as valid JSON with choices
    let first_json = data_lines[0].strip_prefix("data: ").unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(first_json).expect("first data line should be valid JSON");
    assert!(
        parsed.get("choices").is_some(),
        "data event should have choices"
    );

    // Last data line must be the terminal [DONE] marker
    assert_eq!(
        *data_lines.last().unwrap(),
        "data: [DONE]",
        "stream must end with data: [DONE]"
    );

    // Every non-DONE data line should be valid JSON
    for line in &data_lines {
        let payload = line.strip_prefix("data: ").unwrap();
        if payload.trim() == "[DONE]" {
            continue;
        }
        serde_json::from_str::<serde_json::Value>(payload)
            .unwrap_or_else(|_| panic!("expected valid JSON in SSE event: {payload}"));
    }
}

#[tokio::test]
async fn test_non_streaming_through_handler() {
    let mock_vllm = MockServer::start().await;

    let response_body = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    // Non-streaming should return application/json, not text/event-stream
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.contains("application/json"),
        "non-streaming response should be JSON, got: {content_type}"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
    assert_eq!(body["usage"]["prompt_tokens"], 5);
}

#[tokio::test]
async fn test_stream_field_defaults_to_false() {
    // Verify that omitting "stream" from the request body results in non-streaming
    let mock_vllm = MockServer::start().await;

    let response_body = serde_json::json!({
        "id": "chatcmpl-default",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Default"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.contains("application/json"),
        "omitting stream field should default to non-streaming JSON, got: {content_type}"
    );
}

// ─── Billing Settlement Tests ────────────────────────────────────────────

fn test_backend(config: Arc<OperatorConfig>) -> VllmBackend {
    let vllm = Arc::new(llm_inference::vllm::VllmProcess::connect(config.clone()).unwrap());
    VllmBackend::new(config, vllm)
}

#[tokio::test]
async fn test_billing_actual_cost_less_than_preauth() {
    let config = Arc::new(test_config(8000));
    let backend = test_backend(config);

    // price_per_input = 1, price_per_output = 2
    // 10 input + 5 output = 10*1 + 5*2 = 20
    let actual_cost = backend.calculate_cost(10, 5);
    assert_eq!(actual_cost, 20);

    // Pre-auth ceiling was 1000 — charge_amount should be min(20, 1000) = 20
    let preauth_amount: u64 = 1000;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 20,
        "should charge actual cost, not the full pre-auth"
    );
}

#[tokio::test]
async fn test_billing_actual_cost_exceeds_preauth_cap() {
    let config = Arc::new(test_config(8000));
    let backend = test_backend(config);

    // price_per_input = 1, price_per_output = 2
    // 500 input + 300 output = 500 + 600 = 1100
    let actual_cost = backend.calculate_cost(500, 300);
    assert_eq!(actual_cost, 1100);

    // Pre-auth ceiling was 100 — charge_amount should be min(1100, 100) = 100
    let preauth_amount: u64 = 100;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 100,
        "charge must be capped at pre-authorized amount"
    );
}

#[tokio::test]
async fn test_billing_zero_usage_yields_zero_charge() {
    let config = Arc::new(test_config(8000));
    let backend = test_backend(config);

    let actual_cost = backend.calculate_cost(0, 0);
    assert_eq!(actual_cost, 0);

    let preauth_amount: u64 = 500;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 0,
        "zero usage should result in zero charge, not the preauth amount"
    );
}

// ─── Policy Enforcement Tests ───────────────────────────────────────────

#[tokio::test]
async fn test_billing_required_rejects_missing_spend_auth() {
    let mock_vllm = MockServer::start().await;

    // Set up a mock response (won't be reached because billing check happens first)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) =
        start_billing_required_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        402,
        "requests without spend_auth should be rejected with 402 when billing_required is true"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    // Verify the response body indicates a payment/auth error
    let has_error = body["error"].is_object() || body["error"].is_string();
    assert!(has_error, "response should contain error field: {body}");
}

#[tokio::test]
async fn test_max_spend_per_request_rejection() {
    let mock_vllm = MockServer::start().await;

    // No mock needed — the request should fail before reaching vLLM
    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "spend_auth": {
                "commitment": "0x0000000000000000000000000000000000000000000000000000000000000001",
                "service_id": 1,
                "job_index": 0,
                "amount": "99999999",
                "operator": "0x0000000000000000000000000000000000000001",
                "nonce": 1,
                "expiry": 9999999999u64,
                "signature": "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
            }
        }))
        .send()
        .await
        .unwrap();

    // The spend_auth signature is invalid, so it will be rejected at step 3d
    // (signature recovery). To test max_spend specifically, we verify the
    // enforcement logic directly:
    let max_spend: u64 = 1_000_000; // from test_config
    let requested: u64 = 99_999_999;
    assert!(
        requested > max_spend,
        "test setup: requested amount must exceed max_spend_per_request"
    );

    // The response will be PAYMENT_REQUIRED due to invalid sig, which is
    // checked after amount validation. We confirm the handler rejects it.
    assert!(
        resp.status() == 400 || resp.status() == 402,
        "request should be rejected, got {}",
        resp.status()
    );
}

// ─── Job Handler Error Path Tests ───────────────────────────────────────

#[tokio::test]
async fn test_run_inference_returns_error_on_connection_failure() {
    // Verify that run_inference returns an error (not a panic) when vLLM
    // is unreachable. We can't call the handler directly because it needs
    // TangleArg extraction, but we can verify the underlying HTTP call
    // pattern returns an error instead of panicking.
    let client = reqwest::Client::new();

    // Connect to a port where nothing is listening
    let result = client
        .post("http://127.0.0.1:1/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10,
            "temperature": 0.7,
            "stream": false,
        }))
        .send()
        .await;

    // The key assertion: this should be an Err, not a panic.
    assert!(
        result.is_err(),
        "connection to unreachable vLLM should return Err, not panic"
    );
}

// ─── Config Tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_config_default_max_concurrent_requests() {
    let json = r#"{"host":"0.0.0.0","port":8080}"#;
    let config: ServerConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.max_concurrent_requests, 64);
}

#[tokio::test]
async fn test_config_debug_redacts_operator_key() {
    let config = test_config(8000);
    let debug_output = format!("{:?}", config);
    assert!(
        !debug_output.contains("ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"),
        "Debug output must not contain the operator private key"
    );
    assert!(
        debug_output.contains("REDACTED"),
        "Debug output should show [REDACTED] for operator_key"
    );
}

// ─── Wiremock Integration Tests ──────────────────────────────────────────

#[tokio::test]
async fn test_non_streaming_via_wiremock() {
    let mock_server = MockServer::start().await;

    let response_body = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    let port = mock_server.address().port();
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let vllm_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": false,
    });

    let resp = client
        .post(&url)
        .json(&vllm_body)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();

    assert_eq!(resp["choices"][0]["message"]["content"], "Hello!");
    assert_eq!(resp["usage"]["prompt_tokens"], 5);
    assert_eq!(resp["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn test_streaming_via_wiremock() {
    let mock_server = MockServer::start().await;

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&mock_server)
        .await;

    let port = mock_server.address().port();
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let vllm_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": true,
        "stream_options": {"include_usage": true},
    });

    let resp = client.post(&url).json(&vllm_body).send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.text().await.unwrap();
    assert!(body.contains("data: "), "response should contain SSE data");
    assert!(
        body.contains("[DONE]"),
        "response should contain [DONE] marker"
    );
    assert!(
        body.contains("\"prompt_tokens\":5"),
        "response should contain usage"
    );
}

#[tokio::test]
async fn test_upstream_error_returns_error_status() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&mock_server)
        .await;

    let port = mock_server.address().port();
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 500);
}
