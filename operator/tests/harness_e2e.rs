//! BlueprintHarness E2E test — full BPM lifecycle with mock vLLM.
//!
//! This test exercises the complete blueprint-manager flow:
//!   1. Boot Anvil with seeded Tangle contracts (LocalTestnet)
//!   2. Wire the Router + TangleLayer into a BlueprintRunner
//!   3. Start a mock vLLM HTTP server (no GPU required)
//!   4. Submit an inference job on-chain
//!   5. Verify the result is returned on-chain
//!   6. Shutdown cleanly
//!
//! Run:
//!   cargo test --test harness_e2e -- --nocapture
//!
//! Note: requires tnt-core contract artifacts (LocalTestnet broadcast).
//! If missing, the test is skipped gracefully.

use alloy_primitives::Bytes;
use alloy_sol_types::SolValue;
use anyhow::{Context, Result};
use axum::{routing::post, Json, Router as HttpRouter};
use blueprint_anvil_testing_utils::{missing_tnt_core_artifacts, BlueprintHarness};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::timeout;
use llm_inference::{init_for_testing, router, InferenceRequest, InferenceResult, INFERENCE_JOB};

const TEST_TIMEOUT: Duration = Duration::from_secs(120);
const MOCK_MODEL: &str = "test-model";

/// Start a mock vLLM server that responds to /v1/chat/completions
/// with a fixed response. Returns the base URL.
async fn start_mock_vllm() -> String {
    let app = HttpRouter::new().route("/v1/chat/completions", post(mock_completions));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind mock vLLM");
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{port}");

    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    // Wait for listener to be ready
    tokio::time::sleep(Duration::from_millis(50)).await;
    base_url
}

async fn mock_completions(Json(body): Json<Value>) -> Json<Value> {
    let prompt = body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_str())
        .unwrap_or("");
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(32);

    let response_text = format!("Mock response to: {prompt}");
    let prompt_tokens = prompt.split_whitespace().count() as u64;
    let completion_tokens = max_tokens.min(response_text.split_whitespace().count() as u64 + 1);

    Json(json!({
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "model": MOCK_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_inference_job_lifecycle() -> Result<()> {
    timeout(TEST_TIMEOUT, async {
        // 1. Start mock vLLM
        let mock_url = start_mock_vllm().await;
        println!("Mock vLLM at {mock_url}");

        // 2. Boot harness (Anvil + seeded contracts + BlueprintRunner)
        let harness = match BlueprintHarness::builder(router())
            .poll_interval(Duration::from_millis(50))
            .with_pre_spawn_hook(move |_env| {
                let url = mock_url.clone();
                async move {
                    // Initialize the job handler statics to point at mock vLLM
                    init_for_testing(&url, MOCK_MODEL);
                    Ok(())
                }
            })
            .spawn()
            .await
        {
            Ok(h) => h,
            Err(err) => {
                if missing_tnt_core_artifacts(&err) {
                    eprintln!("Skipping: tnt-core artifacts not found: {err}");
                    return Ok(());
                }
                return Err(err);
            }
        };

        println!(
            "Harness ready: blueprint={}, service={}",
            harness.blueprint_id(),
            harness.service_id()
        );

        // 3. Verify service is active
        let client = harness.client();
        let service = client.get_service(harness.service_id()).await?;
        println!("Service status: {:?}", service.status);

        // 4. Submit an inference job
        let request = InferenceRequest {
            prompt: "What is 2+2?".to_string(),
            maxTokens: 32,
            temperature: 700, // 0.7
        };
        let payload = request.abi_encode();
        println!("Submitting inference job ({} bytes)...", payload.len());

        let submission = harness
            .submit_job(INFERENCE_JOB, Bytes::from(payload))
            .await
            .context("failed to submit inference job")?;
        println!("Job submitted: call_id={}", submission.call_id);

        // 5. Wait for result
        let output = harness
            .wait_for_job_result(submission)
            .await
            .context("failed to get job result")?;
        println!("Got result ({} bytes)", output.len());

        // 6. Decode and verify
        let result =
            InferenceResult::abi_decode(&output).context("failed to decode InferenceResult")?;
        println!("  text: {}", result.text);
        println!("  promptTokens: {}", result.promptTokens);
        println!("  completionTokens: {}", result.completionTokens);

        assert!(
            result.text.contains("Mock response"),
            "expected mock response, got: {}",
            result.text
        );
        assert!(result.promptTokens > 0, "expected nonzero prompt tokens");

        println!("\n  ✓ Single job lifecycle passed");

        // 7. Submit multiple sequential jobs
        let prompts = ["Hello", "Explain gravity", "Write a haiku"];

        for prompt in &prompts {
            let request = InferenceRequest {
                prompt: prompt.to_string(),
                maxTokens: 16,
                temperature: 1000,
            };
            let submission = harness
                .submit_job(INFERENCE_JOB, Bytes::from(request.abi_encode()))
                .await?;
            let output = harness.wait_for_job_result(submission).await?;
            let result = InferenceResult::abi_decode(&output)?;

            assert!(
                result.text.contains("Mock response"),
                "job for '{prompt}' failed: {}",
                result.text
            );
            println!("  ✓ Job '{prompt}' → {} tokens", result.completionTokens);
        }

        println!("  ✓ Multiple sequential jobs passed");

        // 8. Shutdown
        harness.shutdown().await;
        Ok(())
    })
    .await
    .context("test timed out")?
}

// Note: Only one harness test per file due to OnceLock statics in the
// inference handler. The init_for_testing() call binds the mock URL once
// per process. Additional tests should be added as steps within
// test_inference_job_lifecycle above.
