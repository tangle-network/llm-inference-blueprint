//! Local E2E integration test — real inference via Ollama
//!
//! Prerequisites:
//!   - Ollama installed and running: `ollama serve`
//!   - A small model pulled: `ollama pull qwen2:0.5b`
//!
//! Run:
//!   OLLAMA_E2E=1 cargo test --test local_e2e -- --nocapture
//!
//! This test:
//!   1. Connects to a local Ollama instance (OpenAI-compatible API)
//!   2. Sends a real prompt
//!   3. Verifies a real completion comes back
//!   4. Checks token usage is reported
//!   5. Verifies the response is valid JSON matching the OpenAI schema

use std::env;

fn should_run() -> bool {
    env::var("OLLAMA_E2E").unwrap_or_default() == "1"
}

#[tokio::test]
async fn test_real_inference_via_ollama() {
    if !should_run() {
        eprintln!("Skipping: set OLLAMA_E2E=1 to run");
        return;
    }

    let base_url =
        env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2:0.5b".to_string());

    let client = reqwest::Client::new();

    // ─── Test 1: Basic completion ─────────────────────────────────
    println!("Test 1: Basic completion...");
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": false
    });

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("Failed to connect to Ollama — is it running?");

    assert_eq!(resp.status(), 200, "Expected 200 OK");

    let json: serde_json::Value = resp.json().await.expect("Invalid JSON response");

    // Verify response structure matches OpenAI schema
    assert!(json["id"].is_string(), "Missing id");
    assert_eq!(json["object"], "chat.completion", "Wrong object type");
    assert!(json["choices"].is_array(), "Missing choices array");
    assert!(!json["choices"].as_array().unwrap().is_empty(), "Empty choices");

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("Missing content");
    println!("  Response: {content}");
    assert!(!content.is_empty(), "Empty response content");
    assert!(content.contains('4'), "Expected '4' in response to 2+2");

    // Verify usage is reported
    let usage = &json["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap_or(0) > 0, "No prompt tokens");
    assert!(
        usage["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "No completion tokens"
    );
    println!(
        "  Tokens: {} prompt, {} completion",
        usage["prompt_tokens"], usage["completion_tokens"]
    );
    println!("  ✓ Basic completion works");

    // ─── Test 2: Longer generation ────────────────────────────────
    println!("\nTest 2: Longer generation...");
    let body2 = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Write a haiku about privacy."}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": false
    });

    let resp2 = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body2)
        .send()
        .await
        .expect("Request failed");

    let json2: serde_json::Value = resp2.json().await.expect("Invalid JSON");
    let haiku = json2["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    println!("  Response: {haiku}");
    assert!(haiku.len() > 10, "Haiku too short");
    println!("  ✓ Longer generation works");

    // ─── Test 3: Model listing ────────────────────────────────────
    println!("\nTest 3: Model listing...");
    let resp3 = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .expect("Models request failed");

    let models: serde_json::Value = resp3.json().await.expect("Invalid JSON");
    assert!(models["data"].is_array(), "No models data");
    let model_list = models["data"].as_array().unwrap();
    assert!(!model_list.is_empty(), "No models available");
    println!(
        "  Available models: {}",
        model_list
            .iter()
            .map(|m| m["id"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ✓ Model listing works");

    // ─── Test 4: Error handling ───────────────────────────────────
    println!("\nTest 4: Invalid model error...");
    let body4 = serde_json::json!({
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false
    });

    let resp4 = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body4)
        .send()
        .await
        .expect("Request failed");

    assert_ne!(resp4.status(), 200, "Expected error for nonexistent model");
    println!("  ✓ Invalid model correctly rejected");

    println!("\n═══════════════════════════════════════");
    println!("  ALL LOCAL E2E TESTS PASSED");
    println!("  Backend: Ollama at {base_url}");
    println!("  Model: {model}");
    println!("═══════════════════════════════════════");
}
