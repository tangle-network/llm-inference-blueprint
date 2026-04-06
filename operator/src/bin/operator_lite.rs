//! Minimal operator binary — runs the HTTP server + billing client without
//! BlueprintRunner or Tangle Substrate. Used for E2E tests that exercise
//! the real Rust operator code path (real signature verification, real
//! authorizeSpend/claimPayment on-chain) without needing a full Tangle node.
//!
//! Expects an external "vLLM" backend (which can actually be an Ollama proxy)
//! reachable at `vllm.host:vllm.port`. This binary never spawns vLLM itself —
//! it uses `VllmProcess::connect()` to attach to an already-running backend.
//!
//! Config is loaded via `OperatorConfig::load()` — file path via first CLI
//! arg, env vars override via `VLLM_OP_*` prefix.

use std::sync::Arc;

use std::sync::RwLock;
use tokio::sync::{watch, Semaphore};

use vllm_inference::billing::BillingClient;
use vllm_inference::config::OperatorConfig;
use vllm_inference::server::{self, AppState, NonceStore};
use vllm_inference::vllm::VllmProcess;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_log();

    // Load config from optional file + VLLM_OP_* env vars
    let path = std::env::args().nth(1);
    let config = Arc::new(OperatorConfig::load(path.as_deref())?);
    tracing::info!(
        rpc_url = %config.tangle.rpc_url,
        shielded_credits = %config.tangle.shielded_credits,
        vllm_host = %config.vllm.host,
        vllm_port = config.vllm.port,
        server_port = config.server.port,
        "operator-lite starting"
    );

    // Connect to an already-running "vLLM" backend (Ollama proxy in tests)
    let vllm = Arc::new(VllmProcess::connect(config.clone())?);

    // Build billing client — reads operator_key + shielded_credits from config
    let billing = Arc::new(BillingClient::new(config.clone()).await?);
    let operator_address = billing.operator_address();
    tracing::info!(%operator_address, "billing client ready");

    // Build the request semaphore (0 = unlimited)
    let max_concurrent = config.server.max_concurrent_requests;
    let semaphore = Arc::new(if max_concurrent == 0 {
        Semaphore::new(Semaphore::MAX_PERMITS)
    } else {
        Semaphore::new(max_concurrent)
    });

    // Persistent nonce replay store
    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));

    // App state
    let state = AppState {
        config: config.clone(),
        vllm,
        billing,
        semaphore,
        nonce_store,
        active_per_account: Arc::new(RwLock::new(std::collections::HashMap::new())),
        operator_address,
    };

    // Start the HTTP server
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = server::start(state, shutdown_rx).await?;
    tracing::info!("operator-lite HTTP server running — Ctrl+C to stop");

    // Wait forever (or until signal). tokio::signal::ctrl_c() for graceful stop.
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("received Ctrl+C, shutting down");
        }
        res = handle => {
            tracing::warn!(?res, "HTTP server task ended");
        }
    }

    Ok(())
}
