//! HTTP operator with core-owned engine and billing.

use std::sync::Arc;

use tokio::sync::watch;

use llm_inference::chat::ChatClient;
use llm_inference::config::OperatorConfig;
use llm_inference::server::{self, VllmBackend};
use llm_inference::{AppStateBuilder, BillingClient, NonceStore};

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_log();

    let path = std::env::args().nth(1);
    let config = Arc::new(OperatorConfig::load(path.as_deref())?);
    tracing::info!(
        vllm_host = %config.vllm.host,
        vllm_port = config.vllm.port,
        server_port = config.server.port,
        "operator-lite starting"
    );

    let managed = !config.vllm.external
        && (config.vllm.base_family.is_some()
            || config.vllm.engine.engine == tangle_inference_core::engine::EngineKind::Sglang);
    let vllm = Arc::new(if managed {
        ChatClient::spawn(config.clone()).await?
    } else {
        ChatClient::connect(config.clone())?
    });

    let billing = Arc::new(BillingClient::new(&config.tangle, &config.billing)?);
    let operator_address = billing.operator_address();
    tracing::info!(%operator_address, "billing client ready");

    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));

    let backend = VllmBackend::new(config.clone(), vllm.clone());
    let state = AppStateBuilder::new()
        .billing(billing)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator_address)
        .backend(backend)
        .build()?;

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = server::start(state, shutdown_rx).await?;
    tracing::info!("operator-lite HTTP server running — Ctrl+C to stop");

    let outcome: anyhow::Result<()> = tokio::select! {
        result = async {
            if managed { vllm.wait_ready().await?; }
            vllm.supervise(shutdown_tx.subscribe()).await
        } => result,
        result = tokio::signal::ctrl_c() => result.map_err(Into::into),
        result = handle => match result {
            Ok(()) => Err(anyhow::anyhow!("HTTP server stopped")),
            Err(error) => Err(error.into()),
        },
    };
    let _ = shutdown_tx.send(true);
    vllm.shutdown().await;
    outcome
}
