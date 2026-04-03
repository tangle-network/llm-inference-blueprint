//! QoS heartbeat — periodically submits operator metrics to the Tangle chain.
//!
//! Standalone implementation (no blueprint-qos dependency). Sends a lightweight
//! heartbeat transaction containing on-chain metrics from metrics::on_chain_metrics().
//!
//! The heartbeat serves two purposes:
//! 1. Liveness proof — the chain knows this operator is alive
//! 2. Metric submission — latency, throughput, error rates feed into reputation

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use alloy::{
    network::EthereumWallet,
    primitives::Address,
    providers::{Provider, ProviderBuilder},
    signers::local::PrivateKeySigner,
    sol,
};

use crate::config::OperatorConfig;
use crate::metrics;

// On-chain heartbeat contract binding.
// The IOperatorStatusRegistry.submitHeartbeat call is the same one used by
// blueprint-qos, but we call it directly to avoid the dependency.
sol! {
    #[sol(rpc)]
    interface IOperatorStatusRegistry {
        struct MetricPair {
            string key;
            uint64 value;
        }

        function submitHeartbeat(
            uint64 serviceId,
            uint64 blueprintId,
            uint64 blockNumber,
            MetricPair[] calldata metrics
        ) external;
    }
}

/// QoS configuration embedded in OperatorConfig.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QoSConfig {
    /// Heartbeat interval in seconds. 0 = disabled.
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,

    /// On-chain address of the IOperatorStatusRegistry contract.
    #[serde(default)]
    pub status_registry_address: Option<String>,
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 0,
            status_registry_address: None,
        }
    }
}

fn default_heartbeat_interval() -> u64 {
    0
}

/// Start the QoS heartbeat loop as a background task.
/// Returns a JoinHandle that runs until the process shuts down.
pub async fn start_heartbeat(
    config: Arc<OperatorConfig>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let qos = config
        .qos
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("qos config missing"))?;

    let interval_secs = qos.heartbeat_interval_secs;
    if interval_secs == 0 {
        anyhow::bail!("heartbeat disabled (interval = 0)");
    }

    let registry_addr: Address = qos
        .status_registry_address
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("status_registry_address not configured"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid status_registry_address: {e}"))?;

    let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
    let wallet = EthereumWallet::from(signer);
    let rpc_url: reqwest::Url = config.tangle.rpc_url.parse()?;
    let service_id = config
        .tangle
        .service_id
        .ok_or_else(|| anyhow::anyhow!("service_id required for QoS heartbeat"))?;
    let blueprint_id = config.tangle.blueprint_id;

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        // Skip the first immediate tick — let the operator warm up
        interval.tick().await;

        loop {
            interval.tick().await;

            match send_heartbeat(
                &wallet,
                &rpc_url,
                registry_addr,
                service_id,
                blueprint_id,
            )
            .await
            {
                Ok(()) => {
                    metrics::HEARTBEATS_SENT.inc();
                    tracing::debug!(service_id, blueprint_id, "heartbeat submitted");
                }
                Err(e) => {
                    metrics::HEARTBEATS_FAILED.inc();
                    tracing::warn!(
                        error = %e,
                        service_id,
                        blueprint_id,
                        "heartbeat submission failed"
                    );
                }
            }
        }
    });

    Ok(handle)
}

async fn send_heartbeat(
    wallet: &EthereumWallet,
    rpc_url: &reqwest::Url,
    registry_addr: Address,
    service_id: u64,
    blueprint_id: u64,
) -> anyhow::Result<()> {
    let provider = ProviderBuilder::new()
        .wallet(wallet.clone())
        .connect_http(rpc_url.clone());

    let block_number = provider.get_block_number().await?;

    let chain_metrics = metrics::on_chain_metrics();
    let metric_pairs: Vec<IOperatorStatusRegistry::MetricPair> = chain_metrics
        .into_iter()
        .map(|(key, value)| IOperatorStatusRegistry::MetricPair { key, value })
        .collect();

    let registry = IOperatorStatusRegistry::new(registry_addr, &provider);
    let call = registry.submitHeartbeat(service_id, blueprint_id, block_number, metric_pairs);

    let tx_hash = call.send().await?.watch().await?;
    tracing::trace!(?tx_hash, "heartbeat tx confirmed");

    Ok(())
}
