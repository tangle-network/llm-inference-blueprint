//! Model and provider configuration around the core engine.
use serde::{Deserialize, Serialize};
use tangle_inference_core::qos::QoSConfig;
pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    pub tangle: TangleConfig,
    pub vllm: VllmConfig,
    pub server: ServerConfig,
    pub billing: BillingConfig,
    pub gpu: GpuConfig,
    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
    /// RLN Mode configuration (optional — enables RLN payment path).
    #[serde(default)]
    pub rln: Option<RLNConfig>,
}
/// Model pricing and provider rules around the shared serving configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    #[serde(flatten)]
    pub engine: tangle_inference_core::engine::EngineConfig,
    pub price_per_input_token: u64,
    pub price_per_output_token: u64,
    #[serde(default)]
    pub providers: Vec<InferenceProvider>,
}
impl std::ops::Deref for VllmConfig {
    type Target = tangle_inference_core::engine::EngineConfig;
    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

/// One OpenAI-compatible inference provider.
///
/// `base_url` is the origin plus optional `/v1`, with or without a trailing
/// slash. The operator appends `/chat/completions`. The bearer token is read
/// from `api_key_env` at request time and is not stored in this config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceProvider {
    pub name: String,
    pub base_url: String,
    /// Model ids this provider serves. An empty list matches nothing.
    #[serde(default)]
    pub models: Vec<String>,
    /// Env var that holds the bearer token. Unset sends no Authorization header.
    #[serde(default)]
    pub api_key_env: Option<String>,
}
impl VllmConfig {
    pub fn provider_for(&self, model: &str) -> Option<&InferenceProvider> {
        self.providers
            .iter()
            .find(|provider| provider.models.iter().any(|name| name == model))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLNConfig {
    /// RLNSettlement contract address.
    pub settlement_address: String,
    /// Path to the snarkjs verification key JSON (optional — MVP skips real verification).
    pub verification_key_path: Option<String>,
    /// How often to batch-settle pending RLN claims (seconds).
    #[serde(default = "default_batch_settle_interval")]
    pub batch_settle_interval_secs: u64,
    /// Maximum claims per batch transaction.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}
fn default_batch_settle_interval() -> u64 {
    60
}
fn default_max_batch_size() -> usize {
    64
}
impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();
        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }
        // Env vars override file config. Prefix: VLLM_OP_ (e.g. VLLM_OP_TANGLE__RPC_URL).
        builder = builder.add_source(
            config::Environment::with_prefix("VLLM_OP")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );
        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}
