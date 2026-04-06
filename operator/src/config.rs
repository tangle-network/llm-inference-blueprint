//! vLLM-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use blueprint_sdk::std::path::PathBuf;
use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// vLLM subprocess + per-token pricing configuration (vllm-specific).
    pub vllm: VllmConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,

    /// RLN Mode configuration (optional — enables RLN payment path).
    #[serde(default)]
    pub rln: Option<RLNConfig>,
}

/// vLLM subprocess + pricing config. This is the only truly vllm-specific
/// config section — everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct").
    pub model: String,

    /// Maximum context length the model will serve.
    pub max_model_len: u32,

    /// Host/port vLLM will listen on internally.
    pub host: String,
    pub port: u16,

    /// Number of GPUs for tensor parallelism.
    pub tensor_parallel_size: u32,

    /// Price per input token in base token units.
    pub price_per_input_token: u64,

    /// Price per output token in base token units.
    pub price_per_output_token: u64,

    /// Additional vLLM CLI args.
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Path to the vLLM Python executable.
    #[serde(default = "default_vllm_command")]
    pub command: String,

    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Custom model download directory.
    pub download_dir: Option<PathBuf>,

    /// Startup timeout in seconds.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
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

fn default_vllm_command() -> String {
    "python3 -m vllm.entrypoints.openai.api_server".to_string()
}

fn default_startup_timeout() -> u64 {
    300
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
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "vllm": {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_model_len": 8192,
                "host": "127.0.0.1",
                "port": 8000,
                "tensor_parallel_size": 1,
                "price_per_input_token": 1,
                "price_per_output_token": 2
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "max_spend_per_request": 1000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 16000
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.vllm.model, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(cfg.vllm.port, 8000);
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.vllm.price_per_input_token, 1);
        assert_eq!(cfg.vllm.price_per_output_token, 2);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_rln_config_optional() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert!(cfg.rln.is_none(), "RLN config should be None by default");
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(
            cfg.vllm.command,
            "python3 -m vllm.entrypoints.openai.api_server"
        );
        assert_eq!(cfg.vllm.startup_timeout_secs, 300);
        assert!(cfg.vllm.extra_args.is_empty());
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_load_from_file() {
        let cfg = OperatorConfig::load(Some("../deploy/config.example")).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.vllm.model, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(cfg.vllm.price_per_output_token, 2);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
