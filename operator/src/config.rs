use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration
    pub tangle: TangleConfig,

    /// vLLM subprocess configuration
    pub vllm: VllmConfig,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration
    pub billing: BillingConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// RLN Mode configuration (optional — enables RLN payment path)
    #[serde(default)]
    pub rln: Option<RLNConfig>,
}

// Redact operator_key in Debug output to prevent accidental logging of secrets.
impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("vllm", &self.vllm)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    pub rpc_url: String,

    /// Chain ID
    pub chain_id: u64,

    /// Operator's private key (hex, without 0x prefix)
    /// In production, use a KMS or hardware signer instead.
    pub operator_key: String,

    /// Tangle core contract address
    pub tangle_core: String,

    /// ShieldedCredits contract address
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    pub service_id: Option<u64>,
}

// Redact operator_key in Debug output.
impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    pub model: String,

    /// Maximum context length the model will serve
    pub max_model_len: u32,

    /// Host/port vLLM will listen on internally
    pub host: String,
    pub port: u16,

    /// Number of GPUs for tensor parallelism
    pub tensor_parallel_size: u32,

    /// Additional vLLM CLI args
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Path to the vLLM Python executable. Defaults to "python3 -m vllm.entrypoints.openai.api_server".
    #[serde(default = "default_vllm_command")]
    pub command: String,

    /// HuggingFace token for gated models
    pub hf_token: Option<String>,

    /// Custom model download directory
    pub download_dir: Option<PathBuf>,

    /// Startup timeout in seconds
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// External host to bind
    #[serde(default = "default_host")]
    pub host: String,

    /// External port to bind
    #[serde(default = "default_port")]
    pub port: u16,

    /// Maximum concurrent requests
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,

    /// Maximum request body size in bytes (default 2 MiB)
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Per-request timeout for streaming connections in seconds (default 300)
    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,

    /// Per-chunk idle timeout in seconds for streaming responses (default 30).
    /// If vLLM doesn't send data within this window, the stream is terminated.
    #[serde(default = "default_idle_chunk_timeout_secs")]
    pub idle_chunk_timeout_secs: u64,

    /// Maximum size of the SSE line buffer in bytes (default 1 MiB).
    /// Prevents unbounded memory growth from malformed upstream data.
    #[serde(default = "default_max_line_buf_bytes")]
    pub max_line_buf_bytes: usize,

    /// Maximum concurrent requests per credit account (commitment).
    /// 0 = unlimited (default). Prevents a single account from monopolizing all slots.
    #[serde(default)]
    pub max_per_account_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing is required for HTTP requests.
    /// When true, requests without a spend_auth are rejected.
    #[serde(default = "default_billing_required")]
    pub required: bool,

    /// Price per input token in tsUSD base units (e.g. 6 decimals: 1 = 0.000001 tsUSD)
    pub price_per_input_token: u64,

    /// Price per output token in tsUSD base units
    pub price_per_output_token: u64,

    /// Maximum amount a single SpendAuth can authorize (anti-abuse)
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account to serve a request
    pub min_credit_balance: u64,

    /// Whether billing (spend_auth) is required on every request.
    /// When true, requests without spend_auth are rejected with 402.
    #[serde(default = "default_billing_required")]
    pub billing_required: bool,

    /// Minimum charge amount per request (gas cost protection).
    /// Requests whose pre-authorized amount is below this are rejected
    /// to prevent operators from losing money on gas fees.
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls.
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks.
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Maximum gas price in gwei the operator is willing to pay for billing txs.
    /// If the current gas price exceeds this, billing transactions are deferred.
    /// 0 = no cap (default).
    #[serde(default)]
    pub max_gas_price_gwei: u64,

    /// Path to persist used nonces across restarts (replay protection).
    /// Defaults to `data/nonces.json` in the working directory.
    /// Without persistence, nonces are lost on restart, allowing replay
    /// of unexpired SpendAuth signatures.
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<PathBuf>,

    /// ERC-20 token address for x402 payment (e.g. tsUSD).
    /// Included in 402 Payment Required responses so clients know which token to use.
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB
    pub min_vram_mib: u32,

    /// GPU model name (e.g. "NVIDIA A100", "RTX 4090") for on-chain registration
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLNConfig {
    /// RLNSettlement contract address
    pub settlement_address: String,

    /// Path to the snarkjs verification key JSON (optional — MVP skips real verification)
    pub verification_key_path: Option<String>,

    /// How often to batch-settle pending RLN claims (seconds)
    #[serde(default = "default_batch_settle_interval")]
    pub batch_settle_interval_secs: u64,

    /// Maximum claims per batch transaction
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

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_concurrent() -> usize {
    64
}

fn default_billing_required() -> bool {
    true
}

fn default_monitor_interval() -> u64 {
    30
}

fn default_max_request_body_bytes() -> usize {
    16 * 1024 * 1024 // 16 MiB
}

fn default_stream_timeout_secs() -> u64 {
    300
}

fn default_idle_chunk_timeout_secs() -> u64 {
    30
}

fn default_max_line_buf_bytes() -> usize {
    1024 * 1024 // 1 MiB
}

fn default_claim_max_retries() -> u32 {
    3
}

fn default_clock_skew_tolerance() -> u64 {
    30
}

fn default_nonce_store_path() -> Option<PathBuf> {
    Some(PathBuf::from("data/nonces.json"))
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: VLLM_OP_ (e.g. VLLM_OP_TANGLE__RPC_URL)
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
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "vllm": {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_model_len": 8192,
                "host": "127.0.0.1",
                "port": 8000,
                "tensor_parallel_size": 1
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "price_per_input_token": 1,
                "price_per_output_token": 2,
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
        assert_eq!(cfg.billing.price_per_input_token, 1);
        assert_eq!(cfg.billing.price_per_output_token, 2);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_rln_config_optional() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert!(cfg.rln.is_none(), "RLN config should be None by default");
    }

    #[test]
    fn test_rln_config_present() {
        let json = r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xdeadbeef",
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "vllm": {
                "model": "test",
                "max_model_len": 4096,
                "host": "127.0.0.1",
                "port": 8000,
                "tensor_parallel_size": 1
            },
            "server": {},
            "billing": {
                "price_per_input_token": 1,
                "price_per_output_token": 2,
                "max_spend_per_request": 1000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 16000
            },
            "rln": {
                "settlement_address": "0x0000000000000000000000000000000000000003",
                "batch_settle_interval_secs": 120,
                "max_batch_size": 32
            }
        }"#;
        let cfg: OperatorConfig = serde_json::from_str(json).unwrap();
        let rln = cfg.rln.unwrap();
        assert_eq!(
            rln.settlement_address,
            "0x0000000000000000000000000000000000000003"
        );
        assert_eq!(rln.batch_settle_interval_secs, 120);
        assert_eq!(rln.max_batch_size, 32);
        assert!(rln.verification_key_path.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        // ServerConfig defaults
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        // VllmConfig defaults
        assert_eq!(
            cfg.vllm.command,
            "python3 -m vllm.entrypoints.openai.api_server"
        );
        assert_eq!(cfg.vllm.startup_timeout_secs, 300);
        assert!(cfg.vllm.extra_args.is_empty());
        // GpuConfig defaults
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_load_from_file() {
        let cfg = OperatorConfig::load(Some("../deploy/config.example")).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.vllm.model, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(cfg.billing.price_per_output_token, 2);
    }

    #[test]
    fn test_roundtrip_serialize() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: OperatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.tangle.chain_id, cfg2.tangle.chain_id);
        assert_eq!(cfg.vllm.model, cfg2.vllm.model);
        assert_eq!(cfg.server.port, cfg2.server.port);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_optional_fields() {
        let json = r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xdeadbeef",
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": 42
            },
            "vllm": {
                "model": "test-model",
                "max_model_len": 4096,
                "host": "127.0.0.1",
                "port": 9000,
                "tensor_parallel_size": 2,
                "hf_token": "hf_secret",
                "download_dir": "/tmp/models"
            },
            "server": {},
            "billing": {
                "price_per_input_token": 5,
                "price_per_output_token": 10,
                "max_spend_per_request": 500000,
                "min_credit_balance": 500
            },
            "gpu": {
                "expected_gpu_count": 2,
                "min_vram_mib": 24000
            }
        }"#;
        let cfg: OperatorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.tangle.service_id, Some(42));
        assert_eq!(cfg.vllm.hf_token.as_deref(), Some("hf_secret"));
        assert_eq!(cfg.vllm.download_dir, Some(PathBuf::from("/tmp/models")));
    }
}
