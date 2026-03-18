use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Price per input token in tsUSD base units (e.g. 6 decimals: 1 = 0.000001 tsUSD)
    pub price_per_input_token: u64,

    /// Price per output token in tsUSD base units
    pub price_per_output_token: u64,

    /// Maximum amount a single SpendAuth can authorize (anti-abuse)
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account to serve a request
    pub min_credit_balance: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB
    pub min_vram_mib: u32,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
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

fn default_monitor_interval() -> u64 {
    30
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
