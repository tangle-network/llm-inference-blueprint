use alloy_sol_types::SolValue;
use llm_inference::config::OperatorConfig;
use llm_inference::InferenceServer;

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.vllm.model.clone(),
        gpu_count,
        total_vram,
        gpu_model,
        endpoint,
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(std::env::var("VLLM_OP_CONFIG").ok().as_deref()),
    registration_payload,
    |_| llm_inference::router(),
    |config| InferenceServer { config },
    qos
);
