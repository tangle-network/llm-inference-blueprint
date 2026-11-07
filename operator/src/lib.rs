pub mod config;
pub use tangle_inference_core::qos;
pub mod chat;
pub mod server;

// Re-export shared infrastructure so downstream crates can `use llm_inference::*`.
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, validate_spend_auth,
};
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerTokenCostModel, RequestGuard, SpendAuthPayload,
};
// Re-export metrics module for tests/downstream use.
pub use tangle_inference_core::metrics;
// Alias billing module path for downstream callers that imported
// `llm_inference::billing::BillingClient` before the refactor.
pub use tangle_inference_core::billing;

#[cfg(feature = "chain")]
mod chain;
#[cfg(feature = "chain")]
pub use chain::*;
