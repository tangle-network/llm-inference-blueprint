# RFP: GPU Support for Tangle Blueprints

## Problem

Blueprints that require GPU (inference, training, rendering) have no standardized way to:
1. Declare GPU requirements in the blueprint definition
2. Provision GPU instances automatically
3. Verify GPU availability at operator registration
4. Match service requests to operators with compatible GPUs

The `blueprint-remote-providers` crate has `ResourceSpec.gpu_count` and AWS GPU instance mapping (p4d, p3, g4dn) but no blueprint has used it yet.

---

## Proposal 1: Service Instance Level (Recommended — smallest change)

**Where:** In the service request config, the customer specifies GPU requirements. The BSM validates at `onRequest`.

**How it works:**
```
Customer: requestService(blueprintId, operators, config={model: "Llama-3.1-70B", gpu: {count: 2, vram_gb: 80, type: "A100"}})
BSM.onRequest: validates requested operators have declared compatible GPUs
Operator: reads config at onServiceInitialized, provisions GPU instance via remote-providers
```

**Changes needed:**
- BSM contract: add GPU validation in `onRequest` (check operator's registered capabilities)
- Operator binary: read GPU config from service config bytes, use `blueprint-remote-providers` to provision
- SDK: add GPU fields to service request config type

**Pros:** No protocol changes. Works with existing blueprint/operator/service hierarchy. Operators self-declare GPU capabilities at registration.

**Cons:** No protocol-level GPU verification. Operator could lie about GPU availability (mitigated by staking/slashing).

---

## Proposal 2: Blueprint Level (Medium change)

**Where:** The blueprint definition itself declares "this blueprint requires GPU." All operators registering must prove GPU capability.

**How it works:**
```
Developer: createBlueprint(definition={..., resourceRequirements: {gpu: {minVram: 24, minCount: 1}}})
Operator: registerOperator(blueprintId, pubkey, rpc, registrationInputs={gpuProof: nvidia-smi output hash})
BSM.onRegister: validates GPU proof meets blueprint requirements
```

**Changes needed:**
- `Types.BlueprintDefinition`: add `resourceRequirements` struct with GPU fields
- `BlueprintServiceManagerBase`: add `getResourceRequirements()` hook
- BSM contracts: validate GPU at registration time
- `cargo tangle blueprint create`: add GPU requirement flags

**Pros:** Blueprint-level enforcement. Operators can't register without meeting GPU requirements.

**Cons:** Requires tnt-core protocol changes (new fields in `BlueprintDefinition`). Rigid — can't adjust GPU requirements per service instance.

---

## Proposal 3: New Execution Target (Large change, like TEE)

**Where:** GPU becomes a first-class execution target alongside TEE, like `ConfidentialityPolicy` but for compute type.

**How it works:**
```
enum ExecutionTarget {
    Any,          // No GPU required
    GpuRequired,  // Must run on GPU
    TeeRequired,  // Must run in TEE
    GpuInTee,     // GPU inside TEE (future: confidential computing)
}

// In service request:
requestService(blueprintId, operators, config, callers, ttl, token, amount, ExecutionTarget.GpuRequired)
```

**Changes needed:**
- `Types.sol`: add `ExecutionTarget` enum
- Tangle core: add execution target to service request + approval flow
- Blueprint SDK: add GPU target support in `BlueprintRunner`
- `blueprint-remote-providers`: add GPU provisioning as a first-class path
- New crate: `blueprint-remote-providers-gpu` with:
  - NVIDIA GPU detection (nvidia-smi parsing)
  - AMD GPU detection (rocm-smi parsing)
  - vLLM/Ollama lifecycle management
  - GPU health monitoring
  - Model download + caching

**Pros:** Protocol-native GPU support. Composable with TEE. Future-proof for GPU-in-TEE confidential inference.

**Cons:** Largest protocol change. Requires tnt-core modifications. Overkill for v1.

---

## Recommendation

**Start with Proposal 1** (service instance level). It works today with zero protocol changes. The vLLM blueprint already has GPU validation in the BSM (`onRegister` checks VRAM and GPU count). What's missing is:

1. Wire operator binary to read model from service config (not static config file)
2. Use `blueprint-remote-providers` to auto-provision GPU instances when needed
3. Add GPU resource fields to the pricing TOML

**Then evolve to Proposal 2** when multiple GPU blueprints exist and we want protocol-level enforcement.

**Save Proposal 3** for when GPU-in-TEE confidential inference is a real requirement.

---

## Pricing TOML for vLLM Blueprint

```toml
# config/pricing.toml
[blueprint]
name = "vllm-inference"

[resources.gpu]
type = "A100"           # or "H100", "4090", etc.
count = 1
vram_gb = 80

[pricing.subscription]
base_rate_per_month = "100000000000000000000"  # 100 tokens/month for GPU hosting

[pricing.per_token]
input_token_price = "1000000000000"    # 0.000001 tokens per input token
output_token_price = "3000000000000"   # 0.000003 tokens per output token

[pricing.models.llama-3-1-70b]
min_vram_gb = 40
context_length = 8192
input_multiplier = 1.0
output_multiplier = 1.0

[pricing.models.llama-3-1-8b]
min_vram_gb = 8
context_length = 4096
input_multiplier = 0.3
output_multiplier = 0.3

[pricing.models.qwen2-0-5b]
min_vram_gb = 2
context_length = 2048
input_multiplier = 0.1
output_multiplier = 0.1
```
