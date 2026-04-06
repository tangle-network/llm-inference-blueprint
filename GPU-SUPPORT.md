# GPU Support Architecture

GPU provisioning and validation operates at two layers: the Blueprint Manager (infrastructure) and the blueprint itself (application).

## Layer 1: Blueprint Manager (Infrastructure)

The manager provisions GPU hardware before the blueprint binary starts. No blueprint code is involved.

### On-chain declaration

Blueprint definitions include GPU requirements in `profilingData` JSON on `BlueprintDefinition.metadata`:

```json
{
  "execution_profile": {
    "gpu": {
      "policy": "required",
      "min_count": 1,
      "min_vram_gb": 16
    }
  }
}
```

Policies:
- **Required** — hard constraint, service won't activate without GPU
- **Preferred** — soft constraint, CPU fallback allowed
- **None** — default

### Manager resolution flow

```
Tangle ServiceActivated event
  → OnChainMetadataProvider::resolve_service()
  → resolve_execution_policies()
    → parses profilingData JSON → GpuRequirements { policy, min_count, min_vram_gb }
  → ensure_service_running()
    → apply_gpu_limits(&gpu_requirements, &mut limits)
    → ResourceLimits { gpu_count, gpu_policy, gpu_min_vram_gb }
    → passed to source handler spawn()
```

### Runtime enforcement

**Kubernetes (container runtime):**

```
GpuPolicy::Required →
  Pod spec: resources.limits["nvidia.com/gpu"] = count
  Node selector: gpu.tangle.tools/enabled = "true"
  Node selector: gpu.tangle.tools/min-vram-gb = "<value>"

GpuPolicy::Preferred →
  Node affinity: preferredDuringSchedulingIgnoredDuringExecution
  Weight 100 for GPU nodes, allows CPU-only fallback
```

**Remote cloud providers (`blueprint-remote-providers`):**

```
GPU workload → CloudProvider::GCP or CloudProvider::AWS
  → CloudProvisioner::provision_with_requirements(ResourceSpec::with_gpu(count))
  → AWS: p4d.24xlarge (A100), g4dn.xlarge (T4)
  → GCP: a2-highgpu (A100), n1-standard + T4
  → TTL-based auto-cleanup (default 1 hour)
```

**Provider selection heuristic:**
- GPU required → GCP (preferred) or AWS
- TEE required → AWS, GCP, or Azure
- High CPU (>8 cores) → Vultr
- High memory (>32GB) → AWS
- Standard → DigitalOcean

## Layer 2: Blueprint (Application)

The blueprint runs on hardware the manager already provisioned. It validates and uses the GPU but does not provision it.

### BSM on-chain validation (InferenceBSM.sol)

At operator registration, the BSM validates GPU capability:

```solidity
function onRegister(address operator, bytes calldata registrationInputs)
    external payable override onlyFromTangle
{
    // Decode: (model, gpuCount, totalVramMib, gpuModel, endpoint)
    if (totalVramMib < mc.minGpuVramMib) {
        revert InsufficientGpuCapability(mc.minGpuVramMib, totalVramMib);
    }
    operatorCaps[operator] = OperatorCapabilities({
        model, gpuCount, totalVramMib, gpuModel, endpoint, active: true
    });
}
```

### Operator GPU detection (health.rs)

At startup, the operator detects available GPUs via `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` and exposes results at `GET /health/gpu`.

### Model-GPU mapping

Each model has a minimum VRAM requirement set via `configureModel()`:

```solidity
configureModel("meta-llama/Llama-3.1-70B-Instruct", 8192, 1, 3, 40960)
//                                                              ^^^^^ minGpuVramMib
```

The operator's `VllmConfig.tensor_parallel_size` controls how many GPUs vLLM uses for tensor parallelism.

## Pricing

GPU resources are priced at two levels:

**Infrastructure (Pricing Engine sidecar):**
```toml
[resources]
gpu = { kind = "GPU", count = 1, price_per_unit_rate = 0.005 }
```

**Application (BSM per-token pricing):**
```solidity
configureModel(model, maxContextLen, pricePerInputToken, pricePerOutputToken, minGpuVramMib)
```

Operators can override BSM defaults via RFQ quotes signed with EIP-712.

## Code references

| What | Where |
|------|-------|
| GpuRequirements struct | `blueprint/crates/manager/src/protocol/tangle/blueprint_metadata.rs` |
| apply_gpu_limits | `blueprint/crates/manager/src/protocol/tangle/event_handler.rs` |
| K8s GPU pod spec | `blueprint/crates/manager/src/rt/container/mod.rs` |
| Cloud GPU routing | `blueprint/crates/manager/src/executor/remote_provider_integration.rs` |
| ResourceSpec.gpu_count | `blueprint/crates/blueprint-remote-providers/src/core/resources.rs` |
| InferenceBSM GPU validation | `contracts/src/InferenceBSM.sol` (onRegister) |
| Operator GPU detection | `operator/src/health.rs` (detect_gpus) |
| Pricing engine | `blueprint/crates/pricing-engine/` |
| Inference blueprint guide | `blueprint/docs/building-an-inference-blueprint.md` |
