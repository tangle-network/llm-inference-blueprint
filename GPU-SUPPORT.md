# GPU support

Use the [Blueprint SDK GPU catalog](https://github.com/tangle-network/blueprint/blob/main/crates/core/src/gpu.rs) for class identifiers and capacity. Resource requirements, device detection, provisioning and provider prices belong to the SDK.

Select the model family, arm, GPU class and tensor parallelism from the [core recipe catalog](https://github.com/tangle-network/tangle-inference-core/blob/master/engine-profiles.json) at the revision pinned in `operator/Cargo.toml`. Core supervises the model replicas and router. The blueprint retains model admission and token pricing.

The current operator configuration supplies declared VRAM. Device detection and class-plus-VRAM registration require the qualified SDK implementation and their integration proof. See [verification and operation](README.md).
