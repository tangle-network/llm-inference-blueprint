# LLM Inference Blueprint

Use [README.md](README.md) for setup, [operator/Cargo.toml](operator/Cargo.toml) for dependency ownership, and [.github/workflows/ci.yml](.github/workflows/ci.yml) for checks.
Keep shared billing, payment validation, nonce handling, health, and metrics in the `tangle-inference-core` dependency.
Read the selected dependency source for API contracts rather than copying versioned examples here.

## Verification

For handler and payment changes, extend [server_tests.rs](operator/tests/server_tests.rs) against the actual server with an external backend substitute.
Streaming changes must cover chunk boundaries, usage extraction, and upstream failures.
Preserve payment rejection and settlement capped by the authorized amount.

[harness_e2e.rs](operator/tests/harness_e2e.rs) exercises on-chain job submission and returned results with a substituted inference backend.
It returns early when required chain artifacts are absent; check that the flow actually ran before claiming success.
Its handler initialization is process-global, so follow the test's isolation constraints.
[local_e2e.rs](operator/tests/local_e2e.rs) checks a real inference backend and requires its documented opt-in and running service.
Neither a skipped test nor a backend substitute proves real model output.
Exercise contract changes with actual deployments under `contracts/`.
