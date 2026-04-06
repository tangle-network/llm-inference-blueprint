# Implementation Plan

## Phase 1: Contracts — Complete

- [x] InferenceBSM with model config, operator registration, GPU validation
- [x] Payment restricted to tsUSD (ShieldedCredits pool token)
- [x] Job schema: input = (prompt, maxTokens, temperatureE4), output = (text, promptTokens, completionTokens)
- [x] Tests for registration, model config, job lifecycle, payment validation
- [x] ShieldedCredits contract with EIP-712 SpendAuth
- [x] RLNSettlement alternative billing path with RLN-based slashing
- [x] Billing E2E tests (BillingE2E.t.sol, RLNBillingE2E.t.sol)
- [x] Import tnt-core as soldeer dependency (v0.10.4, proper imports)
- [x] Deploy script (contracts/script/Deploy.s.sol — deploys ShieldedCredits, RLNSettlement, InferenceBSM with ERC1967Proxy)

## Phase 2: Operator Core — Complete

- [x] Extracted shared infrastructure to [`tangle-inference-core`](../tangle-inference-core/) (billing, metrics, health, nonce store, spend-auth validation, x402 headers, AppState builder)
- [x] Depends on `tangle-inference-core` for billing, metrics, health, server helpers — ~1800 LOC of duplication deleted from this blueprint
- [x] Config system (TOML file + env var overrides)
- [x] vLLM subprocess management (spawn, health check, shutdown, watchdog respawn)
- [x] OpenAI-compatible HTTP API (/v1/chat/completions, /v1/models, /health, /health/gpu)
- [x] SpendAuth off-chain verification (EIP-712 ecrecover)
- [x] Billing client (authorizeSpend + claimPayment via alloy)
- [x] GPU detection via nvidia-smi
- [x] Tangle job handler (Router, TangleArg/TangleResult, ABI encoding)
- [x] Streaming support (SSE for /v1/chat/completions with idle-chunk timeout)
- [x] Graceful shutdown with in-flight request draining
- [x] Request queuing with semaphore-based concurrency control
- [x] Prometheus metrics endpoint (/metrics) — 20+ metrics
- [x] Request/response logging for auditability
- [x] Per-account rate limiting (active request tracking per commitment)
- [x] Nonce replay protection (persistent file-based store)
- [x] vLLM stderr/stdout log draining and structured forwarding
- [x] Pre-authorization mode (authorizeSpend called before serving, claimPayment after with metered cost)

## Phase 3: SDK — Core Done, Discovery/Streaming TODO

- [x] EIP-712 SpendAuth signing with ethers.js
- [x] createInferenceClient() with chat(), listModels(), healthCheck()
- [x] Cost estimation
- [x] Nonce management
- [ ] On-chain credit account query (types defined, operator has get_account_info — SDK needs to expose)
- [ ] Operator discovery from BSM contract (types defined, no contract bindings in SDK)
- [ ] Automatic operator selection (closest, cheapest, most available)
- [ ] Retry logic with failover
- [ ] Streaming support (ReadableStream for SSE — operator supports it, SDK hardcodes stream=false)
- [ ] OpenAI SDK drop-in adapter

## Phase 4: DevOps & Testing — CI Done, Production Infra TODO

- [x] Dockerfile with CUDA + vLLM + Rust binary
- [x] docker-compose for local dev (with nvidia GPU runtime)
- [x] Example config
- [x] GitHub Actions CI (fmt, clippy, test, audit, forge test)
- [x] E2E tests (local_e2e.rs, harness_e2e.rs, server_tests.rs)
- [ ] Multi-arch Docker builds (amd64 + arm64 via buildx)
- [ ] Kubernetes Helm chart
- [ ] Load testing (k6 or locust)

## Phase 5: Production Hardening — Not Started

- [ ] SpendAuth signature malleability audit (no s-value normalization in billing.rs)
- [ ] Billing flow audit (operator overcharge prevention)
- [ ] Model output validation (hash commitment for dispute resolution)
- [ ] Multi-model support per operator (currently single model per config)
- [ ] Model quantization support (GPTQ, AWQ, GGUF via vLLM extra_args)
- [ ] Embedding endpoint (/v1/embeddings)
- [ ] Fine-tune serving (LoRA adapter hot-swap via vLLM --enable-lora)

## Architecture Decisions

### Why vLLM subprocess (not in-process)?
- vLLM is Python; the operator is Rust. Subprocess isolation is clean.
- vLLM manages its own CUDA memory, GPU scheduling, PagedAttention.
- Crash isolation: vLLM can crash without taking down the operator.
- Upgradable independently: swap vLLM version without rebuilding operator.

### Billing: pre-authorize then meter
- Off-chain ecrecover validates SpendAuth signature (instant, free).
- On-chain `authorizeSpend()` reserves payment before inference starts.
- After inference completes, `claimPayment()` claims metered actual cost.
- Operator is protected: payment is reserved before GPU work begins.

### Why tsUSD only?
- ShieldedCredits uses a single pool token per account.
- Stablecoin denomination simplifies pricing (no ETH/USD volatility).

### GPU provisioning
- The Blueprint Manager handles GPU instance provisioning via `blueprint-remote-providers`.
- This blueprint does NOT provision its own GPU — it runs on hardware the manager already set up.
- GPU requirements are declared on-chain in `profilingData` and enforced by the manager at service activation.
- See [GPU-SUPPORT.md](./GPU-SUPPORT.md) for the full architecture.
