![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">vLLM Inference Blueprint</h1>

<p align="center"><em>Anonymous LLM inference on <a href="https://tangle.tools">Tangle</a> — operators serve vLLM, users pay anonymously via shielded credits.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to serve LLM inference via [vLLM](https://github.com/vllm-project/vllm) with anonymous payments through the [Shielded Payment Gateway](https://github.com/tangle-network/shielded-payment-gateway).

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable results on Tangle
- **x402 HTTP** — fast private inference at `/v1/chat/completions`

Single model per instance. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) `0.1.0-alpha.22`.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — vLLM subprocess, HTTP server, SpendAuth billing |
| `contracts/` | Solidity | InferenceBSM — GPU validation, model pricing (24 tests) |
| `sdk/` | TypeScript | Client SDK — OpenAI-compatible interface with spend auth signing |

## Related Repos

- [shielded-payment-gateway](https://github.com/tangle-network/shielded-payment-gateway) — the payment layer
- [blueprint](https://github.com/tangle-network/blueprint) — Blueprint SDK
- [tnt-core](https://github.com/tangle-network/tnt-core) — Tangle core protocol

## Local qualification

Merge `origin/main` before qualifying source changes. Run the affected integration surface before merge through the admitted launcher. The comprehensive reference below retains the former branch checks; workflow and documentation-only changes use YAML, content, link and diff checks.

Provision Rust with rustfmt/clippy, protoc, Foundry and cargo-audit through the maintained host toolchain. The launcher does not provision these Rust/contract prerequisites.

```bash
beelink-gate --no-install auto https://github.com/tangle-network/llm-inference-blueprint.git <full-sha> -- bash -lc 'cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace && cargo audit --ignore RUSTSEC-2025-0055 && (cd contracts && forge soldeer update && forge build && forge test)'
```

The audit exclusion is for the Blueprint dependency's disabled EigenLayer feature. Payment, streaming and chain changes retain the actual-server and on-chain integration requirements in [CLAUDE.md](CLAUDE.md#verification). Missing artifacts or an early-returning harness provide no integration result; a substituted backend does not prove real model output. Automatic PR/push CI is retired. This local gate does not authorize a deployment, model request or payment.

## License

MIT
