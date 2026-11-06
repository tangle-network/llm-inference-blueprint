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

Single model per instance. Dependency versions are owned by [operator/Cargo.toml](operator/Cargo.toml).

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
beelink-gate auto https://github.com/tangle-network/llm-inference-blueprint.git <full-sha> -- bash -lc 'cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace && cargo audit --ignore RUSTSEC-2025-0055 && (cd contracts && forge soldeer update && forge build && forge test)'
```

The audit exclusion is for the Blueprint dependency's disabled EigenLayer feature. Payment, streaming and chain changes retain the actual-server and on-chain integration requirements in [CLAUDE.md](CLAUDE.md#verification). Missing artifacts or an early-returning harness provide no integration result; a substituted backend does not prove real model output. Automatic PR/push CI is retired. This local gate does not authorize a deployment, model request or payment.

## Run an operator

`operator/Dockerfile` builds both binaries in `operator/Cargo.toml` and uses the
[vLLM CUDA image](https://docs.vllm.ai/en/v0.32.0/deployment/docker/).
Edit `deploy/config.example.json` for the chain, model and payment settings.
Set the required SDK RPC, blueprint, service and contract variables listed in
[docker-compose.yml](docker-compose.yml), provision the operator keystore in its
state volume (`--keystore-uri /home/operator/state/keystore`), then run
`docker compose up --build operator` with the NVIDIA container runtime.
`operator-lite` attaches to an existing OpenAI-compatible backend such as the GTR Ollama shim by default. To supervise SGLang, set `vllm.engine` to `sglang` and `vllm.external` to `false`. A derivative model sets `vllm.base_family` and `vllm.model`; select its arm, GPU class and tensor parallelism from the [core catalog](https://github.com/tangle-network/tangle-inference-core/blob/master/engine-profiles.json) at the commit pinned in `operator/Cargo.toml`. On-chain admission and pricing remain model rules in the BSM.

`operator-lite` takes an optional configuration file as its first argument. Without one it reads `VLLM_OP_` environment variables, with `__` between nested fields, such as `VLLM_OP_SERVER__PORT`.

Tagged releases publish both Linux binaries, archive checksums and extracted
binary checksums through [Release](.github/workflows/release.yml).
To install the GTR user service, download a chosen release, verify its archive
with `sha256sum -c operator-lite-x86_64-unknown-linux-gnu.tar.xz.sha256`, extract
it, then verify `sha256sum -c operator-lite.sha256`.
Install the verified binary under `/opt/tangle-operator/releases/<tag>/` and
atomically point `/opt/tangle-operator/current` at that directory. Retain the
previous release for recovery. Install private config at
`/etc/tangle-operator/config.json` (mode 0600), with its nonce store under
`/var/lib/tangle-operator/`; the service user must own config and state.
Install [gtr-operator.service](deploy/systemd/gtr-operator.service) at
`/etc/systemd/user/gtr-operator.service`. Preserve any existing user unit outside
the systemd search path before replacing it. With `ollama-shim.service` running,
wait for `/metrics` active requests to reach zero, then run
`systemctl --user daemon-reload && systemctl --user enable --now gtr-operator`.
For an upgrade, restart the service. Verify `/proc/<MainPID>/exe` against the
release binary checksum, `/health`, `/v1/models` and a real routed completion.
On failure, switch `current` to the retained release and restart, then repeat
these checks. Never reset the nonce store during an upgrade or recovery.

## Live testnet registration

On Tempo Moderato, the serving registration is blueprint **10**, with
InferenceBSM `0x23254045aa898e4b59366a7126047dde80b5d094` and operator
`0xA78D390392ff07b218E9b3b2Ee1340F9C08499c1`. Its public ingress is
`https://drew-gtr-pro.taild10bfc.ts.net`. Read its model and rates from the
running operator and Router APIs. Blueprint **18** is a separate pathUSD
registration; deployment of its manager did not migrate the serving operator.
The [Router billing lane](https://github.com/tangle-network/tangle-router/pull/926)
owns Router price units; coordinate any native-payment price migration with it.

## License

MIT

## Release definitions

`deploy/register-blueprint.sh` reads the prepared definition from this repository's latest GitHub Release. The release workflow packages the actual binary digest, SDK download manifest, and execution profile together. Set `BLUEPRINT_DEFINITION` to a prepared definition for a specific release. For a local Anvil proof only, set `BLUEPRINT_BINARY` to the built operator binary; registration hashes that file and requires a loopback RPC on chain 31337.
