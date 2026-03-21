# Local Development Scripts

## Quick Start

```bash
# Terminal 1: Deploy contracts to local Anvil
./scripts/deploy-local.sh

# Terminal 2 (optional): Start the operator
OLLAMA_URL=http://127.0.0.1:11434 cargo run

# Terminal 3: Run E2E tests
./scripts/test-e2e.sh
```

## Prerequisites

- **Foundry** (`anvil`, `forge`, `cast`)
- **Node.js** (for EIP-712 signing helper)
- **Ollama** with `qwen2:0.5b` (optional, for inference tests)

## Scripts

| Script | Description |
|---|---|
| `deploy-local.sh` | Starts Anvil, deploys MockERC20 + ShieldedCredits + RLNSettlement, funds accounts, writes `.env.local` |
| `test-e2e.sh` | Full E2E test suite: credit mode, RLN mode, malicious inputs, inference API |
| `sign-spend-auth.mjs` | EIP-712 SpendAuthorization signer (used by test-e2e.sh) |

## Environment Overrides

- `ANVIL_PORT` — Anvil port (default: 8645)
- `SKIP_ANVIL=1` — Use existing Anvil instance
- `SKIP_INFERENCE=1` — Skip operator API / Ollama tests
- `OPERATOR_PORT` — Operator HTTP port (default: 9100)
