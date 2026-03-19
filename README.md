![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

# vLLM Inference Blueprint for Tangle

A Tangle Blueprint that enables operators to serve LLM inference via [vLLM](https://github.com/vllm-project/vllm), with anonymous payment through the ShieldedCredits system.

## Architecture

```
User                         Operator                      Tangle Chain
 |                              |                              |
 |  1. Fund ShieldedCredits     |                              |
 |  (one ZK proof) ────────────────────────────────────────────>|
 |                              |                              |
 |  2. Sign SpendAuth (off-chain, free)                        |
 |  3. POST /v1/chat/completions ──>|                          |
 |                              |  4. Verify sig (ecrecover)   |
 |                              |  5. Proxy to vLLM            |
 |  6. Response <───────────────|                              |
 |                              |  7. authorizeSpend ──────────>|
 |                              |  8. claimPayment ────────────>|
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.x drivers
- Docker with NVIDIA Container Toolkit
- Foundry (`curl -L https://foundry.paradigm.xyz | bash`)
- Rust 1.82+ (for building from source)
- Node.js 20+ (for the SDK)

### Local Development

```bash
# 1. Start local chain + operator
cp deploy/config.example.json deploy/config.json
# Edit config.json with your settings

docker compose up -d anvil
docker compose up operator

# 2. Deploy contracts (in another terminal)
cd contracts
forge soldeer update
forge build
forge test

# 3. Test with the SDK
cd sdk
npm install
npm run build
```

### Using the SDK

```typescript
import { createInferenceClient } from "@tangle-network/vllm-inference-sdk";

const client = createInferenceClient({
  operatorUrl: "http://localhost:8080",
  shieldedCreditsAddress: "0x...",
  chainId: 31337,
  commitment: "0x...",
  serviceId: 1n,
  operatorAddress: "0x...",
  spendingKeyPrivate: "0x...",
});

const response = await client.chat([
  { role: "user", content: "Explain zero-knowledge proofs in one sentence." },
]);

console.log(response.choices[0].message.content);
```

### Building the Operator from Source

```bash
cargo build --release -p vllm-operator

# Run with config file
./target/release/vllm-operator --config deploy/config.json

# Or with environment variables
export VLLM_OP_TANGLE__RPC_URL=http://localhost:8545
export VLLM_OP_TANGLE__OPERATOR_KEY=0xac09...
./target/release/vllm-operator --model meta-llama/Llama-3.1-8B-Instruct
```

## Contract: InferenceBSM

The Blueprint Service Manager validates:
- **Operator registration**: GPU capabilities must meet model requirements (VRAM, count)
- **Payment restriction**: Only tsUSD (ShieldedCredits pool token) accepted
- **Model metadata**: Configurable pricing per model (input/output token rates)
- **Job schema**: `(string prompt, uint32 maxTokens, uint32 temperatureE4)` in, `(string text, uint32 promptTokens, uint32 completionTokens)` out

## Billing Flow

1. **User** funds a ShieldedCredits account with one ZK proof (VAnchor withdrawal)
2. **User** signs EIP-712 SpendAuth per request (off-chain, free, anonymous)
3. **Operator** verifies SpendAuth via ecrecover (instant, no gas)
4. **Operator** serves inference via vLLM
5. **Operator** calls `authorizeSpend()` then `claimPayment()` on-chain

The user's real identity is never revealed. The operator only sees the ephemeral spending key.

## Configuration

See `deploy/config.example.json` for all options. Key settings:

| Setting | Description |
|---------|-------------|
| `vllm.model` | HuggingFace model ID |
| `vllm.tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `billing.price_per_input_token` | Price per input token in tsUSD base units |
| `billing.price_per_output_token` | Price per output token in tsUSD base units |
| `gpu.expected_gpu_count` | Required GPU count (startup check) |

## License

MIT OR Apache-2.0
