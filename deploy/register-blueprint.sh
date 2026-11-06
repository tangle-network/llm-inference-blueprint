#!/usr/bin/env bash
# Register the vLLM LLM-inference blueprint on Tangle.
#
# Single-shot flow: deploys InferenceBSM (impl + UUPS proxy + initialize)
# AND calls Tangle.createBlueprint in the same broadcast via
# `contracts/script/Deploy.s.sol`. This replaces the prior
# hand-written cast invocations.
#
# Prerequisites:
#   - forge installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/USDC defaults
#
# Optional overrides:
#   MODEL              vLLM model to advertise (default: meta-llama/Llama-3.1-8B-Instruct)
#   GPU_COUNT          GPUs the operator exposes (default: 1)
#   TOTAL_VRAM         Total VRAM in MiB (default: 48000)
#   GPU_MODEL          GPU model string (default: NVIDIA A100)
#   ENDPOINT           Operator HTTP endpoint (default: https://your-operator.example.com)
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_INFERENCE_BSM_IMPL=<address>
#   DEPLOY_INFERENCE_BSM_PROXY=<address>
#   DEPLOY_INFERENCE_BLUEPRINT_ID=<u64>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
TANGLE_CORE="${TANGLE_CORE:-0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_VRAM="${TOTAL_VRAM:-48000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA A100}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== vLLM Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Tangle Core: ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo "Model:       $MODEL"
echo "GPUs:        $GPU_COUNT x $GPU_MODEL ($TOTAL_VRAM MiB)"
echo "Endpoint:    $ENDPOINT"
echo ""

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --locked | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
cd "$(dirname "$0")/../contracts"

# Deploy BSM (impl + proxy + initialize) AND register the blueprint in one
# forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-0x036CbD53842c5426634e7929541eC2318f3dCF7e}" \
    forge script script/Deploy.s.sol:Deploy \
        --remappings "tangle-inference-core/=$CORE_DIR/contracts/" \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract the BSM proxy address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_INFERENCE_BSM_PROXY=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_INFERENCE_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:        $BLUEPRINT_ID"
echo "InferenceBSM proxy:  $BSM_ADDRESS"
echo ""

# Operator registration is a separate step (per-operator). Encode the
# registration inputs so the operator can call Tangle.registerOperator
# with the right calldata.
REG_INPUTS=$(cast abi-encode \
    "f(string,uint32,uint32,string,string)" \
    "$MODEL" "$GPU_COUNT" "$TOTAL_VRAM" "$GPU_MODEL" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
