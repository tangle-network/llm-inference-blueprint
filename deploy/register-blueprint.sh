#!/usr/bin/env bash
set -euo pipefail

# Register the vLLM inference blueprint on Tangle.
#
# Prerequisites:
#   - forge installed
#   - Contracts deployed (InferenceBSM address known)
#   - Operator wallet funded with TNT for gas + bond
#
# Usage:
#   export RPC_URL=https://rpc.tangle.tools
#   export PRIVATE_KEY=0x...
#   export TANGLE_CORE=0x...
#   export BSM_ADDRESS=0x...
#   ./register-blueprint.sh

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
: "${TANGLE_CORE:?Set TANGLE_CORE}"
: "${BSM_ADDRESS:?Set BSM_ADDRESS}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_VRAM="${TOTAL_VRAM:-48000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA A100}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== vLLM Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Tangle Core: $TANGLE_CORE"
echo "BSM:         $BSM_ADDRESS"
echo "Model:       $MODEL"
echo "GPUs:        $GPU_COUNT x $GPU_MODEL ($TOTAL_VRAM MiB)"
echo "Endpoint:    $ENDPOINT"
echo ""

# Step 1: Deploy the InferenceBSM if not already deployed
if [ "$BSM_ADDRESS" = "deploy" ]; then
    echo "Deploying InferenceBSM..."

    TSUSD_ADDRESS="${TSUSD_ADDRESS:?Set TSUSD_ADDRESS for deployment}"

    BSM_ADDRESS=$(forge create \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        contracts/src/InferenceBSM.sol:InferenceBSM \
        --constructor-args "$TSUSD_ADDRESS" \
        --json | jq -r '.deployedTo')

    echo "InferenceBSM deployed at: $BSM_ADDRESS"
fi

# Step 2: Encode registration inputs
# abi.encode(string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
REG_INPUTS=$(cast abi-encode \
    "f(string,uint32,uint32,string,string)" \
    "$MODEL" "$GPU_COUNT" "$TOTAL_VRAM" "$GPU_MODEL" "$ENDPOINT")

echo "Registration inputs: $REG_INPUTS"

# Step 3: Register operator on Tangle Core
# This calls Tangle.registerOperator(blueprintId, registrationInputs)
# which internally calls bsm.onRegister(operator, registrationInputs)
echo ""
echo "Registering operator..."
echo "NOTE: Use tangle-cli or the Tangle dApp to complete registration."
echo "      The registration inputs to provide: $REG_INPUTS"
echo ""
echo "Manual cast command:"
echo "  cast send $TANGLE_CORE 'registerOperator(uint64,bytes)' <BLUEPRINT_ID> $REG_INPUTS --rpc-url $RPC_URL --private-key $PRIVATE_KEY"

echo ""
echo "Registration data prepared. Complete via Tangle UI or CLI."
