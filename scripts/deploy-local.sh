#!/usr/bin/env bash
# deploy-local.sh — Deploy ShieldedCredits + RLNSettlement + MockERC20 on local Anvil.
#
# Deploys all contracts, funds accounts, registers operator, and writes .env.local.
#
# Prerequisites:
#   - Foundry toolchain: anvil, forge, cast
#
# Usage:
#   ./scripts/deploy-local.sh
#
# Environment overrides:
#   RPC_URL         — Anvil RPC URL (default: http://127.0.0.1:8645)
#   ANVIL_PORT      — Anvil port (default: 8645)
#   SKIP_ANVIL      — Set to 1 to skip Anvil startup (use existing instance)
#   OPERATOR_PORT   — Operator HTTP API port (default: 9100)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACTS_DIR="$ROOT_DIR/contracts"

ANVIL_PORT="${ANVIL_PORT:-8645}"
RPC_URL="${RPC_URL:-http://127.0.0.1:$ANVIL_PORT}"
OPERATOR_PORT="${OPERATOR_PORT:-9100}"

# Anvil deterministic accounts
DEPLOYER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOYER_ADDR="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
OPERATOR_KEY="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
OPERATOR_ADDR="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
USER_KEY="0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"
USER_ADDR="0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"

FUND_AMOUNT="10000000000000000000000" # 10,000 tokens (18 decimals)

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "${ANVIL_PID:-}" ] && kill "$ANVIL_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo "=== vLLM Inference Blueprint — Local Deployment ==="
echo "RPC: $RPC_URL"
echo ""

# ── Helper ──────────────────────────────────────────────────────────────

# Extract deployed address from forge create output
parse_deployed() {
    echo "$1" | grep "Deployed to:" | awk '{print $3}'
}

# ── [0/6] Start Anvil ──────────────────────────────────────────────────

if [ "${SKIP_ANVIL:-0}" = "1" ]; then
    echo "[0/6] Skipping Anvil startup (SKIP_ANVIL=1)"
else
    echo "[0/6] Starting Anvil on port $ANVIL_PORT..."
    # Check if already running
    if cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
        echo "  Anvil already running on $RPC_URL"
    else
        anvil --host 0.0.0.0 --port "$ANVIL_PORT" --silent &
        ANVIL_PID=$!
        sleep 2

        if ! cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
            echo "ERROR: Anvil not responding on $RPC_URL"
            exit 1
        fi
        echo "  Anvil running (PID: $ANVIL_PID)"
    fi
fi

# Verify connectivity
if ! cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to $RPC_URL"
    exit 1
fi

# ── [1/6] Deploy MockERC20 (stablecoin) ───────────────────────────────

echo "[1/6] Deploying MockERC20 (tsUSD stablecoin)..."

# Inline minimal MockERC20 — same as test/BillingE2E.t.sol::MockToken
MOCK_ERC20_SRC='
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
contract MockToken {
    string public name = "Test Shielded USD";
    string public symbol = "tsUSD";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(allowance[from][msg.sender] >= amount, "insufficient allowance");
        require(balanceOf[from] >= amount, "insufficient balance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}
'

# Write temp source for forge create
MOCK_SRC_FILE="$CONTRACTS_DIR/src/_MockToken.sol"
echo "$MOCK_ERC20_SRC" > "$MOCK_SRC_FILE"

FORGE_OUT=$(forge create "$MOCK_SRC_FILE:MockToken" \
    --rpc-url "$RPC_URL" \
    --private-key "$DEPLOYER_KEY" \
    --root "$CONTRACTS_DIR" \
    --broadcast 2>&1)
TOKEN_ADDR=$(parse_deployed "$FORGE_OUT")
rm -f "$MOCK_SRC_FILE"

if [ -z "$TOKEN_ADDR" ]; then
    echo "ERROR: MockToken deployment failed"
    echo "$FORGE_OUT" | tail -20
    exit 1
fi
echo "  tsUSD token: $TOKEN_ADDR"

# ── [2/6] Deploy ShieldedCredits ───────────────────────────────────────

echo "[2/6] Deploying ShieldedCredits..."

FORGE_OUT=$(forge create "$CONTRACTS_DIR/src/ShieldedCredits.sol:ShieldedCredits" \
    --rpc-url "$RPC_URL" \
    --private-key "$DEPLOYER_KEY" \
    --root "$CONTRACTS_DIR" \
    --broadcast 2>&1)
CREDITS_ADDR=$(parse_deployed "$FORGE_OUT")

if [ -z "$CREDITS_ADDR" ]; then
    echo "ERROR: ShieldedCredits deployment failed"
    echo "$FORGE_OUT" | tail -20
    exit 1
fi
echo "  ShieldedCredits: $CREDITS_ADDR"

# ── [3/6] Deploy RLNSettlement + register operator ─────────────────────

echo "[3/6] Deploying RLNSettlement..."

FORGE_OUT=$(forge create "$CONTRACTS_DIR/src/RLNSettlement.sol:RLNSettlement" \
    --rpc-url "$RPC_URL" \
    --private-key "$DEPLOYER_KEY" \
    --root "$CONTRACTS_DIR" \
    --broadcast 2>&1)
RLN_ADDR=$(parse_deployed "$FORGE_OUT")

if [ -z "$RLN_ADDR" ]; then
    echo "ERROR: RLNSettlement deployment failed"
    echo "$FORGE_OUT" | tail -20
    exit 1
fi
echo "  RLNSettlement: $RLN_ADDR"

echo "  Registering operator $OPERATOR_ADDR on RLNSettlement..."
cast send "$RLN_ADDR" \
    "registerOperator(address)" "$OPERATOR_ADDR" \
    --private-key "$DEPLOYER_KEY" \
    --rpc-url "$RPC_URL" \
    --json > /dev/null 2>&1
echo "  Operator registered"

# ── [4/6] Mint tokens + approve contracts ──────────────────────────────

echo "[4/6] Minting tokens and setting approvals..."

# Mint to user
cast send "$TOKEN_ADDR" \
    "mint(address,uint256)" "$USER_ADDR" "$FUND_AMOUNT" \
    --private-key "$DEPLOYER_KEY" \
    --rpc-url "$RPC_URL" \
    --json > /dev/null 2>&1
echo "  Minted $(cast from-wei "$FUND_AMOUNT") tsUSD to user ($USER_ADDR)"

# User approves ShieldedCredits
cast send "$TOKEN_ADDR" \
    "approve(address,uint256)" "$CREDITS_ADDR" "$FUND_AMOUNT" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json > /dev/null 2>&1
echo "  User approved ShieldedCredits for $(cast from-wei "$FUND_AMOUNT") tsUSD"

# User approves RLNSettlement
cast send "$TOKEN_ADDR" \
    "approve(address,uint256)" "$RLN_ADDR" "$FUND_AMOUNT" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json > /dev/null 2>&1
echo "  User approved RLNSettlement for $(cast from-wei "$FUND_AMOUNT") tsUSD"

# ── [5/6] Fund user's ShieldedCredits account ─────────────────────────

echo "[5/6] Funding user's ShieldedCredits account..."

# Generate a deterministic commitment: keccak256(user_addr, "local-test-salt")
COMMITMENT=$(cast keccak "$(cast abi-encode "f(address,string)" "$USER_ADDR" "local-test-salt")")
# Spending key = user address (in local test, user signs spend auths)
SPENDING_KEY="$USER_ADDR"

CREDIT_FUND_AMOUNT="1000000000000000000000" # 1,000 tsUSD
cast send "$CREDITS_ADDR" \
    "fundCredits(address,uint256,bytes32,address)" \
    "$TOKEN_ADDR" "$CREDIT_FUND_AMOUNT" "$COMMITMENT" "$SPENDING_KEY" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json > /dev/null 2>&1
echo "  Funded $(cast from-wei "$CREDIT_FUND_AMOUNT") tsUSD to commitment $COMMITMENT"

# Verify
ACCT_BAL=$(cast call "$CREDITS_ADDR" \
    "getAccount(bytes32)((address,address,uint256,uint256,uint256,uint256))" \
    "$COMMITMENT" \
    --rpc-url "$RPC_URL" 2>/dev/null)
echo "  Account state: $ACCT_BAL"

# ── [6/6] Write .env.local ─────────────────────────────────────────────

echo "[6/6] Writing .env.local..."

ENV_FILE="$ROOT_DIR/.env.local"
cat > "$ENV_FILE" <<EOF
# Generated by deploy-local.sh — $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Do not edit manually.

# Network
RPC_URL=$RPC_URL
CHAIN_ID=31337

# Contracts
TOKEN_ADDR=$TOKEN_ADDR
SHIELDED_CREDITS=$CREDITS_ADDR
RLN_SETTLEMENT=$RLN_ADDR

# Accounts
DEPLOYER_KEY=$DEPLOYER_KEY
DEPLOYER_ADDR=$DEPLOYER_ADDR
OPERATOR_KEY=$OPERATOR_KEY
OPERATOR_ADDR=$OPERATOR_ADDR
USER_KEY=$USER_KEY
USER_ADDR=$USER_ADDR

# Credit account
COMMITMENT=$COMMITMENT
SPENDING_KEY=$SPENDING_KEY
CREDIT_FUND_AMOUNT=$CREDIT_FUND_AMOUNT

# Operator API
OPERATOR_PORT=$OPERATOR_PORT
OPERATOR_API_URL=http://127.0.0.1:$OPERATOR_PORT
EOF

echo "  Written to $ENV_FILE"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Contracts:"
echo "  tsUSD Token:       $TOKEN_ADDR"
echo "  ShieldedCredits:   $CREDITS_ADDR"
echo "  RLNSettlement:     $RLN_ADDR"
echo ""
echo "Accounts:"
echo "  Deployer: $DEPLOYER_ADDR"
echo "  Operator: $OPERATOR_ADDR"
echo "  User:     $USER_ADDR"
echo ""
echo "Credit Account:"
echo "  Commitment:   $COMMITMENT"
echo "  Spending Key: $SPENDING_KEY"
echo "  Balance:      $(cast from-wei "$CREDIT_FUND_AMOUNT") tsUSD"
echo ""
echo "Next steps:"
echo "  # Start the operator (optional, for inference tests):"
echo "  OLLAMA_URL=http://127.0.0.1:11434 cargo run --manifest-path $ROOT_DIR/operator/Cargo.toml"
echo ""
echo "  # Run E2E tests:"
echo "  ./scripts/test-e2e.sh"

# Keep alive if we started Anvil
if [ -n "${ANVIL_PID:-}" ]; then
    echo ""
    echo "Anvil running in background (PID: $ANVIL_PID). Press Ctrl+C to stop."
    wait "$ANVIL_PID"
fi
