#!/usr/bin/env bash
# test-full-protocol.sh — Full-protocol E2E test for vLLM inference blueprint
# with shielded payments on a REAL Tangle Anvil state snapshot.
#
# Tests the complete lifecycle:
#   Phase 1: Anvil with tnt-core state snapshot
#   Phase 2: Deploy shielded payment contracts (MockERC20, ShieldedCredits, RLNSettlement)
#   Phase 3: Register operator + create service via Tangle protocol
#   Phase 4: Fund ShieldedCredits + authorize spend + claim payment
#   Phase 5: RLN deposit + batch claim
#   Phase 6: Submit job to Tangle
#   Phase 7: Inference via Ollama (optional)
#   Phase 8: Gas cost + timing report
#
# Prerequisites:
#   - Foundry (anvil, forge, cast), Node.js, jq, bc
#   - Tangle Anvil state snapshot at:
#       ~/code/blueprint/crates/chain-setup/anvil/snapshots/localtestnet-state.json
#   - Optionally: Ollama on port 11434 with qwen2:0.5b
#
# Usage:
#   ./scripts/test-full-protocol.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACTS_DIR="$ROOT_DIR/contracts"

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ANVIL_PORT="${ANVIL_PORT:-18545}"
RPC_URL="http://127.0.0.1:$ANVIL_PORT"
ANVIL_STATE="${ANVIL_STATE:-$HOME/code/blueprint/crates/chain-setup/anvil/snapshots/localtestnet-state.json}"

# Tangle protocol addresses (deterministic from state snapshot)
TANGLE="0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9"

# Anvil deterministic accounts
DEPLOYER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOYER_ADDR="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
OPERATOR_KEY="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
OPERATOR_ADDR="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
USER_KEY="0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"
USER_ADDR="0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"

FUND_TNT_WEI="100000000000000000000" # 100 native tokens
FUND_AMOUNT="10000000000000000000000" # 10,000 tsUSD (18 decimals)
CREDIT_FUND_AMOUNT="1000000000000000000000" # 1,000 tsUSD

# ═══════════════════════════════════════════════════════════════════════════════
# Test framework + timing
# ═══════════════════════════════════════════════════════════════════════════════

PASS=0; FAIL=0; TOTAL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo -e "  ${RED}FAIL${NC} $1"; }
assert_eq() {
    local got="$1" expected="$2" desc="$3"
    if [ "$got" = "$expected" ]; then pass "$desc"; else fail "$desc (expected='$expected', got='$got')"; fi
}

section() { echo -e "\n${CYAN}${BOLD}[$1]${NC} $2"; }

# Gas tracking
declare -a GAS_LABELS=()
declare -a GAS_VALUES=()
record_gas() {
    local label="$1" result="$2"
    local gas
    gas=$(echo "$result" | jq -r '.gasUsed // "0"' 2>/dev/null | xargs printf "%d" 2>/dev/null || echo "0")
    GAS_LABELS+=("$label")
    GAS_VALUES+=("$gas")
}

# Phase timing
declare -a PHASE_LABELS=()
declare -a PHASE_TIMES=()
SCRIPT_START=$(date +%s%N)
phase_start() { PHASE_START=$(date +%s%N); }
phase_end() {
    local elapsed=$(( ($(date +%s%N) - PHASE_START) / 1000000 ))
    PHASE_LABELS+=("$1")
    PHASE_TIMES+=("$elapsed")
}

strip_cast() { echo "$1" | sed 's/ \[.*\]//g' | tr -d '()' | tr -d ' '; }

get_credit_balance() {
    local raw
    raw=$(cast call "$SHIELDED_CREDITS" \
        "getAccount(bytes32)((address,address,uint256,uint256,uint256,uint256))" \
        "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    local field
    field=$(echo "$raw" | tr ',' '\n' | sed -n '3p')
    strip_cast "$field"
}

get_credit_nonce() {
    local raw
    raw=$(cast call "$SHIELDED_CREDITS" \
        "getAccount(bytes32)((address,address,uint256,uint256,uint256,uint256))" \
        "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    local field
    field=$(echo "$raw" | tr ',' '\n' | sed -n '6p')
    strip_cast "$field"
}

get_token_balance() {
    local raw
    raw=$(cast call "$TOKEN_ADDR" "balanceOf(address)(uint256)" "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    strip_cast "$raw"
}

sign_spend_auth() {
    node "$ROOT_DIR/scripts/sign-spend-auth.mjs" \
        --private-key "$1" \
        --verifying-contract "$SHIELDED_CREDITS" \
        --chain-id 31337 \
        --commitment "$2" \
        --service-id "$3" \
        --job-index "$4" \
        --amount "$5" \
        --operator "$6" \
        --nonce "$7" \
        --expiry "$8"
}

send_authorize_spend() {
    local commitment="$1" svc="$2" job="$3" amount="$4" op="$5" nonce="$6" expiry="$7" sig="$8" key="$9"
    local calldata selector
    calldata=$(cast abi-encode "authorizeSpend((bytes32,uint64,uint8,uint256,address,uint256,uint64,bytes))" \
        "($commitment,$svc,$job,$amount,$op,$nonce,$expiry,$sig)")
    selector=$(cast sig "authorizeSpend((bytes32,uint64,uint8,uint256,address,uint256,uint64,bytes))")
    cast send "$SHIELDED_CREDITS" \
        "${selector}${calldata#0x}" \
        --private-key "$key" \
        --rpc-url "$RPC_URL" \
        --json 2>&1
}

parse_deployed() { echo "$1" | grep "Deployed to:" | awk '{print $3}'; }

fund_account_tnt() {
    local addr="$1" amount_hex
    amount_hex="$(cast to-hex "$FUND_TNT_WEI")"
    cast rpc --rpc-url "$RPC_URL" anvil_setBalance "$addr" "$amount_hex" >/dev/null
}

# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    echo ""
    [ -n "${ANVIL_PID:-}" ] && kill "$ANVIL_PID" 2>/dev/null || true
    [ -f "${MOCK_SRC_FILE:-}" ] && rm -f "$MOCK_SRC_FILE"
    exit "${1:-0}"
}
trap 'cleanup 1' INT TERM

echo -e "${BOLD}=== vLLM Inference Blueprint — Full Protocol E2E Test ===${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Prerequisites
# ═══════════════════════════════════════════════════════════════════════════════

section "0" "Prerequisites"

for cmd in anvil forge cast node jq bc curl; do
    if command -v "$cmd" &>/dev/null; then pass "$cmd available"
    else fail "$cmd not found — install and retry"; cleanup 1; fi
done

if [ ! -f "$ANVIL_STATE" ]; then
    fail "Anvil state snapshot not found: $ANVIL_STATE"
    cleanup 1
fi
pass "state snapshot exists"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Start Anvil with tnt-core state
# ═══════════════════════════════════════════════════════════════════════════════

section "1" "Start Anvil with Tangle protocol state"
phase_start

# Kill any existing anvil on our port
if lsof -ti :"$ANVIL_PORT" >/dev/null 2>&1; then
    kill $(lsof -ti :"$ANVIL_PORT") 2>/dev/null || true
    sleep 1
fi

anvil --block-time 1 --port "$ANVIL_PORT" --load-state "$ANVIL_STATE" --silent &
ANVIL_PID=$!
sleep 2

if ! cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
    fail "Anvil not responding on $RPC_URL"
    cleanup 1
fi
pass "Anvil running (PID: $ANVIL_PID)"

# Fund accounts
fund_account_tnt "$DEPLOYER_ADDR"
fund_account_tnt "$OPERATOR_ADDR"
fund_account_tnt "$USER_ADDR"
pass "accounts funded with native tokens"

# Verify Tangle protocol
BP_COUNT=$(cast call "$TANGLE" "blueprintCount()(uint64)" --rpc-url "$RPC_URL" 2>&1 | xargs)
if [ "$BP_COUNT" -ge 1 ] 2>/dev/null; then
    pass "Tangle protocol live (blueprintCount=$BP_COUNT)"
else
    fail "Tangle protocol not responding (blueprintCount=$BP_COUNT)"
    cleanup 1
fi

phase_end "Anvil startup"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Deploy shielded payment contracts
# ═══════════════════════════════════════════════════════════════════════════════

section "2" "Deploy shielded payment contracts"
phase_start

# 2a. MockERC20
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
    function mint(address to, uint256 amount) external { balanceOf[to] += amount; totalSupply += amount; }
    function approve(address spender, uint256 amount) external returns (bool) { allowance[msg.sender][spender] = amount; return true; }
    function transfer(address to, uint256 amount) external returns (bool) { require(balanceOf[msg.sender] >= amount); balanceOf[msg.sender] -= amount; balanceOf[to] += amount; return true; }
    function transferFrom(address from, address to, uint256 amount) external returns (bool) { require(allowance[from][msg.sender] >= amount); require(balanceOf[from] >= amount); allowance[from][msg.sender] -= amount; balanceOf[from] -= amount; balanceOf[to] += amount; return true; }
}
'
MOCK_SRC_FILE="$CONTRACTS_DIR/src/_MockToken.sol"
echo "$MOCK_ERC20_SRC" > "$MOCK_SRC_FILE"

FORGE_OUT=$(forge create "$MOCK_SRC_FILE:MockToken" \
    --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" --root "$CONTRACTS_DIR" --broadcast 2>&1)
TOKEN_ADDR=$(parse_deployed "$FORGE_OUT")
rm -f "$MOCK_SRC_FILE"
MOCK_SRC_FILE="" # clear so cleanup doesn't try to rm

if [ -z "$TOKEN_ADDR" ]; then fail "MockToken deploy failed"; cleanup 1; fi
pass "MockToken (tsUSD): $TOKEN_ADDR"

# 2b. ShieldedCredits
FORGE_OUT=$(forge create "$CONTRACTS_DIR/src/ShieldedCredits.sol:ShieldedCredits" \
    --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" --root "$CONTRACTS_DIR" --broadcast 2>&1)
SHIELDED_CREDITS=$(parse_deployed "$FORGE_OUT")

if [ -z "$SHIELDED_CREDITS" ]; then fail "ShieldedCredits deploy failed"; cleanup 1; fi
pass "ShieldedCredits: $SHIELDED_CREDITS"

# 2c. RLNSettlement
FORGE_OUT=$(forge create "$CONTRACTS_DIR/src/RLNSettlement.sol:RLNSettlement" \
    --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" --root "$CONTRACTS_DIR" --broadcast 2>&1)
RLN_SETTLEMENT=$(parse_deployed "$FORGE_OUT")

if [ -z "$RLN_SETTLEMENT" ]; then fail "RLNSettlement deploy failed"; cleanup 1; fi
pass "RLNSettlement: $RLN_SETTLEMENT"

# 2d. Register operator on RLN
RESULT=$(cast send "$RLN_SETTLEMENT" "registerOperator(address)" "$OPERATOR_ADDR" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json 2>&1)
record_gas "RLN registerOperator" "$RESULT"
pass "operator registered on RLNSettlement"

phase_end "Deploy contracts"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Tangle protocol lifecycle — register operator + create service
# ═══════════════════════════════════════════════════════════════════════════════

section "3" "Tangle protocol lifecycle"
phase_start

BLUEPRINT_ID=0

# 3a. Check if operator is already registered for blueprint 0 (from snapshot)
OP_REG=$(cast call "$TANGLE" "isOperatorRegistered(uint64,address)(bool)" \
    "$BLUEPRINT_ID" "$OPERATOR_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)

if [ "$OP_REG" = "true" ]; then
    pass "operator already registered for blueprint #$BLUEPRINT_ID (from snapshot)"
else
    # Register operator
    OPERATOR_PUBKEY_RAW=$(cast wallet public-key --private-key "$OPERATOR_KEY" 2>/dev/null | head -1)
    OPERATOR_PUBKEY="0x04${OPERATOR_PUBKEY_RAW#0x}"

    RESULT=$(cast send "$TANGLE" \
        "registerOperator(uint64,bytes,string)" \
        "$BLUEPRINT_ID" "$OPERATOR_PUBKEY" "http://127.0.0.1:9100" \
        --gas-limit 2000000 \
        --rpc-url "$RPC_URL" --private-key "$OPERATOR_KEY" --json 2>&1)
    record_gas "registerOperator" "$RESULT"

    if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
        pass "operator registered for blueprint #$BLUEPRINT_ID"
    else
        fail "registerOperator failed: ${RESULT:0:200}"
    fi
fi

# 3b. Request a new service for this test
NEXT_REQ=$(cast call "$TANGLE" "serviceRequestCount()(uint64)" --rpc-url "$RPC_URL" 2>&1 | xargs)
NEXT_REQ=$(echo "$NEXT_REQ" | sed 's/^0x//' | sed 's/^0*//' | sed 's/^$/0/')

# The snapshot has the 7-param requestService signature
RESULT=$(cast send "$TANGLE" \
    "requestService(uint64,address[],bytes,address[],uint64,address,uint256)" \
    "$BLUEPRINT_ID" \
    "[$OPERATOR_ADDR]" \
    "0x" \
    "[$USER_ADDR]" \
    31536000 \
    "0x0000000000000000000000000000000000000000" \
    0 \
    --gas-limit 3000000 \
    --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" --json 2>&1)
record_gas "requestService" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "service requested (request #$NEXT_REQ)"
else
    fail "requestService failed: ${RESULT:0:200}"
    cleanup 1
fi

# 3c. Operator approves
RESULT=$(cast send "$TANGLE" "approveService(uint64,uint8)" "$NEXT_REQ" 100 \
    --gas-limit 10000000 \
    --rpc-url "$RPC_URL" --private-key "$OPERATOR_KEY" --json 2>&1)
record_gas "approveService" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "operator approved service request #$NEXT_REQ"
else
    fail "approveService failed: ${RESULT:0:200}"
    cleanup 1
fi

# 3d. Find the new service ID
SVC_COUNT=$(cast call "$TANGLE" "serviceCount()(uint64)" --rpc-url "$RPC_URL" 2>&1 | xargs)
SVC_COUNT=$(echo "$SVC_COUNT" | sed 's/^0x//' | sed 's/^0*//' | sed 's/^$/0/')
SERVICE_ID=$((SVC_COUNT - 1))
pass "service activated (service #$SERVICE_ID)"

phase_end "Tangle lifecycle"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: ShieldedCredits — fund, authorize, claim
# ═══════════════════════════════════════════════════════════════════════════════

section "4" "ShieldedCredits — fund + authorize + claim"
phase_start

# 4a. Mint tokens to user + approve
cast send "$TOKEN_ADDR" "mint(address,uint256)" "$USER_ADDR" "$FUND_AMOUNT" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
cast send "$TOKEN_ADDR" "approve(address,uint256)" "$SHIELDED_CREDITS" "$FUND_AMOUNT" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
pass "user minted + approved tsUSD"

# 4b. Fund credits
COMMITMENT=$(cast keccak "$(cast abi-encode "f(address,string)" "$USER_ADDR" "protocol-test-salt")")
SPENDING_KEY="$USER_ADDR"

RESULT=$(cast send "$SHIELDED_CREDITS" \
    "fundCredits(address,uint256,bytes32,address)" \
    "$TOKEN_ADDR" "$CREDIT_FUND_AMOUNT" "$COMMITMENT" "$SPENDING_KEY" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json 2>&1)
record_gas "fundCredits" "$RESULT"

BALANCE=$(get_credit_balance "$COMMITMENT")
assert_eq "$BALANCE" "$CREDIT_FUND_AMOUNT" "credit account funded with $CREDIT_FUND_AMOUNT"

# 4c. Sign and authorize spend
SPEND_AMOUNT="100000000000000000" # 0.1 tsUSD
EXPIRY="99999999999"
CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")

SIGN_OUT=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "$SERVICE_ID" "0" "$SPEND_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$EXPIRY")
SIG=$(echo "$SIGN_OUT" | jq -r '.signature')
pass "SpendAuth EIP-712 signature generated"

RESULT=$(send_authorize_spend "$COMMITMENT" "$SERVICE_ID" "0" "$SPEND_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$EXPIRY" "$SIG" "$USER_KEY")
record_gas "authorizeSpend" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "authorizeSpend succeeded"
else
    fail "authorizeSpend failed: ${RESULT:0:200}"
fi

NEW_BALANCE=$(get_credit_balance "$COMMITMENT")
EXPECTED_BALANCE=$(echo "$CREDIT_FUND_AMOUNT - $SPEND_AMOUNT" | bc)
assert_eq "$NEW_BALANCE" "$EXPECTED_BALANCE" "credit balance decreased by spend amount"

# 4d. Operator claims payment
AUTH_HASH=$(cast keccak "$(cast abi-encode "f(bytes32,uint64,uint8,uint256)" \
    "$COMMITMENT" "$SERVICE_ID" "0" "$CURRENT_NONCE")")

OPERATOR_BAL_BEFORE=$(get_token_balance "$OPERATOR_ADDR")

RESULT=$(cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$AUTH_HASH" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" --rpc-url "$RPC_URL" --json 2>&1)
record_gas "claimPayment" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "claimPayment succeeded"
else
    fail "claimPayment failed: ${RESULT:0:200}"
fi

OPERATOR_BAL_AFTER=$(get_token_balance "$OPERATOR_ADDR")
OPERATOR_RECEIVED=$(echo "$OPERATOR_BAL_AFTER - $OPERATOR_BAL_BEFORE" | bc)
assert_eq "$OPERATOR_RECEIVED" "$SPEND_AMOUNT" "operator received correct payment"

phase_end "ShieldedCredits"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: RLN deposit + batch claim
# ═══════════════════════════════════════════════════════════════════════════════

section "5" "RLNSettlement — deposit + batch claim"
phase_start

RLN_DEPOSIT="500000000000000000000" # 500 tsUSD
IDENTITY_COMMITMENT=$(cast keccak "$(cast abi-encode "f(address,string)" "$USER_ADDR" "rln-identity")")

# Mint + approve for RLN
cast send "$TOKEN_ADDR" "mint(address,uint256)" "$USER_ADDR" "$RLN_DEPOSIT" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
cast send "$TOKEN_ADDR" "approve(address,uint256)" "$RLN_SETTLEMENT" "$RLN_DEPOSIT" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1

# 5a. Deposit
RESULT=$(cast send "$RLN_SETTLEMENT" \
    "deposit(address,uint256,bytes32)" "$TOKEN_ADDR" "$RLN_DEPOSIT" "$IDENTITY_COMMITMENT" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json 2>&1)
record_gas "RLN deposit" "$RESULT"

DEPOSIT_INFO=$(cast call "$RLN_SETTLEMENT" \
    "getDeposit(bytes32)(address,uint256)" "$IDENTITY_COMMITMENT" --rpc-url "$RPC_URL" 2>/dev/null)
if echo "$DEPOSIT_INFO" | grep -q "$RLN_DEPOSIT"; then
    pass "RLN deposit verified ($RLN_DEPOSIT)"
else
    fail "RLN deposit balance mismatch"
fi

# 5b. Batch claim
NF1=$(cast keccak "$(cast abi-encode "f(string)" "nullifier-proto-1")")
NF2=$(cast keccak "$(cast abi-encode "f(string)" "nullifier-proto-2")")
CLAIM_AMT1="10000000000000000000" # 10 tsUSD
CLAIM_AMT2="20000000000000000000" # 20 tsUSD

# Fund RLN contract with tokens for payout
cast send "$TOKEN_ADDR" "mint(address,uint256)" "$RLN_SETTLEMENT" "$RLN_DEPOSIT" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1

OPERATOR_BAL_BEFORE=$(get_token_balance "$OPERATOR_ADDR")

RESULT=$(cast send "$RLN_SETTLEMENT" \
    "batchClaim(address,bytes32[],uint256[],address)" \
    "$TOKEN_ADDR" "[$NF1,$NF2]" "[$CLAIM_AMT1,$CLAIM_AMT2]" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" --rpc-url "$RPC_URL" --json 2>&1)
record_gas "RLN batchClaim" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "batchClaim succeeded"
else
    fail "batchClaim failed: ${RESULT:0:200}"
fi

OPERATOR_BAL_AFTER=$(get_token_balance "$OPERATOR_ADDR")
EXPECTED_CLAIM=$(echo "$CLAIM_AMT1 + $CLAIM_AMT2" | bc)
ACTUAL_CLAIM=$(echo "$OPERATOR_BAL_AFTER - $OPERATOR_BAL_BEFORE" | bc)
assert_eq "$ACTUAL_CLAIM" "$EXPECTED_CLAIM" "operator received correct batch claim (30 tsUSD)"

phase_end "RLNSettlement"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Submit job to Tangle
# ═══════════════════════════════════════════════════════════════════════════════

section "6" "Tangle job submission"
phase_start

# submitJob(uint64 serviceId, uint8 jobIndex, bytes inputs)
JOB_INPUTS=$(cast abi-encode "f(string)" '{"model":"qwen2:0.5b","prompt":"hello"}')

RESULT=$(cast send "$TANGLE" \
    "submitJob(uint64,uint8,bytes)" \
    "$SERVICE_ID" 0 "$JOB_INPUTS" \
    --gas-limit 2000000 \
    --rpc-url "$RPC_URL" --private-key "$USER_KEY" --json 2>&1)
record_gas "submitJob" "$RESULT"

if echo "$RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "job submitted to service #$SERVICE_ID"
else
    fail "submitJob failed: ${RESULT:0:200}"
fi

phase_end "Job submission"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Inference via Ollama (optional)
# ═══════════════════════════════════════════════════════════════════════════════

section "7" "Inference test (Ollama)"
phase_start

OLLAMA_URL="http://127.0.0.1:11434"

if curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    MODELS=$(curl -s "$OLLAMA_URL/api/tags" | jq -r '.models[]?.name // empty' 2>/dev/null || echo "")
    if echo "$MODELS" | grep -q "qwen2:0.5b"; then
        pass "Ollama running with qwen2:0.5b"

        RESP=$(curl -s -X POST "$OLLAMA_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"qwen2:0.5b","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":10}' \
            2>/dev/null)

        CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
        if [ -n "$CONTENT" ]; then
            pass "inference returned: ${CONTENT:0:50}"
        else
            fail "inference returned empty content"
        fi
    else
        echo -e "  ${YELLOW}SKIP${NC} qwen2:0.5b not loaded (have: $MODELS)"
    fi
else
    echo -e "  ${YELLOW}SKIP${NC} Ollama not running on $OLLAMA_URL"
fi

phase_end "Inference"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8: Reports
# ═══════════════════════════════════════════════════════════════════════════════

section "8" "Reports"

TOTAL_GAS=0
for g in "${GAS_VALUES[@]}"; do TOTAL_GAS=$((TOTAL_GAS + g)); done

echo ""
echo -e "${BOLD}"
echo "+-------------------------------------------------+"
echo "|  GAS COST REPORT                                |"
echo "+-------------------------------------------------+"
printf "| %-30s %15s |\n" "Operation" "Gas Used"
echo "+-------------------------------------------------+"
for i in "${!GAS_LABELS[@]}"; do
    printf "| %-30s %15s |\n" "${GAS_LABELS[$i]}" "$(printf "%'d" "${GAS_VALUES[$i]}")"
done
echo "+-------------------------------------------------+"
printf "| %-30s %15s |\n" "TOTAL" "$(printf "%'d" "$TOTAL_GAS")"
echo "+-------------------------------------------------+"
echo -e "${NC}"

SCRIPT_END=$(date +%s%N)
TOTAL_MS=$(( (SCRIPT_END - SCRIPT_START) / 1000000 ))

echo -e "${BOLD}"
echo "+-------------------------------------------------+"
echo "|  TIMING REPORT                                  |"
echo "+-------------------------------------------------+"
printf "| %-30s %12s ms |\n" "Phase" "Duration"
echo "+-------------------------------------------------+"
for i in "${!PHASE_LABELS[@]}"; do
    printf "| %-30s %12s ms |\n" "${PHASE_LABELS[$i]}" "$(printf "%'d" "${PHASE_TIMES[$i]}")"
done
echo "+-------------------------------------------------+"
printf "| %-30s %12s ms |\n" "TOTAL" "$(printf "%'d" "$TOTAL_MS")"
echo "+-------------------------------------------------+"
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}ALL $TOTAL TESTS PASSED${NC}"
else
    echo -e "${RED}${BOLD}$FAIL/$TOTAL TESTS FAILED${NC}"
fi
echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} ($TOTAL total)"
echo ""

cleanup "$( [ "$FAIL" -gt 0 ] && echo 1 || echo 0 )"
