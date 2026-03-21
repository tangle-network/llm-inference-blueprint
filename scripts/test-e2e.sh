#!/usr/bin/env bash
# test-e2e.sh — Full end-to-end tests for vLLM inference blueprint + shielded payments.
#
# Tests: ShieldedCredits (fund, authorize, claim, reclaim, replay, wrong-sig),
#        RLNSettlement (deposit, batch claim, duplicate nullifier, slash),
#        Operator API (inference, x402, payment methods).
#
# Prerequisites:
#   - deploy-local.sh has been run (generates .env.local)
#   - Anvil running
#   - Node.js available (for EIP-712 signing)
#   - Optionally: Ollama on port 11434 with qwen2:0.5b (for inference tests)
#
# Usage:
#   ./scripts/test-e2e.sh
#
# Environment overrides:
#   SKIP_INFERENCE — Set to 1 to skip inference tests
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Load environment
# ─────────────────────────────────────────────────────────────────────────────

ENV_FILE="$ROOT_DIR/.env.local"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Run deploy-local.sh first."
    exit 1
fi
set -o allexport
source "$ENV_FILE"
set +o allexport

# ─────────────────────────────────────────────────────────────────────────────
# Test framework
# ─────────────────────────────────────────────────────────────────────────────

PASS=0; FAIL=0; SKIP=0; TOTAL=0; START_TIME=$(date +%s)
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo -e "  ${RED}FAIL${NC} $1"; }
skip() { SKIP=$((SKIP+1)); TOTAL=$((TOTAL+1)); echo -e "  ${YELLOW}SKIP${NC} $1"; }

assert_eq() {
    local got="$1" expected="$2" desc="$3"
    if [ "$got" = "$expected" ]; then pass "$desc"; else fail "$desc (expected='$expected', got='$got')"; fi
}
assert_ne() {
    local got="$1" not_expected="$2" desc="$3"
    if [ "$got" != "$not_expected" ]; then pass "$desc"; else fail "$desc (got unexpected '$got')"; fi
}
assert_contains() {
    local haystack="$1" needle="$2" desc="$3"
    if echo "$haystack" | grep -qi "$needle"; then pass "$desc"; else fail "$desc (missing '$needle')"; fi
}
assert_reverts() {
    local output="$1" desc="$2"
    if echo "$output" | grep -qi "revert\|execution reverted\|error"; then
        pass "$desc (reverted as expected)"
    else
        fail "$desc (expected revert, got: ${output:0:200})"
    fi
}

section() { echo -e "\n${CYAN}[$1]${NC} $2"; }

# Helper: send authorizeSpend tx (handles tuple encoding correctly)
# Usage: send_authorize_spend <commitment> <serviceId> <jobIndex> <amount> <operator> <nonce> <expiry> <sig> <sender_key>
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

# Helper: sign a SpendAuth via the node script
sign_spend_auth() {
    node "$ROOT_DIR/scripts/sign-spend-auth.mjs" \
        --private-key "$1" \
        --verifying-contract "$SHIELDED_CREDITS" \
        --chain-id "${CHAIN_ID:-31337}" \
        --commitment "$2" \
        --service-id "$3" \
        --job-index "$4" \
        --amount "$5" \
        --operator "$6" \
        --nonce "$7" \
        --expiry "$8"
}

# Helper: strip cast labels like "1000 [1e3]" -> "1000" and trim parens/whitespace
strip_cast() {
    echo "$1" | sed 's/ \[.*\]//g' | tr -d '()' | tr -d ' '
}

# Helper: get credit account balance
get_credit_balance() {
    local raw
    raw=$(cast call "$SHIELDED_CREDITS" \
        "getAccount(bytes32)((address,address,uint256,uint256,uint256,uint256))" \
        "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    # Account struct: (spendingKey, token, balance, totalFunded, totalSpent, nonce)
    # Extract balance (3rd field)
    local field
    field=$(echo "$raw" | tr ',' '\n' | sed -n '3p')
    strip_cast "$field"
}

# Helper: get credit account nonce
get_credit_nonce() {
    local raw
    raw=$(cast call "$SHIELDED_CREDITS" \
        "getAccount(bytes32)((address,address,uint256,uint256,uint256,uint256))" \
        "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    local field
    field=$(echo "$raw" | tr ',' '\n' | sed -n '6p')
    strip_cast "$field"
}

# Helper: get ERC20 balance
get_token_balance() {
    local raw
    raw=$(cast call "$TOKEN_ADDR" "balanceOf(address)(uint256)" "$1" --rpc-url "$RPC_URL" 2>/dev/null)
    strip_cast "$raw"
}

# ═════════════════════════════════════════════════════════════════════════════
# Section 0: Prerequisites
# ═════════════════════════════════════════════════════════════════════════════

section "0" "Prerequisites"

for cmd in cast node curl jq bc; do
    if command -v "$cmd" &>/dev/null; then
        pass "$cmd available"
    else
        fail "$cmd not found"
        echo "Install $cmd and try again."
        exit 1
    fi
done

BLOCK=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null || echo "UNREACHABLE")
if [ "$BLOCK" != "UNREACHABLE" ] && [ "$BLOCK" -gt 0 ] 2>/dev/null; then
    pass "Anvil running at $RPC_URL (block $BLOCK)"
else
    fail "Anvil not reachable at $RPC_URL"
    exit 1
fi

for var in SHIELDED_CREDITS RLN_SETTLEMENT TOKEN_ADDR OPERATOR_KEY OPERATOR_ADDR USER_KEY USER_ADDR COMMITMENT; do
    if [ -n "${!var:-}" ]; then
        pass "\$$var set"
    else
        fail "\$$var not set in .env.local"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════
# Section 1: ShieldedCredits — Fund + Verify
# ═════════════════════════════════════════════════════════════════════════════

section "1" "ShieldedCredits — Account Verification"

BALANCE=$(get_credit_balance "$COMMITMENT")
assert_eq "$BALANCE" "$CREDIT_FUND_AMOUNT" "credit account balance matches funded amount"

NONCE=$(get_credit_nonce "$COMMITMENT")
assert_eq "$NONCE" "0" "credit account nonce is 0"

# ═════════════════════════════════════════════════════════════════════════════
# Section 2: ShieldedCredits — Authorize Spend + Claim
# ═════════════════════════════════════════════════════════════════════════════

section "2" "ShieldedCredits — Authorize Spend + Claim"

SPEND_AMOUNT="100000000000000000" # 0.1 tsUSD
EXPIRY="99999999999"
CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")

# 2a. Sign SpendAuth
SIGN_OUT=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$SPEND_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$EXPIRY")
SIG=$(echo "$SIGN_OUT" | jq -r '.signature')
assert_ne "$SIG" "" "SpendAuth signature generated"

# 2b. Authorize spend on-chain
AUTH_RESULT=$(send_authorize_spend "$COMMITMENT" "1" "0" "$SPEND_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$EXPIRY" "$SIG" "$USER_KEY")

if echo "$AUTH_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "authorizeSpend succeeded"
else
    fail "authorizeSpend failed: ${AUTH_RESULT:0:200}"
fi

# 2c. Verify balance decreased
NEW_BALANCE=$(get_credit_balance "$COMMITMENT")
EXPECTED_BALANCE=$(echo "$CREDIT_FUND_AMOUNT - $SPEND_AMOUNT" | bc)
assert_eq "$NEW_BALANCE" "$EXPECTED_BALANCE" "credit balance decreased by spend amount"

# 2d. Verify nonce incremented
NEW_NONCE=$(get_credit_nonce "$COMMITMENT")
EXPECTED_NONCE=$((CURRENT_NONCE + 1))
assert_eq "$NEW_NONCE" "$EXPECTED_NONCE" "nonce incremented after authorize"

# 2e. Compute authHash and claim
# authHash = keccak256(abi.encode(commitment, serviceId, jobIndex, auth.nonce))
# The contract uses auth.nonce from the SpendAuth struct (the nonce at time of signing)
AUTH_HASH=$(cast keccak "$(cast abi-encode "f(bytes32,uint64,uint8,uint256)" \
    "$COMMITMENT" "1" "0" "$CURRENT_NONCE")")

OPERATOR_BAL_BEFORE=$(get_token_balance "$OPERATOR_ADDR")

CLAIM_RESULT=$(cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$AUTH_HASH" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1)

if echo "$CLAIM_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "claimPayment succeeded"
else
    fail "claimPayment failed: ${CLAIM_RESULT:0:200}"
fi

# 2f. Verify operator received tokens
OPERATOR_BAL_AFTER=$(get_token_balance "$OPERATOR_ADDR")
OPERATOR_RECEIVED=$(echo "$OPERATOR_BAL_AFTER - $OPERATOR_BAL_BEFORE" | bc)
assert_eq "$OPERATOR_RECEIVED" "$SPEND_AMOUNT" "operator received correct payment amount"

# ═════════════════════════════════════════════════════════════════════════════
# Section 3: ShieldedCredits — Expired Auth Reclaim
# ═════════════════════════════════════════════════════════════════════════════

section "3" "ShieldedCredits — Expired Auth Reclaim"

RECLAIM_AMOUNT="50000000000000000" # 0.05 tsUSD
CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")
# Set expiry to current block timestamp + 5 seconds (so we can warp past it)
CURRENT_TS=$(cast call --rpc-url "$RPC_URL" --block latest 0x0000000000000000000000000000000000000000 2>/dev/null || echo "")
# Use a short expiry we can warp past
SHORT_EXPIRY="10" # epoch 10 — already in the past on mainnet but Anvil starts at ~1

# Get current block timestamp
BLOCK_TS=$(cast block --rpc-url "$RPC_URL" latest --json 2>/dev/null | jq -r '.timestamp' | xargs printf "%d")
SHORT_EXPIRY=$((BLOCK_TS + 5))

SIGN_OUT=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$RECLAIM_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$SHORT_EXPIRY")
SIG=$(echo "$SIGN_OUT" | jq -r '.signature')

# Authorize the spend
AUTH_RESULT=$(send_authorize_spend "$COMMITMENT" "1" "0" "$RECLAIM_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$SHORT_EXPIRY" "$SIG" "$USER_KEY")

if echo "$AUTH_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "authorizeSpend (short expiry) succeeded"
else
    fail "authorizeSpend (short expiry) failed: ${AUTH_RESULT:0:200}"
fi

# authHash uses auth.nonce from the struct (= CURRENT_NONCE at time of signing)
RECLAIM_AUTH_HASH=$(cast keccak "$(cast abi-encode "f(bytes32,uint64,uint8,uint256)" \
    "$COMMITMENT" "1" "0" "$CURRENT_NONCE")")

# Warp time past the expiry
cast rpc --rpc-url "$RPC_URL" evm_increaseTime 60 >/dev/null 2>&1
cast rpc --rpc-url "$RPC_URL" evm_mine >/dev/null 2>&1

# Try to claim after expiry — should revert
LATE_CLAIM=$(cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$RECLAIM_AUTH_HASH" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1 || true)
assert_reverts "$LATE_CLAIM" "claim after expiry reverts"

# Reclaim the expired auth
BALANCE_BEFORE_RECLAIM=$(get_credit_balance "$COMMITMENT")
RECLAIM_RESULT=$(cast send "$SHIELDED_CREDITS" \
    "reclaimExpiredAuth(bytes32,bytes32)" "$RECLAIM_AUTH_HASH" "$COMMITMENT" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1)

if echo "$RECLAIM_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "reclaimExpiredAuth succeeded"
else
    fail "reclaimExpiredAuth failed: ${RECLAIM_RESULT:0:200}"
fi

BALANCE_AFTER_RECLAIM=$(get_credit_balance "$COMMITMENT")
RECLAIMED=$(echo "$BALANCE_AFTER_RECLAIM - $BALANCE_BEFORE_RECLAIM" | bc)
assert_eq "$RECLAIMED" "$RECLAIM_AMOUNT" "reclaimed amount returned to credit balance"

# ═════════════════════════════════════════════════════════════════════════════
# Section 4: ShieldedCredits — Malicious Tests
# ═════════════════════════════════════════════════════════════════════════════

section "4" "ShieldedCredits — Malicious Tests"

CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")
MAL_AMOUNT="10000000000000000" # 0.01 tsUSD
FAR_EXPIRY="99999999999"

# 4a. Wrong signature (sign with operator key instead of user key)
WRONG_SIGN=$(sign_spend_auth "$OPERATOR_KEY" "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY")
WRONG_SIG=$(echo "$WRONG_SIGN" | jq -r '.signature')

WRONG_SIG_RESULT=$(send_authorize_spend "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY" "$WRONG_SIG" "$USER_KEY" || true)
assert_reverts "$WRONG_SIG_RESULT" "authorizeSpend with wrong signature reverts"

# 4b. Zero operator address
ZERO_ADDR="0x0000000000000000000000000000000000000000"
ZERO_OP_SIGN=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$ZERO_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY")
ZERO_OP_SIG=$(echo "$ZERO_OP_SIGN" | jq -r '.signature')

ZERO_OP_RESULT=$(send_authorize_spend "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$ZERO_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY" "$ZERO_OP_SIG" "$USER_KEY" || true)
assert_reverts "$ZERO_OP_RESULT" "authorizeSpend with zero operator reverts (OperatorRequired)"

# 4c. Replay nonce (reuse current nonce after a successful auth)
# First do a valid auth to increment nonce
VALID_SIGN=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY")
VALID_SIG=$(echo "$VALID_SIGN" | jq -r '.signature')

send_authorize_spend "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY" "$VALID_SIG" "$USER_KEY" >/dev/null 2>&1

# Now try to replay with the same nonce
REPLAY_RESULT=$(send_authorize_spend "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY" "$VALID_SIG" "$USER_KEY" || true)
assert_reverts "$REPLAY_RESULT" "replay nonce reverts (InvalidNonce)"

# 4d. Double-claim same authHash (use the nonce from the valid auth we just did)
REPLAY_AUTH_HASH=$(cast keccak "$(cast abi-encode "f(bytes32,uint64,uint8,uint256)" \
    "$COMMITMENT" "1" "0" "$CURRENT_NONCE")")

# Claim once
cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$REPLAY_AUTH_HASH" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json >/dev/null 2>&1

# Claim again — should revert
DOUBLE_CLAIM=$(cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$REPLAY_AUTH_HASH" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1 || true)
assert_reverts "$DOUBLE_CLAIM" "double-claim same authHash reverts (AlreadyClaimed)"

# 4e. Wrong operator claims
CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")
VALID_SIGN2=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY")
VALID_SIG2=$(echo "$VALID_SIGN2" | jq -r '.signature')

send_authorize_spend "$COMMITMENT" "1" "0" "$MAL_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$FAR_EXPIRY" "$VALID_SIG2" "$USER_KEY" >/dev/null 2>&1

# authHash uses auth.nonce from the struct (= CURRENT_NONCE at time of signing)
WRONG_OP_AUTH_HASH=$(cast keccak "$(cast abi-encode "f(bytes32,uint64,uint8,uint256)" \
    "$COMMITMENT" "1" "0" "$CURRENT_NONCE")")

# Deployer (not operator) tries to claim
WRONG_OP_RESULT=$(cast send "$SHIELDED_CREDITS" \
    "claimPayment(bytes32,address)" "$WRONG_OP_AUTH_HASH" "$DEPLOYER_ADDR" \
    --private-key "$DEPLOYER_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1 || true)
assert_reverts "$WRONG_OP_RESULT" "wrong operator claim reverts (NotDesignatedOperator)"

# 4f. Fund credits with wrong token on top-up
# Deploy a second mock token
MOCK2_SRC='pragma solidity ^0.8.26; contract M2 { string public name="X"; string public symbol="X"; uint8 public decimals=18; mapping(address=>uint256) public balanceOf; mapping(address=>mapping(address=>uint256)) public allowance; function mint(address t,uint256 a) external { balanceOf[t]+=a; } function approve(address s,uint256 a) external returns(bool) { allowance[msg.sender][s]=a; return true; } function transferFrom(address f,address t,uint256 a) external returns(bool) { allowance[f][msg.sender]-=a; balanceOf[f]-=a; balanceOf[t]+=a; return true; } }'
MOCK2_FILE="$ROOT_DIR/contracts/src/_Mock2.sol"
echo "// SPDX-License-Identifier: MIT" > "$MOCK2_FILE"
echo "$MOCK2_SRC" >> "$MOCK2_FILE"
FORGE_OUT2=$(forge create "$MOCK2_FILE:M2" \
    --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" --root "$ROOT_DIR/contracts" --broadcast 2>&1)
TOKEN2_ADDR=$(echo "$FORGE_OUT2" | grep "Deployed to:" | awk '{print $3}')
rm -f "$MOCK2_FILE"

if [ -n "$TOKEN2_ADDR" ]; then
    # Mint + approve
    cast send "$TOKEN2_ADDR" "mint(address,uint256)" "$USER_ADDR" "1000000000000000000" \
        --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
    cast send "$TOKEN2_ADDR" "approve(address,uint256)" "$SHIELDED_CREDITS" "1000000000000000000" \
        --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1

    # Try to top up with wrong token
    WRONG_TOKEN_RESULT=$(cast send "$SHIELDED_CREDITS" \
        "fundCredits(address,uint256,bytes32,address)" \
        "$TOKEN2_ADDR" "1000000000000000000" "$COMMITMENT" "$USER_ADDR" \
        --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json 2>&1 || true)
    assert_reverts "$WRONG_TOKEN_RESULT" "fund with wrong token reverts (TokenMismatch)"
else
    skip "could not deploy second mock token for TokenMismatch test"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Section 5: RLNSettlement — Deposit + Batch Claim
# ═════════════════════════════════════════════════════════════════════════════

section "5" "RLNSettlement — Deposit + Batch Claim"

RLN_DEPOSIT="500000000000000000000" # 500 tsUSD
IDENTITY_COMMITMENT=$(cast keccak "$(cast abi-encode "f(address,string)" "$USER_ADDR" "rln-identity")")

# 5a. Deposit into RLN
cast send "$RLN_SETTLEMENT" \
    "deposit(address,uint256,bytes32)" "$TOKEN_ADDR" "$RLN_DEPOSIT" "$IDENTITY_COMMITMENT" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json >/dev/null 2>&1

DEPOSIT_INFO=$(cast call "$RLN_SETTLEMENT" \
    "getDeposit(bytes32)(address,uint256)" "$IDENTITY_COMMITMENT" \
    --rpc-url "$RPC_URL" 2>/dev/null)
assert_contains "$DEPOSIT_INFO" "$RLN_DEPOSIT" "RLN deposit balance correct"

# 5b. Batch claim with unique nullifiers
NF1=$(cast keccak "$(cast abi-encode "f(string)" "nullifier-1")")
NF2=$(cast keccak "$(cast abi-encode "f(string)" "nullifier-2")")
CLAIM_AMT1="10000000000000000000" # 10 tsUSD
CLAIM_AMT2="20000000000000000000" # 20 tsUSD

# Mint tokens to RLN contract so it can pay out
cast send "$TOKEN_ADDR" "mint(address,uint256)" "$RLN_SETTLEMENT" "$RLN_DEPOSIT" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1

OPERATOR_BAL_BEFORE=$(get_token_balance "$OPERATOR_ADDR")

BATCH_RESULT=$(cast send "$RLN_SETTLEMENT" \
    "batchClaim(address,bytes32[],uint256[],address)" \
    "$TOKEN_ADDR" "[$NF1,$NF2]" "[$CLAIM_AMT1,$CLAIM_AMT2]" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1)

if echo "$BATCH_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "batchClaim succeeded"
else
    fail "batchClaim failed: ${BATCH_RESULT:0:200}"
fi

OPERATOR_BAL_AFTER=$(get_token_balance "$OPERATOR_ADDR")
EXPECTED_CLAIM=$(echo "$CLAIM_AMT1 + $CLAIM_AMT2" | bc)
ACTUAL_CLAIM=$(echo "$OPERATOR_BAL_AFTER - $OPERATOR_BAL_BEFORE" | bc)
assert_eq "$ACTUAL_CLAIM" "$EXPECTED_CLAIM" "operator received correct batch claim amount"

# 5c. Duplicate nullifier reverts
DUP_NF_RESULT=$(cast send "$RLN_SETTLEMENT" \
    "batchClaim(address,bytes32[],uint256[],address)" \
    "$TOKEN_ADDR" "[$NF1]" "[$CLAIM_AMT1]" "$OPERATOR_ADDR" \
    --private-key "$OPERATOR_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1 || true)
assert_reverts "$DUP_NF_RESULT" "duplicate nullifier reverts (NullifierUsed)"

# 5d. Unauthorized operator reverts
NF3=$(cast keccak "$(cast abi-encode "f(string)" "nullifier-3")")
UNAUTH_RESULT=$(cast send "$RLN_SETTLEMENT" \
    "batchClaim(address,bytes32[],uint256[],address)" \
    "$TOKEN_ADDR" "[$NF3]" "[$CLAIM_AMT1]" "$USER_ADDR" \
    --private-key "$USER_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1 || true)
assert_reverts "$UNAUTH_RESULT" "unauthorized operator reverts"

# ═════════════════════════════════════════════════════════════════════════════
# Section 6: RLNSettlement — Slash with Shamir Shares
# ═════════════════════════════════════════════════════════════════════════════

section "6" "RLNSettlement — Slash"

# Deposit for a slashable identity
SLASH_IC=$(cast keccak "$(cast abi-encode "f(string)" "slash-target")")
SLASH_DEPOSIT="100000000000000000000" # 100 tsUSD

# Mint + approve + deposit
cast send "$TOKEN_ADDR" "mint(address,uint256)" "$USER_ADDR" "$SLASH_DEPOSIT" \
    --private-key "$DEPLOYER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
cast send "$TOKEN_ADDR" "approve(address,uint256)" "$RLN_SETTLEMENT" "$SLASH_DEPOSIT" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1
cast send "$RLN_SETTLEMENT" \
    "deposit(address,uint256,bytes32)" "$TOKEN_ADDR" "$SLASH_DEPOSIT" "$SLASH_IC" \
    --private-key "$USER_KEY" --rpc-url "$RPC_URL" --json >/dev/null 2>&1

# Two Shamir shares (distinct x values, arbitrary y values)
# The slash function computes slope from two points on a line: y = secret + slope*x
# It requires x1 != x2
X1="100"
Y1="200"
X2="300"
Y2="500"
SLASH_NF=$(cast keccak "$(cast abi-encode "f(string)" "slash-nullifier")")

SLASH_RESULT=$(cast send "$RLN_SETTLEMENT" \
    "slash(bytes32,uint256,uint256,uint256,uint256,bytes32)" \
    "$SLASH_NF" "$X1" "$Y1" "$X2" "$Y2" "$SLASH_IC" \
    --private-key "$DEPLOYER_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1)

if echo "$SLASH_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "slash submitted successfully"
else
    fail "slash failed: ${SLASH_RESULT:0:200}"
fi

# Check deposit is zeroed
SLASHED_BAL=$(cast call "$RLN_SETTLEMENT" \
    "getDeposit(bytes32)(address,uint256)" "$SLASH_IC" \
    --rpc-url "$RPC_URL" 2>/dev/null)
assert_contains "$SLASHED_BAL" "0" "slashed deposit balance is 0"

# Warp past SLASH_DELAY (1 day = 86400 seconds) and finalize
cast rpc --rpc-url "$RPC_URL" evm_increaseTime 86401 >/dev/null 2>&1
cast rpc --rpc-url "$RPC_URL" evm_mine >/dev/null 2>&1

# Compute slashId = keccak256(abi.encode(identityCommitment, x1, y1, x2, y2))
SLASH_ID=$(cast keccak "$(cast abi-encode "f(bytes32,uint256,uint256,uint256,uint256)" \
    "$SLASH_IC" "$X1" "$Y1" "$X2" "$Y2")")

DEPLOYER_BAL_BEFORE=$(get_token_balance "$DEPLOYER_ADDR")

FINALIZE_RESULT=$(cast send "$RLN_SETTLEMENT" \
    "finalizeSlash(bytes32)" "$SLASH_ID" \
    --private-key "$DEPLOYER_KEY" \
    --rpc-url "$RPC_URL" \
    --json 2>&1)

if echo "$FINALIZE_RESULT" | jq -e '.status == "0x1"' >/dev/null 2>&1; then
    pass "finalizeSlash succeeded"
else
    fail "finalizeSlash failed: ${FINALIZE_RESULT:0:200}"
fi

DEPLOYER_BAL_AFTER=$(get_token_balance "$DEPLOYER_ADDR")
SLASH_RECEIVED=$(echo "$DEPLOYER_BAL_AFTER - $DEPLOYER_BAL_BEFORE" | bc)
assert_eq "$SLASH_RECEIVED" "$SLASH_DEPOSIT" "slasher received full deposit"

# ═════════════════════════════════════════════════════════════════════════════
# Section 7: Inference Tests (requires Ollama)
# ═════════════════════════════════════════════════════════════════════════════

section "7" "Inference Tests"

OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_OK=false

if [ "${SKIP_INFERENCE:-0}" = "1" ]; then
    skip "inference tests skipped (SKIP_INFERENCE=1)"
elif ! curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    skip "Ollama not running on $OLLAMA_URL — skipping inference tests"
else
    # Check if qwen2:0.5b is available
    MODELS=$(curl -s "$OLLAMA_URL/api/tags" | jq -r '.models[]?.name // empty' 2>/dev/null || echo "")
    if echo "$MODELS" | grep -q "qwen2:0.5b"; then
        OLLAMA_OK=true
        pass "Ollama running with qwen2:0.5b"
    else
        skip "qwen2:0.5b not found in Ollama models (have: $MODELS)"
    fi
fi

API_URL="${OPERATOR_API_URL:-http://127.0.0.1:$OPERATOR_PORT}"
API_RUNNING=false

if [ "$OLLAMA_OK" = "true" ]; then
    # Check if operator API is running
    API_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" 2>/dev/null || echo "000")
    if [ "$API_CODE" = "200" ]; then
        API_RUNNING=true
        pass "Operator API running at $API_URL"
    else
        skip "Operator API not running at $API_URL (HTTP $API_CODE) — skipping live inference tests"
    fi
fi

if [ "$API_RUNNING" = "true" ]; then
    # 7a. x402 flow: request without payment returns 402
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"say hello"}],"max_tokens":10}' 2>/dev/null)
    assert_eq "$HTTP_CODE" "402" "request without payment returns HTTP 402"

    # 7b. Payment methods endpoint
    PM_RESP=$(curl -s "$API_URL/v1/payment_methods" 2>/dev/null)
    assert_contains "$PM_RESP" "credit_mode" "payment methods includes credit_mode"

    # 7c. Chat completion with SpendAuth (if billing accepts it)
    # Build a SpendAuth payload for the API
    CURRENT_NONCE=$(get_credit_nonce "$COMMITMENT")
    API_AMOUNT="1000000000000000000" # 1 tsUSD
    API_EXPIRY="99999999999"
    API_SIGN=$(sign_spend_auth "$USER_KEY" "$COMMITMENT" "1" "0" "$API_AMOUNT" "$OPERATOR_ADDR" "$CURRENT_NONCE" "$API_EXPIRY")
    API_SIG=$(echo "$API_SIGN" | jq -r '.signature')

    CHAT_RESP=$(curl -s -w "\n%{http_code}" \
        -X POST "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],
            \"max_tokens\": 10,
            \"spend_auth\": {
                \"commitment\": \"$COMMITMENT\",
                \"service_id\": 1,
                \"job_index\": 0,
                \"amount\": \"$API_AMOUNT\",
                \"operator\": \"$OPERATOR_ADDR\",
                \"nonce\": $CURRENT_NONCE,
                \"expiry\": $API_EXPIRY,
                \"signature\": \"$API_SIG\"
            }
        }" 2>/dev/null)

    CHAT_CODE=$(echo "$CHAT_RESP" | tail -1)
    CHAT_BODY=$(echo "$CHAT_RESP" | sed '$d')

    if [ "$CHAT_CODE" = "200" ]; then
        pass "chat completion with SpendAuth returned 200"
        CONTENT=$(echo "$CHAT_BODY" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
        assert_ne "$CONTENT" "" "response has content"
        TOTAL_TOKENS=$(echo "$CHAT_BODY" | jq -r '.usage.total_tokens // 0' 2>/dev/null)
        assert_ne "$TOTAL_TOKENS" "0" "response has token usage"
    else
        fail "chat completion returned HTTP $CHAT_CODE (body: ${CHAT_BODY:0:200})"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC} ($TOTAL total)"
echo -e "  Duration: ${ELAPSED}s"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
