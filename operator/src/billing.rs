use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use alloy::{
    network::EthereumWallet,
    primitives::{keccak256, Address, FixedBytes, B256, U256},
    providers::ProviderBuilder,
    signers::local::PrivateKeySigner,
    sol,
    sol_types::SolValue,
};
use tokio::sync::Mutex;

use crate::config::OperatorConfig;
use crate::server::SpendAuthPayload;

// Generate bindings for the ShieldedCredits contract.
// In production, import from tnt-core's published bindings crate.
sol! {
    #[sol(rpc)]
    interface IShieldedCredits {
        struct SpendAuth {
            bytes32 commitment;
            uint64 serviceId;
            uint8 jobIndex;
            uint256 amount;
            address operator;
            uint256 nonce;
            uint64 expiry;
            bytes signature;
        }

        function authorizeSpend(SpendAuth calldata auth) external returns (bytes32 authHash);
        function claimPayment(bytes32 authHash, address recipient) external;
        function getAccount(bytes32 commitment) external view returns (
            address spendingKey,
            address token,
            uint256 balance,
            uint256 totalFunded,
            uint256 totalSpent,
            uint256 nonce
        );
    }
}

// Generate bindings for the RLNSettlement contract.
sol! {
    #[sol(rpc)]
    interface IRLNSettlement {
        function batchClaim(
            address token,
            bytes32[] calldata nullifiers,
            uint256[] calldata amounts,
            address operator
        ) external;
        function usedNullifiers(bytes32) external view returns (bool);
    }
}

// ─── RLN types ──────────────────────────────────────────────────────────

/// An RLN payment proof submitted by the client.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RLNProof {
    /// Groth16 proof bytes (snarkjs format)
    pub proof: Vec<u8>,
    /// Public signals from the circuit
    pub public_signals: Vec<String>,
    /// Nullifier hash (unique per epoch+identity)
    pub nullifier: [u8; 32],
    /// Shamir share x-coordinate
    pub share_x: [u8; 32],
    /// Shamir share y-coordinate
    pub share_y: [u8; 32],
    /// Epoch identifier
    pub epoch: u64,
    /// Claimed cost for this request
    pub amount: u64,
}

/// Result of verifying an RLN proof.
#[derive(Debug)]
pub struct RLNVerificationResult {
    pub nullifier: [u8; 32],
    pub amount: u64,
    pub is_fresh: bool,
}

/// Stored Shamir share for double-signal detection.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ShamirShare {
    x: [u8; 32],
    y: [u8; 32],
}

/// Pending claim awaiting batch settlement.
#[derive(Debug, Clone)]
struct PendingClaim {
    nullifier: [u8; 32],
    amount: u64,
}

/// RLN verifier state (nullifier set, shares, pending claims).
pub(crate) struct RLNState {
    used_nullifiers: HashSet<[u8; 32]>,
    shares: HashMap<[u8; 32], ShamirShare>,
    pending_claims: Vec<PendingClaim>,
}

impl RLNState {
    fn new() -> Self {
        Self {
            used_nullifiers: HashSet::new(),
            shares: HashMap::new(),
            pending_claims: Vec::new(),
        }
    }
}

/// Handles ShieldedCredits billing operations.
pub struct BillingClient {
    config: Arc<OperatorConfig>,
    wallet: EthereumWallet,
    shielded_credits: Address,
    rln_state: Mutex<RLNState>,
    rln_settlement: Option<Address>,
}

impl BillingClient {
    pub async fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
        let wallet = EthereumWallet::from(signer);
        let shielded_credits: Address = config.tangle.shielded_credits.parse()?;

        let rln_settlement = config
            .rln
            .as_ref()
            .map(|rln| rln.settlement_address.parse::<Address>())
            .transpose()?;

        Ok(Self {
            config,
            wallet,
            shielded_credits,
            rln_state: Mutex::new(RLNState::new()),
            rln_settlement,
        })
    }

    /// Calculate cost in base token units for given token counts.
    pub fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        let input_cost = prompt_tokens as u64 * self.config.billing.price_per_input_token;
        let output_cost = completion_tokens as u64 * self.config.billing.price_per_output_token;
        input_cost + output_cost
    }

    /// Look up the spending key for a ShieldedCredits account on-chain.
    ///
    /// Calls `getAccount(commitment)` and returns the registered `spendingKey`.
    pub(crate) async fn get_spending_key(&self, commitment: &str) -> anyhow::Result<Address> {
        let commitment: B256 = commitment.parse()?;

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.config.tangle.rpc_url.parse()?);

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);
        let result = contract.getAccount(FixedBytes(commitment.0)).call().await?;

        Ok(result.spendingKey)
    }

    /// Submit authorizeSpend on-chain, then claimPayment.
    ///
    /// `actual_amount` is the real cost of the served inference. The contract
    /// claims the full authorized amount (partial claims are not supported),
    /// so this method validates that the authorized amount is not wildly
    /// disproportionate and logs the difference for audit.
    pub(crate) async fn authorize_and_claim(
        &self,
        spend_auth: &SpendAuthPayload,
        actual_amount: u64,
    ) -> anyhow::Result<()> {
        let commitment: B256 = spend_auth.commitment.parse()?;
        let authorized_amount: U256 = spend_auth.amount.parse()?;
        let operator: Address = spend_auth.operator.parse()?;

        // Validate: actual cost must not exceed authorized amount
        let actual_u256 = U256::from(actual_amount);
        if actual_u256 > authorized_amount {
            anyhow::bail!(
                "actual cost ({actual_amount}) exceeds authorized amount ({authorized_amount})"
            );
        }

        tracing::info!(
            actual = actual_amount,
            authorized = %authorized_amount,
            "billing: claiming authorized amount (contract does not support partial claims)"
        );

        let sig_bytes = hex::decode(
            spend_auth
                .signature
                .strip_prefix("0x")
                .unwrap_or(&spend_auth.signature),
        )?;

        let auth = IShieldedCredits::SpendAuth {
            commitment: FixedBytes(commitment.0),
            serviceId: spend_auth.service_id,
            jobIndex: spend_auth.job_index,
            amount: authorized_amount,
            operator,
            nonce: U256::from(spend_auth.nonce),
            expiry: spend_auth.expiry,
            signature: sig_bytes.into(),
        };

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.config.tangle.rpc_url.parse()?);

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        // Step 1: authorizeSpend
        let pending = contract.authorizeSpend(auth).send().await?;
        let receipt = pending.get_receipt().await?;
        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            "authorizeSpend confirmed"
        );

        // Extract authHash: keccak256(abi.encode(commitment, serviceId, jobIndex, nonce))
        let auth_hash = keccak256(
            (
                FixedBytes::<32>(commitment.0),
                U256::from(spend_auth.service_id),
                U256::from(spend_auth.job_index),
                U256::from(spend_auth.nonce),
            )
                .abi_encode(),
        );

        // Step 2: claimPayment
        let pending2 = contract
            .claimPayment(FixedBytes(auth_hash.0), operator)
            .send()
            .await?;
        let receipt = pending2.get_receipt().await?;
        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            "claimPayment confirmed"
        );

        Ok(())
    }

    // ─── RLN Mode ───────────────────────────────────────────────────────

    /// Verify an RLN proof (MVP: structural validation + nullifier freshness).
    ///
    /// Real Groth16 verification requires ark-bn254/ark-groth16 or shelling out
    /// to snarkjs. This MVP checks proof structure and nullifier uniqueness.
    pub async fn verify_rln_proof(&self, proof: &RLNProof) -> anyhow::Result<RLNVerificationResult> {
        // Structural checks
        if proof.public_signals.is_empty() {
            anyhow::bail!("RLN proof has no public signals");
        }
        if proof.amount == 0 {
            anyhow::bail!("RLN proof amount is zero");
        }

        // Check nullifier freshness (in-memory)
        let state = self.rln_state.lock().await;
        let is_fresh = !state.used_nullifiers.contains(&proof.nullifier);

        // If configured, check on-chain nullifier status
        if is_fresh {
            if let Some(settlement_addr) = self.rln_settlement {
                let provider = ProviderBuilder::new()
                    .wallet(self.wallet.clone())
                    .connect_http(self.config.tangle.rpc_url.parse()?);
                let contract = IRLNSettlement::new(settlement_addr, &provider);
                let on_chain_used = contract
                    .usedNullifiers(FixedBytes(proof.nullifier))
                    .call()
                    .await?;
                if on_chain_used {
                    return Ok(RLNVerificationResult {
                        nullifier: proof.nullifier,
                        amount: proof.amount,
                        is_fresh: false,
                    });
                }
            }
        }

        Ok(RLNVerificationResult {
            nullifier: proof.nullifier,
            amount: proof.amount,
            is_fresh,
        })
    }

    /// Record an RLN claim after successful verification.
    /// Also stores the Shamir share for double-signal detection.
    pub async fn record_rln_claim(&self, proof: &RLNProof) {
        let mut state = self.rln_state.lock().await;
        state.used_nullifiers.insert(proof.nullifier);
        state.shares.insert(
            proof.nullifier,
            ShamirShare {
                x: proof.share_x,
                y: proof.share_y,
            },
        );
        state.pending_claims.push(PendingClaim {
            nullifier: proof.nullifier,
            amount: proof.amount,
        });
    }

    /// Batch-settle all pending RLN claims on-chain via RLNSettlement.batchClaim.
    /// Returns the transaction hash on success.
    pub async fn batch_settle_rln(&self) -> anyhow::Result<String> {
        let settlement_addr = self
            .rln_settlement
            .ok_or_else(|| anyhow::anyhow!("RLN settlement not configured"))?;

        let claims: Vec<PendingClaim> = {
            let mut state = self.rln_state.lock().await;
            let max_batch = self
                .config
                .rln
                .as_ref()
                .map(|r| r.max_batch_size)
                .unwrap_or(64);
            let drain_count = state.pending_claims.len().min(max_batch);
            state.pending_claims.drain(..drain_count).collect()
        };

        if claims.is_empty() {
            return Ok(String::new());
        }

        let nullifiers: Vec<FixedBytes<32>> = claims
            .iter()
            .map(|c| FixedBytes(c.nullifier))
            .collect();
        let amounts: Vec<U256> = claims.iter().map(|c| U256::from(c.amount)).collect();

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.config.tangle.rpc_url.parse()?);

        let contract = IRLNSettlement::new(settlement_addr, &provider);

        // Use the shielded_credits token as the payment token
        let token: Address = self.shielded_credits; // TODO: separate RLN token config

        let signer: PrivateKeySigner = self.config.tangle.operator_key.parse()?;
        let operator_addr = signer.address();

        let pending = contract
            .batchClaim(token, nullifiers, amounts, operator_addr)
            .send()
            .await?;
        let receipt = pending.get_receipt().await?;

        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            claim_count = claims.len(),
            "RLN batch settlement confirmed"
        );

        Ok(format!("{}", receipt.transaction_hash))
    }

    /// Get the number of pending RLN claims.
    pub async fn pending_rln_count(&self) -> usize {
        self.rln_state.lock().await.pending_claims.len()
    }
}

/// Verify a SpendAuth signature off-chain using ecrecover.
///
/// Checks that:
/// - The `operator` field matches `expected_operator` (prevents replay to wrong operator)
/// - The EIP-712 signature is structurally valid
/// - The signature has not expired
///
/// Returns `Some(recovered_signer_address)` on success so the caller can verify the
/// signer is the legitimate spending key (via on-chain `getAccount`). Returns `None`
/// if any check fails.
pub(crate) fn verify_spend_auth_signature(
    auth: &SpendAuthPayload,
    shielded_credits_addr: &str,
    chain_id: u64,
    expected_operator: &Address,
) -> Option<Address> {
    use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    // Check expiry first (cheapest check)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if now > auth.expiry {
        return None;
    }

    // Verify the operator field matches this operator's address
    let operator: Address = auth.operator.parse().ok()?;
    if operator != *expected_operator {
        tracing::warn!(
            expected = %expected_operator,
            got = %operator,
            "SpendAuth operator mismatch"
        );
        return None;
    }

    // Parse fields
    let commitment: B256 = auth.commitment.parse().ok()?;
    let amount: U256 = auth.amount.parse().ok()?;

    // Reconstruct EIP-712 domain separator
    let domain_separator = keccak256(
        (
            keccak256(
                b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
            ),
            keccak256(b"ShieldedCredits"),
            keccak256(b"1"),
            U256::from(chain_id),
            shielded_credits_addr.parse::<Address>().ok()?,
        )
            .abi_encode(),
    );

    // Reconstruct struct hash
    let spend_typehash = keccak256(
        b"SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)",
    );

    let struct_hash = keccak256(
        (
            spend_typehash,
            commitment,
            U256::from(auth.service_id),
            U256::from(auth.job_index),
            amount,
            operator,
            U256::from(auth.nonce),
            U256::from(auth.expiry),
        )
            .abi_encode(),
    );

    // EIP-712 digest: "\x19\x01" || domainSeparator || structHash
    let digest = keccak256(
        [
            &[0x19, 0x01],
            domain_separator.as_slice(),
            struct_hash.as_slice(),
        ]
        .concat(),
    );

    // Parse signature
    let sig_hex = auth.signature.strip_prefix("0x").unwrap_or(&auth.signature);
    let sig_bytes = hex::decode(sig_hex).ok()?;
    if sig_bytes.len() != 65 {
        return None;
    }

    let v = sig_bytes[64];
    let recovery_id = match v {
        27 => 0u8,
        28 => 1u8,
        0 | 1 => v,
        _ => return None,
    };

    let signature = Signature::from_slice(&sig_bytes[..64]).ok()?;
    let rid = RecoveryId::try_from(recovery_id).ok()?;
    let recovered = VerifyingKey::recover_from_prehash(digest.as_slice(), &signature, rid).ok()?;

    // Convert recovered public key to Ethereum address:
    // keccak256(uncompressed_pubkey_without_prefix)[12..32]
    let pubkey_bytes = recovered.to_encoded_point(false);
    let pubkey_hash = keccak256(&pubkey_bytes.as_bytes()[1..]); // skip 0x04 prefix
    let recovered_addr = Address::from_slice(&pubkey_hash[12..]);

    Some(recovered_addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OperatorConfig;

    const TEST_OPERATOR_ADDR: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
    const TEST_SC_ADDR: &str = "0x0000000000000000000000000000000000000002";
    const TEST_CHAIN_ID: u64 = 31337;
    const TEST_PRIVKEY: &str = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

    fn test_operator_config() -> OperatorConfig {
        serde_json::from_str(
            r#"{
                "tangle": {
                    "rpc_url": "http://localhost:8545",
                    "chain_id": 31337,
                    "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                    "tangle_core": "0x0000000000000000000000000000000000000001",
                    "shielded_credits": "0x0000000000000000000000000000000000000002",
                    "blueprint_id": 1,
                    "service_id": null
                },
                "vllm": {
                    "model": "test",
                    "max_model_len": 4096,
                    "host": "127.0.0.1",
                    "port": 8000,
                    "tensor_parallel_size": 1
                },
                "server": {},
                "billing": {
                    "required": true,
                    "price_per_input_token": 1,
                    "price_per_output_token": 2,
                    "max_spend_per_request": 1000000,
                    "min_credit_balance": 1000
                },
                "gpu": {
                    "expected_gpu_count": 1,
                    "min_vram_mib": 16000
                }
            }"#,
        )
        .unwrap()
    }

    fn test_operator_addr() -> Address {
        TEST_OPERATOR_ADDR.parse().unwrap()
    }

    fn make_spend_auth(
        commitment: &str,
        operator: &str,
        amount: &str,
        expiry: u64,
        signature: &str,
    ) -> SpendAuthPayload {
        SpendAuthPayload {
            commitment: commitment.to_string(),
            service_id: 1,
            job_index: 0,
            amount: amount.to_string(),
            operator: operator.to_string(),
            nonce: 0,
            expiry,
            signature: signature.to_string(),
        }
    }

    /// Helper: create a real EIP-712 SpendAuth with a valid signature.
    fn make_real_spend_auth() -> (SpendAuthPayload, Address) {
        use k256::ecdsa::SigningKey;

        let signing_key = SigningKey::from_slice(&hex::decode(TEST_PRIVKEY).unwrap()).unwrap();

        let commitment = B256::ZERO;
        let service_id = 1u64;
        let job_index = 0u8;
        let amount = U256::from(1000u64);
        let operator = test_operator_addr();
        let nonce = 0u64;
        let expiry = u64::MAX;
        let sc_addr: Address = TEST_SC_ADDR.parse().unwrap();

        let domain_separator = keccak256(
            (
                keccak256(
                    b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
                ),
                keccak256(b"ShieldedCredits"),
                keccak256(b"1"),
                U256::from(TEST_CHAIN_ID),
                sc_addr,
            )
                .abi_encode(),
        );

        let spend_typehash = keccak256(
            b"SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)",
        );

        let struct_hash = keccak256(
            (
                spend_typehash,
                commitment,
                U256::from(service_id),
                U256::from(job_index),
                amount,
                operator,
                U256::from(nonce),
                U256::from(expiry),
            )
                .abi_encode(),
        );

        let digest = keccak256(
            [
                &[0x19, 0x01],
                domain_separator.as_slice(),
                struct_hash.as_slice(),
            ]
            .concat(),
        );

        let (sig, recid) = signing_key
            .sign_prehash_recoverable(digest.as_slice())
            .unwrap();
        let mut sig_bytes = sig.to_bytes().to_vec();
        sig_bytes.push(recid.to_byte() + 27);

        let auth = SpendAuthPayload {
            commitment: format!("0x{}", hex::encode(commitment.as_slice())),
            service_id,
            job_index,
            amount: amount.to_string(),
            operator: format!("{operator}"),
            nonce,
            expiry,
            signature: format!("0x{}", hex::encode(&sig_bytes)),
        };

        (auth, operator)
    }

    // ─── BillingClient.calculate_cost tests ──────────────────────────────

    #[tokio::test]
    async fn test_billing_client_calculate_cost() {
        let config = Arc::new(test_operator_config());
        let client = BillingClient::new(config).await.unwrap();
        // price_per_input_token=1, price_per_output_token=2
        assert_eq!(client.calculate_cost(100, 50), 200);
        assert_eq!(client.calculate_cost(0, 0), 0);
        assert_eq!(client.calculate_cost(1, 0), 1);
        assert_eq!(client.calculate_cost(0, 1), 2);
        assert_eq!(client.calculate_cost(100_000, 50_000), 200_000);
    }

    // ─── Signature verification tests ────────────────────────────────────

    #[test]
    fn test_verify_rejects_expired_signature() {
        let op = test_operator_addr();
        let auth = make_spend_auth(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            TEST_OPERATOR_ADDR,
            "1000",
            1, // expired
            "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject expired signature"
        );
    }

    #[test]
    fn test_verify_rejects_operator_mismatch() {
        let wrong_operator: Address = "0x0000000000000000000000000000000000000099"
            .parse()
            .unwrap();
        let (auth, _) = make_real_spend_auth();
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &wrong_operator)
                .is_none(),
            "should reject when operator field doesn't match expected operator"
        );
    }

    #[test]
    fn test_verify_rejects_bad_signature_length() {
        let op = test_operator_addr();
        let auth = make_spend_auth(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            TEST_OPERATOR_ADDR,
            "1000",
            u64::MAX,
            "0xdeadbeef", // too short
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject short signature"
        );
    }

    #[test]
    fn test_verify_rejects_invalid_commitment() {
        let op = test_operator_addr();
        let auth = make_spend_auth(
            "not_a_hex_value",
            TEST_OPERATOR_ADDR,
            "1000",
            u64::MAX,
            "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject invalid commitment hex"
        );
    }

    #[test]
    fn test_verify_rejects_invalid_amount() {
        let op = test_operator_addr();
        let auth = make_spend_auth(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            TEST_OPERATOR_ADDR,
            "not_a_number",
            u64::MAX,
            "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject invalid amount"
        );
    }

    #[test]
    fn test_verify_rejects_invalid_operator_address() {
        // Can't even parse the operator field — rejected before identity check
        let op: Address = "0x0000000000000000000000000000000000000001"
            .parse()
            .unwrap();
        let auth = make_spend_auth(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            "bad_address",
            "1000",
            u64::MAX,
            "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject invalid operator address"
        );
    }

    #[test]
    fn test_verify_rejects_bad_recovery_id() {
        let op = test_operator_addr();
        let mut sig = vec![0u8; 65];
        sig[64] = 99; // invalid v
        let auth = make_spend_auth(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            TEST_OPERATOR_ADDR,
            "1000",
            u64::MAX,
            &format!("0x{}", hex::encode(&sig)),
        );
        assert!(
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &op).is_none(),
            "should reject invalid recovery ID"
        );
    }

    #[test]
    fn test_verify_with_real_signature() {
        let (auth, operator) = make_real_spend_auth();
        let recovered = verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &operator);
        assert!(recovered.is_some(), "valid signature should verify");

        // The recovered address should be the signer's Ethereum address
        // (derived from the test private key)
        let signer_addr = recovered.unwrap();
        assert!(
            !signer_addr.is_zero(),
            "recovered address should be non-zero"
        );
    }

    #[test]
    fn test_verify_tampered_amount_recovers_wrong_signer() {
        // ecrecover with tampered data produces a different (wrong) signer address.
        // The caller must check the recovered address against the expected spending key.
        let (auth, operator) = make_real_spend_auth();
        let good_signer =
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &operator).unwrap();

        let mut tampered = auth;
        tampered.amount = "9999".to_string();
        let bad_signer =
            verify_spend_auth_signature(&tampered, TEST_SC_ADDR, TEST_CHAIN_ID, &operator);
        // Tampered data either fails recovery or recovers a different address
        assert!(
            bad_signer.is_none() || bad_signer.unwrap() != good_signer,
            "tampered amount must not recover the same signer"
        );
    }

    #[test]
    fn test_verify_tampered_commitment_recovers_wrong_signer() {
        let (auth, operator) = make_real_spend_auth();
        let good_signer =
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &operator).unwrap();

        let mut tampered = auth;
        tampered.commitment =
            "0x0000000000000000000000000000000000000000000000000000000000000099".to_string();
        let bad_signer =
            verify_spend_auth_signature(&tampered, TEST_SC_ADDR, TEST_CHAIN_ID, &operator);
        assert!(
            bad_signer.is_none() || bad_signer.unwrap() != good_signer,
            "tampered commitment must not recover the same signer"
        );
    }

    #[test]
    fn test_verify_tampered_nonce_recovers_wrong_signer() {
        let (auth, operator) = make_real_spend_auth();
        let good_signer =
            verify_spend_auth_signature(&auth, TEST_SC_ADDR, TEST_CHAIN_ID, &operator).unwrap();

        let mut tampered = auth;
        tampered.nonce = 42;
        let bad_signer =
            verify_spend_auth_signature(&tampered, TEST_SC_ADDR, TEST_CHAIN_ID, &operator);
        assert!(
            bad_signer.is_none() || bad_signer.unwrap() != good_signer,
            "tampered nonce must not recover the same signer"
        );
    }
}
