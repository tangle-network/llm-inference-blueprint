use std::sync::Arc;
use std::time::Duration;

use alloy::{
    network::EthereumWallet,
    primitives::{keccak256, Address, FixedBytes, B256, U256},
    providers::{Provider, ProviderBuilder},
    signers::local::PrivateKeySigner,
    sol,
    sol_types::SolValue,
};

use crate::config::OperatorConfig;
use crate::server::SpendAuthPayload;

// Generate bindings for the ShieldedCredits contract.
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

/// On-chain account info returned by getAccount.
pub struct AccountInfo {
    pub spending_key: Address,
    pub balance: U256,
}

/// Handles ShieldedCredits billing operations.
pub struct BillingClient {
    config: Arc<OperatorConfig>,
    wallet: EthereumWallet,
    shielded_credits: Address,
    rpc_url: reqwest::Url,
    /// This operator's Ethereum address, derived from operator_key.
    operator_address: Address,
}

impl BillingClient {
    pub async fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
        let operator_address = signer.address();

        // Warn about plaintext key storage. In production, operators should use
        // a KMS, hardware signer, or encrypted keystore instead of raw hex.
        if std::env::var("VLLM_OP_TANGLE__OPERATOR_KEY").is_ok() {
            tracing::warn!(
                "operator_key loaded from environment variable — \
                 use a KMS or encrypted keystore in production"
            );
        } else {
            tracing::warn!(
                "operator_key loaded from plaintext config — \
                 use a KMS or encrypted keystore in production"
            );
        }

        let wallet = EthereumWallet::from(signer);
        let shielded_credits: Address = config.tangle.shielded_credits.parse()?;
        let rpc_url: reqwest::Url = config.tangle.rpc_url.parse()?;

        Ok(Self {
            config,
            wallet,
            shielded_credits,
            rpc_url,
            operator_address,
        })
    }

    /// Returns the operator's Ethereum address.
    pub fn operator_address(&self) -> Address {
        self.operator_address
    }

    /// Calculate cost in base token units for given token counts.
    pub fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        let input_cost = prompt_tokens as u64 * self.config.billing.price_per_input_token;
        let output_cost = completion_tokens as u64 * self.config.billing.price_per_output_token;
        input_cost + output_cost
    }

    fn build_auth(
        &self,
        spend_auth: &SpendAuthPayload,
    ) -> anyhow::Result<IShieldedCredits::SpendAuth> {
        let commitment: B256 = spend_auth.commitment.parse()?;
        let amount: U256 = spend_auth.amount.parse()?;
        let operator: Address = spend_auth.operator.parse()?;
        let sig_bytes = hex::decode(
            spend_auth
                .signature
                .strip_prefix("0x")
                .unwrap_or(&spend_auth.signature),
        )?;

        Ok(IShieldedCredits::SpendAuth {
            commitment: FixedBytes(commitment.0),
            serviceId: spend_auth.service_id,
            jobIndex: spend_auth.job_index,
            amount,
            operator,
            nonce: U256::from(spend_auth.nonce),
            expiry: spend_auth.expiry,
            signature: sig_bytes.into(),
        })
    }

    fn auth_hash(spend_auth: &SpendAuthPayload) -> anyhow::Result<FixedBytes<32>> {
        let commitment: B256 = spend_auth.commitment.parse()?;
        let hash = keccak256(
            (
                FixedBytes::<32>(commitment.0),
                U256::from(spend_auth.service_id),
                U256::from(spend_auth.job_index),
                U256::from(spend_auth.nonce),
            )
                .abi_encode(),
        );
        Ok(FixedBytes(hash.0))
    }

    /// Check current gas price against the configured cap.
    /// Returns Ok(()) if gas price is acceptable, Err if it exceeds the cap.
    async fn check_gas_price(&self) -> anyhow::Result<()> {
        let max_gwei = self.config.billing.max_gas_price_gwei;
        if max_gwei == 0 {
            return Ok(());
        }

        let provider = ProviderBuilder::new().connect_http(self.rpc_url.clone());
        let gas_price = provider.get_gas_price().await?;
        let gas_price_gwei = gas_price / 1_000_000_000;

        if gas_price_gwei > max_gwei as u128 {
            anyhow::bail!(
                "gas price {gas_price_gwei} gwei exceeds cap {max_gwei} gwei — deferring tx"
            );
        }

        Ok(())
    }

    /// Pre-authorize spending on-chain. Must be called before serving inference.
    pub async fn authorize_spend(&self, spend_auth: &SpendAuthPayload) -> anyhow::Result<()> {
        self.check_gas_price().await?;

        let auth = self.build_auth(spend_auth)?;

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.rpc_url.clone());

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let pending = contract.authorizeSpend(auth).send().await?;
        let receipt = pending.get_receipt().await?;
        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            "authorizeSpend confirmed"
        );

        Ok(())
    }

    /// Claim payment on-chain after inference is served.
    ///
    /// IMPORTANT: The ShieldedCredits contract `claimPayment(bytes32, address)`
    /// always settles the full pre-authorized amount. There is no partial
    /// settlement. The `actual_amount` parameter is logged for auditing and
    /// metrics only. To minimize overcharging, the HTTP handler validates that
    /// the user's pre-auth amount is reasonable relative to `max_tokens`.
    ///
    /// Retries up to `claim_max_retries` times on failure with exponential backoff.
    pub async fn claim_payment(
        &self,
        spend_auth: &SpendAuthPayload,
        actual_amount: u64,
    ) -> anyhow::Result<()> {
        let auth_hash = Self::auth_hash(spend_auth)?;
        let operator: Address = spend_auth.operator.parse()?;
        let max_retries = self.config.billing.claim_max_retries;

        tracing::info!(
            actual_amount = actual_amount,
            preauth_amount = %spend_auth.amount,
            "claiming payment (actual metered cost)"
        );

        let mut last_err = None;
        for attempt in 0..=max_retries {
            // Check gas price before each attempt (price may change between retries)
            if let Err(e) = self.check_gas_price().await {
                tracing::warn!(error = %e, attempt, "gas price check failed for claimPayment");
                last_err = Some(e);
                let delay = Duration::from_millis(500 * 2u64.pow(attempt));
                tokio::time::sleep(delay).await;
                continue;
            }
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "retrying claimPayment"
                );
                tokio::time::sleep(delay).await;
            }

            let provider = ProviderBuilder::new()
                .wallet(self.wallet.clone())
                .connect_http(self.rpc_url.clone());

            let contract = IShieldedCredits::new(self.shielded_credits, &provider);

            match contract.claimPayment(auth_hash, operator).send().await {
                Ok(pending) => match pending.get_receipt().await {
                    Ok(receipt) => {
                        tracing::info!(
                            tx_hash = %receipt.transaction_hash,
                            actual_amount = actual_amount,
                            attempt,
                            "claimPayment confirmed"
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        last_err = Some(e.into());
                    }
                },
                Err(e) => {
                    last_err = Some(e.into());
                }
            }
        }

        let err = last_err.unwrap_or_else(|| anyhow::anyhow!("claimPayment failed"));
        tracing::error!(
            error = %err,
            auth_hash = %auth_hash,
            actual_amount,
            commitment = %spend_auth.commitment,
            "claimPayment FAILED after {} retries — operator served inference for free. Manual recovery required.",
            max_retries
        );
        Err(err)
    }

    /// Query on-chain account info (spending key + balance) for a ShieldedCredits account.
    pub async fn get_account_info(&self, commitment: &str) -> anyhow::Result<AccountInfo> {
        let commitment: B256 = commitment.parse()?;

        let provider = ProviderBuilder::new().connect_http(self.rpc_url.clone());

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let result = contract.getAccount(FixedBytes(commitment.0)).call().await?;
        Ok(AccountInfo {
            spending_key: result.spendingKey,
            balance: result.balance,
        })
    }
}

/// Recover the signer address from a SpendAuth EIP-712 signature and verify it
/// matches the expected spending key.
///
/// Returns Ok(recovered_address) if the signature is valid and the signer matches
/// `expected_spending_key`. Returns Err with a reason string if:
/// - The signature is malformed or cannot be recovered
/// - The SpendAuth has expired (with clock skew tolerance)
/// - The recovered signer does NOT match `expected_spending_key`
pub fn verify_spend_auth_signature(
    auth: &SpendAuthPayload,
    expected_spending_key: Address,
    shielded_credits_addr: &str,
    chain_id: u64,
    clock_skew_tolerance_secs: u64,
) -> Result<Address, String> {
    let recovered = recover_spend_auth_signer(
        auth,
        shielded_credits_addr,
        chain_id,
        clock_skew_tolerance_secs,
    )?;

    if recovered != expected_spending_key {
        return Err(format!(
            "recovered signer ({recovered}) does not match expected spending key ({expected_spending_key})"
        ));
    }

    Ok(recovered)
}

/// Recover the signer address from a SpendAuth EIP-712 signature.
///
/// Returns the recovered Ethereum address on success. The caller MUST compare
/// this against the account's on-chain spending key to authenticate the request.
///
/// Also checks expiry with the given clock skew tolerance.
pub fn recover_spend_auth_signer(
    auth: &SpendAuthPayload,
    shielded_credits_addr: &str,
    chain_id: u64,
    clock_skew_tolerance_secs: u64,
) -> Result<Address, String> {
    use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    let shielded_addr: Address = shielded_credits_addr
        .parse()
        .map_err(|e| format!("invalid shielded_credits address: {e}"))?;

    let domain_separator = keccak256(
        (
            keccak256(
                b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
            ),
            keccak256(b"ShieldedCredits"),
            keccak256(b"1"),
            U256::from(chain_id),
            shielded_addr,
        )
            .abi_encode(),
    );

    let spend_typehash = keccak256(
        b"SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)",
    );

    let commitment: B256 = auth
        .commitment
        .parse()
        .map_err(|e| format!("invalid commitment: {e}"))?;
    let amount: U256 = auth
        .amount
        .parse()
        .map_err(|e| format!("invalid amount: {e}"))?;
    let operator: Address = auth
        .operator
        .parse()
        .map_err(|e| format!("invalid operator address: {e}"))?;

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

    let digest = keccak256(
        [
            &[0x19, 0x01],
            domain_separator.as_slice(),
            struct_hash.as_slice(),
        ]
        .concat(),
    );

    let sig_hex = auth.signature.strip_prefix("0x").unwrap_or(&auth.signature);
    let sig_bytes = hex::decode(sig_hex).map_err(|e| format!("invalid signature hex: {e}"))?;
    if sig_bytes.len() != 65 {
        return Err(format!(
            "invalid signature length: expected 65, got {}",
            sig_bytes.len()
        ));
    }

    let v = sig_bytes[64];
    let recovery_id = match v {
        27 => 0u8,
        28 => 1u8,
        0 | 1 => v,
        _ => return Err(format!("invalid signature recovery byte: {v}")),
    };

    let signature =
        Signature::from_slice(&sig_bytes[..64]).map_err(|e| format!("invalid signature: {e}"))?;
    let rid = RecoveryId::try_from(recovery_id).map_err(|e| format!("invalid recovery id: {e}"))?;
    let recovered = VerifyingKey::recover_from_prehash(digest.as_slice(), &signature, rid)
        .map_err(|e| format!("ecrecover failed: {e}"))?;

    // Convert the recovered public key to an Ethereum address
    let pubkey_bytes = recovered.to_encoded_point(false);
    let pubkey_hash = keccak256(&pubkey_bytes.as_bytes()[1..]);
    let recovered_address = Address::from_slice(&pubkey_hash[12..]);

    // Check expiry with clock skew tolerance
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| "system clock is before UNIX epoch".to_string())?
        .as_secs();
    if now > auth.expiry.saturating_add(clock_skew_tolerance_secs) {
        return Err(format!(
            "SpendAuth expired: now={now}, expiry={}, tolerance={clock_skew_tolerance_secs}s",
            auth.expiry
        ));
    }

    Ok(recovered_address)
}
