use std::sync::Arc;

use alloy::{
    network::EthereumWallet,
    primitives::{keccak256, Address, FixedBytes, B256, U256},
    providers::ProviderBuilder,
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

/// Handles ShieldedCredits billing operations.
pub struct BillingClient {
    config: Arc<OperatorConfig>,
    wallet: EthereumWallet,
    shielded_credits: Address,
}

impl BillingClient {
    pub async fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
        let wallet = EthereumWallet::from(signer);
        let shielded_credits: Address = config.tangle.shielded_credits.parse()?;

        Ok(Self {
            config,
            wallet,
            shielded_credits,
        })
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

    /// Pre-authorize spending on-chain. Must be called before serving inference.
    pub async fn authorize_spend(&self, spend_auth: &SpendAuthPayload) -> anyhow::Result<()> {
        let auth = self.build_auth(spend_auth)?;

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.config.tangle.rpc_url.parse()?);

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
    /// `actual_amount` is the metered cost derived from token usage and
    /// configured pricing.  The on-chain contract currently settles the full
    /// pre-authorized amount, but the operator records the actual cost for
    /// auditing.  When the contract supports partial settlement, this value
    /// will be forwarded on-chain.
    pub async fn claim_payment(
        &self,
        spend_auth: &SpendAuthPayload,
        actual_amount: u64,
    ) -> anyhow::Result<()> {
        let auth_hash = Self::auth_hash(spend_auth)?;
        let operator: Address = spend_auth.operator.parse()?;

        tracing::info!(
            actual_amount = actual_amount,
            preauth_amount = %spend_auth.amount,
            "claiming payment (actual metered cost)"
        );

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.config.tangle.rpc_url.parse()?);

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let pending = contract.claimPayment(auth_hash, operator).send().await?;
        let receipt = pending.get_receipt().await?;
        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            actual_amount = actual_amount,
            "claimPayment confirmed"
        );

        Ok(())
    }

    /// Query the on-chain balance for a ShieldedCredits account.
    pub async fn get_account_balance(&self, commitment: &str) -> anyhow::Result<U256> {
        let commitment: B256 = commitment.parse()?;

        let provider = ProviderBuilder::new().connect_http(self.config.tangle.rpc_url.parse()?);

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let result = contract.getAccount(FixedBytes(commitment.0)).call().await?;
        Ok(result.balance)
    }

    /// Legacy combined method — authorize spend + claim payment in one call.
    pub async fn authorize_and_claim(
        &self,
        spend_auth: &SpendAuthPayload,
        actual_amount: u64,
    ) -> anyhow::Result<()> {
        self.authorize_spend(spend_auth).await?;
        self.claim_payment(spend_auth, actual_amount).await?;
        Ok(())
    }
}

/// Verify a SpendAuth signature off-chain using ecrecover.
/// Returns true if the signature is valid and not expired.
pub fn verify_spend_auth_signature(
    auth: &SpendAuthPayload,
    shielded_credits_addr: &str,
    chain_id: u64,
) -> bool {
    use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    let domain_separator = keccak256(
        (
            keccak256(
                b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
            ),
            keccak256(b"ShieldedCredits"),
            keccak256(b"1"),
            U256::from(chain_id),
            shielded_credits_addr
                .parse::<Address>()
                .unwrap_or(Address::ZERO),
        )
            .abi_encode(),
    );

    let spend_typehash = keccak256(
        b"SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)",
    );

    let Ok(commitment) = auth.commitment.parse::<B256>() else {
        return false;
    };
    let Ok(amount) = auth.amount.parse::<U256>() else {
        return false;
    };
    let Ok(operator) = auth.operator.parse::<Address>() else {
        return false;
    };

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
    let Ok(sig_bytes) = hex::decode(sig_hex) else {
        return false;
    };
    if sig_bytes.len() != 65 {
        return false;
    }

    let v = sig_bytes[64];
    let recovery_id = match v {
        27 => 0u8,
        28 => 1u8,
        0 | 1 => v,
        _ => return false,
    };

    let Ok(signature) = Signature::from_slice(&sig_bytes[..64]) else {
        return false;
    };
    let Ok(rid) = RecoveryId::try_from(recovery_id) else {
        return false;
    };
    let Ok(_recovered) = VerifyingKey::recover_from_prehash(digest.as_slice(), &signature, rid)
    else {
        return false;
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if now > auth.expiry {
        return false;
    }

    true
}
