//! Register through the shipped CLI against the SDK's real Tangle snapshot.
use alloy::{providers::ProviderBuilder, sol};
use blueprint_anvil_testing_utils::TangleHarness;
use std::path::Path;

sol! {
    #[sol(rpc)]
    interface Registry {
        struct Metadata {
            string name;
            string description;
            string author;
            string category;
            string codeRepository;
            string logo;
            string website;
            string license;
            string profilingData;
        }
        function blueprintCount() external view returns (uint64);
        function blueprintMetadata(uint64 id) external view returns (Metadata metadata, string uri, bytes32 hash);
    }
}

#[tokio::test]
async fn registration_preserves_definition_gpu_requirement() -> anyhow::Result<()> {
    let chain = TangleHarness::start(true).await?;
    let provider = ProviderBuilder::new().connect_http(chain.http_endpoint().clone());
    let registry = Registry::new(chain.tangle_contract, &provider);
    let id = registry.blueprintCount().call().await?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = tokio::process::Command::new("bash")
        .arg(root.join("deploy/register-blueprint.sh"))
        .current_dir(root)
        .env("RPC_URL", chain.http_endpoint().as_str())
        .env("TANGLE_CORE", chain.tangle_contract.to_string())
        .env(
            "PAYMENT_TOKEN",
            "0x0000000000000000000000000000000000000001",
        )
        // Public deterministic Anvil fixture, never a funded network key.
        .env(
            "PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        .output()
        .await?;
    anyhow::ensure!(
        output.status.success(),
        "registration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(registry.blueprintCount().call().await?, id + 1);
    let metadata = registry.blueprintMetadata(id).call().await?.metadata;
    let definition: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("deploy/definition.json"))?)?;
    let expected = definition["metadata"]["profiling_data"].as_str().unwrap();
    assert_eq!(
        metadata.profilingData, expected,
        "definition GPU policy must reach the chain unchanged"
    );
    let profile: serde_json::Value = serde_json::from_str(&metadata.profilingData)?;
    assert_eq!(profile["execution_profile"]["gpu"]["policy"], "required");
    assert!(
        profile["execution_profile"]["gpu"]["min_count"]
            .as_u64()
            .unwrap()
            > 0
    );
    Ok(())
}
