//! The operator-lite process bills paid chats on Anvil: authorize, serve, settle, and a SIGKILL
//! between serving and settling is paid exactly once by the next start.
use alloy::{
    network::{Ethereum, EthereumWallet, Network, TransactionBuilder},
    primitives::{Address, Bytes, B256, U256},
    providers::{Provider, ProviderBuilder},
    signers::{local::PrivateKeySigner, SignerSync},
    sol,
    sol_types::{eip712_domain, SolStruct},
};
use serde_json::{json, Value};
use std::{
    net::TcpListener,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::process::{Child, Command};

const OPERATOR_KEY: &str = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const FUNDER_KEY: &str = "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
const SERVICE_ID: u64 = 1;
const FUNDED: u64 = 10_000;
const PRICE_IN: u64 = 3;
const PRICE_OUT: u64 = 5;
const MAX_TOKENS: u64 = 16;
/// Within [estimate, 1.5x estimate] for every prompt below, so authorize_request accepts it.
const PREAUTH: u64 = 100;
const MODEL: &str = "test-model";

// Minimal ERC-20 compiled from tangle-inference-core tests/TestERC20.sol with solc 0.8.26 --optimize --metadata-hash none
sol! {
    #[sol(rpc, bytecode = "6080604052348015600e575f80fd5b506105268061001c5f395ff3fe608060405234801561000f575f80fd5b5060043610610060575f3560e01c8063095ea7b31461006457806323b872dd1461008c57806340c10f191461009f57806370a08231146100b4578063a9059cbb146100e1578063dd62ed3e146100f4575b5f80fd5b61007761007236600461042c565b61011e565b60405190151581526020015b60405180910390f35b61007761009a366004610454565b61014b565b6100b26100ad36600461042c565b6102bc565b005b6100d36100c236600461048e565b5f6020819052908152604090205481565b604051908152602001610083565b6100776100ef36600461042c565b61032c565b6100d36101023660046104ae565b600160209081525f928352604080842090915290825290205481565b335f9081526001602081815260408084206001600160a01b03871685529091529091208290555b92915050565b6001600160a01b0383165f90815260208190526040812054821180159061019457506001600160a01b0384165f9081526001602090815260408083203384529091529020548211155b6101d45760405162461bcd60e51b815260206004820152600c60248201526b1a5b9cdd59999a58da595b9d60a21b60448201526064015b60405180910390fd5b6001600160a01b0384165f908152600160209081526040808320338452909152812080548492906102069084906104f3565b90915550506001600160a01b0384165f90815260208190526040812080548492906102329084906104f3565b90915550506001600160a01b0383165f908152602081905260408120805484929061025e908490610506565b92505081905550826001600160a01b0316846001600160a01b03167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef846040516102aa91815260200190565b60405180910390a35060019392505050565b6001600160a01b0382165f90815260208190526040812080548392906102e3908490610506565b90915550506040518181526001600160a01b038316905f907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef9060200160405180910390a35050565b335f908152602081905260408120548211156103795760405162461bcd60e51b815260206004820152600c60248201526b1a5b9cdd59999a58da595b9d60a21b60448201526064016101cb565b335f90815260208190526040812080548492906103979084906104f3565b90915550506001600160a01b0383165f90815260208190526040812080548492906103c3908490610506565b90915550506040518281526001600160a01b0384169033907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef9060200160405180910390a350600192915050565b80356001600160a01b0381168114610427575f80fd5b919050565b5f806040838503121561043d575f80fd5b61044683610411565b946020939093013593505050565b5f805f60608486031215610466575f80fd5b61046f84610411565b925061047d60208501610411565b929592945050506040919091013590565b5f6020828403121561049e575f80fd5b6104a782610411565b9392505050565b5f80604083850312156104bf575f80fd5b6104c883610411565b91506104d660208401610411565b90509250929050565b634e487b7160e01b5f52601160045260245ffd5b81810381811115610145576101456104df565b80820180821115610145576101456104df56fea164736f6c634300081a000a")]
    contract TestToken {
        function mint(address to, uint256 amount) external;
        function balanceOf(address owner) external view returns (uint256);
        function approve(address spender, uint256 amount) external returns (bool);
    }
}

// tests/fixtures/ShieldedCredits.bin is tangle-inference-core's fixture of the same name:
// src/shielded/ShieldedCredits.sol from github.com/tangle-network/shielded-payment-gateway rev
// 65f6b4064503e79fcc750799bc3f69a4e1f6c4ad, solc 0.8.26, optimizer 200 runs, via_ir, cancun.
sol! {
    struct SpendAuthorization {
        bytes32 commitment;
        uint64 serviceId;
        uint8 jobIndex;
        uint256 amount;
        address operator;
        uint256 nonce;
        uint64 expiry;
    }

    #[sol(rpc)]
    contract ShieldedCredits {
        function fundCredits(address token, uint256 amount, bytes32 commitment, address spendingKey) external;
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

/// An OpenAI-compatible upstream with deterministic usage: prompt tokens are the prompt's length,
/// completion tokens are 9. Each request is logged on arrival and held while the hold file exists.
const UPSTREAM: &str = r#"import json, os, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
port, arrivals, hold = int(sys.argv[1]), sys.argv[2], sys.argv[3]
class Upstream(BaseHTTPRequestHandler):
    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self.reply({'object': 'list', 'data': [{'id': 'test-model'}]} if self.path == '/v1/models' else {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        content = body['messages'][-1]['content']
        with open(arrivals, 'a') as log: log.write(content + '\n')
        while os.path.exists(hold): time.sleep(0.02)
        prompt, completion = len(content), 9
        self.reply({'id': 'upstream', 'object': 'chat.completion', 'created': 1, 'model': body['model'],
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
                      'prompt_tokens_details': {'cached_tokens': 0}}})
    def log_message(self, *args): pass
ThreadingHTTPServer(('127.0.0.1', port), Upstream).serve_forever()
"#;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

struct Account {
    commitment: B256,
    key: PrivateKeySigner,
}

struct Chain {
    rpc: String,
    chain_id: u64,
    credits: Address,
    token: Address,
    operator: Address,
}

/// Deploy ShieldedCredits and a test token, and fund `count` credit accounts with FUNDED each.
async fn deploy_shielded(rpc: String, count: u8) -> anyhow::Result<(Chain, Vec<Account>)> {
    let operator: PrivateKeySigner = OPERATOR_KEY.parse()?;
    let operator_address = operator.address();
    let deployer = ProviderBuilder::new()
        .wallet(EthereumWallet::from(operator))
        .connect_http(rpc.parse()?);
    let code: Bytes = include_str!("fixtures/ShieldedCredits.bin")
        .trim()
        .parse()?;
    let deploy = <Ethereum as Network>::TransactionRequest::default().with_deploy_code(code);
    let credits = deployer
        .send_transaction(deploy)
        .await?
        .get_receipt()
        .await?
        .contract_address
        .expect("ShieldedCredits deployment");
    let token = *TestToken::deploy(&deployer).await?.address();
    let funder: PrivateKeySigner = FUNDER_KEY.parse()?;
    let funder_address = funder.address();
    let funder = ProviderBuilder::new()
        .wallet(EthereumWallet::from(funder))
        .connect_http(rpc.parse()?);
    let funder_token = TestToken::new(token, &funder);
    let total = U256::from(FUNDED) * U256::from(count);
    funder_token
        .mint(funder_address, total)
        .send()
        .await?
        .get_receipt()
        .await?;
    funder_token
        .approve(credits, total)
        .send()
        .await?
        .get_receipt()
        .await?;
    let mut accounts = Vec::new();
    for tag in 1..=count {
        let account = Account {
            commitment: B256::with_last_byte(tag),
            key: PrivateKeySigner::from_bytes(&B256::with_last_byte(0x10 + tag))?,
        };
        ShieldedCredits::new(credits, &funder)
            .fundCredits(
                token,
                U256::from(FUNDED),
                account.commitment,
                account.key.address(),
            )
            .send()
            .await?
            .get_receipt()
            .await?;
        accounts.push(account);
    }
    let chain = Chain {
        chain_id: deployer.get_chain_id().await?,
        rpc,
        credits,
        token,
        operator: operator_address,
    };
    Ok((chain, accounts))
}

impl Chain {
    /// The SpendAuth a client puts in the chat body.
    fn sign(&self, account: &Account, nonce: u64, expiry: u64) -> Value {
        let typed = SpendAuthorization {
            commitment: account.commitment,
            serviceId: SERVICE_ID,
            jobIndex: 0,
            amount: U256::from(PREAUTH),
            operator: self.operator,
            nonce: U256::from(nonce),
            expiry,
        };
        let domain = eip712_domain! {
            name: "ShieldedCredits",
            version: "1",
            chain_id: self.chain_id,
            verifying_contract: self.credits,
        };
        let signature = account
            .key
            .sign_hash_sync(&typed.eip712_signing_hash(&domain))
            .unwrap();
        json!({
            "commitment": format!("{:#x}", account.commitment),
            "serviceId": SERVICE_ID,
            "jobIndex": 0,
            "amount": PREAUTH.to_string(),
            "operator": format!("{:#x}", self.operator),
            "nonce": nonce,
            "expiry": expiry,
            "signature": alloy::hex::encode_prefixed(signature.as_bytes()),
        })
    }

    /// On-chain (balance, nonce): authorizeSpend locks the pre-auth, settlement refunds the rest.
    async fn on_chain(&self, account: &Account) -> (u64, u64) {
        let provider = ProviderBuilder::new().connect_http(self.rpc.parse().unwrap());
        let view = ShieldedCredits::new(self.credits, provider)
            .getAccount(account.commitment)
            .call()
            .await
            .unwrap();
        (view.balance.to(), view.nonce.to())
    }

    async fn operator_paid(&self) -> u64 {
        let provider = ProviderBuilder::new().connect_http(self.rpc.parse().unwrap());
        TestToken::new(self.token, provider)
            .balanceOf(self.operator)
            .call()
            .await
            .unwrap()
            .to()
    }

    /// Set the next block's base fee and mine it, so the operator's gas-price cap holds or passes.
    async fn set_base_fee(&self, gwei: u64) -> anyhow::Result<()> {
        let provider = ProviderBuilder::new().connect_http(self.rpc.parse()?);
        let wei = U256::from(gwei) * U256::from(1_000_000_000u64);
        provider
            .raw_request::<_, Value>("anvil_setNextBlockBaseFeePerGas".into(), (wei,))
            .await?;
        provider
            .raw_request::<_, Value>("evm_mine".into(), ())
            .await?;
        Ok(())
    }
}

/// Owed minus settled records in the journal file: settlements a restart must still send.
fn owed_in(journal: &Path) -> usize {
    let text = std::fs::read_to_string(journal).unwrap_or_default();
    text.matches("{\"owed\"").count() - text.matches("{\"settled\"").count()
}

/// Poll `check` every 50 ms until it holds or `secs` pass.
async fn until<F: std::future::Future<Output = bool>>(
    secs: u64,
    what: &str,
    mut check: impl FnMut() -> F,
) -> anyhow::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while !check().await {
        anyhow::ensure!(Instant::now() < deadline, "timed out: {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}

struct Operator {
    child: Child,
    url: String,
    log: PathBuf,
}

/// Start the operator-lite binary on `config` and wait until /health reaches the upstream.
async fn start_operator(dir: &Path, config: &Path, run: &str) -> anyhow::Result<Operator> {
    let port: u16 = serde_json::from_slice::<Value>(&std::fs::read(config)?)?["server"]["port"]
        .as_u64()
        .unwrap() as u16;
    let log = dir.join(format!("operator-{run}.log"));
    let out = std::fs::File::create(&log)?;
    let mut child = Command::new(env!("CARGO_BIN_EXE_operator-lite"))
        .arg(config)
        .current_dir(dir)
        .env_remove("PRODUCTION")
        .env("RUST_LOG", "info")
        .stdout(Stdio::from(out.try_clone()?))
        .stderr(Stdio::from(out))
        .kill_on_drop(true)
        .spawn()?;
    let url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "operator {run} exited before readiness ({status}):\n{}",
                std::fs::read_to_string(&log)?
            );
        }
        if client
            .get(format!("{url}/health"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(Operator { child, url, log });
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "operator {run} never became healthy"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

impl Operator {
    /// `kill -9`: no signal handler, no drain, no shutdown path.
    async fn sigkill(mut self) -> anyhow::Result<()> {
        let pid = self.child.id().expect("running operator").to_string();
        let killed = Command::new("kill").args(["-KILL", &pid]).status().await?;
        anyhow::ensure!(killed.success(), "kill -KILL {pid} failed");
        let status = self.child.wait().await?;
        anyhow::ensure!(
            status.signal() == Some(9),
            "operator ended with {status}, not SIGKILL"
        );
        Ok(())
    }
}

/// A paid chat; returns the HTTP status and body.
async fn paid_chat(url: &str, content: &str, spend_auth: Value) -> anyhow::Result<(u16, Value)> {
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .timeout(Duration::from_secs(60))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
            "spend_auth": spend_auth,
        }))
        .send()
        .await?;
    let status = response.status().as_u16();
    let text = response.text().await?;
    Ok((
        status,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    ))
}

/// The charge for a served chat: its reported usage at the configured per-token prices.
fn cost_of(body: &Value) -> u64 {
    body["usage"]["prompt_tokens"].as_u64().unwrap() * PRICE_IN
        + body["usage"]["completion_tokens"].as_u64().unwrap() * PRICE_OUT
}

#[tokio::test(flavor = "multi_thread")]
async fn killed_operator_process_settles_paid_chats_exactly_once_on_anvil() -> anyhow::Result<()> {
    if std::env::var("ANVIL_E2E").as_deref() != Ok("1") {
        eprintln!("set ANVIL_E2E=1 to run the chain journey");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let anvil_port = free_port();
    let _anvil = Command::new("anvil")
        .args(["--port", &anvil_port.to_string(), "--silent"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("anvil must be installed (foundry)");
    let rpc = format!("http://127.0.0.1:{anvil_port}");
    let probe = ProviderBuilder::new().connect_http(rpc.parse()?);
    until(10, "anvil to answer", || async {
        probe.get_chain_id().await.is_ok()
    })
    .await?;
    let (chain, accounts) = deploy_shielded(rpc.clone(), 4).await?;

    let upstream_port = free_port();
    let arrivals = dir.path().join("arrivals.log");
    let hold = dir.path().join("hold");
    let script = dir.path().join("upstream.py");
    std::fs::write(&script, UPSTREAM)?;
    let _upstream = Command::new("python3")
        .arg(&script)
        .arg(upstream_port.to_string())
        .arg(&arrivals)
        .arg(&hold)
        .kill_on_drop(true)
        .spawn()?;

    let journal = dir.path().join("data/settlements.jsonl");
    let mut config: Value = serde_json::from_str(include_str!("../../deploy/config.example.json"))?;
    config["tangle"]["rpc_url"] = json!(rpc);
    config["tangle"]["chain_id"] = json!(chain.chain_id);
    config["tangle"]["operator_key"] = json!(format!("0x{OPERATOR_KEY}"));
    config["tangle"]["shielded_credits"] = json!(format!("{:#x}", chain.credits));
    config["tangle"]["service_id"] = json!(SERVICE_ID);
    config["vllm"]["model"] = json!(MODEL);
    config["vllm"]["external"] = json!(true);
    config["vllm"]["host"] = json!("127.0.0.1");
    config["vllm"]["port"] = json!(upstream_port);
    config["vllm"]["price_per_input_token"] = json!(PRICE_IN);
    config["vllm"]["price_per_output_token"] = json!(PRICE_OUT);
    config["server"]["host"] = json!("127.0.0.1");
    config["server"]["port"] = json!(free_port());
    config["billing"] = json!({
        "payment_rails": {"shielded": true},
        "billing_required": true,
        "max_spend_per_request": 0,
        "min_credit_balance": 0,
        "min_charge_amount": 0,
        "max_gas_price_gwei": 100,
        // Backoff long enough that the batch's settlements are still retrying at the kill.
        "claim_max_retries": 10,
        "nonce_store_path": dir.path().join("data/nonces.json"),
        "settlement_journal_path": journal,
        "direct_replay_store_path": dir.path().join("data/used-tx.json"),
        "payment_token_address": format!("{:#x}", chain.token),
    });
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_vec(&config)?)?;

    // One paid chat: the operator is paid its usage cost and the payer refunded the rest.
    let first = start_operator(dir.path(), &config_path, "first").await?;
    assert_eq!(chain.operator_paid().await, 0);
    let (status, body) = paid_chat(
        &first.url,
        "Hello",
        chain.sign(&accounts[0], 0, now() + 600),
    )
    .await?;
    assert_eq!(status, 200, "paid chat refused: {body}");
    assert_eq!(
        (
            body["usage"]["prompt_tokens"].as_u64(),
            body["usage"]["completion_tokens"].as_u64()
        ),
        (Some(5), Some(9))
    );
    let single = cost_of(&body);
    assert_eq!(single, 5 * PRICE_IN + 9 * PRICE_OUT);
    until(30, "the single chat to settle", || async {
        chain.operator_paid().await == single
    })
    .await?;
    assert_eq!(chain.on_chain(&accounts[0]).await, (FUNDED - single, 1));
    assert_eq!(owed_in(&journal), 0);
    eprintln!(
        "single chat: expected {single}, operator paid {}, payer balance {:?}",
        chain.operator_paid().await,
        chain.on_chain(&accounts[0]).await
    );

    // A batch is authorized on-chain and held in the upstream; a gas spike then holds every
    // settlement, the batch is served, and the operator is killed while settlements retry.
    std::fs::write(&hold, b"hold")?;
    let batch = ["a", "bb bb", "ccc ccc ccc"];
    let expiry = now() + 600;
    let mut requests = Vec::new();
    for (index, (&content, account)) in batch.iter().zip(&accounts[1..]).enumerate() {
        let auth = chain.sign(account, 0, expiry);
        let url = first.url.clone();
        requests.push(tokio::spawn(
            async move { paid_chat(&url, content, auth).await },
        ));
        // One authorizeSpend at a time: wait for this request to reach the upstream.
        until(30, "the batch to reach the upstream", || async {
            std::fs::read_to_string(&arrivals)
                .unwrap_or_default()
                .lines()
                .count()
                == index + 2
        })
        .await?;
    }
    for account in &accounts[1..] {
        assert_eq!(chain.on_chain(account).await, (FUNDED - PREAUTH, 1));
    }
    chain.set_base_fee(1_000).await?;
    std::fs::remove_file(&hold)?;
    let mut owed = 0;
    for request in requests {
        let (status, body) = request.await??;
        assert_eq!(status, 200, "batch chat refused: {body}");
        owed += cost_of(&body);
    }
    let costs: Vec<u64> = batch
        .iter()
        .map(|content| content.len() as u64 * PRICE_IN + 9 * PRICE_OUT)
        .collect();
    assert_eq!(owed, costs.iter().sum::<u64>());
    until(10, "the batch's settlements to be journaled", || async {
        owed_in(&journal) == batch.len()
    })
    .await?;
    let log = first.log.clone();
    first.sigkill().await?;
    assert!(std::fs::read_to_string(&log)?.contains("gas price check failed"));
    assert_eq!(owed_in(&journal), batch.len());
    assert_eq!(chain.operator_paid().await, single);
    for account in &accounts[1..] {
        assert_eq!(chain.on_chain(account).await, (FUNDED - PREAUTH, 1));
    }
    let journaled = std::fs::read(&journal)?;

    // The same binary on the same journal settles every served chat once, before expiry.
    chain.set_base_fee(1).await?;
    let second = start_operator(dir.path(), &config_path, "second").await?;
    until(60, "the restart to settle the journal", || async {
        owed_in(&journal) == 0
    })
    .await?;
    assert!(now() < expiry, "settled after the SpendAuth expiry");
    let paid = chain.operator_paid().await;
    eprintln!(
        "kill/restart batch: expected {owed} (costs {costs:?}), operator paid {}",
        paid - single
    );
    assert_eq!(paid, single + owed);
    for (account, cost) in accounts[1..].iter().zip(&costs) {
        assert_eq!(chain.on_chain(account).await, (FUNDED - cost, 1));
    }

    // Killed after confirming but before forgetting: another restart charges nothing more.
    second.sigkill().await?;
    std::fs::write(&journal, journaled)?;
    let third = start_operator(dir.path(), &config_path, "third").await?;
    until(60, "the replayed journal to be skipped", || async {
        owed_in(&journal) == 0
    })
    .await?;
    assert_eq!(chain.operator_paid().await, single + owed);
    assert_eq!(chain.on_chain(&accounts[0]).await, (FUNDED - single, 1));
    for (account, cost) in accounts[1..].iter().zip(&costs) {
        assert_eq!(chain.on_chain(account).await, (FUNDED - cost, 1));
    }
    third.sigkill().await?;
    Ok(())
}
