//! Public CLI -> core recipe -> child argv -> HTTP readiness; only the GPU process is substituted.
use serde_json::{json, Value};
use std::{net::TcpListener, process::Stdio, time::Duration};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test]
async fn operator_lite_launches_inherited_sglang_recipe() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let engine_port = free_port();
    let operator_port = free_port();
    let argv_path = dir.path().join("argv.json");
    let router_argv_path = dir.path().join("router-argv.json");
    let ready_path = dir.path().join("ready");
    let stop_path = dir.path().join("stop");
    let peer = dir.path().join("engine.py");
    std::fs::write(
        &peer,
        r#"import sys, json, os, time, http.client, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
parent = os.getppid()
with open(sys.argv[1], 'w') as output: json.dump(sys.argv[4:], output)
while not os.path.exists(sys.argv[2]) and not os.path.exists(sys.argv[3]) and os.getppid() == parent: time.sleep(0.02)
if os.path.exists(sys.argv[3]) or os.getppid() != parent: sys.exit(0)
port = int(sys.argv[sys.argv.index('--port') + 1])
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {'data':[{'id':'example/GLM-derivative'}]} if self.path == '/v1/models' else {}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        if '--worker-urls' in sys.argv:
            worker = urllib.parse.urlsplit(sys.argv[sys.argv.index('--worker-urls') + 1])
            connection = http.client.HTTPConnection(worker.hostname, worker.port)
            connection.request('POST', self.path, body, {'Content-Type':'application/json'})
            response = connection.getresponse()
            self.send_response(response.status); self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(response.read()); connection.close(); return
        response={'id':'core-endpoint','object':'chat.completion','created':1,'model':'example/GLM-derivative',
          'choices':[{'index':0,'message':{'role':'assistant','content':'core endpoint'},'finish_reason':'stop'}],
          'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2,'prompt_tokens_details':{'cached_tokens':0}}}
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', port), Health)
server.timeout = 0.05
while not os.path.exists(sys.argv[3]) and os.getppid() == parent: server.handle_request()
"#,
    )?;
    let mut config: Value = serde_json::from_str(include_str!("../../deploy/config.example.json"))?;
    config["vllm"]["engine"] = json!("sglang");
    config["vllm"]["base_family"] = json!("glm-5.3");
    config["vllm"]["gpu_class"] = json!("nvidia-h200");
    config["vllm"]["tensor_parallel_size"] = json!(8);
    config["vllm"]["model"] = json!("example/GLM-derivative");
    config["vllm"]["command"] = json!(format!(
        "python3 {} {} {} {}",
        peer.display(),
        argv_path.display(),
        ready_path.display(),
        stop_path.display()
    ));
    config["vllm"]["router_command"] = json!(format!(
        "python3 {} {} {} {}",
        peer.display(),
        router_argv_path.display(),
        ready_path.display(),
        stop_path.display()
    ));
    config["vllm"]["host"] = json!("127.0.0.2");
    config["vllm"]["port"] = json!(engine_port);
    config["vllm"]["startup_timeout_secs"] = json!(10);
    config["vllm"]["external"] = json!(false);
    config["server"]["host"] = json!("127.0.0.1");
    config["server"]["port"] = json!(operator_port);
    config["billing"]["billing_required"] = json!(false);
    config["billing"]["payment_rails"] = json!({"shielded": false, "direct": false});
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(&config)?)?;
    let log_path = dir.path().join("operator.log");
    let log = std::fs::File::create(&log_path)?;
    let mut operator = tokio::process::Command::new(env!("CARGO_BIN_EXE_operator-lite"))
        .arg(path)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .kill_on_drop(true)
        .spawn()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    let ready = tokio::time::timeout(Duration::from_secs(12), async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if client
                    .get(format!("http://127.0.0.1:{operator_port}/metrics"))
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await?;
        assert_eq!(
            client
                .get(format!("http://127.0.0.1:{operator_port}/health"))
                .send()
                .await?
                .status(),
            503
        );
        std::fs::write(&ready_path, b"ready")?;
        loop {
            if let Some(status) = operator.try_wait()? {
                let log = std::fs::read_to_string(&log_path)?;
                let reason = if log.contains("stack overflow") { "stack overflow" }
                    else { log.lines().find(|line| line.starts_with("Error:")).unwrap_or("no error diagnostic") };
                anyhow::bail!("operator-lite exited before readiness ({status}): {reason}");
            }
            if client
                .get(format!("http://127.0.0.1:{operator_port}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                let response = client.post(format!("http://127.0.0.1:{operator_port}/v1/chat/completions"))
                    .json(&json!({"model":"example/GLM-derivative","messages":[{"role":"user","content":"Hello"}]}))
                    .send().await?;
                anyhow::ensure!(response.status().is_success(), "chat did not reach the effective core endpoint: {}", response.status());
                assert_eq!(response.json::<Value>().await?["id"], "core-endpoint");
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    let metadata = async {
        if !matches!(&ready, Ok(Ok(()))) {
            return Ok::<_, anyhow::Error>(());
        }
        // The first call hashes /proc/self/exe once; a ~590 MB debug binary on a loaded host takes tens of seconds.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()?;
        let info: Value = client
            .get(format!("http://127.0.0.1:{operator_port}/v1/operator"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let runtime = &info["runtime"];
        assert_eq!(
            runtime["ready"], true,
            "operator omitted core runtime facts"
        );
        assert_eq!(runtime["served_models"], json!(["example/GLM-derivative"]));
        assert_eq!(runtime["recipe"]["gpu_class"], "nvidia-h200");
        assert_eq!(runtime["processes"].as_array().map(Vec::len), Some(2));
        assert_eq!(runtime["operator_sha256"].as_str().map(str::len), Some(64));
        assert!(runtime["verified_image_digest"].is_null());
        Ok(())
    }
    .await;
    if operator.try_wait()?.is_none() {
        if let Some(pid) = operator.id() {
            tokio::process::Command::new("kill")
                .args(["-INT", &pid.to_string()])
                .status()
                .await?;
        }
    }
    let stopped = tokio::time::timeout(Duration::from_secs(8), operator.wait()).await;
    std::fs::write(&stop_path, b"stop")?;
    ready??;
    metadata?;
    assert!(stopped??.success());
    let argv: Vec<String> = serde_json::from_slice(&std::fs::read(argv_path)?)?;
    assert!(argv
        .windows(2)
        .any(|args| args == ["--model-path", "example/GLM-derivative"]));
    assert!(argv
        .windows(2)
        .any(|args| args == ["--tool-call-parser", "auto"]));
    assert!(argv.iter().any(|arg| arg == "--enable-cache-report"));
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn private_operator_requires_selected_sdk_key() -> anyhow::Result<()> {
    use blueprint_keystore::{
        backends::Backend,
        crypto::{k256::K256Ecdsa, BytesEncoding, KeyType},
        Keystore, KeystoreConfig,
    };
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir()?;
    let keys = dir.path().join("keys");
    std::fs::create_dir(&keys)?;
    std::fs::set_permissions(&keys, std::fs::Permissions::from_mode(0o700))?;
    let store = Keystore::new(KeystoreConfig::new().fs_root(&keys))?;
    let public = store.generate::<K256Ecdsa>(None)?;
    let token = hex::encode(store.get_secret::<K256Ecdsa>(&public)?.to_bytes());
    let port = free_port();
    let mut config: Value = serde_json::from_str(include_str!("../../deploy/config.example.json"))?;
    config["vllm"]["external"] = json!(true);
    config["vllm"]["port"] = json!(free_port());
    config["server"]["host"] = json!("127.0.0.1");
    config["server"]["port"] = json!(port);
    config["server"]["endpoint_key"] =
        json!({"kind":"secret-ref","key":hex::encode(public.to_bytes()),"format":"bearer"});
    config["billing"]["billing_required"] = json!(false);
    config["billing"]["payment_rails"] = json!({"shielded":false,"direct":false});
    config["billing"]["nonce_store_path"] = json!(dir.path().join("nonces.json"));
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(&config)?)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    let mut operator = tokio::process::Command::new(env!("CARGO_BIN_EXE_operator-lite"))
        .arg(&path)
        .current_dir(dir.path())
        .env_clear()
        .env("KEYSTORE_URI", &keys)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    let url = format!("http://127.0.0.1:{port}");
    let result = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if client.get(format!("{url}/v1/models")).send().await.is_ok() {
                break;
            }
            anyhow::ensure!(
                operator.try_wait()?.is_none(),
                "private operator exited before binding"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let mut refused = Vec::new();
        for route in ["/v1/models", "/health", "/metrics", "/v1/chat/completions"] {
            let request = if route.ends_with("completions") {
                client
                    .post(format!("{url}{route}"))
                    .json(&json!({"messages":[{"role":"user","content":"Hello"}]}))
            } else {
                client.get(format!("{url}{route}"))
            };
            refused.push((route, request.send().await?.status()));
        }
        let wrong = client
            .get(format!("{url}/v1/models"))
            .bearer_auth("invalid")
            .send()
            .await?
            .status();
        let valid = client
            .get(format!("{url}/v1/models"))
            .bearer_auth(&token)
            .send()
            .await?
            .status();
        let cold = client
            .get(format!("{url}/health"))
            .bearer_auth(&token)
            .send()
            .await?
            .status();
        let probe = include_str!("../Dockerfile")
            .lines()
            .find(|line| line.starts_with("HEALTHCHECK "))
            .and_then(|line| line.split_once(" CMD "))
            .ok_or_else(|| anyhow::anyhow!("serving image has no HEALTHCHECK CMD"))?
            .1
            .replace("8080", &port.to_string());
        let argv: Vec<String> = if probe.starts_with('[') {
            serde_json::from_str(&probe)?
        } else {
            vec!["/bin/sh".into(), "-c".into(), probe]
        };
        let mut probe = tokio::process::Command::new(&argv[0])
            .args(&argv[1..])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let live = match tokio::time::timeout(Duration::from_secs(5), probe.wait()).await {
            Ok(status) => status?.success(),
            Err(_) => {
                probe.kill().await?;
                probe.wait().await?;
                false
            }
        };
        Ok::<_, anyhow::Error>((refused, wrong, valid, cold, live))
    })
    .await;
    operator.kill().await?;
    operator.wait().await?;
    let (refused, wrong, valid, cold, live) = result??;
    for (route, status) in refused {
        assert_eq!(status, 401, "unauthenticated {route}");
    }
    assert_eq!(wrong, 401);
    assert_eq!(valid, 200);
    assert_eq!(cold, 503);
    assert!(
        live,
        "Dockerfile HEALTHCHECK rejected private operator liveness"
    );
    let wallet = <K256Ecdsa as KeyType>::Secret::from_bytes(&hex::decode(
        config["tangle"]["operator_key"]
            .as_str()
            .unwrap()
            .trim_start_matches("0x"),
    )?)?;
    store.insert::<K256Ecdsa>(&wallet)?;
    let wallet_public = hex::encode(K256Ecdsa::public_from_secret(&wallet).to_bytes());
    let missing =
        Keystore::new(KeystoreConfig::new().in_memory(true))?.generate::<K256Ecdsa>(None)?;
    for case in ["root", "format", "key", "permissions", "wallet"] {
        let mut invalid = config.clone();
        if case == "format" {
            invalid["server"]["endpoint_key"]
                .as_object_mut()
                .unwrap()
                .remove("format");
        }
        if case == "key" {
            invalid["server"]["endpoint_key"]["key"] = json!(hex::encode(missing.to_bytes()));
        }
        if case == "wallet" {
            invalid["server"]["endpoint_key"]["key"] = json!(wallet_public);
        }
        if case == "permissions" {
            std::fs::set_permissions(&keys, std::fs::Permissions::from_mode(0o755))?;
        }
        std::fs::write(&path, serde_json::to_vec(&invalid)?)?;
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_operator-lite"));
        command
            .arg(&path)
            .current_dir(dir.path())
            .env_clear()
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if case != "root" {
            command.env("KEYSTORE_URI", &keys);
        }
        let mut rejected = command.spawn()?;
        let outcome = tokio::time::timeout(Duration::from_secs(3), rejected.wait()).await;
        if outcome.is_err() {
            rejected.kill().await?;
            rejected.wait().await?;
        }
        std::fs::set_permissions(&keys, std::fs::Permissions::from_mode(0o700))?;
        assert!(
            outcome.is_ok_and(|status| status.is_ok_and(|status| !status.success())),
            "private operator accepted invalid {case} configuration"
        );
        assert!(
            tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_err(),
            "invalid {case} configuration left an HTTP listener"
        );
    }
    Ok(())
}
