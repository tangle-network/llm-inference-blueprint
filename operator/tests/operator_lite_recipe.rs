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
    let ready_path = dir.path().join("ready");
    let stop_path = dir.path().join("stop");
    let peer = dir.path().join("engine.py");
    std::fs::write(
        &peer,
        r#"import sys, json, os, time
from http.server import HTTPServer, BaseHTTPRequestHandler
parent = os.getppid()
with open(sys.argv[1], 'w') as output: json.dump(sys.argv[4:], output)
while not os.path.exists(sys.argv[2]) and not os.path.exists(sys.argv[3]) and os.getppid() == parent: time.sleep(0.02)
if os.path.exists(sys.argv[3]) or os.getppid() != parent: sys.exit(0)
port = int(sys.argv[sys.argv.index('--port') + 1])
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
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
    config["vllm"]["port"] = json!(engine_port);
    config["vllm"]["startup_timeout_secs"] = json!(10);
    config["vllm"]["external"] = json!(false);
    config["server"]["host"] = json!("127.0.0.1");
    config["server"]["port"] = json!(operator_port);
    config["billing"]["billing_required"] = json!(false);
    config["billing"]["payment_rails"] = json!({"shielded": false, "direct": false});
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(&config)?)?;
    let mut operator = tokio::process::Command::new(env!("CARGO_BIN_EXE_operator-lite"))
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
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
            if operator.try_wait()?.is_some() {
                anyhow::bail!("operator-lite exited before readiness");
            }
            if client
                .get(format!("http://127.0.0.1:{operator_port}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
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
