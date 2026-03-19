use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::config::OperatorConfig;
use crate::server::{ChatCompletionRequest, ChatCompletionResponse};

/// Manages a vLLM subprocess.
pub struct VllmProcess {
    child: Mutex<Option<Child>>,
    config: Arc<OperatorConfig>,
    client: reqwest::Client,
}

impl VllmProcess {
    /// Connect to an already-running vLLM instance without spawning a subprocess.
    /// Useful for testing or when vLLM is managed externally.
    pub fn connect(config: Arc<OperatorConfig>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("failed to build HTTP client");
        Self {
            child: Mutex::new(None),
            config,
            client,
        }
    }

    /// Spawn the vLLM OpenAI-compatible server as a subprocess.
    pub async fn spawn(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let vllm_url = format!("http://{}:{}", config.vllm.host, config.vllm.port);

        let parts: Vec<&str> = config.vllm.command.split_whitespace().collect();
        let (program, base_args) = parts
            .split_first()
            .ok_or_else(|| anyhow::anyhow!("empty vllm command"))?;

        let mut cmd = Command::new(program);
        cmd.args(base_args)
            .arg("--model")
            .arg(&config.vllm.model)
            .arg("--host")
            .arg(&config.vllm.host)
            .arg("--port")
            .arg(config.vllm.port.to_string())
            .arg("--max-model-len")
            .arg(config.vllm.max_model_len.to_string())
            .arg("--tensor-parallel-size")
            .arg(config.vllm.tensor_parallel_size.to_string());

        if let Some(ref token) = config.vllm.hf_token {
            cmd.env("HF_TOKEN", token);
        }
        if let Some(ref dir) = config.vllm.download_dir {
            cmd.arg("--download-dir").arg(dir);
        }
        for arg in &config.vllm.extra_args {
            cmd.arg(arg);
        }

        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn()?;
        tracing::info!(pid = child.id(), url = %vllm_url, "spawned vLLM process");

        // Drain stdout and stderr asynchronously so the OS pipe buffer never
        // fills up and blocks the child process.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(target: "vllm::stdout", "{}", line);
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(target: "vllm::stderr", "{}", line);
                }
            });
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self {
            child: Mutex::new(Some(child)),
            config,
            client,
        })
    }

    /// Block until vLLM's /health endpoint returns 200.
    pub async fn wait_ready(&self) -> anyhow::Result<()> {
        let url = format!(
            "http://{}:{}/health",
            self.config.vllm.host, self.config.vllm.port
        );
        let timeout = Duration::from_secs(self.config.vllm.startup_timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!(
                    "vLLM failed to become ready within {}s",
                    self.config.vllm.startup_timeout_secs
                );
            }

            {
                let mut guard = self.child.lock().await;
                if let Some(ref mut child) = *guard {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            anyhow::bail!("vLLM process exited during startup: {status}");
                        }
                        Ok(None) => {}
                        Err(e) => {
                            anyhow::bail!("failed to poll vLLM process: {e}");
                        }
                    }
                }
            }

            match self.client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    /// Check if vLLM is currently healthy.
    pub async fn is_healthy(&self) -> bool {
        let url = format!(
            "http://{}:{}/health",
            self.config.vllm.host, self.config.vllm.port
        );
        matches!(
            self.client
                .get(&url)
                .timeout(Duration::from_secs(5))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        )
    }

    fn completions_url(&self) -> String {
        format!(
            "http://{}:{}/v1/chat/completions",
            self.config.vllm.host, self.config.vllm.port
        )
    }

    fn build_vllm_body(&self, req: &ChatCompletionRequest, stream: bool) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": req.model.as_deref().unwrap_or(&self.config.vllm.model),
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": stream,
            "top_p": req.top_p,
            "frequency_penalty": req.frequency_penalty,
            "presence_penalty": req.presence_penalty,
            "stop": req.stop,
        });

        // When streaming, request usage in the final chunk
        if stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        body
    }

    /// Proxy a non-streaming chat completion request to vLLM.
    pub async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> anyhow::Result<ChatCompletionResponse> {
        let vllm_body = self.build_vllm_body(req, false);

        let resp = self
            .client
            .post(self.completions_url())
            .json(&vllm_body)
            .send()
            .await?
            .error_for_status()?
            .json::<ChatCompletionResponse>()
            .await?;

        Ok(resp)
    }

    /// Start a streaming chat completion, returning the raw reqwest::Response.
    /// The caller proxies the byte stream to the client.
    pub async fn chat_completion_stream(
        &self,
        req: &ChatCompletionRequest,
    ) -> anyhow::Result<reqwest::Response> {
        let vllm_body = self.build_vllm_body(req, true);

        let resp = self
            .client
            .post(self.completions_url())
            .json(&vllm_body)
            .send()
            .await?
            .error_for_status()?;

        Ok(resp)
    }

    /// Shut down the vLLM subprocess.
    pub async fn shutdown(&self) {
        let mut guard = self.child.lock().await;
        if let Some(ref mut child) = *guard {
            tracing::info!(pid = child.id(), "shutting down vLLM");
            let _ = child.kill().await;
        }
        *guard = None;
    }
}

impl Drop for VllmProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.start_kill();
            }
        }
    }
}
