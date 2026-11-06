use crate::config::OperatorConfig;
use crate::server::{ChatCompletionRequest, ChatCompletionResponse};
use std::sync::Arc;
use tangle_inference_core::engine::EngineProcess;

/// Model request and provider routing adapter; core owns the serving process.
pub struct ChatClient {
    process: EngineProcess,
    config: Arc<OperatorConfig>,
}
impl ChatClient {
    pub fn connect(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        Ok(Self {
            process: EngineProcess::connect(config.vllm.engine.clone())?,
            config,
        })
    }
    pub async fn spawn(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        Ok(Self {
            process: EngineProcess::spawn(config.vllm.engine.clone()).await?,
            config,
        })
    }
    pub async fn wait_ready(&self) -> anyhow::Result<()> {
        self.process.wait_ready().await
    }
    pub async fn is_healthy(&self) -> bool {
        self.process.is_healthy().await
    }
    pub async fn shutdown(&self) {
        self.process.shutdown().await;
    }
    pub async fn supervise(
        &self,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> anyhow::Result<()> {
        self.process.supervise(shutdown).await
    }
    fn completions_url(&self, model: &str) -> String {
        self.config.vllm.completions_url_for(model)
    }
    fn authorized(&self, model: &str, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let Some(provider) = self.config.vllm.provider_for(model) else {
            return builder;
        };
        let Some(env_name) = provider.api_key_env.as_deref() else {
            return builder;
        };
        match std::env::var(env_name) {
            Ok(key) if !key.is_empty() => builder.bearer_auth(key),
            _ => builder,
        }
    }
    fn build_vllm_body(&self, req: &ChatCompletionRequest, stream: bool) -> serde_json::Value {
        let mut body = req.options.clone();
        body.extend(
            serde_json::json!({
                "model": req.model.as_deref().unwrap_or(&self.config.vllm.model),
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stream": stream,
                "top_p": req.top_p,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty,
                "stop": req.stop,
            })
            .as_object()
            .expect("chat request object")
            .clone(),
        );
        // When streaming, request usage in the final chunk
        if stream {
            let mut options = body
                .get("stream_options")
                .and_then(|value| value.as_object())
                .cloned()
                .unwrap_or_default();
            options.insert("include_usage".into(), serde_json::json!(true));
            body.insert("stream_options".into(), serde_json::Value::Object(options));
        }
        serde_json::Value::Object(body)
    }
    /// Proxy a non-streaming chat completion request to vLLM.
    pub async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> anyhow::Result<ChatCompletionResponse> {
        let model = req.model.as_deref().unwrap_or(&self.config.vllm.model);
        let vllm_body = self.build_vllm_body(req, false);
        let resp = self
            .authorized(
                model,
                self.process.client().post(self.completions_url(model)),
            )
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
        let model = req.model.as_deref().unwrap_or(&self.config.vllm.model);
        let vllm_body = self.build_vllm_body(req, true);
        let resp = self
            .authorized(
                model,
                self.process.client().post(self.completions_url(model)),
            )
            .json(&vllm_body)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp)
    }
}
