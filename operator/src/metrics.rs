use std::sync::LazyLock;
use std::time::Instant;

use prometheus::{
    Encoder, Gauge, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};

// ─── Global metrics registry ─────────────────────────────────────────────

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

pub static REQUEST_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new(
        "vllm_operator_request_count",
        "Total inference requests handled",
    )
    .namespace("vllm_operator");
    let counter = IntCounterVec::new(opts, &["status"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new(
        "vllm_operator_request_duration_seconds",
        "Request duration in seconds",
    )
    .namespace("vllm_operator")
    .buckets(vec![
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ]);
    let hist = HistogramVec::new(opts, &["status"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static TOKENS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("vllm_operator_tokens_total", "Total tokens processed")
        .namespace("vllm_operator");
    let counter = IntCounterVec::new(opts, &["type"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static ACTIVE_REQUESTS: LazyLock<Gauge> = LazyLock::new(|| {
    let gauge = Gauge::new(
        "vllm_operator_active_requests",
        "Number of currently active inference requests",
    )
    .unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

// ─── RAII guard for request lifecycle ────────────────────────────────────

impl Default for RequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RequestGuard {
    start: Instant,
    prompt_tokens: u32,
    completion_tokens: u32,
    success: bool,
}

impl RequestGuard {
    pub fn new() -> Self {
        ACTIVE_REQUESTS.inc();
        Self {
            start: Instant::now(),
            prompt_tokens: 0,
            completion_tokens: 0,
            success: false,
        }
    }

    pub fn set_tokens(&mut self, prompt: u32, completion: u32) {
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
    }

    pub fn set_success(&mut self) {
        self.success = true;
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();

        let status = if self.success { "success" } else { "error" };
        let elapsed = self.start.elapsed().as_secs_f64();

        REQUEST_COUNT.with_label_values(&[status]).inc();
        REQUEST_DURATION
            .with_label_values(&[status])
            .observe(elapsed);

        if self.prompt_tokens > 0 {
            TOKENS_TOTAL
                .with_label_values(&["prompt"])
                .inc_by(self.prompt_tokens as u64);
        }
        if self.completion_tokens > 0 {
            TOKENS_TOTAL
                .with_label_values(&["completion"])
                .inc_by(self.completion_tokens as u64);
        }
    }
}

// ─── Gather all metrics as Prometheus text ───────────────────────────────

pub fn gather() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
