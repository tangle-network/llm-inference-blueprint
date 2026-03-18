FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
RUN python3 -m pip install --no-cache-dir vllm==0.6.4

# ── Build Rust operator ──────────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder

WORKDIR /build
COPY Cargo.toml ./
COPY operator/ operator/

# Build release binary
RUN cargo build --release -p vllm-operator

# ── Final image ──────────────────────────────────────────────────────────
FROM base

# Copy operator binary
COPY --from=builder /build/target/release/vllm-operator /usr/local/bin/vllm-operator

# Create non-root user
RUN useradd -m -s /bin/bash operator
USER operator
WORKDIR /home/operator

# Default config location
ENV VLLM_OP_CONFIG=/home/operator/config.json

# Expose operator HTTP port and vLLM internal port
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["vllm-operator"]
