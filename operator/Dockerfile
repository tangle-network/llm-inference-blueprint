FROM rust:1.93-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libssl-dev pkg-config clang cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY operator/ operator/
RUN cargo build --locked --release -p llm-inference --bins

FROM vllm/vllm-openai:v0.32.0-cu129-ubuntu2404
COPY --from=builder /build/target/release/llm-operator /build/target/release/operator-lite /usr/local/bin/
RUN useradd -m -s /bin/bash operator
USER operator
WORKDIR /home/operator
RUN mkdir -p /home/operator/.cache/huggingface /home/operator/state
EXPOSE 8080 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8080/health || exit 1
ENTRYPOINT ["llm-operator"]
ENV VLLM_OP_CONFIG=/home/operator/config.json HF_HOME=/home/operator/.cache/huggingface DATA_DIR=/home/operator/state KEYSTORE_URI=/home/operator/state/keystore PROTOCOL=tangle VLLM_OP_BILLING__NONCE_STORE_PATH=/home/operator/state/nonces.json
CMD ["run"]
