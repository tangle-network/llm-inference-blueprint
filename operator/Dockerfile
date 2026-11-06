ARG ENGINE_IMAGE=scratch
FROM rust:1.93-bookworm AS build-base
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev libssl-dev pkg-config clang cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY operator/ operator/
FROM build-base AS operator-lite-builder
RUN --mount=type=cache,target=/build/target,sharing=locked --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked --mount=type=cache,target=/usr/local/cargo/git,sharing=locked cargo build --locked --release --no-default-features -p llm-inference --bin operator-lite && mkdir /out && cp target/release/operator-lite /out/ && strip /out/operator-lite
FROM build-base AS builder
RUN --mount=type=cache,target=/build/target,sharing=locked --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked --mount=type=cache,target=/usr/local/cargo/git,sharing=locked cargo build --locked --release -p llm-inference --bin llm-operator && mkdir /out && cp target/release/llm-operator /out/
COPY --from=operator-lite-builder /out/operator-lite /out/

FROM ${ENGINE_IMAGE} AS operator-lite
ARG ENGINE_IMAGE
ARG VCS_REF
ARG CORE_REVISION
LABEL org.opencontainers.image.source="https://github.com/tangle-network/llm-inference-blueprint" org.opencontainers.image.revision=$VCS_REF org.opencontainers.image.base.name=$ENGINE_IMAGE tools.tangle.inference-core.revision=$CORE_REVISION
USER root
COPY --from=operator-lite-builder /out/operator-lite /usr/local/bin/operator-lite
RUN test -n "$VCS_REF" && test -n "$CORE_REVISION" && groupadd -f operator && useradd -m -g operator -s /bin/bash operator && install -d -o operator -g operator /home/operator/.cache/huggingface /home/operator/state
USER operator
WORKDIR /home/operator
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080"]
ENV HF_HOME=/home/operator/.cache/huggingface VLLM_OP_BILLING__NONCE_STORE_PATH=/home/operator/state/nonces.json
ENTRYPOINT ["/usr/local/bin/operator-lite"]
CMD []

FROM vllm/vllm-openai:v0.32.0-cu129-ubuntu2404
COPY --from=builder /out/ /usr/local/bin/
RUN groupadd -f operator && useradd -m -g operator -s /bin/bash operator
USER operator
WORKDIR /home/operator
RUN mkdir -p /home/operator/.cache/huggingface /home/operator/state
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080"]
ENTRYPOINT ["llm-operator"]
ENV VLLM_OP_CONFIG=/home/operator/config.json HF_HOME=/home/operator/.cache/huggingface DATA_DIR=/home/operator/state KEYSTORE_URI=/home/operator/state/keystore PROTOCOL=tangle VLLM_OP_BILLING__NONCE_STORE_PATH=/home/operator/state/nonces.json
CMD ["run"]
