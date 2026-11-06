import type { PublicClient } from "viem";

import { readCreditAccount } from "./account";
import { signSpendAuth, toSpendAuthPayload } from "./spendAuth";
import type {
  ChatCompletionResponse,
  ChatMessage,
  ChatOptions,
  ErrorResponse,
  InferenceClientConfig,
  ModelList,
  OperatorInfo,
  SpendAuthPayload,
} from "./types";

const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_EXPIRY_SECONDS = 300;
const CHARS_PER_TOKEN = 4;

/** A streamed token plus the running accumulation. */
export interface StreamChunk {
  token: string;
  accumulated: string;
}

export interface ChatStreamHandlers {
  onToken: (chunk: StreamChunk) => void;
  onDone: (full: string) => void;
  onError: (error: Error) => void;
}

function estimateInputTokens(messages: ChatMessage[]): number {
  return messages.reduce(
    (sum, m) => sum + Math.ceil(m.content.length / CHARS_PER_TOKEN),
    0,
  );
}

/**
 * Inference client for the vLLM blueprint with ShieldedCredits billing.
 *
 * Uses an ephemeral signer from createLocalSpendSigner. An optional
 * PublicClient synchronizes the on-chain spend nonce.
 */
export function createInferenceClient(config: InferenceClientConfig) {
  const fetchImpl = config.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const priceIn = config.pricePerInputToken ?? 1n;
  const priceOut = config.pricePerOutputToken ?? 2n;
  const expirySeconds = config.expirySeconds ?? DEFAULT_EXPIRY_SECONDS;
  let currentNonce = 0n;

  function estimateCost(inputTokens: number, maxOutputTokens: number): bigint {
    return BigInt(inputTokens) * priceIn + BigInt(maxOutputTokens) * priceOut;
  }

  /** Sync the spend nonce (and verify balance) from on-chain state. */
  async function syncNonce(publicClient: PublicClient): Promise<bigint> {
    const account = await readCreditAccount(
      publicClient,
      config.shieldedCreditsAddress,
      config.commitment,
    );
    currentNonce = account.nonce;
    return currentNonce;
  }

  async function authorize(
    messages: ChatMessage[],
    maxTokens: number,
    authorizedAmount?: bigint,
  ): Promise<SpendAuthPayload> {
    const amount =
      authorizedAmount ??
      estimateCost(estimateInputTokens(messages), maxTokens);
    const expiry =
      BigInt(Math.floor(Date.now() / 1000)) + BigInt(expirySeconds);
    const auth = await signSpendAuth({
      signer: config.signer,
      chainId: config.chainId,
      shieldedCreditsAddress: config.shieldedCreditsAddress,
      commitment: config.commitment,
      serviceId: config.serviceId,
      operator: config.operatorAddress,
      amount,
      nonce: currentNonce,
      expiry,
    });
    currentNonce += 1n;
    return toSpendAuthPayload(auth);
  }

  function requestBody(
    messages: ChatMessage[],
    options: ChatOptions,
    maxTokens: number,
    stream: boolean,
    spendAuth: SpendAuthPayload,
  ) {
    return {
      model: options.model ?? config.model,
      messages,
      max_tokens: maxTokens,
      temperature: options.temperature ?? 0.7,
      stream,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      stop: options.stop,
      spend_auth: spendAuth,
    };
  }

  async function raiseForStatus(response: Response): Promise<never> {
    let message = response.statusText;
    try {
      const err = (await response.json()) as ErrorResponse;
      message = err.error?.message ?? message;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new Error(`Inference request failed (${response.status}): ${message}`);
  }

  /** Non-streaming chat completion. */
  async function chat(
    messages: ChatMessage[],
    options: ChatOptions & { authorizedAmount?: bigint } = {},
  ): Promise<ChatCompletionResponse> {
    const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    const spendAuth = await authorize(
      messages,
      maxTokens,
      options.authorizedAmount,
    );
    const response = await fetchImpl(
      `${config.operatorUrl}/v1/chat/completions`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          requestBody(messages, options, maxTokens, false, spendAuth),
        ),
      },
    );
    if (!response.ok) return raiseForStatus(response);
    return response.json() as Promise<ChatCompletionResponse>;
  }

  /**
   * Streaming chat completion over SSE. Returns an abort function. Tokens are
   * delivered via `handlers.onToken`; `[DONE]` resolves `onToken`/`onDone`.
   */
  function chatStream(
    messages: ChatMessage[],
    handlers: ChatStreamHandlers,
    options: ChatOptions & { authorizedAmount?: bigint } = {},
  ): () => void {
    const controller = new AbortController();
    const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;

    (async () => {
      let accumulated = "";
      try {
        const spendAuth = await authorize(
          messages,
          maxTokens,
          options.authorizedAmount,
        );
        const response = await fetchImpl(
          `${config.operatorUrl}/v1/chat/completions`,
          {
            method: "POST",
            signal: controller.signal,
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
            },
            body: JSON.stringify(
              requestBody(messages, options, maxTokens, true, spendAuth),
            ),
          },
        );
        if (!response.ok) return raiseForStatus(response);
        if (!response.body) throw new Error("operator returned no stream body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const line = frame
              .split("\n")
              .find((l) => l.startsWith("data:"));
            if (!line) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") {
              handlers.onDone(accumulated);
              return;
            }
            try {
              const json = JSON.parse(payload) as {
                choices?: { delta?: { content?: string } }[];
              };
              const token = json.choices?.[0]?.delta?.content;
              if (token) {
                accumulated += token;
                handlers.onToken({ token, accumulated });
              }
            } catch {
              // keep-alive comment or partial frame; ignore
            }
          }
        }
        handlers.onDone(accumulated);
      } catch (err) {
        if (controller.signal.aborted) return;
        handlers.onError(err instanceof Error ? err : new Error(String(err)));
      }
    })();

    return () => controller.abort();
  }

  async function listModels(): Promise<ModelList> {
    const response = await fetchImpl(`${config.operatorUrl}/v1/models`);
    if (!response.ok) return raiseForStatus(response);
    return response.json() as Promise<ModelList>;
  }

  async function operatorInfo(): Promise<OperatorInfo> {
    const response = await fetchImpl(`${config.operatorUrl}/v1/operator`);
    if (!response.ok) return raiseForStatus(response);
    return response.json() as Promise<OperatorInfo>;
  }

  async function health(): Promise<boolean> {
    try {
      const response = await fetchImpl(`${config.operatorUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  return {
    chat,
    chatStream,
    listModels,
    operatorInfo,
    health,
    estimateCost,
    syncNonce,
    setNonce: (nonce: bigint) => {
      currentNonce = nonce;
    },
    getNonce: () => currentNonce,
    get spendingKey() {
      return config.signer.account;
    },
  };
}

export type InferenceClient = ReturnType<typeof createInferenceClient>;
