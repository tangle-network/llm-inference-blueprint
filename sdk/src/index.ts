import { ethers } from "ethers";
import type {
  ChatMessage,
  ChatOptions,
  ChatCompletionResponse,
  InferenceClientConfig,
  SpendAuth,
  ModelList,
  ErrorResponse,
} from "./types";

export * from "./types";

// EIP-712 constants matching ShieldedCredits.sol
const SPEND_TYPEHASH = ethers.keccak256(
  ethers.toUtf8Bytes(
    "SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)"
  )
);

const EIP712_DOMAIN = {
  name: "ShieldedCredits",
  version: "1",
};

/**
 * High-level inference client that handles SpendAuth signing and request routing.
 *
 * Usage:
 * ```ts
 * const client = createInferenceClient({
 *   operatorUrl: "https://op1.example.com",
 *   shieldedCreditsAddress: "0x...",
 *   chainId: 1,
 *   commitment: "0x...",
 *   serviceId: 1n,
 *   operatorAddress: "0x...",
 *   spendingKeyPrivate: "0x...",
 * });
 *
 * const response = await client.chat([
 *   { role: "user", content: "What is Tangle Network?" }
 * ]);
 * console.log(response.choices[0].message.content);
 * ```
 */
export function createInferenceClient(config: InferenceClientConfig) {
  const spendingWallet = new ethers.Wallet(config.spendingKeyPrivate);
  let currentNonce = 0n;

  /**
   * Sign a SpendAuth for a given amount.
   */
  async function signSpendAuth(
    amount: bigint,
    nonce: bigint,
    expirySeconds: number = 300
  ): Promise<SpendAuth> {
    const expiry = BigInt(Math.floor(Date.now() / 1000)) + BigInt(expirySeconds);

    const domain = {
      ...EIP712_DOMAIN,
      chainId: config.chainId,
      verifyingContract: config.shieldedCreditsAddress,
    };

    const types = {
      SpendAuthorization: [
        { name: "commitment", type: "bytes32" },
        { name: "serviceId", type: "uint64" },
        { name: "jobIndex", type: "uint8" },
        { name: "amount", type: "uint256" },
        { name: "operator", type: "address" },
        { name: "nonce", type: "uint256" },
        { name: "expiry", type: "uint64" },
      ],
    };

    const value = {
      commitment: config.commitment,
      serviceId: config.serviceId,
      jobIndex: 0, // inference job
      amount: amount,
      operator: config.operatorAddress,
      nonce: nonce,
      expiry: expiry,
    };

    const signature = await spendingWallet.signTypedData(domain, types, value);

    return {
      commitment: config.commitment,
      serviceId: config.serviceId,
      jobIndex: 0,
      amount,
      operator: config.operatorAddress,
      nonce,
      expiry,
      signature,
    };
  }

  /**
   * Estimate cost for a request based on estimated token counts.
   * This is a rough pre-authorization estimate; actual cost is determined post-inference.
   */
  function estimateCost(
    estimatedInputTokens: number,
    maxOutputTokens: number,
    pricePerInputToken: bigint,
    pricePerOutputToken: bigint
  ): bigint {
    return (
      BigInt(estimatedInputTokens) * pricePerInputToken +
      BigInt(maxOutputTokens) * pricePerOutputToken
    );
  }

  /**
   * Send a chat completion request with automatic SpendAuth signing.
   */
  async function chat(
    messages: ChatMessage[],
    options: ChatOptions & {
      /** Pre-authorized amount. If not set, a default estimate is used. */
      authorizedAmount?: bigint;
      /** Price per input token (from model config) */
      pricePerInputToken?: bigint;
      /** Price per output token (from model config) */
      pricePerOutputToken?: bigint;
    } = {}
  ): Promise<ChatCompletionResponse> {
    const maxTokens = options.maxTokens ?? 512;
    const priceIn = options.pricePerInputToken ?? 1n;
    const priceOut = options.pricePerOutputToken ?? 2n;

    // Rough estimate: 4 chars per token for input
    const estimatedInputTokens = messages.reduce(
      (sum, m) => sum + Math.ceil(m.content.length / 4),
      0
    );

    const amount =
      options.authorizedAmount ??
      estimateCost(estimatedInputTokens, maxTokens, priceIn, priceOut);

    const spendAuth = await signSpendAuth(amount, currentNonce);
    currentNonce++;

    const body = {
      messages,
      max_tokens: maxTokens,
      temperature: options.temperature ?? 0.7,
      stream: false,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      stop: options.stop,
      spend_auth: {
        commitment: spendAuth.commitment,
        service_id: Number(spendAuth.serviceId),
        job_index: spendAuth.jobIndex,
        amount: spendAuth.amount.toString(),
        operator: spendAuth.operator,
        nonce: Number(spendAuth.nonce),
        expiry: Number(spendAuth.expiry),
        signature: spendAuth.signature,
      },
    };

    const response = await fetch(`${config.operatorUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error: ErrorResponse = await response.json();
      throw new Error(
        `Inference request failed (${response.status}): ${error.error?.message ?? response.statusText}`
      );
    }

    return response.json();
  }

  /**
   * List available models from the operator.
   */
  async function listModels(): Promise<ModelList> {
    const response = await fetch(`${config.operatorUrl}/v1/models`);
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Check operator health.
   */
  async function healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${config.operatorUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Set the current nonce (e.g., after querying the credit account on-chain).
   */
  function setNonce(nonce: bigint) {
    currentNonce = nonce;
  }

  return {
    chat,
    listModels,
    healthCheck,
    signSpendAuth,
    estimateCost,
    setNonce,
    get address() {
      return spendingWallet.address;
    },
  };
}

export type InferenceClient = ReturnType<typeof createInferenceClient>;
