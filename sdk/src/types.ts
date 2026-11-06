import type { Address, Hex } from "viem";
import type { createLocalSpendSigner } from "./spendAuth";

/** Chat message in OpenAI format */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Options for a chat completion request */
export interface ChatOptions {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string[];
}

/** Response from a non-streaming chat completion */
export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
}

export interface Choice {
  index: number;
  message: ChatMessage;
  finish_reason: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** Model listing (`/v1/models`) */
export interface ModelInfo {
  id: string;
  object: string;
  owned_by: string;
}

export interface ModelList {
  object: string;
  data: ModelInfo[];
}

/** Operator self-description (`/v1/operator`) */
export interface OperatorInfo {
  operator: Address;
  model: string;
  pricing: {
    price_per_input_token: number;
    price_per_output_token: number;
    currency: string;
  };
  gpu: {
    count: number;
    min_vram_mib: number;
    model: string | null;
    detected: unknown;
  };
  server: {
    max_concurrent_requests: number;
    max_context_length: number;
  };
  billing_required: boolean;
  payment_token: string | null;
}

/**
 * A signed ShieldedCredits spend authorization. Field order and types mirror
 * `ShieldedCredits.SPEND_TYPEHASH` exactly — changing any of this breaks
 * on-chain recovery and the operator's signature check.
 */
export interface SpendAuth {
  commitment: Hex;
  serviceId: bigint;
  jobIndex: number;
  amount: bigint;
  operator: Address;
  nonce: bigint;
  expiry: bigint;
  signature: Hex;
}

/**
 * Wire shape the operator parses (`SpendAuthPayload`, camelCase, numbers as
 * strings for bigint safety). This is what goes in the request body's
 * `spend_auth` field or the `X-Payment-Signature` header.
 */
export interface SpendAuthPayload {
  commitment: Hex;
  serviceId: number;
  jobIndex: number;
  amount: string;
  operator: Address;
  nonce: number;
  expiry: number;
  signature: Hex;
}

/** On-chain credit account state (`getAccount`). */
export interface CreditAccount {
  spendingKey: Address;
  token: Address;
  balance: bigint;
  totalFunded: bigint;
  totalSpent: bigint;
  nonce: bigint;
}

export type SpendAuthSigner = ReturnType<typeof createLocalSpendSigner>;

/** An unsigned EVM transaction request (funding flow). */
export interface TxRequest {
  to: Address;
  data: Hex;
  value: bigint;
}

/** Client configuration. */
export interface InferenceClientConfig {
  /** Operator HTTP endpoint URL */
  operatorUrl: string;
  /** ShieldedCredits contract address (EIP-712 verifyingContract) */
  shieldedCreditsAddress: Address;
  /** Chain ID for the EIP-712 domain */
  chainId: number;
  /** Credit account commitment */
  commitment: Hex;
  /** Service ID on Tangle */
  serviceId: bigint;
  /** Operator's on-chain address (SpendAuth designated recipient) */
  operatorAddress: Address;
  /** Signer for spend authorizations (ephemeral spending key) */
  signer: SpendAuthSigner;
  /** Default model (falls back to operator default if unset) */
  model?: string;
  /** Per-input-token price (wei) for cost estimation */
  pricePerInputToken?: bigint;
  /** Per-output-token price (wei) for cost estimation */
  pricePerOutputToken?: bigint;
  /** SpendAuth validity window in seconds (default 300) */
  expirySeconds?: number;
  /** Inject a fetch impl (tests / non-browser). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

/** Error response envelope from the operator. */
export interface ErrorResponse {
  error: {
    message: string;
    type: string;
    code: string;
  };
}
