/** Chat message in OpenAI format */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Options for a chat completion request */
export interface ChatOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string[];
}

/** Response from chat completion */
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

/** Model listing */
export interface ModelInfo {
  id: string;
  object: string;
  owned_by: string;
}

export interface ModelList {
  object: string;
  data: ModelInfo[];
}

/** Operator info from the BSM contract */
export interface OperatorInfo {
  address: string;
  model: string;
  gpuCount: number;
  totalVramMib: number;
  gpuModel: string;
  endpoint: string;
  active: boolean;
}

/** Model pricing config from the BSM contract */
export interface ModelConfig {
  maxContextLen: number;
  pricePerInputToken: bigint;
  pricePerOutputToken: bigint;
  minGpuVramMib: number;
  enabled: boolean;
}

/** ShieldedCredits spend authorization */
export interface SpendAuth {
  commitment: string;
  serviceId: bigint;
  jobIndex: number;
  amount: bigint;
  operator: string;
  nonce: bigint;
  expiry: bigint;
  signature: string;
}

/** Credit account state */
export interface CreditAccount {
  spendingKey: string;
  token: string;
  balance: bigint;
  totalFunded: bigint;
  totalSpent: bigint;
  nonce: bigint;
}

/** Client configuration */
export interface InferenceClientConfig {
  /** Operator HTTP endpoint URL */
  operatorUrl: string;

  /** ShieldedCredits contract address */
  shieldedCreditsAddress: string;

  /** Chain ID for EIP-712 domain */
  chainId: number;

  /** Credit account commitment (keccak256(spendingKey, salt)) */
  commitment: string;

  /** Service ID on Tangle */
  serviceId: bigint;

  /** Operator's on-chain address (for SpendAuth designation) */
  operatorAddress: string;

  /** Spending key private key (ephemeral, for signing SpendAuths) */
  spendingKeyPrivate: string;
}

/** Error response from operator */
export interface ErrorResponse {
  error: {
    message: string;
    type: string;
    code: string;
  };
}
