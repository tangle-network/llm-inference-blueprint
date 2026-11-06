import { signSpendAuth as signTcloudSpendAuth } from "@tangle-network/tcloud/shielded";
import { privateKeyToAccount } from "viem/accounts";
import type { Address, Hex } from "viem";
import type { SpendAuth, SpendAuthPayload } from "./types";

type Authorization = {
  chainId: number;
  shieldedCreditsAddress: Address;
  commitment: Hex;
  serviceId: bigint;
  operator: Address;
  amount: bigint;
  nonce: bigint;
  jobIndex?: number;
  expiry: bigint;
};

class LocalSpendSigner {
  readonly account: Address;
  #privateKey: Hex;

  constructor(privateKey: Hex) {
    this.#privateKey = privateKey;
    this.account = privateKeyToAccount(privateKey).address;
  }

  async sign(params: Authorization): Promise<SpendAuth> {
    const jobIndex = params.jobIndex ?? 0;
    const { signature } = await signTcloudSpendAuth(
      { privateKey: this.#privateKey, commitment: params.commitment },
      { chainId: params.chainId, creditsAddress: params.shieldedCreditsAddress,
        serviceId: params.serviceId, jobIndex, operator: params.operator,
        amount: params.amount, nonce: params.nonce, expiry: params.expiry },
    );
    return { commitment: params.commitment, serviceId: params.serviceId, jobIndex,
      operator: params.operator, amount: params.amount, nonce: params.nonce,
      expiry: params.expiry, signature };
  }
}

/** Keep the ephemeral key inside the factory's opaque handle. */
export function createLocalSpendSigner(spendingKeyPrivate: Hex) {
  return new LocalSpendSigner(spendingKeyPrivate);
}

export function signSpendAuth(params: Authorization & { signer: LocalSpendSigner }): Promise<SpendAuth> {
  return params.signer.sign(params);
}

export function toSpendAuthPayload(auth: SpendAuth): SpendAuthPayload {
  return {
    commitment: auth.commitment, serviceId: auth.serviceId.toString(), jobIndex: auth.jobIndex,
    amount: auth.amount.toString(), operator: auth.operator, nonce: auth.nonce.toString(),
    expiry: auth.expiry.toString(), signature: auth.signature,
  };
}
