import { sendTransaction } from "@wagmi/core";
import type { Config } from "@wagmi/core";
import type { Hex } from "viem";

import type { TxRequest } from "./types";

/**
 * Send the funding transactions (ERC-20 approve + fundCredits) through the
 * connected wagmi wallet, in order. Returns the transaction hashes.
 */
export async function sendFundingTxs(
  config: Config,
  txs: TxRequest[],
): Promise<Hex[]> {
  const hashes: Hex[] = [];
  for (const tx of txs) {
    const hash = await sendTransaction(config, {
      to: tx.to,
      data: tx.data,
      value: tx.value,
    });
    hashes.push(hash);
  }
  return hashes;
}
