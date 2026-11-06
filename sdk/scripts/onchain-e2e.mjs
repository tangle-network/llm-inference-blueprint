// On-chain end-to-end proof for the viem SDK against a live ShieldedCredits
// deployment prepared through its maintained deploy path. It signs a SpendAuthorization
// with the SDK and submits `authorizeSpend` on-chain — so the contract's own
// ECDSA.recover must accept the SDK's EIP-712 signature, not just our unit
// reconstruction. Reads ../../.env.local for addresses + keys.
//
//   node sdk/scripts/onchain-e2e.mjs
//
// Exits non-zero on any assertion failure.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import {
  createPublicClient,
  createWalletClient,
  http,
  getAddress,
  parseAbi,
} from 'viem'
import { privateKeyToAccount } from 'viem/accounts'

import {
  createLocalSpendSigner,
  signSpendAuth,
  readCreditAccount,
  SHIELDED_CREDITS_ABI,
} from '../dist/index.js'

const here = dirname(fileURLToPath(import.meta.url))
const envPath = resolve(here, '../../.env.local')

function loadEnv(path) {
  const env = {}
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const m = line.match(/^([A-Z0-9_]+)=(.*)$/)
    if (m) env[m[1]] = m[2].trim()
  }
  return env
}

let failed = 0
function assert(cond, desc, detail = '') {
  if (cond) console.log(`  PASS ${desc}`)
  else {
    failed++
    console.error(`  FAIL ${desc} ${detail}`)
  }
}

const e = loadEnv(envPath)
const RPC_URL = e.RPC_URL
const CHAIN_ID = Number(e.CHAIN_ID ?? '31337')
const SHIELDED = getAddress(e.SHIELDED_CREDITS)
const COMMITMENT = e.COMMITMENT
const OPERATOR = getAddress(e.OPERATOR_ADDR)
const USER_KEY = e.USER_KEY // private key whose address == account spendingKey
const DEPLOYER_KEY = e.DEPLOYER_KEY // pays gas to submit authorizeSpend

const chain = {
  id: CHAIN_ID,
  name: 'anvil',
  nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: [RPC_URL] } },
}

const publicClient = createPublicClient({ chain, transport: http(RPC_URL) })
const submitter = createWalletClient({
  account: privateKeyToAccount(DEPLOYER_KEY),
  chain,
  transport: http(RPC_URL),
})

const AUTHORIZE_ABI = parseAbi([
  'function authorizeSpend((bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry,bytes signature) auth) returns (bytes32 authHash)',
])

async function main() {
  console.log(`[onchain-e2e] ShieldedCredits=${SHIELDED} chain=${CHAIN_ID}`)

  // 1. The funded account's on-chain state (spendingKey + nonce).
  const before = await readCreditAccount(publicClient, SHIELDED, COMMITMENT)
  assert(before.balance > 0n, 'credit account is funded', `balance=${before.balance}`)

  // 2. Sign a SpendAuth with the SDK using the ephemeral spending key.
  const signer = createLocalSpendSigner(USER_KEY)
  assert(
    getAddress(signer.account) === getAddress(before.spendingKey),
    'SDK signer address == on-chain spendingKey',
    `signer=${signer.account} onchain=${before.spendingKey}`,
  )

  const amount = 100_000_000_000_000_000n // 0.1 token
  const expiry = BigInt(Math.floor(Date.now() / 1000) + 3600)
  const auth = await signSpendAuth({
    signer,
    chainId: CHAIN_ID,
    shieldedCreditsAddress: SHIELDED,
    commitment: COMMITMENT,
    serviceId: 1n,
    operator: OPERATOR,
    amount,
    nonce: before.nonce,
    expiry,
  })

  // 3. Submit authorizeSpend on-chain — the contract recovers the signer from
  //    the SDK's EIP-712 signature and must match spendingKey, else it reverts.
  const hash = await submitter.writeContract({
    address: SHIELDED,
    abi: AUTHORIZE_ABI,
    functionName: 'authorizeSpend',
    args: [
      {
        commitment: auth.commitment,
        serviceId: auth.serviceId,
        jobIndex: auth.jobIndex,
        amount: auth.amount,
        operator: auth.operator,
        nonce: auth.nonce,
        expiry: auth.expiry,
        signature: auth.signature,
      },
    ],
  })
  const receipt = await publicClient.waitForTransactionReceipt({ hash })
  assert(
    receipt.status === 'success',
    'authorizeSpend accepted the SDK signature on-chain',
    `status=${receipt.status}`,
  )

  // 4. Balance decreased by amount, nonce incremented — the contract processed it.
  const after = await readCreditAccount(publicClient, SHIELDED, COMMITMENT)
  assert(
    after.balance === before.balance - amount,
    'credit balance decreased by the authorized amount',
    `before=${before.balance} after=${after.balance}`,
  )
  assert(
    after.nonce === before.nonce + 1n,
    'spend nonce incremented',
    `before=${before.nonce} after=${after.nonce}`,
  )

  console.log(failed === 0 ? '\n[onchain-e2e] ALL PASS' : `\n[onchain-e2e] ${failed} FAILED`)
  process.exit(failed === 0 ? 0 : 1)
}

main().catch((err) => {
  console.error('[onchain-e2e] error:', err)
  process.exit(1)
})
