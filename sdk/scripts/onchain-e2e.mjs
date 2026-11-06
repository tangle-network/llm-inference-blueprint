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
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import {
  createPublicClient,
  createWalletClient,
  http,
  getAddress,
  parseAbi,
} from 'viem'
import { privateKeyToAccount } from 'viem/accounts'

const args = process.argv.slice(2)
if (args.length && (args[0] !== '--http-peer' || args.length > 2)) throw new Error('Expected --http-peer [SDK module] or no arguments')
const {
  createLocalSpendSigner,
  createInferenceClient,
  generateCreditIdentity,
  signSpendAuth,
  readCreditAccount,
  SHIELDED_CREDITS_ABI,
} = await import(args[1] ? pathToFileURL(resolve(args[1])).href : new URL('../dist/index.js', import.meta.url).href)

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

async function main() {
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


async function httpPeer(context) {
  const { createServer } = await import('node:http')
  const { buildSpendAuthMessage } = await import('@tangle-network/tcloud/shielded')
  const { recoverTypedDataAddress, getAddress } = await import('viem')
  let requestIndex = 0
  const server = createServer(async (request, response) => {
    let bytes = ''
    for await (const chunk of request) bytes += chunk
    let payload, verified = false
    try {
      const body = JSON.parse(bytes)
      const { signature, ...fields } = body.spend_auth
      payload = fields
      const recovered = await recoverTypedDataAddress({
        ...buildSpendAuthMessage({ ...fields, serviceId: BigInt(fields.serviceId),
          jobIndex: Number(fields.jobIndex), amount: BigInt(fields.amount),
          nonce: BigInt(fields.nonce), expiry: BigInt(fields.expiry) },
          context.chainId, context.credits), signature,
      })
      const expected = context.cases[requestIndex]
      verified = getAddress(recovered) === getAddress(context.account)
        && String(fields.serviceId) === expected.serviceId
        && String(fields.nonce) === expected.nonce && fields.amount === '17'
        && request.method === 'POST' && request.url === '/v1/chat/completions'
        && body.model === 'fixture' && body.max_tokens === 1 && body.stream === false
        && body.messages[0].role === 'user' && body.messages[0].content === 'Hello'
    } catch {}
    console.log(JSON.stringify({ type: 'authorization', index: requestIndex++, payload, verified }))
    response.writeHead(verified ? 200 : 422, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(verified ? {
      id: 'sdk-http-peer', object: 'chat.completion', created: 1, model: 'fixture',
      choices: [{ index: 0, message: { role: 'assistant', content: 'accepted' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    } : { error: { message: 'Authorization did not verify', type: 'invalid_request_error' } }))
  })
  server.listen(0, '127.0.0.1', () => console.log(JSON.stringify({ type: 'ready', port: server.address().port })))
}

async function proveHttpPeer() {
  const identity = generateCreditIdentity()
  const signer = createLocalSpendSigner(identity.spendingKeyPrivate)
  const context = { account: signer.account, chainId: 31337,
    credits: '0x0000000000000000000000000000000000000001',
    cases: [{ serviceId: '7', nonce: '3' },
      { serviceId: '9007199254740993', nonce: '9007199254740993' }] }
  const child = spawn(process.execPath, ['--input-type=module', '-'], {
    cwd: resolve(here, '..'), env: {}, stdio: ['pipe', 'pipe', 'inherit'],
  })
  const captured = []
  let resolveCaptured
  const allCaptured = new Promise(resolveCapture => { resolveCaptured = resolveCapture })
  const lines = createInterface({ input: child.stdout })
  let timer
  const ready = new Promise((resolvePort, reject) => {
    timer = setTimeout(() => reject(new Error('HTTP peer startup timed out')), 5000)
    child.once('error', reject)
    child.once('exit', () => reject(new Error('HTTP peer exited before startup')))
    lines.on('line', line => {
      const message = JSON.parse(line)
      if (message.type === 'ready') { clearTimeout(timer); resolvePort(message.port) }
      else { captured.push(message); if (captured.length === context.cases.length) resolveCaptured() }
    })
  })
  child.stdin.end(`(${httpPeer.toString()})(${JSON.stringify(context)}).catch(() => process.exit(1))`)
  try {
    const port = await ready
    for (const example of context.cases) {
      const client = createInferenceClient({ operatorUrl: `http://127.0.0.1:${port}`,
        shieldedCreditsAddress: context.credits, chainId: context.chainId,
        commitment: identity.commitment, serviceId: BigInt(example.serviceId),
        operatorAddress: '0x0000000000000000000000000000000000000002', signer,
        fetchImpl: (url, init) => fetch(url, { ...init, signal: AbortSignal.timeout(5000) }) })
      client.setNonce(BigInt(example.nonce))
      const response = await client.chat([{ role: 'user', content: 'Hello' }],
        { model: 'fixture', maxTokens: 1, authorizedAmount: 17n })
      if (response.id !== 'sdk-http-peer') throw new Error('Unexpected HTTP peer response')
    }
    await Promise.race([allCaptured, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("Authorization capture timed out")), 5000)
    })])
    if (captured.length !== context.cases.length || captured.some(item => !item.verified)) {
      throw new Error('SDK authorization capture was incomplete')
    }
  } finally {
    clearTimeout(timer)
    const exited = new Promise(resolveExit => child.once('exit', resolveExit))
    child.kill('SIGTERM')
    await Promise.race([exited, new Promise(resolveDelay => setTimeout(resolveDelay, 2000))])
    if (child.exitCode === null && child.signalCode === null) { child.kill('SIGKILL'); await exited }
    lines.close()
    console.log('SDK_HTTP_PEER_RECEIPT=' + JSON.stringify({ cases: captured, chain_exercised: false,
      cleanup: child.exitCode !== null || child.signalCode !== null }))
  }
}


;(args.length ? proveHttpPeer() : main()).catch((err) => {
  console.error('[onchain-e2e] error:', err)
  process.exit(1)
})
