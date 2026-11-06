// Exercise a supplied operator-lite artifact, or the existing SpendAuth SDK flow.
import { readFileSync, mkdirSync, writeFileSync, openSync, closeSync, statSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import check from 'node:assert/strict'
import { createServer } from 'node:net'

const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
async function stop(child) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return
  child.kill('SIGINT')
  await Promise.race([new Promise(resolve => child.once('exit', resolve)), pause(5000)])
  if (child.exitCode === null && child.signalCode === null) { child.kill('SIGKILL'); await Promise.race([new Promise(resolve => child.once('exit', resolve)), pause(2000)]) }
}

async function qualifyOperator() {
  const argv = process.argv.slice(2), split = argv.indexOf('--')
  const options = split < 0 ? argv : argv.slice(0, split)
  check.equal(options.length % 2, 0, 'Options require values')
  const args = Object.fromEntries(Array.from({ length: options.length / 2 }, (_, i) => [options[2 * i], options[2 * i + 1]]))
  for (const key of ['--operator-lite-bin', '--config', '--provenance', '--receipt']) check(args[key], `Missing ${key}`)
  const allowed = ['--operator-lite-bin', '--config', '--provenance', '--receipt', '--port', '--forward-host', '--forward-port']
  check(Object.keys(args).every(key => allowed.includes(key)), 'Unknown option')
  const binary = resolve(args['--operator-lite-bin']), config = resolve(args['--config'])
  const receipt = resolve(args['--receipt']), state = dirname(receipt), port = Number(args['--port'] ?? 9910)
  check.equal(statSync(config).mode & 0o077, 0, 'Operator config must be private')
  check(Number.isInteger(port) && port > 0 && port <= 65535, 'Invalid port')
  const provenance = JSON.parse(readFileSync(args['--provenance'], 'utf8'))
  const digest = createHash('sha256').update(readFileSync(binary)).digest('hex')
  check.equal(provenance.sha256, digest, 'Artifact digest must match gate provenance')
  check.match(provenance.source, /^[a-f0-9]{40}$/)
  check.match(provenance.core, /^[a-f0-9]{40}$/)
  let model
  try { model = JSON.parse(readFileSync(config, 'utf8')).vllm.model } catch { throw new Error('Invalid private operator configuration') }
  check(!existsSync(receipt), 'Use a new receipt path')
  mkdirSync(state, { recursive: true, mode: 0o700 })
  const log = openSync(`${receipt}.log`, 'wx', 0o600)
  const proof = { ...provenance, sha256: digest, endpoint: `http://127.0.0.1:${port}`, turns: [], status: 'running' }
  let candidate, forward, consumer
  const interrupted = new AbortController()
  const onSignal = () => interrupted.abort(new Error('Qualification interrupted'))
  process.once('SIGINT', onSignal); process.once('SIGTERM', onSignal)
  try {
    const env = Object.fromEntries(['PATH', 'HOME', 'LD_LIBRARY_PATH'].filter(k => process.env[k]).map(k => [k, process.env[k]]))
    Object.assign(env, { VLLM_OP_SERVER__HOST: '127.0.0.1', VLLM_OP_SERVER__PORT: String(port), VLLM_OP_BILLING__NONCE_STORE_PATH: `${receipt}.nonces`, RUST_LOG: 'info' })
    const reservation = createServer()
    await new Promise((resolve, reject) => { reservation.once('error', reject); reservation.listen(port, '127.0.0.1', resolve) })
    await new Promise(resolve => reservation.close(resolve))
    candidate = spawn(binary, [config], { env, stdio: ['ignore', log, log] })
    proof.pid = candidate.pid
    candidate.once('error', onSignal)
    const ready = Date.now() + 30_000
    while (true) {
      interrupted.signal.throwIfAborted()
      check(candidate.exitCode === null && candidate.signalCode === null, 'Candidate exited; private log retained')
      try { if ((await fetch(`${proof.endpoint}/health`, { signal: AbortSignal.timeout(2000) })).ok) break } catch {}
      check(Date.now() < ready, 'Candidate readiness deadline')
      await pause(250)
    }
    for (let turn = 0; turn < 2; turn++) {
      check(candidate.exitCode === null && candidate.signalCode === null, 'Candidate stopped')
      const response = await fetch(`${proof.endpoint}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.any([interrupted.signal, AbortSignal.timeout(120_000)]), body: JSON.stringify({ model, messages: [{ role: 'user', content: 'Reply exactly ROUTER_OK.' }], max_tokens: 128, temperature: 0, stream: false }) })
      check.equal(response.status, 200)
      const value = await response.json()
      check.equal(value.choices[0].message.content, 'ROUTER_OK')
      const cached = value.usage.prompt_tokens_details.cached_tokens
      check(Number.isSafeInteger(cached) && cached >= 0 && cached <= value.usage.prompt_tokens)
      proof.turns.push({ content: 'ROUTER_OK', usage: value.usage })
    }
    check(candidate.exitCode === null && candidate.signalCode === null, 'Candidate stopped during requests')
    check(proof.turns.at(-1).usage.prompt_tokens_details.cached_tokens > 0, 'Warm request must preserve cached usage')
    if (split >= 0) {
      check(args['--forward-host'] && args['--forward-port'], 'Router proof requires the WSL SSH endpoint and port')
      const remotePort = Number(args['--forward-port'])
      check(Number.isInteger(remotePort) && remotePort > 0 && remotePort <= 65535)
      forward = spawn('ssh', ['-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-R', `127.0.0.1:${remotePort}:127.0.0.1:${port}`, '-N', args['--forward-host']], { stdio: ['ignore', log, log] })
      forward.once('error', onSignal)
      await pause(1500)
      check(forward.exitCode === null && forward.signalCode === null, 'SSH forward failed')
      const command = argv.slice(split + 1)
      check(command.length > 0, 'Missing Router proof command')
      proof.router = { command, forwardHost: args['--forward-host'], forwardPort: remotePort }
      consumer = spawn(command[0], command.slice(1), { stdio: 'inherit', signal: interrupted.signal })
      const timer = setTimeout(() => interrupted.abort(new Error('Router qualification deadline')), 1_800_000)
      try { proof.router_exit = await new Promise((resolve, reject) => { consumer.once('error', reject); consumer.once('exit', resolve) }) } finally { clearTimeout(timer) }
      check.equal(proof.router_exit, 0, 'Router billing proof failed')
      check(candidate.exitCode === null && candidate.signalCode === null, 'Candidate stopped during Router proof')
      check(forward.exitCode === null && forward.signalCode === null, 'SSH forward stopped during Router proof')
    }
    proof.status = 'passed'
  } catch (error) {
    proof.status = 'failed'; proof.error = error.message
    throw error
  } finally {
    await stop(consumer); await stop(forward); await stop(candidate)
    process.removeListener('SIGINT', onSignal); process.removeListener('SIGTERM', onSignal)
    closeSync(log)
    proof.cleanup = { candidateStopped: !candidate?.pid || candidate.exitCode !== null || candidate.signalCode !== null, forwardStopped: !forward?.pid || forward.exitCode !== null || forward.signalCode !== null }
    if (!Object.values(proof.cleanup).every(Boolean)) { proof.status = 'failed'; proof.error = 'Qualification process cleanup failed' }
    writeFileSync(receipt, JSON.stringify(proof, null, 2) + '\n', { flag: 'wx', mode: 0o600 })
    check(Object.values(proof.cleanup).every(Boolean), proof.error)
    if (proof.status === 'passed') console.log(JSON.stringify(proof))
  }
}

async function billedSdkFlow() {
  const { createPublicClient, http, getAddress } = await import('viem')
  const { createInferenceClient, createLocalSpendSigner } = await import('../dist/index.js')
const here = dirname(fileURLToPath(import.meta.url))
const env = Object.fromEntries(
  readFileSync(resolve(here, '../../.env.local'), 'utf8')
    .split('\n')
    .map((l) => l.match(/^([A-Z0-9_]+)=(.*)$/))
    .filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
)

const OPERATOR_API = process.env.OPERATOR_API ?? 'http://127.0.0.1:9100'
const CHAIN_ID = Number(env.CHAIN_ID ?? '31337')

let failed = 0
const assert = (cond, desc, detail = '') =>
  cond
    ? console.log(`  PASS ${desc}`)
    : (failed++, console.error(`  FAIL ${desc} ${detail}`))

const chain = {
  id: CHAIN_ID,
  name: 'anvil',
  nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: [env.RPC_URL] } },
}
const publicClient = createPublicClient({ chain, transport: http(env.RPC_URL) })

const client = createInferenceClient({
  operatorUrl: OPERATOR_API,
  shieldedCreditsAddress: getAddress(env.SHIELDED_CREDITS),
  chainId: CHAIN_ID,
  commitment: env.COMMITMENT,
  serviceId: 1n,
  operatorAddress: getAddress(env.OPERATOR_ADDR),
  signer: createLocalSpendSigner(env.USER_KEY),
  model: 'claude-code/sonnet',
  pricePerInputToken: 1n,
  pricePerOutputToken: 2n,
})

async function main() {
  console.log(`[http-e2e] operator=${OPERATOR_API} chain=${CHAIN_ID}`)

  const nonce = await client.syncNonce(publicClient)
  console.log(`[http-e2e] synced spend nonce=${nonce}`)

  console.log('[http-e2e] sending SpendAuth-gated chat (real backend, may take a while)…')
  const res = await client.chat(
    [{ role: 'user', content: 'Reply with exactly one word: hello' }],
    { maxTokens: 64 },
  )

  const content = res?.choices?.[0]?.message?.content ?? ''
  console.log(`[http-e2e] model=${res?.model} content=${JSON.stringify(content)}`)
  console.log(`[http-e2e] usage=${JSON.stringify(res?.usage)}`)

  assert(typeof content === 'string' && content.length > 0, 'completion has content')
  assert((res?.usage?.total_tokens ?? 0) > 0, 'usage reports tokens', `usage=${JSON.stringify(res?.usage)}`)

  // The operator consumed the SpendAuth → on-chain nonce advanced.
  const after = await client.syncNonce(publicClient)
  assert(after === nonce + 1n, 'on-chain spend nonce advanced after the request', `before=${nonce} after=${after}`)

  console.log(failed === 0 ? '\n[http-e2e] ALL PASS' : `\n[http-e2e] ${failed} FAILED`)
  process.exit(failed === 0 ? 0 : 1)
}

  await main()
}

(process.argv.includes('--operator-lite-bin') ? qualifyOperator() : billedSdkFlow()).catch(error => { console.error(error.message); process.exitCode = 1 })
