#!/usr/bin/env node
// sign-spend-auth.mjs — Sign an EIP-712 SpendAuthorization for ShieldedCredits.
//
// Usage:
//   node scripts/sign-spend-auth.mjs \
//     --private-key 0x... \
//     --verifying-contract 0x... \
//     --chain-id 31337 \
//     --commitment 0x... \
//     --service-id 1 \
//     --job-index 0 \
//     --amount 1000000000000000000 \
//     --operator 0x... \
//     --nonce 0 \
//     --expiry 9999999999
//
// Outputs JSON: { "signature": "0x...", "digest": "0x..." }
//
// Uses only Node.js built-ins + ethers (no extra deps). If ethers is not
// installed, falls back to viem. If neither is available, uses manual signing
// via cast (exec fallback).

import { execSync } from "child_process";

// Parse CLI args
const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, "").replace(/-/g, "_");
  args[key] = process.argv[i + 1];
}

const required = [
  "private_key",
  "verifying_contract",
  "chain_id",
  "commitment",
  "service_id",
  "job_index",
  "amount",
  "operator",
  "nonce",
  "expiry",
];
for (const k of required) {
  if (!args[k]) {
    console.error(`Missing --${k.replace(/_/g, "-")}`);
    process.exit(1);
  }
}

// EIP-712 domain and types
const domain = {
  name: "ShieldedCredits",
  version: "1",
  chainId: parseInt(args.chain_id),
  verifyingContract: args.verifying_contract,
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
  commitment: args.commitment,
  serviceId: parseInt(args.service_id),
  jobIndex: parseInt(args.job_index),
  amount: args.amount,
  operator: args.operator,
  nonce: parseInt(args.nonce),
  expiry: parseInt(args.expiry),
};

// Compute EIP-712 digest and sign manually using cast
// This avoids requiring ethers/viem as a dependency.

function keccak256(hex) {
  return execSync(`cast keccak "${hex}"`, { encoding: "utf-8" }).trim();
}

function abiEncode(types_str, ...values) {
  const args_str = values.map((v) => `"${v}"`).join(" ");
  return execSync(`cast abi-encode "f(${types_str})" ${args_str}`, {
    encoding: "utf-8",
  }).trim();
}

// SPEND_TYPEHASH = keccak256("SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)")
const SPEND_TYPEHASH = keccak256(
  "0x" +
    Buffer.from(
      "SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)"
    ).toString("hex")
);

// Domain separator
const DOMAIN_TYPEHASH = keccak256(
  "0x" +
    Buffer.from(
      "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    ).toString("hex")
);

const nameHash = keccak256(
  "0x" + Buffer.from("ShieldedCredits").toString("hex")
);
const versionHash = keccak256("0x" + Buffer.from("1").toString("hex"));

const domainEncoded = abiEncode(
  "bytes32,bytes32,bytes32,uint256,address",
  DOMAIN_TYPEHASH,
  nameHash,
  versionHash,
  domain.chainId.toString(),
  domain.verifyingContract
);
const DOMAIN_SEPARATOR = keccak256(domainEncoded);

// Struct hash
const structEncoded = abiEncode(
  "bytes32,bytes32,uint64,uint8,uint256,address,uint256,uint64",
  SPEND_TYPEHASH,
  value.commitment,
  value.serviceId.toString(),
  value.jobIndex.toString(),
  value.amount,
  value.operator,
  value.nonce.toString(),
  value.expiry.toString()
);
const structHash = keccak256(structEncoded);

// EIP-712 digest = keccak256("\x19\x01" || domainSeparator || structHash)
// "\x19\x01" = 0x1901
const digestInput =
  "0x1901" +
  DOMAIN_SEPARATOR.slice(2) +
  structHash.slice(2);
const digest = keccak256(digestInput);

// Sign the digest with cast
const signature = execSync(
  `cast wallet sign --private-key ${args.private_key} --no-hash ${digest}`,
  { encoding: "utf-8" }
).trim();

console.log(JSON.stringify({ signature, digest, domainSeparator: DOMAIN_SEPARATOR, structHash }));
