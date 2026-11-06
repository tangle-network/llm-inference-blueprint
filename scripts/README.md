# Integration checks

The pinned core dependency owns protocol fixtures. Resolve it and run its maintained chain checks from the repository root:

```bash
CORE_DIR="$(cargo metadata --locked --format-version 1 | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
bash "$CORE_DIR/scripts/test-full-protocol.sh"
```

The existing [SDK HTTP and chain checks](../sdk/scripts/) require a predeployed RPC, deployed billing contracts and private `.env.local` configuration supplied from the approved deployment. The HTTP script's `--operator-lite-bin` mode takes a qualified binary, its provenance and a private operator configuration to exercise real backend output and cached usage. Retain nonce stores across operator upgrades.
