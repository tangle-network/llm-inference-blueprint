use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=src");

    let blueprint_metadata = serde_json::json!({
        "name": "llm-inference",
        "description": "vLLM-backed LLM inference operator with shielded billing",
        "version": env!("CARGO_PKG_VERSION"),
        "manager": {
            "Evm": "InferenceBSM"
        },
        "master_revision": "Latest",
        "jobs": [
            {
                "name": "inference",
                "job_index": 0,
                "description": "Run LLM inference via vLLM backend (prompt → completion)",
                "inputs": ["(string,uint32,uint64)"],
                "outputs": ["(string,uint32,uint32)"],
                "required_results": 1,
                "execution": "local"
            }
        ]
    });

    let json = serde_json::to_string_pretty(&blueprint_metadata).unwrap();

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .expect("Failed to find workspace root");

    std::fs::write(workspace_root.join("blueprint.json"), json.as_bytes()).unwrap();
}
