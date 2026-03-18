use serde::{Deserialize, Serialize};
use std::process::Command;

/// Information about a single GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mib: u32,
    pub memory_used_mib: u32,
    pub memory_free_mib: u32,
    pub temperature_c: Option<u32>,
    pub utilization_pct: Option<u32>,
    pub driver_version: String,
}

/// Detect available NVIDIA GPUs via nvidia-smi.
pub async fn detect_gpus() -> anyhow::Result<Vec<GpuInfo>> {
    // Run nvidia-smi in a blocking thread to avoid blocking the async runtime
    let output = tokio::task::spawn_blocking(|| {
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
    })
    .await??;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("nvidia-smi failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let fields: Vec<&str> = line.split(", ").collect();
        if fields.len() < 8 {
            continue;
        }

        gpus.push(GpuInfo {
            index: fields[0].trim().parse().unwrap_or(0),
            name: fields[1].trim().to_string(),
            memory_total_mib: fields[2].trim().parse().unwrap_or(0),
            memory_used_mib: fields[3].trim().parse().unwrap_or(0),
            memory_free_mib: fields[4].trim().parse().unwrap_or(0),
            temperature_c: fields[5].trim().parse().ok(),
            utilization_pct: fields[6].trim().parse().ok(),
            driver_version: fields[7].trim().to_string(),
        });
    }

    Ok(gpus)
}

/// Summary of GPU capabilities for registration.
#[derive(Debug, Serialize)]
pub struct GpuCapabilities {
    pub gpu_count: u32,
    pub total_vram_mib: u32,
    pub gpu_model: String,
}

/// Build a capabilities summary for blueprint registration.
pub async fn get_capabilities() -> anyhow::Result<GpuCapabilities> {
    let gpus = detect_gpus().await?;

    if gpus.is_empty() {
        anyhow::bail!("no GPUs detected");
    }

    Ok(GpuCapabilities {
        gpu_count: gpus.len() as u32,
        total_vram_mib: gpus.iter().map(|g| g.memory_total_mib).sum(),
        gpu_model: gpus[0].name.clone(),
    })
}
