use anyhow::{Context, Result};
use serde::Serialize;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub(super) struct RdmaBenchRun {
    payload_bytes: usize,
    command: Vec<String>,
    ok: bool,
    elapsed_millis: u128,
    output: String,
}

pub(super) fn run_ib_send_bw_server(
    payload_bytes: usize,
    port: u16,
    duration_secs: u64,
) -> Result<RdmaBenchRun> {
    run_ib_send_bw(
        payload_bytes,
        vec![
            "--report_gbits".to_owned(),
            "-s".to_owned(),
            payload_bytes.to_string(),
            "-p".to_owned(),
            port.to_string(),
            "-D".to_owned(),
            duration_secs.to_string(),
        ],
    )
}

pub(super) fn run_ib_send_bw_client(
    payload_bytes: usize,
    port: u16,
    duration_secs: u64,
    peer: &str,
) -> Result<RdmaBenchRun> {
    run_ib_send_bw(
        payload_bytes,
        vec![
            "--report_gbits".to_owned(),
            "-s".to_owned(),
            payload_bytes.to_string(),
            "-p".to_owned(),
            port.to_string(),
            "-D".to_owned(),
            duration_secs.to_string(),
            peer.to_owned(),
        ],
    )
}

fn run_ib_send_bw(payload_bytes: usize, args: Vec<String>) -> Result<RdmaBenchRun> {
    let timeout_secs = args
        .windows(2)
        .find(|window| window[0] == "-D")
        .and_then(|window| window[1].parse::<u64>().ok())
        .unwrap_or(2)
        + 20;
    let mut command = vec![
        "timeout".to_owned(),
        timeout_secs.to_string(),
        "ib_send_bw".to_owned(),
    ];
    command.extend(args);
    let started = Instant::now();
    let output = Command::new(&command[0])
        .args(&command[1..])
        .output()
        .with_context(|| format!("running {}", command.join(" ")))?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(RdmaBenchRun {
        payload_bytes,
        command,
        ok: output.status.success(),
        elapsed_millis: started.elapsed().as_millis(),
        output: text.trim().to_owned(),
    })
}
