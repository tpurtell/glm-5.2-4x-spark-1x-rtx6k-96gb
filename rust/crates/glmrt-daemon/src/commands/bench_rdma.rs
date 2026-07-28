use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

use crate::cli::BenchRdmaArgs;
use crate::{command_probe, Probe};

use app::{
    glmrt_verbs_app_benchmark_attempt, run_glmrt_verbs_app_benchmark, GlmrtVerbsAppBenchmarkAttempt,
};
use vendor::{run_ib_send_bw_client, run_ib_send_bw_server, RdmaBenchRun};

mod app;
mod vendor;

const VENDOR_PERFTEST_BENCHMARK_KIND: &str = "vendor-perftest-ib_send_bw";
const GLMRT_VERBS_APP_BENCHMARK_KIND: &str = "glmrt-verbs-app-protocol-v2";
const DEFAULT_VENDOR_PAYLOAD_BYTES: &str = "4096,8192,12288,16384,32768,65536";
const DEFAULT_APP_PROTOCOL_V2_PAYLOAD_BYTES: &str =
    "12288,24576,49152,98304,196608,786432,3145728,6291456";

#[derive(Debug, Serialize)]
struct RdmaBenchReport {
    benchmark_kind: String,
    mode: String,
    resolved_mode: String,
    peer: Option<String>,
    hostname: String,
    port: u16,
    duration_secs: u64,
    payload_bytes: Vec<usize>,
    dev_infiniband_present: bool,
    ibv_devices: Probe,
    rdma_link: Probe,
    ib_send_bw: Probe,
    skipped: bool,
    skip_reason: Option<String>,
    runs: Vec<RdmaBenchRun>,
    #[serde(skip_serializing_if = "Option::is_none")]
    glmrt_app_benchmark: Option<GlmrtVerbsAppBenchmarkAttempt>,
}

pub(crate) fn run_bench_rdma(args: BenchRdmaArgs) -> Result<()> {
    let hostname = command_probe("hostname", &["-s"]).output;
    let resolved_mode = resolve_rdma_mode(&args.mode, args.peer.as_deref(), &hostname)?;
    let app_mode = is_glmrt_app_mode(&resolved_mode);
    let payload_spec = if app_mode && args.payload_bytes == DEFAULT_VENDOR_PAYLOAD_BYTES {
        DEFAULT_APP_PROTOCOL_V2_PAYLOAD_BYTES
    } else {
        &args.payload_bytes
    };
    let payload_bytes = parse_payload_bytes(payload_spec)?;
    let ibv_devices = command_probe("ibv_devices", &[]);
    let rdma_link = command_probe("rdma", &["link"]);
    let ib_send_bw = command_probe("ib_send_bw", &["--version"]);
    let dev_infiniband_present = Path::new("/dev/infiniband").is_dir();
    let benchmark_kind = if app_mode {
        GLMRT_VERBS_APP_BENCHMARK_KIND
    } else {
        VENDOR_PERFTEST_BENCHMARK_KIND
    };
    let mut report = RdmaBenchReport {
        benchmark_kind: benchmark_kind.to_owned(),
        mode: args.mode.clone(),
        resolved_mode: resolved_mode.clone(),
        peer: args.peer.clone(),
        hostname,
        port: args.port,
        duration_secs: args.duration_secs,
        payload_bytes,
        dev_infiniband_present,
        ibv_devices,
        rdma_link,
        ib_send_bw,
        skipped: false,
        skip_reason: None,
        runs: Vec::new(),
        glmrt_app_benchmark: None,
    };

    if app_mode {
        let mut attempt = glmrt_verbs_app_benchmark_attempt(
            &resolved_mode,
            args.peer.clone(),
            &report.payload_bytes,
        )?;
        if resolved_mode == "app-capability" {
            report.skipped = true;
            report.skip_reason = Some("GLMRT verbs app capability mode only".to_owned());
        } else if !attempt.preflight_ok {
            report.skipped = true;
            report.skip_reason = Some(match attempt.preflight_error.as_deref() {
                Some(err) => format!("GLMRT verbs app benchmark preflight failed: {err}"),
                None => "GLMRT verbs app benchmark preflight failed".to_owned(),
            });
        } else {
            match run_glmrt_verbs_app_benchmark(
                &resolved_mode,
                args.peer.as_deref(),
                args.port,
                &attempt.payloads,
                args.duration_secs,
            ) {
                Ok(runs) => {
                    attempt.app_transport_runs = runs;
                    report.skipped = false;
                    report.skip_reason = None;
                }
                Err(err) => {
                    let err = err.to_string();
                    attempt.app_transport_error = Some(err.clone());
                    report.skipped = true;
                    report.skip_reason = Some(format!("GLMRT verbs app benchmark failed: {err}"));
                }
            }
        }
        report.glmrt_app_benchmark = Some(attempt);
    } else if !report.dev_infiniband_present {
        report.skipped = true;
        report.skip_reason = Some("/dev/infiniband is absent".to_owned());
    } else if !report.ibv_devices.ok {
        report.skipped = true;
        report.skip_reason = Some("ibv_devices failed".to_owned());
    } else if !report.ib_send_bw.ok {
        report.skipped = true;
        report.skip_reason = Some("ib_send_bw is unavailable".to_owned());
    } else if resolved_mode == "capability" {
        report.skipped = true;
        report.skip_reason = Some("capability mode only".to_owned());
    }

    if !app_mode && !report.skipped {
        match resolved_mode.as_str() {
            "server" => {
                for payload in &report.payload_bytes {
                    report.runs.push(run_ib_send_bw_server(
                        *payload,
                        args.port,
                        args.duration_secs,
                    )?);
                }
            }
            "client" => {
                let peer = args
                    .peer
                    .as_deref()
                    .context("client RDMA benchmark requires --peer")?;
                for payload in &report.payload_bytes {
                    report.runs.push(run_ib_send_bw_client(
                        *payload,
                        args.port,
                        args.duration_secs,
                        peer,
                    )?);
                }
            }
            other => anyhow::bail!("unsupported resolved RDMA mode: {other}"),
        }
    }

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_payload_bytes(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid payload byte size: {part}"))
        })
        .collect()
}

fn resolve_rdma_mode(mode: &str, peer: Option<&str>, hostname: &str) -> Result<String> {
    match mode {
        "capability" | "server" | "client" => Ok(mode.to_owned()),
        "app" | "app-capability" => Ok("app-capability".to_owned()),
        "app-server" | "app-client" => Ok(mode.to_owned()),
        "auto" => {
            let Some(peer) = peer else {
                return Ok("capability".to_owned());
            };
            let peer_short = peer.split('.').next().unwrap_or(peer);
            if peer == hostname || peer_short == hostname {
                Ok("server".to_owned())
            } else {
                Ok("client".to_owned())
            }
        }
        "app-auto" => {
            let Some(peer) = peer else {
                return Ok("app-capability".to_owned());
            };
            let peer_short = peer.split('.').next().unwrap_or(peer);
            if peer == hostname || peer_short == hostname {
                Ok("app-server".to_owned())
            } else {
                Ok("app-client".to_owned())
            }
        }
        other => anyhow::bail!("unsupported RDMA benchmark mode: {other}"),
    }
}

fn is_glmrt_app_mode(resolved_mode: &str) -> bool {
    matches!(
        resolved_mode,
        "app-capability" | "app-server" | "app-client"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_auto_mode_resolves_to_client_or_capability() {
        assert_eq!(
            resolve_rdma_mode("app-auto", None, "kiwi").unwrap(),
            "app-capability"
        );
        assert_eq!(
            resolve_rdma_mode("app-auto", Some("emu"), "kiwi").unwrap(),
            "app-client"
        );
        assert_eq!(
            resolve_rdma_mode("app-auto", Some("kiwi"), "kiwi").unwrap(),
            "app-server"
        );
    }
}
