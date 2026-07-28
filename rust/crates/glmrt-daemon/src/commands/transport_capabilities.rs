use anyhow::{Context, Result};
use glmrt_core::{
    TransportCapabilities, TransportPrefillBandwidthMeasurement, TransportRttMeasurement,
    GLM52_HIDDEN_SIZE,
};
use serde::Serialize;
use std::fs;
use std::path::Path;

use crate::cli::TransportCapabilitiesArgs;

#[derive(Debug, Serialize)]
struct TransportCapabilitiesReport {
    benchmark_jsonl: Option<String>,
    transports: Vec<TransportCapabilities>,
    rdma_component_perftest_rtt_by_size: Vec<TransportRttMeasurement>,
}

pub(crate) fn run_transport_capabilities(args: TransportCapabilitiesArgs) -> Result<()> {
    let mut transports = vec![
        glmrt_transport::inproc_capabilities(),
        glmrt_transport::tcp_capabilities(),
        glmrt_transport::verbs_host_capabilities(),
    ];
    let mut rdma_component_perftest_rtt_by_size = Vec::new();
    if let Some(path) = args.benchmark_jsonl.as_deref() {
        annotate_transport_capabilities(
            &mut transports,
            &mut rdma_component_perftest_rtt_by_size,
            path,
        )?;
    }

    let report = TransportCapabilitiesReport {
        benchmark_jsonl: args
            .benchmark_jsonl
            .as_ref()
            .map(|path| path.display().to_string()),
        transports,
        rdma_component_perftest_rtt_by_size,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.out {
        fs::write(&path, format!("{encoded}\n"))
            .with_context(|| format!("writing transport capabilities to {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn annotate_transport_capabilities(
    transports: &mut [TransportCapabilities],
    rdma_component_perftest_rtt_by_size: &mut Vec<TransportRttMeasurement>,
    path: &Path,
) -> Result<()> {
    let input = fs::read_to_string(path)
        .with_context(|| format!("reading benchmark JSONL from {}", path.display()))?;
    for (line_idx, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parsing benchmark JSONL line {}", line_idx + 1))?;
        let Some(benchmark) = row.get("benchmark").and_then(serde_json::Value::as_str) else {
            continue;
        };
        match benchmark {
            "tcp_expert_roundtrip" => {
                let measurement = TransportRttMeasurement {
                    payload_bytes: json_usize(&row, "payload_bytes")?,
                    avg_ms: json_f64(&row, "avg_ms")?,
                    min_ms: json_f64_opt(&row, "min_ms"),
                    max_ms: json_f64_opt(&row, "max_ms"),
                    p99_ms: None,
                    iterations: json_usize_opt(&row, "iterations"),
                    source: benchmark.to_owned(),
                };
                transport_mut(transports, "tcp")?
                    .measured_rtt_by_size
                    .push(measurement);
            }
            "tcp_expert_75_layer_prefill_chain" | "tcp_expert_prefill_roundtrip" => {
                let logical_payload_bytes = json_usize(&row, "logical_payload_bytes")?;
                let hops = json_usize(&row, "hops")?;
                let total_ms = json_f64(&row, "total_ms")?;
                let aggregate_logical_gbps =
                    aggregate_logical_gbps(logical_payload_bytes, hops, total_ms);
                let measurement = TransportPrefillBandwidthMeasurement {
                    row_count: json_usize(&row, "row_count")?,
                    logical_payload_bytes,
                    hops,
                    total_ms,
                    avg_ms: json_f64(&row, "avg_ms")?,
                    effective_prefill_tokens_per_sec: json_f64(
                        &row,
                        "effective_prefill_tokens_per_sec",
                    )?,
                    aggregate_logical_gbps,
                    source: benchmark.to_owned(),
                };
                transport_mut(transports, "tcp")?
                    .measured_prefill_payload_bandwidth
                    .push(measurement);
            }
            "spark_roce_75hop_send_lat" => {
                let source = row
                    .get("pair")
                    .and_then(serde_json::Value::as_str)
                    .map(|pair| format!("{benchmark}:{pair}"))
                    .unwrap_or_else(|| benchmark.to_owned());
                let measurement = TransportRttMeasurement {
                    payload_bytes: json_usize(&row, "payload_bytes")?,
                    avg_ms: json_f64(&row, "avg_us")? / 1000.0,
                    min_ms: None,
                    max_ms: json_f64_opt(&row, "max_us").map(|value| value / 1000.0),
                    p99_ms: json_f64_opt(&row, "p99_us").map(|value| value / 1000.0),
                    iterations: json_usize_opt(&row, "hops"),
                    source,
                };
                rdma_component_perftest_rtt_by_size.push(measurement);
            }
            "spark_verbs_app_protocol_v2" => {
                let Some(rtt_measurement) = verbs_app_rtt_measurement(&row)? else {
                    continue;
                };
                let prefill_measurement =
                    verbs_app_prefill_bandwidth_measurement(&row, &rtt_measurement)?;
                let transport = transport_mut(transports, "verbs-host")?;
                transport.measured_rtt_by_size.push(rtt_measurement);
                if let Some(prefill_measurement) = prefill_measurement {
                    transport
                        .measured_prefill_payload_bandwidth
                        .push(prefill_measurement);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn verbs_app_rtt_measurement(row: &serde_json::Value) -> Result<Option<TransportRttMeasurement>> {
    if row.get("role").and_then(serde_json::Value::as_str) != Some("client") {
        return Ok(None);
    }
    if !json_bool_opt(row, "ok").unwrap_or(false)
        || !json_bool_opt(row, "request_payload_matches").unwrap_or(false)
        || !json_bool_opt(row, "response_payload_matches").unwrap_or(false)
    {
        return Ok(None);
    }
    let roundtrips = verbs_app_roundtrips(row);
    if let Some(roundtrips) = roundtrips {
        if json_usize_opt(row, "send_completions").unwrap_or(0) < roundtrips
            || json_usize_opt(row, "recv_completions").unwrap_or(0) < roundtrips
        {
            return Ok(None);
        }
    }
    let Some(avg_us) = json_f64_opt(row, "roundtrip_latency_micros_avg") else {
        return Ok(None);
    };
    let payload_bytes = verbs_app_payload_bytes(row)?;
    Ok(Some(TransportRttMeasurement {
        payload_bytes,
        avg_ms: avg_us / 1000.0,
        min_ms: json_f64_opt(row, "roundtrip_latency_micros_min").map(|value| value / 1000.0),
        max_ms: json_f64_opt(row, "roundtrip_latency_micros_max").map(|value| value / 1000.0),
        p99_ms: None,
        iterations: roundtrips,
        source: verbs_app_measurement_source(row),
    }))
}

fn verbs_app_prefill_bandwidth_measurement(
    row: &serde_json::Value,
    rtt: &TransportRttMeasurement,
) -> Result<Option<TransportPrefillBandwidthMeasurement>> {
    if row.get("source_kind").and_then(serde_json::Value::as_str) != Some("prefill") {
        return Ok(None);
    }
    let row_count = json_usize(row, "row_count")?;
    let hops = json_usize_opt(row, "hops_per_iteration")
        .filter(|hops| *hops > 0)
        .unwrap_or(1);
    let total_ms = rtt.avg_ms * hops as f64;
    Ok(Some(TransportPrefillBandwidthMeasurement {
        row_count,
        logical_payload_bytes: rtt.payload_bytes,
        hops,
        total_ms,
        avg_ms: rtt.avg_ms,
        effective_prefill_tokens_per_sec: row_count as f64 / (total_ms / 1000.0),
        aggregate_logical_gbps: aggregate_logical_gbps(rtt.payload_bytes, hops, total_ms),
        source: rtt.source.clone(),
    }))
}

fn verbs_app_payload_bytes(row: &serde_json::Value) -> Result<usize> {
    if let Some(payload_bytes) = json_usize_opt(row, "request_logical_payload_bytes") {
        if payload_bytes > 0 {
            return Ok(payload_bytes);
        }
    }
    if let Some(row_count) = json_usize_opt(row, "row_count") {
        if row_count > 0 {
            return row_count
                .checked_mul(GLM52_HIDDEN_SIZE)
                .and_then(|value| value.checked_mul(2))
                .context("verbs app ProtocolV2 logical payload byte count overflow");
        }
    }
    json_usize_opt(row, "request_wire_bytes")
        .context("missing verbs app ProtocolV2 request payload byte count")
}

fn verbs_app_roundtrips(row: &serde_json::Value) -> Option<usize> {
    json_usize_opt(row, "roundtrips").or_else(|| {
        let iterations = json_usize_opt(row, "iterations")?;
        let hops = json_usize_opt(row, "hops_per_iteration").unwrap_or(1);
        Some(iterations.saturating_mul(hops))
    })
}

fn verbs_app_measurement_source(row: &serde_json::Value) -> String {
    let mut source = "spark_verbs_app_protocol_v2".to_owned();
    for key in ["pair", "run_kind", "source_kind", "artifact"] {
        if let Some(value) = row.get(key).and_then(serde_json::Value::as_str) {
            source.push(':');
            source.push_str(value);
        }
    }
    source
}

fn aggregate_logical_gbps(logical_payload_bytes: usize, hops: usize, total_ms: f64) -> f64 {
    ((logical_payload_bytes * hops * 2) as f64 * 8.0) / (total_ms / 1000.0) / 1e9
}

fn transport_mut<'a>(
    transports: &'a mut [TransportCapabilities],
    name: &str,
) -> Result<&'a mut TransportCapabilities> {
    transports
        .iter_mut()
        .find(|transport| transport.name == name)
        .with_context(|| format!("missing transport capability entry for {name}"))
}

fn json_usize(row: &serde_json::Value, key: &str) -> Result<usize> {
    json_usize_opt(row, key).with_context(|| format!("missing integer benchmark field {key}"))
}

fn json_usize_opt(row: &serde_json::Value, key: &str) -> Option<usize> {
    row.get(key)?.as_u64().map(|value| value as usize)
}

fn json_f64(row: &serde_json::Value, key: &str) -> Result<f64> {
    json_f64_opt(row, key).with_context(|| format!("missing numeric benchmark field {key}"))
}

fn json_f64_opt(row: &serde_json::Value, key: &str) -> Option<f64> {
    row.get(key)?.as_f64()
}

fn json_bool_opt(row: &serde_json::Value, key: &str) -> Option<bool> {
    row.get(key)?.as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_capabilities_parse_prefill_roundtrip_measurements() {
        let path = std::env::temp_dir().join(format!(
            "glmrt-prefill-roundtrip-{}-{}.jsonl",
            std::process::id(),
            "test"
        ));
        fs::write(
            &path,
            r#"{"benchmark":"tcp_expert_prefill_roundtrip","row_count":512,"logical_payload_bytes":6291456,"hops":1,"total_ms":50.0,"avg_ms":50.0,"effective_prefill_tokens_per_sec":10240.0}"#,
        )
        .unwrap();
        let mut transports = vec![
            glmrt_transport::inproc_capabilities(),
            glmrt_transport::tcp_capabilities(),
            glmrt_transport::verbs_host_capabilities(),
        ];
        let mut rdma_component_perftest_rtt_by_size = Vec::new();

        annotate_transport_capabilities(
            &mut transports,
            &mut rdma_component_perftest_rtt_by_size,
            &path,
        )
        .unwrap();
        let _ = fs::remove_file(&path);

        let tcp = transports
            .iter()
            .find(|transport| transport.name == "tcp")
            .unwrap();
        assert_eq!(tcp.measured_prefill_payload_bandwidth.len(), 1);
        let measurement = &tcp.measured_prefill_payload_bandwidth[0];
        assert_eq!(measurement.row_count, 512);
        assert_eq!(measurement.hops, 1);
        assert_eq!(measurement.source, "tcp_expert_prefill_roundtrip");
        assert!((measurement.aggregate_logical_gbps - 2.01326592).abs() < 1.0e-9);
        assert!(rdma_component_perftest_rtt_by_size.is_empty());
    }

    #[test]
    fn transport_capabilities_keep_roce_perftest_separate_from_app_rtt() {
        let path = std::env::temp_dir().join(format!(
            "glmrt-roce-perftest-{}-{}.jsonl",
            std::process::id(),
            "test"
        ));
        fs::write(
            &path,
            r#"{"benchmark":"spark_roce_75hop_send_lat","pair":"kiwi-emu","payload_bytes":12288,"avg_us":5.59,"max_us":5.85,"p99_us":5.85,"hops":75}"#,
        )
        .unwrap();
        let mut transports = vec![
            glmrt_transport::inproc_capabilities(),
            glmrt_transport::tcp_capabilities(),
            glmrt_transport::verbs_host_capabilities(),
        ];
        let mut rdma_component_perftest_rtt_by_size = Vec::new();

        annotate_transport_capabilities(
            &mut transports,
            &mut rdma_component_perftest_rtt_by_size,
            &path,
        )
        .unwrap();
        let _ = fs::remove_file(&path);

        let verbs_host = transports
            .iter()
            .find(|transport| transport.name == "verbs-host")
            .unwrap();
        assert!(verbs_host.app_transport_implemented);
        assert_eq!(
            verbs_host.app_transport_status,
            glmrt_transport::VERBS_HOST_APP_TRANSPORT_STATUS
        );
        assert!(verbs_host.measured_rtt_by_size.is_empty());
        assert_eq!(rdma_component_perftest_rtt_by_size.len(), 1);
        let perftest = &rdma_component_perftest_rtt_by_size[0];
        assert_eq!(perftest.payload_bytes, 12288);
        assert_eq!(perftest.iterations, Some(75));
        assert_eq!(perftest.source, "spark_roce_75hop_send_lat:kiwi-emu");
        assert!((perftest.avg_ms - 0.00559).abs() < 1.0e-12);
    }

    #[test]
    fn transport_capabilities_annotate_verbs_app_client_rtt() {
        let path = std::env::temp_dir().join(format!(
            "glmrt-verbs-app-rtt-{}-{}.jsonl",
            std::process::id(),
            "test"
        ));
        fs::write(
            &path,
            concat!(
                r#"{"benchmark":"spark_verbs_app_protocol_v2","artifact":"verbs_app_ostrich_dodo_client.json","pair":"ostrich->dodo","role":"client","run_kind":"roundtrip","source_kind":"decode","ok":true,"row_count":1,"request_wire_bytes":12434,"roundtrips":3,"roundtrip_latency_micros_avg":47.66933333333333,"roundtrip_latency_micros_min":45.0,"roundtrip_latency_micros_max":52.0,"request_payload_matches":true,"response_payload_matches":true,"send_completions":3,"recv_completions":3}"#,
                "\n",
                r#"{"benchmark":"spark_verbs_app_protocol_v2","artifact":"verbs_app_ostrich_dodo_server.json","pair":"dodo->ostrich","role":"server","run_kind":"roundtrip","source_kind":"decode","ok":true,"request_wire_bytes":12434,"roundtrips":3,"roundtrip_latency_micros_avg":47.66933333333333,"request_payload_matches":true,"response_payload_matches":true,"send_completions":3,"recv_completions":3}"#,
                "\n",
                r#"{"benchmark":"spark_verbs_app_protocol_v2","artifact":"verbs_app_bad_client.json","pair":"ostrich->dodo","role":"client","run_kind":"roundtrip","source_kind":"decode","ok":true,"request_wire_bytes":12434,"roundtrips":3,"roundtrip_latency_micros_avg":99.0,"request_payload_matches":true,"response_payload_matches":true,"send_completions":2,"recv_completions":3}"#,
            ),
        )
        .unwrap();
        let mut transports = vec![
            glmrt_transport::inproc_capabilities(),
            glmrt_transport::tcp_capabilities(),
            glmrt_transport::verbs_host_capabilities(),
        ];
        let mut rdma_component_perftest_rtt_by_size = Vec::new();

        annotate_transport_capabilities(
            &mut transports,
            &mut rdma_component_perftest_rtt_by_size,
            &path,
        )
        .unwrap();
        let _ = fs::remove_file(&path);

        let verbs_host = transports
            .iter()
            .find(|transport| transport.name == "verbs-host")
            .unwrap();
        assert_eq!(verbs_host.measured_rtt_by_size.len(), 1);
        let measurement = &verbs_host.measured_rtt_by_size[0];
        assert_eq!(measurement.payload_bytes, 12288);
        assert_eq!(measurement.iterations, Some(3));
        assert_eq!(
            measurement.source,
            "spark_verbs_app_protocol_v2:ostrich->dodo:roundtrip:decode:verbs_app_ostrich_dodo_client.json"
        );
        assert!((measurement.avg_ms - 0.04766933333333333).abs() < 1.0e-12);
        assert_eq!(measurement.min_ms, Some(0.045));
        assert_eq!(measurement.max_ms, Some(0.052));
        assert!(rdma_component_perftest_rtt_by_size.is_empty());
    }

    #[test]
    fn transport_capabilities_annotate_verbs_app_prefill_bandwidth() {
        let path = std::env::temp_dir().join(format!(
            "glmrt-verbs-app-prefill-{}-{}.jsonl",
            std::process::id(),
            "test"
        ));
        fs::write(
            &path,
            r#"{"benchmark":"spark_verbs_app_protocol_v2","artifact":"verbs_app_ostrich_dodo_client.json","pair":"ostrich->dodo","role":"client","run_kind":"chain_75hop","source_kind":"prefill","ok":true,"row_count":16,"hops_per_iteration":75,"roundtrips":75,"roundtrip_latency_micros_avg":1000.0,"request_payload_matches":true,"response_payload_matches":true,"send_completions":75,"recv_completions":75}"#,
        )
        .unwrap();
        let mut transports = vec![
            glmrt_transport::inproc_capabilities(),
            glmrt_transport::tcp_capabilities(),
            glmrt_transport::verbs_host_capabilities(),
        ];
        let mut rdma_component_perftest_rtt_by_size = Vec::new();

        annotate_transport_capabilities(
            &mut transports,
            &mut rdma_component_perftest_rtt_by_size,
            &path,
        )
        .unwrap();
        let _ = fs::remove_file(&path);

        let verbs_host = transports
            .iter()
            .find(|transport| transport.name == "verbs-host")
            .unwrap();
        assert_eq!(verbs_host.measured_rtt_by_size.len(), 1);
        assert_eq!(verbs_host.measured_prefill_payload_bandwidth.len(), 1);
        let measurement = &verbs_host.measured_prefill_payload_bandwidth[0];
        assert_eq!(measurement.row_count, 16);
        assert_eq!(measurement.logical_payload_bytes, 196_608);
        assert_eq!(measurement.hops, 75);
        assert_eq!(measurement.total_ms, 75.0);
        assert_eq!(measurement.avg_ms, 1.0);
        assert!((measurement.effective_prefill_tokens_per_sec - 213.33333333333334).abs() < 1.0e-9);
        assert!((measurement.aggregate_logical_gbps - 3.145728).abs() < 1.0e-12);
        assert_eq!(
            measurement.source,
            "spark_verbs_app_protocol_v2:ostrich->dodo:chain_75hop:prefill:verbs_app_ostrich_dodo_client.json"
        );
        assert!(rdma_component_perftest_rtt_by_size.is_empty());
    }
}
