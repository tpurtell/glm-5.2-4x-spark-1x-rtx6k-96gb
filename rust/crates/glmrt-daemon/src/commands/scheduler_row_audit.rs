use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::cli::SchedulerRowAuditArgs;

pub(crate) fn run_scheduler_row_audit(args: SchedulerRowAuditArgs) -> Result<()> {
    let inputs = scheduler_row_audit_inputs(&args.inputs, &args.input_lists)?;
    let report = scheduler_row_audit_report(&inputs, args.next_window_count)?;
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, format!("{encoded}\n"))
            .with_context(|| format!("writing {}", out.display()))?;
    }
    println!("{encoded}");
    Ok(())
}

fn scheduler_row_audit_inputs(inputs: &[PathBuf], input_lists: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut expanded = inputs.to_vec();
    for input_list in input_lists {
        let text = fs::read_to_string(input_list)
            .with_context(|| format!("reading {}", input_list.display()))?;
        for (line_index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let path = PathBuf::from(trimmed);
            if path.as_os_str().is_empty() {
                anyhow::bail!(
                    "empty scheduler-row audit input path in {} line {}",
                    input_list.display(),
                    line_index + 1
                );
            }
            expanded.push(path);
        }
    }
    if expanded.is_empty() {
        anyhow::bail!("provide at least one --input or --input-list entry");
    }
    Ok(expanded)
}

fn scheduler_row_audit_report(
    inputs: &[PathBuf],
    next_window_count: usize,
) -> Result<SchedulerRowAuditReport> {
    let mut probes = Vec::new();
    for path in inputs {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let extracted = extract_scheduler_row_probe_values(&text);
        for value in extracted {
            let probe: SchedulerRowProbe = serde_json::from_value(value)
                .with_context(|| format!("parsing scheduler-row probe in {}", path.display()))?;
            probes.push(SourcedSchedulerRowProbe {
                source: path.display().to_string(),
                probe,
            });
        }
    }
    Ok(audit_scheduler_row_probes_with_next_window_count(
        probes,
        next_window_count,
    ))
}

#[derive(Debug, Serialize)]
struct SchedulerRowAuditReport {
    status: &'static str,
    input_files: usize,
    parsed_probes: usize,
    accepted_windows: usize,
    rejected_probes: usize,
    planned_source_rows: usize,
    planned_decode_source_rows: usize,
    planned_prefill_source_rows: usize,
    planned_mtp_verify_source_rows: usize,
    covered_source_rows: usize,
    covered_decode_source_rows: usize,
    covered_prefill_source_rows: usize,
    covered_mtp_verify_source_rows: usize,
    duplicate_source_rows: usize,
    missing_source_rows: usize,
    missing_ranges: Vec<RowRange>,
    next_missing_window: Option<RowRange>,
    next_missing_window_env: Option<String>,
    duplicate_ranges: Vec<RowRange>,
    total_layers_executed: usize,
    total_routes_executed: usize,
    output_checksum_sum: f64,
    final_residual_checksum_sum: f64,
    all_windows_passed: bool,
    all_windows_cover_sparse_layers: bool,
    all_windows_cover_full_top_k: bool,
    covers_all_scheduler_rows: bool,
    windows: Vec<SchedulerRowAuditWindow>,
    rejected: Vec<SchedulerRowAuditRejectedProbe>,
}

#[derive(Debug, Serialize)]
struct SchedulerRowAuditWindow {
    source: String,
    start: usize,
    end: usize,
    rows: usize,
    executed_decode_source_rows: usize,
    executed_prefill_source_rows: usize,
    executed_mtp_verify_source_rows: usize,
    layers_executed: usize,
    routes_executed: usize,
    output_checksum: Option<f64>,
    final_residual_checksum: Option<f64>,
}

#[derive(Debug, Serialize)]
struct SchedulerRowAuditRejectedProbe {
    source: String,
    status: String,
    row_mode: String,
    reason: String,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
struct RowRange {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct SourcedSchedulerRowProbe {
    source: String,
    probe: SchedulerRowProbe,
}

#[derive(Debug, Deserialize)]
struct SchedulerRowProbe {
    status: String,
    row_mode: String,
    source_rows_executed: usize,
    planned_source_rows: usize,
    planned_decode_source_rows: usize,
    planned_prefill_source_rows: usize,
    planned_mtp_verify_source_rows: usize,
    executed_decode_source_rows: usize,
    executed_prefill_source_rows: usize,
    executed_mtp_verify_source_rows: usize,
    source_row_window_start: usize,
    source_row_window_end: usize,
    uses_source_row_window: bool,
    uses_full_scheduler_rows: bool,
    covers_all_scheduler_rows: bool,
    layers_executed: usize,
    routes_executed: usize,
    covers_all_sparse_layers: bool,
    covers_full_top_k: bool,
    output_checksum: Option<f64>,
    final_residual_checksum: Option<f64>,
    passed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SourceModeCounts {
    decode: usize,
    prefill: usize,
    mtp_verify: usize,
}

impl SourceModeCounts {
    fn total(self) -> usize {
        self.decode + self.prefill + self.mtp_verify
    }
}

#[cfg(test)]
fn audit_scheduler_row_probes(probes: Vec<SourcedSchedulerRowProbe>) -> SchedulerRowAuditReport {
    audit_scheduler_row_probes_with_next_window_count(probes, 1)
}

fn audit_scheduler_row_probes_with_next_window_count(
    probes: Vec<SourcedSchedulerRowProbe>,
    next_window_count: usize,
) -> SchedulerRowAuditReport {
    let input_files = probes
        .iter()
        .map(|probe| probe.source.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let parsed_probes = probes.len();
    let mut planned = SourceModeCounts::default();
    let mut planned_source_rows = 0_usize;
    let mut coverage = Vec::<usize>::new();
    let mut windows = Vec::new();
    let mut rejected = Vec::new();
    let mut total_layers_executed = 0_usize;
    let mut total_routes_executed = 0_usize;
    let mut output_checksum_sum = 0.0_f64;
    let mut final_residual_checksum_sum = 0.0_f64;
    let mut all_windows_passed = true;
    let mut all_windows_cover_sparse_layers = true;
    let mut all_windows_cover_full_top_k = true;

    for sourced in probes {
        let probe = sourced.probe;
        if planned_source_rows == 0 {
            planned = SourceModeCounts {
                decode: probe.planned_decode_source_rows,
                prefill: probe.planned_prefill_source_rows,
                mtp_verify: probe.planned_mtp_verify_source_rows,
            };
            planned_source_rows = probe.planned_source_rows;
            coverage.resize(planned_source_rows, 0);
        }

        if let Some(reason) = reject_probe(&probe, planned_source_rows, planned) {
            rejected.push(SchedulerRowAuditRejectedProbe {
                source: sourced.source,
                status: probe.status,
                row_mode: probe.row_mode,
                reason,
            });
            continue;
        }

        let start = if probe.uses_source_row_window {
            probe.source_row_window_start
        } else {
            0
        };
        let end = if probe.uses_source_row_window {
            probe.source_row_window_end
        } else {
            probe.planned_source_rows
        };
        for row_index in start..end {
            coverage[row_index] += 1;
        }

        total_layers_executed += probe.layers_executed;
        total_routes_executed += probe.routes_executed;
        output_checksum_sum += probe.output_checksum.unwrap_or(0.0);
        final_residual_checksum_sum += probe.final_residual_checksum.unwrap_or(0.0);
        all_windows_passed &= probe.passed;
        all_windows_cover_sparse_layers &= probe.covers_all_sparse_layers;
        all_windows_cover_full_top_k &= probe.covers_full_top_k;
        windows.push(SchedulerRowAuditWindow {
            source: sourced.source,
            start,
            end,
            rows: end - start,
            executed_decode_source_rows: probe.executed_decode_source_rows,
            executed_prefill_source_rows: probe.executed_prefill_source_rows,
            executed_mtp_verify_source_rows: probe.executed_mtp_verify_source_rows,
            layers_executed: probe.layers_executed,
            routes_executed: probe.routes_executed,
            output_checksum: probe.output_checksum,
            final_residual_checksum: probe.final_residual_checksum,
        });
    }

    let covered_source_rows = coverage.iter().filter(|count| **count > 0).count();
    let duplicate_source_rows = coverage.iter().filter(|count| **count > 1).count();
    let missing_source_rows = coverage.iter().filter(|count| **count == 0).count();
    let missing_ranges = ranges_for_count(&coverage, |count| count == 0);
    let next_missing_window = next_missing_window(&missing_ranges, next_window_count);
    let next_missing_window_env =
        next_missing_window.map(|range| format!("{}:{}", range.start, range.end - range.start));
    let duplicate_ranges = ranges_for_count(&coverage, |count| count > 1);
    let covered_by_mode = covered_source_modes(&coverage, planned);
    let accepted_windows = windows.len();
    let covers_all_scheduler_rows = planned_source_rows > 0
        && accepted_windows > 0
        && rejected.is_empty()
        && missing_source_rows == 0
        && duplicate_source_rows == 0
        && all_windows_passed
        && all_windows_cover_sparse_layers
        && all_windows_cover_full_top_k;
    let status = if covers_all_scheduler_rows {
        "complete"
    } else if accepted_windows > 0 {
        "partial"
    } else {
        "no-valid-windows"
    };

    SchedulerRowAuditReport {
        status,
        input_files,
        parsed_probes,
        accepted_windows,
        rejected_probes: rejected.len(),
        planned_source_rows,
        planned_decode_source_rows: planned.decode,
        planned_prefill_source_rows: planned.prefill,
        planned_mtp_verify_source_rows: planned.mtp_verify,
        covered_source_rows,
        covered_decode_source_rows: covered_by_mode.decode,
        covered_prefill_source_rows: covered_by_mode.prefill,
        covered_mtp_verify_source_rows: covered_by_mode.mtp_verify,
        duplicate_source_rows,
        missing_source_rows,
        missing_ranges,
        next_missing_window,
        next_missing_window_env,
        duplicate_ranges,
        total_layers_executed,
        total_routes_executed,
        output_checksum_sum,
        final_residual_checksum_sum,
        all_windows_passed,
        all_windows_cover_sparse_layers,
        all_windows_cover_full_top_k,
        covers_all_scheduler_rows,
        windows,
        rejected,
    }
}

fn next_missing_window(missing_ranges: &[RowRange], requested_count: usize) -> Option<RowRange> {
    let first = *missing_ranges.first()?;
    let count = requested_count.max(1);
    let end = first.start.saturating_add(count).min(first.end);
    Some(RowRange {
        start: first.start,
        end,
    })
}

fn reject_probe(
    probe: &SchedulerRowProbe,
    planned_source_rows: usize,
    planned: SourceModeCounts,
) -> Option<String> {
    if !probe.uses_full_scheduler_rows {
        return Some("probe does not use full scheduler rows".to_owned());
    }
    if !probe.passed {
        return Some("probe did not pass".to_owned());
    }
    if probe.row_mode != "all-scheduler-rows" {
        return Some(format!("probe row_mode is {}", probe.row_mode));
    }
    if probe.planned_source_rows != planned_source_rows {
        return Some(format!(
            "planned_source_rows {} differs from audit plan {}",
            probe.planned_source_rows, planned_source_rows
        ));
    }
    let probe_planned = SourceModeCounts {
        decode: probe.planned_decode_source_rows,
        prefill: probe.planned_prefill_source_rows,
        mtp_verify: probe.planned_mtp_verify_source_rows,
    };
    if probe_planned != planned || probe_planned.total() != probe.planned_source_rows {
        return Some("probe planned source-mode counts do not match audit plan".to_owned());
    }

    let start = if probe.uses_source_row_window {
        probe.source_row_window_start
    } else {
        0
    };
    let end = if probe.uses_source_row_window {
        probe.source_row_window_end
    } else {
        probe.planned_source_rows
    };
    if start >= end || end > probe.planned_source_rows {
        return Some(format!("invalid source-row window {start}..{end}"));
    }
    if end - start != probe.source_rows_executed {
        return Some(format!(
            "window rows {} do not match source_rows_executed {}",
            end - start,
            probe.source_rows_executed
        ));
    }
    let executed = SourceModeCounts {
        decode: probe.executed_decode_source_rows,
        prefill: probe.executed_prefill_source_rows,
        mtp_verify: probe.executed_mtp_verify_source_rows,
    };
    if executed.total() != probe.source_rows_executed {
        return Some("executed source-mode counts do not sum to source_rows_executed".to_owned());
    }
    let expected = expected_source_modes_for_range(start, end, planned);
    if executed != expected {
        return Some(format!(
            "executed source-mode counts {:?} do not match planned row range {:?}",
            executed, expected
        ));
    }
    if !probe.uses_source_row_window && !probe.covers_all_scheduler_rows {
        return Some("non-window full probe does not cover all scheduler rows".to_owned());
    }
    None
}

fn expected_source_modes_for_range(
    start: usize,
    end: usize,
    planned: SourceModeCounts,
) -> SourceModeCounts {
    let decode_end = planned.decode;
    let prefill_end = planned.decode + planned.prefill;
    SourceModeCounts {
        decode: intersect_len(start, end, 0, decode_end),
        prefill: intersect_len(start, end, decode_end, prefill_end),
        mtp_verify: intersect_len(start, end, prefill_end, planned.total()),
    }
}

fn covered_source_modes(coverage: &[usize], planned: SourceModeCounts) -> SourceModeCounts {
    let decode_end = planned.decode;
    let prefill_end = planned.decode + planned.prefill;
    let mut counts = SourceModeCounts::default();
    for (row_index, count) in coverage.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        if row_index < decode_end {
            counts.decode += 1;
        } else if row_index < prefill_end {
            counts.prefill += 1;
        } else {
            counts.mtp_verify += 1;
        }
    }
    counts
}

fn intersect_len(start: usize, end: usize, range_start: usize, range_end: usize) -> usize {
    end.min(range_end).saturating_sub(start.max(range_start))
}

fn ranges_for_count<F>(coverage: &[usize], predicate: F) -> Vec<RowRange>
where
    F: Fn(usize) -> bool,
{
    let mut ranges = Vec::new();
    let mut start = None;
    for (index, count) in coverage.iter().copied().enumerate() {
        if predicate(count) {
            start.get_or_insert(index);
        } else if let Some(range_start) = start.take() {
            ranges.push(RowRange {
                start: range_start,
                end: index,
            });
        }
    }
    if let Some(range_start) = start {
        ranges.push(RowRange {
            start: range_start,
            end: coverage.len(),
        });
    }
    ranges
}

fn extract_scheduler_row_probe_values(text: &str) -> Vec<Value> {
    extract_json_values(text)
        .into_iter()
        .filter(|value| {
            value.get("row_mode").is_some()
                && value.get("source_rows_executed").is_some()
                && value.get("source_row_window_start").is_some()
        })
        .collect()
}

fn extract_json_values(text: &str) -> Vec<Value> {
    let mut values = Vec::new();
    let mut start = None;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(index);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start_index) = start.take() {
                        let end_index = index + ch.len_utf8();
                        if let Ok(value) =
                            serde_json::from_str::<Value>(&text[start_index..end_index])
                        {
                            values.push(value);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::{
        audit_scheduler_row_probes, audit_scheduler_row_probes_with_next_window_count,
        extract_scheduler_row_probe_values, scheduler_row_audit_inputs, SchedulerRowProbe,
        SourcedSchedulerRowProbe,
    };
    use std::path::PathBuf;

    #[test]
    fn extracts_scheduler_row_probe_from_cargo_log() {
        let log = r#"
running 1 test
{
  "status": "numeric-real-nvfp4-full-scheduler-row-window",
  "row_mode": "all-scheduler-rows",
  "source_rows_executed": 1,
  "planned_source_rows": 4,
  "planned_decode_source_rows": 1,
  "planned_prefill_source_rows": 2,
  "planned_mtp_verify_source_rows": 1,
  "executed_decode_source_rows": 1,
  "executed_prefill_source_rows": 0,
  "executed_mtp_verify_source_rows": 0,
  "source_row_window_start": 0,
  "source_row_window_end": 1,
  "uses_source_row_window": true,
  "uses_full_scheduler_rows": true,
  "covers_all_scheduler_rows": false,
  "layers_executed": 75,
  "routes_executed": 600,
  "covers_all_sparse_layers": true,
  "covers_full_top_k": true,
  "output_checksum": 1.5,
  "final_residual_checksum": 2.5,
  "passed": true
}
test result: ok
"#;

        let values = extract_scheduler_row_probe_values(log);

        assert_eq!(values.len(), 1);
        assert_eq!(values[0]["source_row_window_end"], 1);
    }

    #[test]
    fn audit_reports_complete_exact_window_coverage() {
        let probes = vec![
            sourced_probe("a", probe(0, 1, 1, 0, 0)),
            sourced_probe("b", probe(1, 3, 0, 2, 0)),
            sourced_probe("c", probe(3, 4, 0, 0, 1)),
        ];

        let report = audit_scheduler_row_probes(probes);

        assert_eq!(report.status, "complete");
        assert!(report.covers_all_scheduler_rows);
        assert_eq!(report.accepted_windows, 3);
        assert_eq!(report.covered_source_rows, 4);
        assert_eq!(report.covered_decode_source_rows, 1);
        assert_eq!(report.covered_prefill_source_rows, 2);
        assert_eq!(report.covered_mtp_verify_source_rows, 1);
        assert!(report.missing_ranges.is_empty());
        assert!(report.next_missing_window.is_none());
        assert!(report.next_missing_window_env.is_none());
        assert!(report.duplicate_ranges.is_empty());
    }

    #[test]
    fn audit_reports_partial_duplicate_and_rejected_probes() {
        let mut rejected = probe(0, 1, 1, 0, 0);
        rejected.uses_full_scheduler_rows = false;
        let probes = vec![
            sourced_probe("a", probe(0, 1, 1, 0, 0)),
            sourced_probe("b", probe(0, 1, 1, 0, 0)),
            sourced_probe("c", rejected),
        ];

        let report = audit_scheduler_row_probes(probes);

        assert_eq!(report.status, "partial");
        assert!(!report.covers_all_scheduler_rows);
        assert_eq!(report.accepted_windows, 2);
        assert_eq!(report.rejected_probes, 1);
        assert_eq!(report.duplicate_source_rows, 1);
        assert_eq!(report.missing_source_rows, 3);
        assert_eq!(report.duplicate_ranges[0].start, 0);
        assert_eq!(report.missing_ranges[0].start, 1);
        assert_eq!(report.next_missing_window.unwrap().start, 1);
        assert_eq!(report.next_missing_window.unwrap().end, 2);
        assert_eq!(report.next_missing_window_env.as_deref(), Some("1:1"));
    }

    #[test]
    fn audit_reports_next_missing_window_for_requested_chunk_size() {
        let probes = vec![sourced_probe("a", probe(0, 1, 1, 0, 0))];

        let report = audit_scheduler_row_probes_with_next_window_count(probes, 2);

        assert_eq!(report.status, "partial");
        assert_eq!(report.missing_ranges[0].start, 1);
        assert_eq!(report.missing_ranges[0].end, 4);
        assert_eq!(report.next_missing_window.unwrap().start, 1);
        assert_eq!(report.next_missing_window.unwrap().end, 3);
        assert_eq!(report.next_missing_window_env.as_deref(), Some("1:2"));
    }

    #[test]
    fn input_list_expands_non_comment_paths_after_inline_inputs() {
        let list_path = std::env::temp_dir().join(format!(
            "glmrt-scheduler-row-audit-inputs-{}.txt",
            std::process::id()
        ));
        std::fs::write(
            &list_path,
            "\n# scheduler window inputs\nreports/window-a.log\nreports/window-b.log\n",
        )
        .expect("writing scheduler-row audit input-list fixture");

        let inputs =
            scheduler_row_audit_inputs(&[PathBuf::from("inline.log")], &[list_path.clone()])
                .expect("expanding scheduler-row audit input list");

        std::fs::remove_file(&list_path).ok();
        assert_eq!(
            inputs,
            vec![
                PathBuf::from("inline.log"),
                PathBuf::from("reports/window-a.log"),
                PathBuf::from("reports/window-b.log"),
            ]
        );
    }

    #[test]
    fn input_list_requires_at_least_one_expanded_input() {
        let err = scheduler_row_audit_inputs(&[], &[]).unwrap_err();

        assert!(err.to_string().contains("--input"));
    }

    fn sourced_probe(source: &str, probe: SchedulerRowProbe) -> SourcedSchedulerRowProbe {
        SourcedSchedulerRowProbe {
            source: source.to_owned(),
            probe,
        }
    }

    fn probe(
        start: usize,
        end: usize,
        executed_decode: usize,
        executed_prefill: usize,
        executed_mtp_verify: usize,
    ) -> SchedulerRowProbe {
        SchedulerRowProbe {
            status: "numeric-real-nvfp4-full-scheduler-row-window".to_owned(),
            row_mode: "all-scheduler-rows".to_owned(),
            source_rows_executed: end - start,
            planned_source_rows: 4,
            planned_decode_source_rows: 1,
            planned_prefill_source_rows: 2,
            planned_mtp_verify_source_rows: 1,
            executed_decode_source_rows: executed_decode,
            executed_prefill_source_rows: executed_prefill,
            executed_mtp_verify_source_rows: executed_mtp_verify,
            source_row_window_start: start,
            source_row_window_end: end,
            uses_source_row_window: true,
            uses_full_scheduler_rows: true,
            covers_all_scheduler_rows: false,
            layers_executed: 75 * (end - start),
            routes_executed: 600 * (end - start),
            covers_all_sparse_layers: true,
            covers_full_top_k: true,
            output_checksum: Some(start as f64),
            final_residual_checksum: Some(end as f64),
            passed: true,
        }
    }
}
