use super::*;

#[test]
fn scheduler_smoke_prioritizes_decode_under_background_prefill() {
    let report = scheduler_smoke_report(SchedulerSmokeArgs {
        prefill_tokens: 512,
        chunk_tokens: 16,
        decode_arrivals: 32,
        decode_period_iterations: 1,
        max_prefill_tokens_per_iteration: 16,
        max_active_prefill_chunks: 1,
    });

    assert_eq!(report.selected_decode_rows, 32);
    assert_eq!(report.selected_prefill_rows, 512);
    assert_eq!(report.selected_prefill_chunks, 32);
    assert_eq!(report.p99_decode_admission_delay_iterations, 0);
    assert_eq!(report.max_decode_admission_delay_iterations, 0);
    assert_eq!(report.p50_decode_inter_token_iterations, 1);
    assert_eq!(report.p99_decode_inter_token_iterations, 1);
    assert_eq!(report.prefill_completion_iterations, 32);
    assert_eq!(report.ttft_iteration_estimate, 32);
}

#[test]
fn scheduler_smoke_reports_prefill_rows_per_expert_distribution() {
    let report = scheduler_smoke_report(SchedulerSmokeArgs {
        prefill_tokens: 512,
        chunk_tokens: 16,
        decode_arrivals: 0,
        decode_period_iterations: 1,
        max_prefill_tokens_per_iteration: 16,
        max_active_prefill_chunks: 1,
    });

    assert_eq!(report.prefill_active_experts, GLM52_ROUTED_EXPERTS);
    assert_eq!(report.prefill_rows_per_expert_min, 16);
    assert_eq!(report.prefill_rows_per_expert_max, 16);
    assert!((report.prefill_rows_per_expert_avg - 16.0).abs() < 1.0e-9);
}
