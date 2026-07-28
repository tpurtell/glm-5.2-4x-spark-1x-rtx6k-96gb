use super::super::types::RealGlmFullPreflightReport;

mod execution_scheduler;
mod expert_execution;
mod kv_attention;
mod residual_sampling;

pub(super) fn assert_real_full_preflight_report(report: &RealGlmFullPreflightReport) {
    execution_scheduler::assert_execution_scheduler_report(report);
    kv_attention::assert_kv_attention_report(report);
    residual_sampling::assert_residual_sampling_report(report);
    expert_execution::assert_expert_execution_report(report);
}
