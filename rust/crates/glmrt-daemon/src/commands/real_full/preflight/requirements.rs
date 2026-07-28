use glmrt_core::{KvCacheConfig, TensorCatalog};

use super::super::types::*;

mod availability;
mod available;
mod blocked;

use availability::real_full_preflight_availability;
use available::available_runtime_requirements;
use blocked::{blocked_runtime_requirements, BlockedRuntimeRequirementInputs};

pub(super) struct RealFullPreflightRequirementInputs<'a> {
    pub(super) catalog: &'a TensorCatalog,
    pub(super) coverage: &'a FullModelTensorCoverage,
    pub(super) kv_config: &'a KvCacheConfig,
    pub(super) expert_hosts: &'a [String],
    pub(super) execution_plan: &'a RealFullExecutionPlan,
    pub(super) residual_stream_dry_run: &'a RealFullResidualStreamDryRun,
    pub(super) sampling_dry_run: &'a RealFullSamplingDryRun,
    pub(super) expert_execution_dry_run: &'a RealFullExpertExecutionDryRun,
    pub(super) scheduler_dry_run: &'a RealFullSchedulerDryRun,
    pub(super) scheduler_execution_dry_run: &'a RealFullSchedulerExecutionDryRun,
    pub(super) kv_backing_store_dry_run: &'a RealFullKvBackingStoreDryRun,
    pub(super) attention_kv_io_dry_run: &'a RealFullAttentionKvIoDryRun,
    pub(super) attention_kv_binding_dry_run: &'a RealFullAttentionKvBindingDryRun,
    pub(super) coordinator_resident_preload: &'a RealFullCoordinatorResidentPreloadPlan,
}

pub(super) fn real_full_preflight_requirements(
    inputs: RealFullPreflightRequirementInputs<'_>,
) -> Vec<RealFullRequirement> {
    let availability = real_full_preflight_availability(&inputs);
    let mut requirements = available_runtime_requirements(&inputs, &availability);

    requirements.extend(blocked_runtime_requirements(
        BlockedRuntimeRequirementInputs {
            attention_kv_binding_available: availability.attention_kv_binding_available,
            attention_kv_binding_dry_run: inputs.attention_kv_binding_dry_run,
            dense_prefix_probe: availability.dense_prefix_probe,
            attention_residual_prefix_probe: availability.attention_residual_prefix_probe,
            mla_rope_attention_probe: availability.mla_rope_attention_probe,
            dsa_indexer_attention_probe: availability.dsa_indexer_attention_probe,
            attention_dense_sparse_prefix_probe: availability.attention_dense_sparse_prefix_probe,
            scheduler_real_tensor_catalog_available: availability
                .scheduler_real_tensor_catalog_available,
            expert_execution_dry_run: inputs.expert_execution_dry_run,
            expert_numeric_probe: availability.expert_numeric_probe,
            expert_all_layer_probe: availability.expert_all_layer_probe,
            expert_residual_chain_probe: availability.expert_residual_chain_probe,
            expert_shared_chain_probe: availability.expert_shared_chain_probe,
            expert_scheduler_rows_probe: availability.expert_scheduler_rows_probe,
            layer_ordered_prefix_probe: &inputs
                .residual_stream_dry_run
                .real_layer_ordered_prefix_probe,
            layer_ordered_execution_probe: &inputs
                .residual_stream_dry_run
                .real_layer_ordered_execution_probe,
            scheduler_execution_dry_run: inputs.scheduler_execution_dry_run,
            residual_kernel: availability.residual_kernel,
            sampling_dry_run: inputs.sampling_dry_run,
            sampling_real_lm_head_probe: availability.sampling_real_lm_head_probe,
        },
    ));
    requirements
}

pub(super) fn coordinator_resident_preload_requirement(
    coordinator_resident_preload: &RealFullCoordinatorResidentPreloadPlan,
) -> RealFullRequirement {
    available::coordinator_resident_preload_requirement(coordinator_resident_preload)
}
