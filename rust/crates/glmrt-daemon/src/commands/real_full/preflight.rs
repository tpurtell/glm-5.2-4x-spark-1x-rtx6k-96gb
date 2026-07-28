use anyhow::Result;
use glmrt_core::{KvCacheConfig, KvCacheDType, ModelFacts, TensorCatalog, TransportCapabilities};

use crate::cli::CoordinatorArgs;

mod requirements;

use super::attention::real_full_attention_kv_binding_dry_run;
use super::constants::REAL_GLM_FULL_BLOCKER;
use super::coverage::tensor_coverage;
use super::execution_plan::real_full_execution_plan;
use super::experts::real_full_expert_execution_dry_run;
use super::kv::{real_full_attention_kv_io_dry_run, real_full_kv_backing_store_dry_run};
use super::residency::real_full_coordinator_resident_preload_plan;
use super::residual::real_full_residual_stream_dry_run;
use super::sampling::real_full_sampling_dry_run;
use super::scheduler::{real_full_scheduler_dry_run, real_full_scheduler_execution_dry_run};
use super::types::{
    RealFullCoordinatorResidentPreloadPlan, RealFullKvPlan, RealFullRequirement,
    RealFullSparseTransportPlan, RealGlmFullPreflightReport,
};
use requirements::{real_full_preflight_requirements, RealFullPreflightRequirementInputs};

pub(super) fn real_glm_full_preflight_report(
    args: &CoordinatorArgs,
    catalog_source: &str,
    catalog: &TensorCatalog,
) -> Result<RealGlmFullPreflightReport> {
    let coordinator_resident_preload = real_full_coordinator_resident_preload_plan(catalog);
    real_glm_full_preflight_report_with_coordinator_resident_preload(
        args,
        catalog_source,
        catalog,
        coordinator_resident_preload,
    )
}

pub(super) fn coordinator_resident_preload_requirement(
    coordinator_resident_preload: &RealFullCoordinatorResidentPreloadPlan,
) -> RealFullRequirement {
    requirements::coordinator_resident_preload_requirement(coordinator_resident_preload)
}

pub(super) fn real_glm_full_preflight_report_with_coordinator_resident_preload(
    args: &CoordinatorArgs,
    catalog_source: &str,
    catalog: &TensorCatalog,
    coordinator_resident_preload: RealFullCoordinatorResidentPreloadPlan,
) -> Result<RealGlmFullPreflightReport> {
    let coverage = tensor_coverage(catalog);
    let catalog_hash = catalog.content_hash();
    let kv_config = real_full_kv_cache_config(args)?;
    let sparse_transport = real_full_sparse_transport_plan(args);
    let expert_hosts = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let execution_plan = real_full_execution_plan(&expert_hosts, kv_config.bytes_per_token());
    let residual_stream_dry_run = real_full_residual_stream_dry_run(&execution_plan, catalog);
    let sampling_dry_run = real_full_sampling_dry_run(catalog, &execution_plan)?;
    let scheduler_dry_run = real_full_scheduler_dry_run(&catalog_hash)?;
    let scheduler_execution_dry_run =
        real_full_scheduler_execution_dry_run(kv_config.clone(), catalog)?;
    let expert_execution_dry_run =
        real_full_expert_execution_dry_run(catalog, &expert_hosts, &scheduler_execution_dry_run);
    let kv_backing_store_dry_run = real_full_kv_backing_store_dry_run(kv_config.clone())?;
    let attention_kv_io_dry_run = real_full_attention_kv_io_dry_run(kv_config.clone())?;
    let attention_kv_binding_dry_run =
        real_full_attention_kv_binding_dry_run(catalog, &kv_config, &attention_kv_io_dry_run);
    let requirements = real_full_preflight_requirements(RealFullPreflightRequirementInputs {
        catalog,
        coverage: &coverage,
        kv_config: &kv_config,
        expert_hosts: &expert_hosts,
        execution_plan: &execution_plan,
        residual_stream_dry_run: &residual_stream_dry_run,
        sampling_dry_run: &sampling_dry_run,
        expert_execution_dry_run: &expert_execution_dry_run,
        scheduler_dry_run: &scheduler_dry_run,
        scheduler_execution_dry_run: &scheduler_execution_dry_run,
        kv_backing_store_dry_run: &kv_backing_store_dry_run,
        attention_kv_io_dry_run: &attention_kv_io_dry_run,
        attention_kv_binding_dry_run: &attention_kv_binding_dry_run,
        coordinator_resident_preload: &coordinator_resident_preload,
    });
    Ok(RealGlmFullPreflightReport {
        backend: "real-glm-full",
        status: "blocked",
        model_id: args.model_id.clone(),
        catalog_path: catalog_source.to_owned(),
        snapshot_path: catalog.snapshot_path.clone(),
        catalog_hash,
        tensor_count: catalog.tensors.len(),
        listen: args.listen.clone(),
        transport: args.transport.clone(),
        sparse_transport,
        expert_hosts,
        model_facts: catalog.facts.clone(),
        expected_facts: ModelFacts::default(),
        role_counts: catalog.summary_by_role(),
        full_model_tensor_coverage: coverage,
        kv_plan: RealFullKvPlan {
            layout: kv_config.layout_label(),
            dtype: kv_config.dtype_label(),
            max_tokens: kv_config.max_tokens,
            bytes_per_token: kv_config.bytes_per_token(),
            capacity_bytes: kv_config.capacity_bytes(),
        },
        execution_plan,
        residual_stream_dry_run,
        sampling_dry_run,
        expert_execution_dry_run,
        scheduler_dry_run,
        scheduler_execution_dry_run,
        kv_backing_store_dry_run,
        attention_kv_io_dry_run,
        attention_kv_binding_dry_run,
        coordinator_resident_preload,
        requirements,
        blocker: REAL_GLM_FULL_BLOCKER,
    })
}

pub(in crate::commands::real_full) fn real_full_sparse_transport_plan(
    args: &CoordinatorArgs,
) -> RealFullSparseTransportPlan {
    match args.transport.as_str() {
        "tcp" => {
            let targets_configured = args
                .expert_hosts
                .split(',')
                .map(str::trim)
                .any(|target| !target.is_empty());
            sparse_transport_plan_from_capabilities(
                args,
                glmrt_transport::tcp_capabilities(),
                if targets_configured {
                    "ready-tcp-protocol-v2"
                } else {
                    "blocked-missing-expert-hosts"
                },
                targets_configured,
                Some("tcp-protocol-v2-persistent-client"),
                true,
                None,
                Some(glmrt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL),
                (!targets_configured).then_some(
                    "real-glm-full TCP sparse dispatch requires --expert-hosts".to_owned(),
                ),
            )
        }
        "inproc" => sparse_transport_plan_from_capabilities(
            args,
            glmrt_transport::inproc_capabilities(),
            "disabled-inproc",
            false,
            None,
            true,
            None,
            None,
            Some(
                "real-glm-full inproc transport disables sparse expert dispatch; use --transport tcp for live sparse serving"
                    .to_owned(),
            ),
        ),
        "tcp-debug-json" | "debug-json" => sparse_transport_plan_from_capabilities(
            args,
            glmrt_transport::tcp_capabilities(),
            "blocked-debug-json-not-supported",
            false,
            None,
            true,
            None,
            Some(glmrt_transport::DEBUG_JSON_FRAME_PROTOCOL),
            Some(
                "real-glm-full sparse serving requires ProtocolV2 TCP, not debug-json framing"
                    .to_owned(),
            ),
        ),
        "verbs-host" => {
            let targets_configured = args
                .expert_hosts
                .split(',')
                .map(str::trim)
                .any(|target| !target.is_empty());
            let preflight = glmrt_transport::verbs_host_preflight();
            let (preflight_ok, preflight_error) = match preflight {
                Ok(_) => (true, None),
                Err(error) => (false, Some(error.to_string())),
            };
            let blocker = if let Some(error) = &preflight_error {
                Some(format!(
                    "real-glm-full verbs-host sparse dispatch RDMA preflight failed: {error}"
                ))
            } else if !targets_configured {
                Some("real-glm-full verbs-host sparse dispatch requires --expert-hosts".to_owned())
            } else {
                None
            };
            let sparse_dispatch_available = preflight_ok && targets_configured;
            sparse_transport_plan_from_capabilities(
                args,
                glmrt_transport::verbs_host_capabilities(),
                if sparse_dispatch_available {
                    "ready-verbs-host-protocol-v2"
                } else if preflight_ok {
                    "blocked-missing-expert-hosts"
                } else {
                    "blocked-rdma-preflight-failed"
                },
                sparse_dispatch_available,
                sparse_dispatch_available.then_some("verbs-host-protocol-v2-rc-qp"),
                preflight_ok,
                preflight_error,
                Some(glmrt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL),
                blocker,
            )
        }
        other => RealFullSparseTransportPlan {
            transport: other.to_owned(),
            status: "blocked-unsupported-transport".to_owned(),
            sparse_dispatch_available: false,
            scheduler_dispatch_backend: None,
            supports_rdma: false,
            supports_host_registered_buffers: false,
            requires_pinned_host_memory: false,
            app_transport_implemented: false,
            app_transport_status: "unsupported".to_owned(),
            preflight_ok: false,
            preflight_error: Some(format!("unsupported real-glm-full sparse transport: {other}")),
            frame_protocol: None,
            blocker: Some(format!(
                "unsupported real-glm-full sparse transport: {other}"
            )),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn sparse_transport_plan_from_capabilities(
    args: &CoordinatorArgs,
    capabilities: TransportCapabilities,
    status: &str,
    sparse_dispatch_available: bool,
    scheduler_dispatch_backend: Option<&str>,
    preflight_ok: bool,
    preflight_error: Option<String>,
    frame_protocol: Option<&str>,
    blocker: Option<String>,
) -> RealFullSparseTransportPlan {
    RealFullSparseTransportPlan {
        transport: args.transport.clone(),
        status: status.to_owned(),
        sparse_dispatch_available,
        scheduler_dispatch_backend: scheduler_dispatch_backend.map(str::to_owned),
        supports_rdma: capabilities.supports_rdma,
        supports_host_registered_buffers: capabilities.supports_host_registered_buffers,
        requires_pinned_host_memory: capabilities.requires_pinned_host_memory,
        app_transport_implemented: capabilities.app_transport_implemented,
        app_transport_status: capabilities.app_transport_status,
        preflight_ok,
        preflight_error,
        frame_protocol: frame_protocol.map(str::to_owned),
        blocker,
    }
}

pub(super) fn real_full_kv_cache_config(args: &CoordinatorArgs) -> Result<KvCacheConfig> {
    anyhow::ensure!(
        args.max_context_tokens > 0,
        "real-glm-full --max-context-tokens must be a positive integer"
    );
    let dtype = KvCacheDType::parse_glm52_cache_dtype(&args.kv_cache_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported real-glm-full --kv-cache-dtype {}; expected bf16, fp8, or nvfp4",
            args.kv_cache_dtype
        )
    })?;
    KvCacheConfig::glm52_compressed(args.max_context_tokens, dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported real-glm-full compressed KV cache dtype {}; expected bf16, fp8, or nvfp4",
            dtype.label()
        )
    })
}
