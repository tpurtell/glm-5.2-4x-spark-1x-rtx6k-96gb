use anyhow::{Context, Result};
use glmrt_core::{KvCacheAllocator, KvCacheConfig, KvCacheDType};

use crate::cli::CoordinatorArgs;
use crate::commands::real_full::{load_real_full_serving, run_real_glm_full_preflight};
use crate::python_graph_capture::{
    finish_coordinator_python_capture_startup, initialize_coordinator_python_capture_from_env,
};

pub(crate) async fn run_coordinator(args: CoordinatorArgs) -> Result<()> {
    if matches!(args.backend.as_str(), "real-glm-full" | "cuda-reference") && args.preflight_only {
        return run_real_glm_full_preflight(&args);
    }
    let backend = match args.backend.as_str() {
        "tiny" | "synthetic-glm-layer" | "real-glm-full" => {
            glmrt_api::ApiBackend::parse(&args.backend).expect("matched coordinator backend parses")
        }
        "cuda-reference" => glmrt_api::ApiBackend::RealGlmFull,
        "real-glm-slice" => anyhow::bail!(
            "real-glm-slice coordinator probes were superseded by real-glm-full execution stepper coverage"
        ),
        other => anyhow::bail!("unsupported coordinator backend: {other}"),
    };
    let transport = glmrt_api::ApiTransport::parse(&args.transport)
        .ok_or_else(|| anyhow::anyhow!("unsupported coordinator transport: {}", args.transport))?;
    let python_capture = if args.backend == "cuda-reference" {
        None
    } else {
        initialize_coordinator_python_capture_from_env()
            .context("initializing coordinator Python graph-capture bridge")?
    };
    let expert_targets = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let real_full_serving = if backend == glmrt_api::ApiBackend::RealGlmFull {
        Some(load_real_full_serving(&args)?)
    } else {
        None
    };
    finish_coordinator_python_capture_startup();
    let api_config = glmrt_api::ApiConfig {
        backend,
        transport,
        model_id: args.model_id.clone(),
        expert_targets,
        real_slice: None,
        real_full: real_full_serving
            .as_ref()
            .map(|serving| serving.info.clone()),
        real_full_executor: real_full_serving.map(|serving| serving.executor),
    };
    let kv_allocator = KvCacheAllocator::new(coordinator_kv_cache_config(&args)?);
    let kv_snapshot = kv_allocator.snapshot();
    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("binding coordinator API to {}", args.listen))?;
    println!(
        "starting coordinator backend={} transport={} model_id={} expert_hosts={} listen={}",
        args.backend, args.transport, args.model_id, args.expert_hosts, args.listen
    );
    if let Some(status) = python_capture.as_ref() {
        println!(
            "python_graph_capture status=enabled gate={} modules={}",
            status.gate_env,
            status.imported_modules.join(",")
        );
    }
    println!(
        "kv_cache layout={:?} dtype={:?} layers={} key_value_width={} dsa_indexer_layers={} dsa_index_head_dim={} max_tokens={} bytes_per_token={} capacity_bytes={}",
        kv_snapshot.config.layout,
        kv_snapshot.config.dtype,
        kv_snapshot.config.layers,
        kv_snapshot.config.key_value_width,
        kv_snapshot.config.dsa_indexer_layers,
        kv_snapshot.config.dsa_index_head_dim,
        kv_snapshot.config.max_tokens,
        kv_snapshot.bytes_per_token,
        kv_snapshot.capacity_bytes
    );
    axum::serve(listener, glmrt_api::router_with_config(api_config)).await?;
    Ok(())
}

fn coordinator_kv_cache_config(args: &CoordinatorArgs) -> Result<KvCacheConfig> {
    anyhow::ensure!(
        args.max_context_tokens > 0,
        "coordinator --max-context-tokens must be a positive integer"
    );
    let dtype = KvCacheDType::parse_glm52_cache_dtype(&args.kv_cache_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported coordinator --kv-cache-dtype {}; expected bf16, fp8, or nvfp4",
            args.kv_cache_dtype
        )
    })?;
    KvCacheConfig::glm52_compressed(args.max_context_tokens, dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported GLM-5.2 compressed KV cache dtype {}; expected bf16, fp8, or nvfp4",
            dtype.label()
        )
    })
}
