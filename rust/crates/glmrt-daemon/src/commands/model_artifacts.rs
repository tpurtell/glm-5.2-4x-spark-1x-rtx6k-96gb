use anyhow::{Context, Result};
use glmrt_core::{
    owner_for_expert, PlacementPolicy, TensorCatalog, TensorRole, EXPERT_HOSTS, SUPPORTED_MODEL_IDS,
};
use glmrt_loader::{
    build_catalog, build_load_plan, classification_summary_markdown, encode_tokenizer_text,
    load_tensor_bytes_with_options, resolve_snapshot, LoadedTensorSummary, TensorLoadOptions,
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs::{self, File},
    str::FromStr,
};

use crate::cli::{InspectModelArgs, LoadTensorsArgs, MakeLoadPlanArgs, TokenizeArgs};

mod loadplan;

pub(crate) use glmrt_core::ExpertOwnerLookup;
pub(crate) use loadplan::read_expert_owner_lookup;
use loadplan::write_node_load_plans;
pub(crate) use loadplan::{read_expert_loadplan, read_expert_serving_loadplan};

pub(crate) fn build_runtime_catalog(model_id: &str) -> Result<TensorCatalog> {
    anyhow::ensure!(
        SUPPORTED_MODEL_IDS.contains(&model_id),
        "unsupported production checkpoint {model_id:?}; supported checkpoints: {}",
        SUPPORTED_MODEL_IDS.join(", ")
    );
    build_catalog(model_id, None)
        .with_context(|| format!("building runtime tensor catalog for {model_id}"))
}

pub(crate) fn build_runtime_owner_lookup(catalog: &TensorCatalog) -> Result<ExpertOwnerLookup> {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|host| (*host).to_owned())
        .collect::<Vec<_>>();
    let experts =
        catalog
            .tensors
            .iter()
            .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
            .map(|tensor| {
                Ok((
                    tensor.layer_id.with_context(|| {
                        format!("routed tensor missing layer id: {}", tensor.name)
                    })? as usize,
                    tensor.expert_id.with_context(|| {
                        format!("routed tensor missing expert id: {}", tensor.name)
                    })? as usize,
                ))
            })
            .collect::<Result<BTreeSet<_>>>()?;
    anyhow::ensure!(
        !experts.is_empty(),
        "runtime checkpoint {} contains no routed experts",
        catalog.model_id
    );
    Ok(ExpertOwnerLookup::from_pairs(experts.into_iter().map(
        |(layer_id, expert_id)| {
            let owner = owner_for_expert(layer_id, expert_id, &hosts, PlacementPolicy::Modulo)
                .expect("runtime expert hosts are non-empty");
            ((layer_id, expert_id), owner)
        },
    )))
}

pub(crate) fn validate_runtime_expert_role(role: Option<&str>) -> Result<&str> {
    let role = role.context(
        "inferred runtime placement requires --role spark-0, spark-1, spark-2, or spark-3",
    )?;
    anyhow::ensure!(
        EXPERT_HOSTS.contains(&role),
        "unknown inferred runtime role {role:?}; expected one of {}",
        EXPERT_HOSTS.join(", ")
    );
    Ok(role)
}

pub(crate) fn run_inspect_model(args: InspectModelArgs) -> Result<()> {
    let catalog = build_catalog(&args.model_id, None)?;
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.summary.parent() {
        fs::create_dir_all(parent)?;
    }
    serde_json::to_writer(File::create(&args.out)?, &catalog)
        .with_context(|| format!("writing {}", args.out.display()))?;
    fs::write(&args.summary, classification_summary_markdown(&catalog))
        .with_context(|| format!("writing {}", args.summary.display()))?;
    println!(
        "wrote catalog tensors={} hash={} out={} summary={}",
        catalog.tensors.len(),
        catalog.content_hash(),
        args.out.display(),
        args.summary.display()
    );
    Ok(())
}

pub(crate) fn run_tokenize(args: TokenizeArgs) -> Result<()> {
    let resolution = resolve_snapshot(&args.model_id, args.hf_home.as_deref())?;
    let snapshot_path = resolution
        .snapshot_path
        .as_ref()
        .with_context(|| format!("no local snapshot found for {}", args.model_id))?;
    let summary = encode_tokenizer_text(snapshot_path, &args.text, args.add_special_tokens)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub(crate) fn run_make_loadplan(args: MakeLoadPlanArgs) -> Result<()> {
    let catalog: TensorCatalog = serde_json::from_reader(
        File::open(&args.catalog).with_context(|| format!("opening {}", args.catalog.display()))?,
    )
    .with_context(|| format!("parsing {}", args.catalog.display()))?;
    let policy = PlacementPolicy::from_str(&args.policy)?;
    let hosts = args
        .hosts
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let plan = build_load_plan(&catalog, policy, hosts)?;
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    serde_json::to_writer(File::create(&args.out)?, &plan)
        .with_context(|| format!("writing {}", args.out.display()))?;
    write_node_load_plans(&args.out, &plan)?;
    println!(
        "wrote loadplan assignments={} placement_version={} out={}",
        plan.assignments.len(),
        plan.placement_version,
        args.out.display()
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct LoadTensorsReport {
    catalog: String,
    verify_hashes: bool,
    tensor_count: usize,
    total_bytes_read: u64,
    total_elapsed_micros: u128,
    tensors: Vec<LoadedTensorSummary>,
}

pub(crate) fn run_load_tensors(args: LoadTensorsArgs) -> Result<()> {
    let catalog: TensorCatalog = serde_json::from_reader(
        File::open(&args.catalog).with_context(|| format!("opening {}", args.catalog.display()))?,
    )
    .with_context(|| format!("parsing {}", args.catalog.display()))?;
    let tensor_names = if args.tensors.is_empty() {
        default_load_tensor_names(&catalog)
    } else {
        args.tensors
    };
    if tensor_names.is_empty() {
        anyhow::bail!("no tensors selected for loading");
    }

    let mut summaries = Vec::with_capacity(tensor_names.len());
    let load_options = if args.verify_hashes {
        TensorLoadOptions::verify_hashes()
    } else {
        TensorLoadOptions::default()
    };
    for tensor_name in tensor_names {
        let loaded = load_tensor_bytes_with_options(&catalog, &tensor_name, load_options)?;
        let summary = loaded.summary();
        let sha256 = if summary.sha256.is_empty() {
            "disabled"
        } else {
            summary.sha256.as_str()
        };
        println!(
            "loaded tensor={} bytes={} elapsed_us={} read_gbps={:.3} sha256={}",
            summary.tensor_name,
            summary.bytes_read,
            summary.elapsed_micros,
            summary.read_gbps,
            sha256
        );
        summaries.push(summary);
    }
    let total_bytes_read = summaries.iter().map(|summary| summary.bytes_read).sum();
    let total_elapsed_micros = summaries.iter().map(|summary| summary.elapsed_micros).sum();
    let report = LoadTensorsReport {
        catalog: args.catalog.display().to_string(),
        verify_hashes: args.verify_hashes,
        tensor_count: summaries.len(),
        total_bytes_read,
        total_elapsed_micros,
        tensors: summaries,
    };
    if let Some(parent) = args.summary.parent() {
        fs::create_dir_all(parent)?;
    }
    serde_json::to_writer_pretty(File::create(&args.summary)?, &report)
        .with_context(|| format!("writing {}", args.summary.display()))?;
    println!(
        "wrote load tensor summary count={} bytes={} out={}",
        report.tensor_count,
        report.total_bytes_read,
        args.summary.display()
    );
    Ok(())
}

pub(crate) fn default_load_tensor_names(catalog: &TensorCatalog) -> Vec<String> {
    let selectors: [Box<dyn Fn(&glmrt_core::TensorInfo) -> bool>; 4] = [
        Box::new(|tensor| tensor.role == TensorRole::Norm),
        Box::new(|tensor| tensor.role == TensorRole::Router),
        Box::new(|tensor| {
            tensor.role == TensorRole::RoutedExpert
                && tensor.layer_id == Some(3)
                && tensor.expert_id == Some(0)
                && !tensor.is_quantization_metadata
        }),
        Box::new(|tensor| {
            tensor.role == TensorRole::SharedExpert && !tensor.is_quantization_metadata
        }),
    ];
    let mut names = Vec::new();
    for selector in selectors {
        if let Some(tensor) = catalog.tensors.iter().find(|tensor| selector(tensor)) {
            if !names.contains(&tensor.name) {
                names.push(tensor.name.clone());
            }
        }
    }
    names
}
