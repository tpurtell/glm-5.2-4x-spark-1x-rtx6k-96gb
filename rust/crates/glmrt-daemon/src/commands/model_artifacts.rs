use anyhow::{Context, Result};
use glmrt_core::{
    owner_for_expert, PlacementPolicy, TensorCatalog, TensorRole, EXPERT_HOSTS, SUPPORTED_MODEL_IDS,
};
use glmrt_loader::{
    build_catalog, build_catalog_for_snapshot, build_load_plan, classification_summary_markdown,
    encode_tokenizer_text, load_tensor_bytes_with_options, resolve_snapshot, LoadedTensorSummary,
    TensorLoadOptions,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use crate::cli::{InspectModelArgs, LoadTensorsArgs, MakeLoadPlanArgs, TokenizeArgs};

mod loadplan;

pub(crate) use glmrt_core::ExpertOwnerLookup;
pub(crate) use loadplan::read_expert_owner_lookup;
use loadplan::write_node_load_plans;
pub(crate) use loadplan::{read_expert_loadplan, read_expert_serving_loadplan};

const RUNTIME_CATALOG_CACHE_DIR_ENV: &str = "GLMRT_RUNTIME_CATALOG_CACHE_DIR";
const RUNTIME_CATALOG_CACHE_SCHEMA: u32 = 2;
const RUNTIME_CATALOG_SOURCE_IDENTITY_SCHEMA: &[u8] = b"glmrt-runtime-catalog-v2";
const RUNTIME_CATALOG_QUANTIZATION_CONFIG_FILES: [&str; 2] =
    ["quantize_config.json", "quantization_config.json"];

#[derive(Debug, Deserialize)]
struct RuntimeCatalogSafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct RuntimeCatalogCacheManifest {
    schema: u32,
    model_id: String,
    snapshot_path: String,
    source_identity: String,
    payload_bytes: u64,
    payload_sha256: String,
}

pub(crate) fn build_runtime_catalog(model_id: &str) -> Result<TensorCatalog> {
    anyhow::ensure!(
        SUPPORTED_MODEL_IDS.contains(&model_id),
        "unsupported production checkpoint {model_id:?}; supported checkpoints: {}",
        SUPPORTED_MODEL_IDS.join(", ")
    );
    let Some(cache_dir) = env::var_os(RUNTIME_CATALOG_CACHE_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        return build_catalog(model_id, None)
            .with_context(|| format!("building runtime tensor catalog for {model_id}"));
    };

    let resolution = resolve_snapshot(model_id, None)?;
    let snapshot_path = resolution
        .snapshot_path
        .as_deref()
        .with_context(|| format!("no local snapshot found for {model_id}"))?;
    let identity_started = Instant::now();
    let source_identity = runtime_catalog_source_identity(model_id, snapshot_path)?;
    let identity_ms = identity_started.elapsed().as_secs_f64() * 1_000.0;
    let load_started = Instant::now();
    match read_runtime_catalog_cache(&cache_dir, model_id, snapshot_path, &source_identity) {
        Ok(Some(catalog)) => {
            eprintln!(
                "runtime_catalog_cache status=hit identity={} identity_ms={identity_ms:.3} load_ms={:.3} tensors={} path={}",
                source_identity,
                load_started.elapsed().as_secs_f64() * 1_000.0,
                catalog.tensors.len(),
                cache_dir.display(),
            );
            return Ok(catalog);
        }
        Ok(None) => {}
        Err(error) => eprintln!(
            "runtime_catalog_cache status=invalid identity={} path={} error={error:#}",
            source_identity,
            cache_dir.display(),
        ),
    }

    let build_started = Instant::now();
    let catalog = build_catalog_for_snapshot(model_id, snapshot_path)
        .with_context(|| format!("building runtime tensor catalog for {model_id}"))?;
    let build_ms = build_started.elapsed().as_secs_f64() * 1_000.0;
    let write_started = Instant::now();
    if let Err(error) = write_runtime_catalog_cache(
        &cache_dir,
        model_id,
        snapshot_path,
        &source_identity,
        &catalog,
    ) {
        eprintln!(
            "runtime_catalog_cache status=write-failed identity={} path={} error={error:#}",
            source_identity,
            cache_dir.display(),
        );
    } else {
        eprintln!(
            "runtime_catalog_cache status=miss identity={} identity_ms={identity_ms:.3} build_ms={build_ms:.3} write_ms={:.3} tensors={} path={}",
            source_identity,
            write_started.elapsed().as_secs_f64() * 1_000.0,
            catalog.tensors.len(),
            cache_dir.display(),
        );
    }
    Ok(catalog)
}

fn runtime_catalog_source_identity(model_id: &str, snapshot_path: &Path) -> Result<String> {
    let config_path = snapshot_path.join("config.json");
    let index_path = snapshot_path.join("model.safetensors.index.json");
    let config = fs::read(&config_path).with_context(|| {
        format!(
            "reading runtime catalog identity source {}",
            config_path.display()
        )
    })?;
    let index_bytes = fs::read(&index_path).with_context(|| {
        format!(
            "reading runtime catalog identity source {}",
            index_path.display()
        )
    })?;
    let index: RuntimeCatalogSafetensorsIndex =
        serde_json::from_slice(&index_bytes).with_context(|| {
            format!(
                "parsing runtime catalog identity source {}",
                index_path.display()
            )
        })?;
    let files = index.weight_map.into_values().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        !files.is_empty(),
        "runtime catalog index contains no weight shards"
    );

    let mut hasher = Sha256::new();
    update_catalog_identity_field(&mut hasher, RUNTIME_CATALOG_SOURCE_IDENTITY_SCHEMA);
    update_catalog_identity_field(&mut hasher, model_id.as_bytes());
    update_catalog_identity_field(&mut hasher, snapshot_path.as_os_str().as_encoded_bytes());
    update_catalog_identity_field(&mut hasher, &config);
    update_catalog_identity_field(&mut hasher, &index_bytes);
    for file_name in RUNTIME_CATALOG_QUANTIZATION_CONFIG_FILES {
        let path = snapshot_path.join(file_name);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).with_context(|| {
            format!("reading runtime catalog identity source {}", path.display())
        })?;
        update_catalog_identity_field(&mut hasher, file_name.as_bytes());
        update_catalog_identity_field(&mut hasher, &bytes);
    }
    for file_name in files {
        let shard_path = snapshot_path.join(&file_name);
        let mut shard = File::open(&shard_path).with_context(|| {
            format!(
                "opening runtime catalog identity shard {}",
                shard_path.display()
            )
        })?;
        let mut header_length_bytes = [0_u8; 8];
        shard
            .read_exact(&mut header_length_bytes)
            .with_context(|| {
                format!("reading safetensors header length {}", shard_path.display())
            })?;
        let header_length = u64::from_le_bytes(header_length_bytes);
        let shard_length = shard.metadata()?.len();
        anyhow::ensure!(
            header_length <= shard_length.saturating_sub(8),
            "invalid safetensors header length {header_length} for {} bytes in {}",
            shard_length,
            shard_path.display(),
        );
        let header_length = usize::try_from(header_length)
            .context("safetensors header length does not fit usize")?;
        let mut header = vec![0_u8; header_length];
        shard
            .read_exact(&mut header)
            .with_context(|| format!("reading safetensors header {}", shard_path.display()))?;
        update_catalog_identity_field(&mut hasher, file_name.as_bytes());
        update_catalog_identity_field(&mut hasher, &header_length_bytes);
        update_catalog_identity_field(&mut hasher, &header);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn update_catalog_identity_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn runtime_catalog_cache_paths(cache_dir: &Path, source_identity: &str) -> (PathBuf, PathBuf) {
    (
        cache_dir.join(format!("{source_identity}.catalog.bin")),
        cache_dir.join(format!("{source_identity}.manifest.json")),
    )
}

fn read_runtime_catalog_cache(
    cache_dir: &Path,
    model_id: &str,
    snapshot_path: &Path,
    source_identity: &str,
) -> Result<Option<TensorCatalog>> {
    let (payload_path, manifest_path) = runtime_catalog_cache_paths(cache_dir, source_identity);
    if !payload_path.is_file() || !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: RuntimeCatalogCacheManifest =
        serde_json::from_reader(File::open(&manifest_path).with_context(|| {
            format!(
                "opening runtime catalog manifest {}",
                manifest_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "parsing runtime catalog manifest {}",
                manifest_path.display()
            )
        })?;
    anyhow::ensure!(
        manifest.schema == RUNTIME_CATALOG_CACHE_SCHEMA,
        "runtime catalog cache schema mismatch"
    );
    anyhow::ensure!(
        manifest.model_id == model_id,
        "runtime catalog cache model mismatch"
    );
    anyhow::ensure!(
        manifest.snapshot_path == snapshot_path.display().to_string(),
        "runtime catalog cache snapshot mismatch"
    );
    anyhow::ensure!(
        manifest.source_identity == source_identity,
        "runtime catalog cache identity mismatch"
    );
    let payload = fs::read(&payload_path)
        .with_context(|| format!("reading runtime catalog payload {}", payload_path.display()))?;
    anyhow::ensure!(
        payload.len() as u64 == manifest.payload_bytes,
        "runtime catalog cache payload length mismatch"
    );
    let payload_sha256 = format!("{:x}", Sha256::digest(&payload));
    anyhow::ensure!(
        payload_sha256 == manifest.payload_sha256,
        "runtime catalog cache payload hash mismatch"
    );
    let catalog: TensorCatalog = bincode::deserialize(&payload)
        .with_context(|| format!("parsing runtime catalog payload {}", payload_path.display()))?;
    anyhow::ensure!(
        catalog.model_id == model_id,
        "cached runtime catalog model mismatch"
    );
    anyhow::ensure!(
        catalog.snapshot_path == snapshot_path.display().to_string(),
        "cached runtime catalog snapshot mismatch"
    );
    Ok(Some(catalog))
}

fn write_runtime_catalog_cache(
    cache_dir: &Path,
    model_id: &str,
    snapshot_path: &Path,
    source_identity: &str,
    catalog: &TensorCatalog,
) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating runtime catalog cache {}", cache_dir.display()))?;
    // The versioned manifest supplies the schema and integrity envelope, so the
    // local cache payload can use Serde's compact binary representation.
    let payload = bincode::serialize(catalog).context("serializing runtime catalog cache")?;
    let manifest = RuntimeCatalogCacheManifest {
        schema: RUNTIME_CATALOG_CACHE_SCHEMA,
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        source_identity: source_identity.to_owned(),
        payload_bytes: payload.len() as u64,
        payload_sha256: format!("{:x}", Sha256::digest(&payload)),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .context("serializing runtime catalog cache manifest")?;
    let (payload_path, manifest_path) = runtime_catalog_cache_paths(cache_dir, source_identity);
    write_atomic_runtime_catalog_cache_file(&payload_path, &payload)?;
    write_atomic_runtime_catalog_cache_file(&manifest_path, &manifest_bytes)?;
    Ok(())
}

fn write_atomic_runtime_catalog_cache_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}.tmp.{}",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("cache"),
        std::process::id(),
    ));
    let mut file = File::create(&temporary)
        .with_context(|| format!("creating runtime catalog cache {}", temporary.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing runtime catalog cache {}", temporary.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing runtime catalog cache {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "installing runtime catalog cache {} -> {}",
            temporary.display(),
            path.display()
        )
    })?;
    Ok(())
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

#[cfg(test)]
mod runtime_catalog_cache_tests {
    use super::*;
    use glmrt_core::{DType, ModelFacts, TensorInfo};
    use tempfile::tempdir;

    fn test_catalog(model_id: &str, snapshot_path: &Path) -> TensorCatalog {
        TensorCatalog {
            model_id: model_id.to_owned(),
            snapshot_path: snapshot_path.display().to_string(),
            facts: ModelFacts {
                model_id: model_id.to_owned(),
                ..ModelFacts::default()
            },
            tensors: vec![TensorInfo {
                name: "model.layers.3.mlp.experts.0.gate_proj.weight".to_owned(),
                file: "model-00001-of-00001.safetensors".to_owned(),
                dtype: DType::F4,
                shape: vec![8, 8],
                byte_offset: 16,
                byte_length: 32,
                role: TensorRole::RoutedExpert,
                layer_id: Some(3),
                expert_id: Some(0),
                is_quantization_metadata: false,
            }],
        }
    }

    #[test]
    fn runtime_catalog_cache_validates_payload_integrity() {
        let temporary = tempdir().expect("temporary directory");
        let snapshot_path = temporary.path().join("snapshot");
        let cache_dir = temporary.path().join("cache");
        fs::create_dir_all(&snapshot_path).expect("snapshot directory");
        let model_id = SUPPORTED_MODEL_IDS[0];
        let source_identity = "a".repeat(64);
        let catalog = test_catalog(model_id, &snapshot_path);

        write_runtime_catalog_cache(
            &cache_dir,
            model_id,
            &snapshot_path,
            &source_identity,
            &catalog,
        )
        .expect("write cache");
        let loaded =
            read_runtime_catalog_cache(&cache_dir, model_id, &snapshot_path, &source_identity)
                .expect("read cache")
                .expect("cache hit");
        assert_eq!(loaded.tensors.len(), 1);
        assert_eq!(loaded.tensors[0].name, catalog.tensors[0].name);

        let (payload_path, _) = runtime_catalog_cache_paths(&cache_dir, &source_identity);
        let mut payload = fs::read(&payload_path).expect("cached payload");
        payload[0] ^= 1;
        fs::write(&payload_path, payload).expect("corrupt cached payload");
        let error =
            read_runtime_catalog_cache(&cache_dir, model_id, &snapshot_path, &source_identity)
                .expect_err("corrupt cache must fail closed");
        assert!(error.to_string().contains("payload hash mismatch"));
    }

    #[test]
    fn runtime_catalog_identity_changes_with_safetensors_header() {
        let temporary = tempdir().expect("temporary directory");
        let snapshot_path = temporary.path().join("snapshot");
        fs::create_dir_all(&snapshot_path).expect("snapshot directory");
        fs::write(snapshot_path.join("config.json"), b"{}\n").expect("config");
        fs::write(
            snapshot_path.join("model.safetensors.index.json"),
            br#"{"weight_map":{"tensor":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("index");
        let shard_path = snapshot_path.join("model-00001-of-00001.safetensors");
        let write_shard = |header: &[u8]| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.push(0);
            fs::write(&shard_path, bytes).expect("shard");
        };
        write_shard(b"{}");
        let first = runtime_catalog_source_identity(SUPPORTED_MODEL_IDS[0], &snapshot_path)
            .expect("first identity");
        write_shard(b"{ }");
        let second = runtime_catalog_source_identity(SUPPORTED_MODEL_IDS[0], &snapshot_path)
            .expect("second identity");

        assert_ne!(first, second);
    }

    #[test]
    fn runtime_catalog_identity_changes_with_external_quantization_config() {
        let temporary = tempdir().expect("temporary directory");
        let snapshot_path = temporary.path().join("snapshot");
        fs::create_dir_all(&snapshot_path).expect("snapshot directory");
        fs::write(snapshot_path.join("config.json"), b"{}\n").expect("config");
        fs::write(
            snapshot_path.join("model.safetensors.index.json"),
            br#"{"weight_map":{"tensor":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("index");
        let shard_path = snapshot_path.join("model-00001-of-00001.safetensors");
        let mut shard = 2_u64.to_le_bytes().to_vec();
        shard.extend_from_slice(b"{}");
        shard.push(0);
        fs::write(shard_path, shard).expect("shard");
        let quantization_path = snapshot_path.join("quantize_config.json");
        fs::write(&quantization_path, b"{\"recipe\":1}\n").expect("quantization config");
        let first = runtime_catalog_source_identity(SUPPORTED_MODEL_IDS[0], &snapshot_path)
            .expect("first identity");
        fs::write(&quantization_path, b"{\"recipe\":2}\n").expect("quantization config");
        let second = runtime_catalog_source_identity(SUPPORTED_MODEL_IDS[0], &snapshot_path)
            .expect("second identity");

        assert_ne!(first, second);
    }
}
