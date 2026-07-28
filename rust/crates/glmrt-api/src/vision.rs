use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

use crate::{ApiError, RealFullVisionEmbedding};

const IMAGE_TOKEN_ID: usize = 154_854;
// Image rows are overwritten with projected vision embeddings before layer 0, so
// these valid low-vocabulary token IDs affect cache identity but not model input.
const VISION_CACHE_TOKEN_BASE: usize = 256;
const TEXT_HIDDEN_SIZE: usize = 6_144;
const MAX_IMAGES_PER_REQUEST: usize = 16;
const VISION_ENABLED_ENV: &str = "GLMRT_VISION_ENABLED";
const VISION_MODEL_ENV: &str = "GLMRT_VISION_MODEL";
const VISION_PYTHON_ENV: &str = "GLMRT_VISION_PYTHON";
const VISION_WORKER_ENV: &str = "GLMRT_VISION_WORKER";

static VISION_WORKER: OnceLock<Mutex<Option<VisionWorker>>> = OnceLock::new();

#[derive(Debug)]
pub(crate) struct PreparedVisionPrompt {
    pub(crate) prompt_token_ids: Arc<Vec<usize>>,
    pub(crate) embeddings: Arc<Vec<RealFullVisionEmbedding>>,
}

#[derive(Serialize)]
struct WorkerRequest<'a> {
    request_id: &'a str,
    images: &'a [String],
}

#[derive(Deserialize)]
struct WorkerReady {
    status: String,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct WorkerResponse {
    status: String,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    images: Vec<WorkerImage>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct WorkerImage {
    path: PathBuf,
    rows: usize,
    hidden_size: usize,
    bytes: usize,
    sha256: String,
    #[serde(default)]
    cache_hit: bool,
}

struct VisionWorker {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl VisionWorker {
    fn spawn() -> Result<Self> {
        let repo_root = repo_root();
        let model = env_path(VISION_MODEL_ENV)
            .with_context(|| format!("{VISION_MODEL_ENV} is required when vision is enabled"))?
            .canonicalize()
            .context("resolving GLM-5.2 vision checkpoint directory")?;
        for name in ["vision_tower.safetensors", "mm_projector.safetensors"] {
            anyhow::ensure!(
                model.join(name).is_file(),
                "vision checkpoint is missing {}",
                model.join(name).display()
            );
        }
        let python =
            env_path(VISION_PYTHON_ENV).unwrap_or_else(|| repo_root.join(".venv/bin/python"));
        let worker = env_path(VISION_WORKER_ENV)
            .unwrap_or_else(|| repo_root.join("python/tools/glmrt_vision_worker.py"));
        anyhow::ensure!(
            python.is_file(),
            "vision Python interpreter does not exist: {}",
            python.display()
        );
        anyhow::ensure!(
            worker.is_file(),
            "vision worker does not exist: {}",
            worker.display()
        );
        let mut child = Command::new(&python)
            .arg(&worker)
            .arg("--weights-dir")
            .arg(&model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("starting GLM-5.2 vision worker with {}", python.display()))?;
        let input = child
            .stdin
            .take()
            .context("vision worker stdin was not piped")?;
        let output = child
            .stdout
            .take()
            .context("vision worker stdout was not piped")?;
        let mut this = Self {
            child,
            input: BufWriter::new(input),
            output: BufReader::new(output),
        };
        let ready: WorkerReady = this.read_json_line("vision worker ready message")?;
        anyhow::ensure!(
            ready.status == "ready",
            "vision worker failed startup: {}",
            ready.error.unwrap_or_else(|| ready.status)
        );
        Ok(this)
    }

    fn encode(&mut self, request_id: &str, images: &[String]) -> Result<Vec<WorkerImage>> {
        if let Some(status) = self.child.try_wait().context("polling vision worker")? {
            bail!("vision worker exited before request with status {status}");
        }
        serde_json::to_writer(&mut self.input, &WorkerRequest { request_id, images })
            .context("serializing vision worker request")?;
        self.input
            .write_all(b"\n")
            .context("terminating vision worker request")?;
        self.input
            .flush()
            .context("flushing vision worker request")?;
        let response: WorkerResponse = self.read_json_line("vision worker response")?;
        anyhow::ensure!(
            response.request_id.as_deref() == Some(request_id),
            "vision worker response request id {:?} did not match {request_id}",
            response.request_id
        );
        anyhow::ensure!(
            response.status == "ok",
            "vision worker rejected request: {}",
            response.error.unwrap_or_else(|| response.status)
        );
        anyhow::ensure!(
            response.images.len() == images.len(),
            "vision worker returned {} images for {} inputs",
            response.images.len(),
            images.len()
        );
        Ok(response.images)
    }

    fn read_json_line<T: for<'de> Deserialize<'de>>(&mut self, label: &str) -> Result<T> {
        let mut line = String::new();
        let bytes = self
            .output
            .read_line(&mut line)
            .with_context(|| format!("reading {label}"))?;
        anyhow::ensure!(bytes > 0, "{label} ended at EOF");
        serde_json::from_str(&line).with_context(|| format!("parsing {label}: {line:?}"))
    }
}

pub(crate) fn vision_enabled() -> bool {
    match env::var(VISION_ENABLED_ENV) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

pub(crate) fn prepare_vision_prompt(
    initial_prompt_token_ids: Vec<usize>,
    image_sources: &[String],
) -> Result<PreparedVisionPrompt, ApiError> {
    if image_sources.is_empty() {
        return Ok(PreparedVisionPrompt {
            prompt_token_ids: Arc::new(initial_prompt_token_ids),
            embeddings: Arc::new(Vec::new()),
        });
    }
    if !vision_enabled() {
        return Err(crate::invalid_request(
            format!("image input requires {VISION_ENABLED_ENV}=1 and a vision checkpoint"),
            Some("messages"),
        ));
    }
    if image_sources.len() > MAX_IMAGES_PER_REQUEST {
        return Err(crate::invalid_request(
            format!(
                "image input count {} exceeds maximum {MAX_IMAGES_PER_REQUEST}",
                image_sources.len()
            ),
            Some("messages"),
        ));
    }
    let placeholders = initial_prompt_token_ids
        .iter()
        .filter(|token_id| **token_id == IMAGE_TOKEN_ID)
        .count();
    if placeholders != image_sources.len() {
        return Err(crate::runtime_error(format!(
            "rendered vision prompt contains {placeholders} image placeholders for {} images",
            image_sources.len()
        )));
    }

    let request_id = format!("vision-{}", Uuid::new_v4());
    let worker_images = encode_images(&request_id, image_sources).map_err(crate::runtime_error)?;
    let mut expanded = Vec::with_capacity(
        initial_prompt_token_ids.len()
            + worker_images
                .iter()
                .map(|image| image.rows.saturating_sub(1))
                .sum::<usize>(),
    );
    let mut embeddings = Vec::with_capacity(worker_images.len());
    let mut images = worker_images.into_iter();
    for token_id in initial_prompt_token_ids {
        if token_id != IMAGE_TOKEN_ID {
            expanded.push(token_id);
            continue;
        }
        let image = images
            .next()
            .expect("placeholder count was checked against image outputs");
        let hidden = read_worker_embedding(&image).map_err(crate::runtime_error)?;
        let token_start = expanded.len();
        expanded.extend(
            vision_cache_identity_token_ids(image.rows, &image.sha256)
                .map_err(crate::runtime_error)?,
        );
        eprintln!(
            "real_full_vision_image request_id={} token_start={} rows={} bytes={} cache_hit={} sha256={}",
            request_id,
            token_start,
            image.rows,
            image.bytes,
            image.cache_hit,
            image.sha256,
        );
        embeddings.push(RealFullVisionEmbedding {
            token_start,
            rows: image.rows,
            hidden_size: image.hidden_size,
            image_sha256: image.sha256,
            hidden_bf16: Arc::new(hidden),
        });
    }
    Ok(PreparedVisionPrompt {
        prompt_token_ids: Arc::new(expanded),
        embeddings: Arc::new(embeddings),
    })
}

fn vision_cache_identity_token_ids(rows: usize, sha256: &str) -> Result<Vec<usize>> {
    anyhow::ensure!(rows > 0, "vision embedding must contain at least one row");
    anyhow::ensure!(
        sha256.len() == 64,
        "vision image SHA-256 must contain 64 hexadecimal digits"
    );
    let mut token_ids = Vec::with_capacity(rows);
    for byte in sha256.bytes().take(rows) {
        let nibble = match byte {
            b'0'..=b'9' => usize::from(byte - b'0'),
            b'a'..=b'f' => usize::from(byte - b'a') + 10,
            b'A'..=b'F' => usize::from(byte - b'A') + 10,
            _ => bail!("vision image SHA-256 contains a non-hexadecimal digit"),
        };
        token_ids.push(VISION_CACHE_TOKEN_BASE + nibble);
    }
    token_ids.resize(rows, IMAGE_TOKEN_ID);
    Ok(token_ids)
}

fn encode_images(request_id: &str, image_sources: &[String]) -> Result<Vec<WorkerImage>> {
    let worker = VISION_WORKER.get_or_init(|| Mutex::new(None));
    let mut worker = worker
        .lock()
        .map_err(|error| anyhow::anyhow!("locking vision worker failed: {error}"))?;
    if worker.is_none() {
        *worker = Some(VisionWorker::spawn()?);
    }
    let result = worker
        .as_mut()
        .expect("vision worker was initialized")
        .encode(request_id, image_sources);
    if result.is_err() {
        if let Some(mut failed) = worker.take() {
            let _ = failed.child.kill();
            let _ = failed.child.wait();
        }
    }
    result
}

fn read_worker_embedding(image: &WorkerImage) -> Result<Vec<u8>> {
    let expected_bytes = image
        .rows
        .checked_mul(TEXT_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("vision embedding byte count overflows usize")?;
    anyhow::ensure!(
        image.rows > 0 && image.hidden_size == TEXT_HIDDEN_SIZE && image.bytes == expected_bytes,
        "vision embedding shape {}x{} reports {} bytes, expected {}",
        image.rows,
        image.hidden_size,
        image.bytes,
        expected_bytes
    );
    let parent = image
        .path
        .parent()
        .context("vision worker output has no parent directory")?;
    anyhow::ensure!(
        parent == Path::new("/dev/shm")
            && image
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("glmrt-vision-") && name.ends_with(".bf16")),
        "vision worker returned unsafe output path {}",
        image.path.display()
    );
    let result = fs::read(&image.path)
        .with_context(|| format!("reading vision embedding {}", image.path.display()));
    let cleanup = fs::remove_file(&image.path)
        .with_context(|| format!("removing vision embedding {}", image.path.display()));
    let hidden = result?;
    cleanup?;
    anyhow::ensure!(
        hidden.len() == expected_bytes,
        "vision embedding file has {} bytes, expected {expected_bytes}",
        hidden.len()
    );
    Ok(hidden)
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn repo_root() -> PathBuf {
    let current = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if current
        .join("python/tools/glmrt_vision_worker.py")
        .is_file()
    {
        return current;
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_repo_root_contains_worker() {
        assert!(repo_root()
            .join("python/tools/glmrt_vision_worker.py")
            .is_file());
    }

    #[test]
    fn image_sha_is_part_of_radix_cache_identity() {
        let first =
            vision_cache_identity_token_ids(1_024, &format!("{}{}", "0".repeat(63), "1")).unwrap();
        let second =
            vision_cache_identity_token_ids(1_024, &format!("{}{}", "0".repeat(63), "2")).unwrap();

        assert_ne!(first, second);
        assert_eq!(first.len(), 1_024);
        assert!(first[..64]
            .iter()
            .all(|token| (VISION_CACHE_TOKEN_BASE..VISION_CACHE_TOKEN_BASE + 16).contains(token)));
        assert!(first[64..].iter().all(|token| *token == IMAGE_TOKEN_ID));
    }

    #[test]
    fn image_cache_identity_rejects_invalid_sha() {
        assert!(vision_cache_identity_token_ids(1, "not-a-sha").is_err());
        assert!(vision_cache_identity_token_ids(1, &"g".repeat(64)).is_err());
    }
}
