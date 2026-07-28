use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo, TensorRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedTensor {
    pub info: TensorInfo,
    pub source_path: PathBuf,
    pub bytes: Vec<u8>,
    pub elapsed_micros: u128,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedTensorSummary {
    pub tensor_name: String,
    pub source_path: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub role: TensorRole,
    pub layer_id: Option<u32>,
    pub expert_id: Option<u32>,
    pub byte_offset: u64,
    pub bytes_requested: u64,
    pub bytes_read: u64,
    pub elapsed_micros: u128,
    pub read_gbps: f64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedTensorRows {
    pub info: TensorInfo,
    pub source_path: PathBuf,
    pub start_row: usize,
    pub row_count: usize,
    pub row_width: usize,
    pub bytes_per_scalar: usize,
    pub bytes: Vec<u8>,
    pub elapsed_micros: u128,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedTensorRowsSummary {
    pub tensor_name: String,
    pub source_path: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start_row: usize,
    pub row_count: usize,
    pub row_width: usize,
    pub bytes_per_scalar: usize,
    pub byte_offset: u64,
    pub bytes_read: u64,
    pub elapsed_micros: u128,
    pub read_gbps: f64,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TensorLoadOptions {
    pub compute_sha256: bool,
}

impl TensorLoadOptions {
    pub fn verify_hashes() -> Self {
        Self {
            compute_sha256: true,
        }
    }
}

pub fn load_tensor_bytes(catalog: &TensorCatalog, tensor_name: &str) -> Result<LoadedTensor> {
    load_tensor_bytes_with_options(catalog, tensor_name, TensorLoadOptions::default())
}

pub fn load_tensor_bytes_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    options: TensorLoadOptions,
) -> Result<LoadedTensor> {
    let info = find_catalog_tensor(catalog, tensor_name)
        .with_context(|| format!("tensor {tensor_name} not found in catalog"))?
        .clone();
    let source_path = Path::new(&catalog.snapshot_path).join(&info.file);
    let mut file =
        File::open(&source_path).with_context(|| format!("opening {}", source_path.display()))?;
    let mut bytes = vec![0_u8; info.byte_length as usize];
    let start = Instant::now();
    file.seek(SeekFrom::Start(info.byte_offset))
        .with_context(|| format!("seeking {} to {}", source_path.display(), info.byte_offset))?;
    file.read_exact(&mut bytes).with_context(|| {
        format!(
            "reading tensor {tensor_name} from {}",
            source_path.display()
        )
    })?;
    let elapsed_micros = start.elapsed().as_micros();
    let sha256 = maybe_sha256(&bytes, options);
    Ok(LoadedTensor {
        info,
        source_path,
        bytes,
        elapsed_micros,
        sha256,
    })
}

pub fn read_tensor_bytes_into(
    catalog: &TensorCatalog,
    tensor_name: &str,
    dst: &mut [u8],
) -> Result<LoadedTensorSummary> {
    read_tensor_bytes_into_with_options(catalog, tensor_name, dst, TensorLoadOptions::default())
}

pub fn read_tensor_bytes_into_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    dst: &mut [u8],
    options: TensorLoadOptions,
) -> Result<LoadedTensorSummary> {
    let info = find_catalog_tensor(catalog, tensor_name)
        .with_context(|| format!("tensor {tensor_name} not found in catalog"))?
        .clone();
    let bytes_to_read: usize = info
        .byte_length
        .try_into()
        .context("tensor byte length does not fit in usize")?;
    if dst.len() < bytes_to_read {
        anyhow::bail!(
            "destination buffer for tensor {tensor_name} has {} bytes, needs {bytes_to_read}",
            dst.len()
        );
    }

    let source_path = Path::new(&catalog.snapshot_path).join(&info.file);
    let mut file =
        File::open(&source_path).with_context(|| format!("opening {}", source_path.display()))?;
    let read_dst = &mut dst[..bytes_to_read];
    let start = Instant::now();
    file.seek(SeekFrom::Start(info.byte_offset))
        .with_context(|| format!("seeking {} to {}", source_path.display(), info.byte_offset))?;
    file.read_exact(read_dst).with_context(|| {
        format!(
            "reading tensor {tensor_name} from {}",
            source_path.display()
        )
    })?;
    let elapsed_micros = start.elapsed().as_micros();
    let sha256 = maybe_sha256(read_dst, options);
    Ok(LoadedTensorSummary {
        tensor_name: info.name,
        source_path: source_path.display().to_string(),
        dtype: info.dtype,
        shape: info.shape,
        role: info.role,
        layer_id: info.layer_id,
        expert_id: info.expert_id,
        byte_offset: info.byte_offset,
        bytes_requested: info.byte_length,
        bytes_read: bytes_to_read as u64,
        elapsed_micros,
        read_gbps: read_gbps(bytes_to_read as u64, elapsed_micros),
        sha256,
    })
}

pub fn load_tensor_rows(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
) -> Result<LoadedTensorRows> {
    load_tensor_rows_with_options(
        catalog,
        tensor_name,
        start_row,
        row_count,
        TensorLoadOptions::default(),
    )
}

pub fn load_tensor_rows_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    options: TensorLoadOptions,
) -> Result<LoadedTensorRows> {
    let window = tensor_row_window(catalog, tensor_name, start_row, row_count)?;
    let source_path = Path::new(&catalog.snapshot_path).join(&window.info.file);
    let mut file =
        File::open(&source_path).with_context(|| format!("opening {}", source_path.display()))?;
    let mut bytes = vec![0_u8; window.bytes_to_read];
    let start = Instant::now();
    file.seek(SeekFrom::Start(window.absolute_offset))
        .with_context(|| {
            format!(
                "seeking {} to {}",
                source_path.display(),
                window.absolute_offset
            )
        })?;
    file.read_exact(&mut bytes).with_context(|| {
        format!(
            "reading tensor {tensor_name} rows {start_row}..{} from {}",
            window.end_row,
            source_path.display()
        )
    })?;
    let elapsed_micros = start.elapsed().as_micros();
    let sha256 = maybe_sha256(&bytes, options);
    Ok(LoadedTensorRows {
        info: window.info,
        source_path,
        start_row,
        row_count,
        row_width: window.row_width,
        bytes_per_scalar: window.bytes_per_scalar,
        bytes,
        elapsed_micros,
        sha256,
    })
}

pub fn read_tensor_rows_into(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    dst: &mut [u8],
) -> Result<LoadedTensorRowsSummary> {
    read_tensor_rows_into_with_options(
        catalog,
        tensor_name,
        start_row,
        row_count,
        dst,
        TensorLoadOptions::default(),
    )
}

pub fn read_tensor_rows_into_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    dst: &mut [u8],
    options: TensorLoadOptions,
) -> Result<LoadedTensorRowsSummary> {
    let window = tensor_row_window(catalog, tensor_name, start_row, row_count)?;
    if dst.len() < window.bytes_to_read {
        anyhow::bail!(
            "destination buffer for tensor {tensor_name} rows {start_row}..{} has {} bytes, needs {}",
            window.end_row,
            dst.len(),
            window.bytes_to_read
        );
    }
    let source_path = Path::new(&catalog.snapshot_path).join(&window.info.file);
    let mut file =
        File::open(&source_path).with_context(|| format!("opening {}", source_path.display()))?;
    let read_dst = &mut dst[..window.bytes_to_read];
    let start = Instant::now();
    file.seek(SeekFrom::Start(window.absolute_offset))
        .with_context(|| {
            format!(
                "seeking {} to {}",
                source_path.display(),
                window.absolute_offset
            )
        })?;
    file.read_exact(read_dst).with_context(|| {
        format!(
            "reading tensor {tensor_name} rows {start_row}..{} from {}",
            window.end_row,
            source_path.display()
        )
    })?;
    let elapsed_micros = start.elapsed().as_micros();
    let sha256 = maybe_sha256(read_dst, options);
    Ok(LoadedTensorRowsSummary {
        tensor_name: window.info.name,
        source_path: source_path.display().to_string(),
        dtype: window.info.dtype,
        shape: window.info.shape,
        start_row,
        row_count,
        row_width: window.row_width,
        bytes_per_scalar: window.bytes_per_scalar,
        byte_offset: window.absolute_offset,
        bytes_read: window.bytes_to_read as u64,
        elapsed_micros,
        read_gbps: read_gbps(window.bytes_to_read as u64, elapsed_micros),
        sha256,
    })
}

pub fn read_tensor_row_prefix_into(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    prefix_width: usize,
    dst: &mut [u8],
) -> Result<LoadedTensorRowsSummary> {
    read_tensor_row_prefix_into_with_options(
        catalog,
        tensor_name,
        start_row,
        row_count,
        prefix_width,
        dst,
        TensorLoadOptions::default(),
    )
}

pub fn read_tensor_row_prefix_into_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    prefix_width: usize,
    dst: &mut [u8],
    options: TensorLoadOptions,
) -> Result<LoadedTensorRowsSummary> {
    if prefix_width == 0 {
        anyhow::bail!("row prefix width must be greater than zero");
    }
    let row_width = tensor_row_window(catalog, tensor_name, start_row, row_count)?.row_width;
    if prefix_width > row_width {
        anyhow::bail!(
            "row prefix width {prefix_width} exceeds tensor {tensor_name} row width {row_width}"
        );
    }
    read_tensor_row_window_into_with_options(
        catalog,
        tensor_name,
        start_row,
        row_count,
        0,
        prefix_width,
        dst,
        options,
    )
}

pub fn read_tensor_row_window_into(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    start_column: usize,
    column_count: usize,
    dst: &mut [u8],
) -> Result<LoadedTensorRowsSummary> {
    read_tensor_row_window_into_with_options(
        catalog,
        tensor_name,
        start_row,
        row_count,
        start_column,
        column_count,
        dst,
        TensorLoadOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn read_tensor_row_window_into_with_options(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
    start_column: usize,
    column_count: usize,
    dst: &mut [u8],
    options: TensorLoadOptions,
) -> Result<LoadedTensorRowsSummary> {
    if column_count == 0 {
        anyhow::bail!("row window column count must be greater than zero");
    }
    let window = tensor_row_window(catalog, tensor_name, start_row, row_count)?;
    let end_column = start_column
        .checked_add(column_count)
        .context("row window column end overflow")?;
    if end_column > window.row_width {
        anyhow::bail!(
            "row window columns {start_column}..{end_column} exceed tensor {tensor_name} row width {}",
            window.row_width
        );
    }
    let compact_row_bytes = column_count
        .checked_mul(window.bytes_per_scalar)
        .context("row window byte width overflow")?;
    let bytes_to_read = row_count
        .checked_mul(compact_row_bytes)
        .context("row window byte length overflow")?;
    if dst.len() < bytes_to_read {
        anyhow::bail!(
            "destination buffer for tensor {tensor_name} rows {start_row}..{} columns {start_column}..{end_column} has {} bytes, needs {bytes_to_read}",
            window.end_row,
            dst.len()
        );
    }
    let full_row_bytes = window
        .row_width
        .checked_mul(window.bytes_per_scalar)
        .context("full row byte width overflow")?;
    let column_offset_bytes = start_column
        .checked_mul(window.bytes_per_scalar)
        .context("row window column byte offset overflow")?;
    let first_byte_offset = window
        .absolute_offset
        .checked_add(column_offset_bytes as u64)
        .context("row window absolute byte offset overflow")?;
    let source_path = Path::new(&catalog.snapshot_path).join(&window.info.file);
    let mut file =
        File::open(&source_path).with_context(|| format!("opening {}", source_path.display()))?;
    let read_dst = &mut dst[..bytes_to_read];
    let start = Instant::now();
    for row_offset in 0..row_count {
        let source_offset = first_byte_offset
            .checked_add(
                row_offset
                    .checked_mul(full_row_bytes)
                    .context("row window source offset overflow")? as u64,
            )
            .context("row window absolute source offset overflow")?;
        let dst_start = row_offset
            .checked_mul(compact_row_bytes)
            .context("row window destination offset overflow")?;
        let dst_end = dst_start
            .checked_add(compact_row_bytes)
            .context("row window destination end overflow")?;
        file.seek(SeekFrom::Start(source_offset))
            .with_context(|| format!("seeking {} to {source_offset}", source_path.display()))?;
        file.read_exact(&mut read_dst[dst_start..dst_end])
            .with_context(|| {
                format!(
                    "reading tensor {tensor_name} row {} columns {start_column}..{end_column} from {}",
                    start_row + row_offset,
                    source_path.display()
                )
            })?;
    }
    let elapsed_micros = start.elapsed().as_micros();
    let sha256 = maybe_sha256(read_dst, options);
    Ok(LoadedTensorRowsSummary {
        tensor_name: window.info.name,
        source_path: source_path.display().to_string(),
        dtype: window.info.dtype,
        shape: window.info.shape,
        start_row,
        row_count,
        row_width: column_count,
        bytes_per_scalar: window.bytes_per_scalar,
        byte_offset: first_byte_offset,
        bytes_read: bytes_to_read as u64,
        elapsed_micros,
        read_gbps: read_gbps(bytes_to_read as u64, elapsed_micros),
        sha256,
    })
}

struct TensorRowWindow {
    info: TensorInfo,
    end_row: usize,
    row_width: usize,
    bytes_per_scalar: usize,
    bytes_to_read: usize,
    absolute_offset: u64,
}

fn tensor_row_window(
    catalog: &TensorCatalog,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
) -> Result<TensorRowWindow> {
    if row_count == 0 {
        anyhow::bail!("row_count must be greater than zero");
    }
    let info = find_catalog_tensor(catalog, tensor_name)
        .with_context(|| format!("tensor {tensor_name} not found in catalog"))?
        .clone();
    if info.shape.len() != 2 {
        anyhow::bail!(
            "tensor {tensor_name} must be rank-2 for row loading, got shape {:?}",
            info.shape
        );
    }
    let rows = info.shape[0];
    let row_width = info.shape[1];
    let end_row = start_row
        .checked_add(row_count)
        .context("row window end overflow")?;
    if start_row >= rows || end_row > rows {
        anyhow::bail!(
            "row window [{start_row}, {}) exceeds tensor {tensor_name} row count {rows}",
            end_row
        );
    }
    let bytes_per_scalar = dtype_byte_width(&info.dtype)?;
    let row_bytes = row_width
        .checked_mul(bytes_per_scalar)
        .context("row byte width overflow")?;
    let relative_offset = start_row
        .checked_mul(row_bytes)
        .context("row byte offset overflow")?;
    let bytes_to_read = row_count
        .checked_mul(row_bytes)
        .context("row byte length overflow")?;
    let end_offset = relative_offset
        .checked_add(bytes_to_read)
        .context("row byte end offset overflow")?;
    let tensor_byte_length: usize = info
        .byte_length
        .try_into()
        .context("tensor byte length does not fit in usize")?;
    if end_offset > tensor_byte_length {
        anyhow::bail!(
            "row window for tensor {tensor_name} exceeds recorded byte length {}",
            info.byte_length
        );
    }
    let absolute_offset = info.byte_offset + relative_offset as u64;
    Ok(TensorRowWindow {
        info,
        end_row,
        row_width,
        bytes_per_scalar,
        bytes_to_read,
        absolute_offset,
    })
}

fn find_catalog_tensor<'a>(
    catalog: &'a TensorCatalog,
    tensor_name: &str,
) -> Option<&'a TensorInfo> {
    catalog
        .tensors
        .binary_search_by(|tensor| tensor.name.as_str().cmp(tensor_name))
        .ok()
        .and_then(|index| catalog.tensors.get(index))
        .or_else(|| {
            catalog
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
        })
}

fn maybe_sha256(bytes: &[u8], options: TensorLoadOptions) -> String {
    if options.compute_sha256 {
        format!("{:x}", Sha256::digest(bytes))
    } else {
        String::new()
    }
}

fn read_gbps(bytes_read: u64, elapsed_micros: u128) -> f64 {
    let elapsed_secs = (elapsed_micros as f64 / 1_000_000.0).max(1.0e-9);
    bytes_read as f64 / elapsed_secs / 1.0e9
}

pub fn dtype_byte_width(dtype: &DType) -> Result<usize> {
    match dtype {
        DType::Bf16 | DType::F16 | DType::I16 => Ok(2),
        DType::F32 | DType::I32 => Ok(4),
        DType::I8 | DType::U8 | DType::F8E4M3 | DType::F8E5M2 => Ok(1),
        DType::F4 => anyhow::bail!("packed F4 row loading needs explicit packing metadata"),
        DType::Unknown(value) => anyhow::bail!("unknown dtype byte width: {value}"),
    }
}

impl LoadedTensor {
    pub fn summary(&self) -> LoadedTensorSummary {
        let elapsed_secs = (self.elapsed_micros as f64 / 1_000_000.0).max(1.0e-9);
        let bytes_read = self.bytes.len() as u64;
        LoadedTensorSummary {
            tensor_name: self.info.name.clone(),
            source_path: self.source_path.display().to_string(),
            dtype: self.info.dtype.clone(),
            shape: self.info.shape.clone(),
            role: self.info.role.clone(),
            layer_id: self.info.layer_id,
            expert_id: self.info.expert_id,
            byte_offset: self.info.byte_offset,
            bytes_requested: self.info.byte_length,
            bytes_read,
            elapsed_micros: self.elapsed_micros,
            read_gbps: bytes_read as f64 / elapsed_secs / 1.0e9,
            sha256: self.sha256.clone(),
        }
    }
}

impl LoadedTensorRows {
    pub fn summary(&self) -> LoadedTensorRowsSummary {
        let elapsed_secs = (self.elapsed_micros as f64 / 1_000_000.0).max(1.0e-9);
        let bytes_read = self.bytes.len() as u64;
        let row_byte_offset = self.info.byte_offset
            + (self.start_row * self.row_width * self.bytes_per_scalar) as u64;
        LoadedTensorRowsSummary {
            tensor_name: self.info.name.clone(),
            source_path: self.source_path.display().to_string(),
            dtype: self.info.dtype.clone(),
            shape: self.info.shape.clone(),
            start_row: self.start_row,
            row_count: self.row_count,
            row_width: self.row_width,
            bytes_per_scalar: self.bytes_per_scalar,
            byte_offset: row_byte_offset,
            bytes_read,
            elapsed_micros: self.elapsed_micros,
            read_gbps: bytes_read as f64 / elapsed_secs / 1.0e9,
            sha256: self.sha256.clone(),
        }
    }
}
