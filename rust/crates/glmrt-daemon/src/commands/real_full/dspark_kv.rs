use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum DsparkKvStorage {
    Bf16,
    Fp8,
}

impl DsparkKvStorage {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "fp8" | "f8" | "e4m3" => Some(Self::Fp8),
            _ => None,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }

    pub(super) fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkPagedKvMetadataBuffers {
    pub(super) query_indptr: GlmrtDeviceBuffer,
    pub(super) kv_indptr: GlmrtDeviceBuffer,
    pub(super) page_indices: GlmrtDeviceBuffer,
    pub(super) last_page_len: GlmrtDeviceBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DsparkPagedKvMetadata {
    pub(super) query_indptr: Vec<i32>,
    pub(super) kv_indptr: Vec<i32>,
    pub(super) page_indices: Vec<i32>,
    pub(super) last_page_len: Vec<i32>,
    pub(super) active_page_indices: usize,
}

impl DsparkPagedKvMetadata {
    pub(super) fn for_lengths(
        lengths: &[i32],
        query_rows: usize,
        page_size: usize,
        physical_pages_per_request: usize,
    ) -> Result<Self> {
        let page_tables = (0..lengths.len())
            .map(|request| {
                let base = request
                    .checked_mul(physical_pages_per_request)
                    .context("dSpark request physical page base overflow")?;
                (0..physical_pages_per_request)
                    .map(|page| {
                        base.checked_add(page)
                            .context("dSpark physical page ID overflow")?
                            .try_into()
                            .context("dSpark physical page ID does not fit i32")
                    })
                    .collect::<Result<Vec<i32>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Self::for_page_tables(
            lengths,
            query_rows,
            page_size,
            &page_tables,
            lengths
                .len()
                .checked_mul(physical_pages_per_request)
                .context("dSpark total physical page count overflow")?,
        )
    }

    pub(super) fn for_page_tables(
        lengths: &[i32],
        query_rows: usize,
        page_size: usize,
        page_tables: &[Vec<i32>],
        page_index_capacity: usize,
    ) -> Result<Self> {
        anyhow::ensure!(!lengths.is_empty(), "dSpark paged KV lengths are empty");
        anyhow::ensure!(query_rows > 0, "dSpark paged KV query rows are zero");
        anyhow::ensure!(page_size > 0, "dSpark paged KV page size is zero");
        anyhow::ensure!(
            page_tables.len() == lengths.len(),
            "dSpark page-table request count {} does not match length count {}",
            page_tables.len(),
            lengths.len()
        );
        anyhow::ensure!(
            page_index_capacity > 0,
            "dSpark page-index capacity is zero"
        );
        let mut query_indptr = Vec::with_capacity(lengths.len() + 1);
        let mut kv_indptr = Vec::with_capacity(lengths.len() + 1);
        let mut page_indices = Vec::with_capacity(page_index_capacity);
        let mut last_page_len = Vec::with_capacity(lengths.len());
        query_indptr.push(0);
        kv_indptr.push(0);

        for (request, length) in lengths.iter().copied().enumerate() {
            let length =
                usize::try_from(length).context("dSpark paged KV length must be positive")?;
            anyhow::ensure!(length > 0, "dSpark paged KV length must be positive");
            let pages = length.div_ceil(page_size);
            anyhow::ensure!(
                pages <= page_tables[request].len(),
                "dSpark request {request} needs {pages} pages, table has {}",
                page_tables[request].len()
            );
            anyhow::ensure!(
                page_indices.len().saturating_add(pages) <= page_index_capacity,
                "dSpark active page count exceeds page-index capacity {page_index_capacity}"
            );

            let next_query = query_indptr
                .last()
                .copied()
                .unwrap_or(0_i32)
                .checked_add(
                    query_rows
                        .try_into()
                        .context("dSpark query rows do not fit i32")?,
                )
                .context("dSpark query indptr overflow")?;
            query_indptr.push(next_query);
            let next_page = kv_indptr
                .last()
                .copied()
                .unwrap_or(0_i32)
                .checked_add(
                    pages
                        .try_into()
                        .context("dSpark request page count does not fit i32")?,
                )
                .context("dSpark KV indptr overflow")?;
            kv_indptr.push(next_page);

            anyhow::ensure!(
                page_tables[request][..pages].iter().all(|page| *page >= 0),
                "dSpark request {request} page table contains a negative physical page"
            );
            page_indices.extend_from_slice(&page_tables[request][..pages]);
            last_page_len.push(
                ((length - 1) % page_size + 1)
                    .try_into()
                    .context("dSpark last-page length does not fit i32")?,
            );
        }

        let active_page_indices = page_indices.len();
        page_indices.resize(page_index_capacity, 0);
        Ok(Self {
            query_indptr,
            kv_indptr,
            page_indices,
            last_page_len,
            active_page_indices,
        })
    }

    pub(super) fn upload(
        &self,
        library: &'static NativeLibrary,
        buffers: DsparkPagedKvMetadataBuffers,
    ) -> Result<()> {
        library
            .copy_h2d(buffers.query_indptr, as_bytes(&self.query_indptr))
            .context("uploading dSpark query indptr")?;
        library
            .copy_h2d(buffers.kv_indptr, as_bytes(&self.kv_indptr))
            .context("uploading dSpark KV indptr")?;
        library
            .copy_h2d(buffers.page_indices, as_bytes(&self.page_indices))
            .context("uploading dSpark page indices")?;
        library
            .copy_h2d(buffers.last_page_len, as_bytes(&self.last_page_len))
            .context("uploading dSpark last-page lengths")
    }
}

pub(super) fn i32_buffer_bytes(entries: usize, label: &str) -> Result<usize> {
    entries
        .checked_mul(std::mem::size_of::<i32>())
        .with_context(|| format!("dSpark {label} byte count overflow"))
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{DsparkKvStorage, DsparkPagedKvMetadata};

    #[test]
    fn parses_only_implemented_dspark_cache_formats() {
        assert_eq!(DsparkKvStorage::parse("bf16"), Some(DsparkKvStorage::Bf16));
        assert_eq!(DsparkKvStorage::parse("FP8"), Some(DsparkKvStorage::Fp8));
        assert_eq!(DsparkKvStorage::parse("nvfp4"), None);
    }

    #[test]
    fn packs_request_pages_without_padding_active_ranges() {
        let metadata = DsparkPagedKvMetadata::for_lengths(&[65, 17], 16, 64, 2).unwrap();
        assert_eq!(metadata.query_indptr, vec![0, 16, 32]);
        assert_eq!(metadata.kv_indptr, vec![0, 2, 3]);
        assert_eq!(metadata.page_indices, vec![0, 1, 2, 0]);
        assert_eq!(metadata.last_page_len, vec![1, 17]);
        assert_eq!(metadata.active_page_indices, 3);
    }

    #[test]
    fn preserves_chronological_order_for_rotated_physical_pages() {
        let metadata =
            DsparkPagedKvMetadata::for_page_tables(&[129], 16, 64, &[vec![7, 3, 11, 5]], 4)
                .unwrap();
        assert_eq!(metadata.query_indptr, vec![0, 16]);
        assert_eq!(metadata.kv_indptr, vec![0, 3]);
        assert_eq!(metadata.page_indices, vec![7, 3, 11, 0]);
        assert_eq!(metadata.last_page_len, vec![1]);
        assert_eq!(metadata.active_page_indices, 3);
    }
}
