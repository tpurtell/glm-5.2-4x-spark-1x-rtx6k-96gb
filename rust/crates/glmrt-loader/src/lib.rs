mod catalog;
mod placement;
mod snapshot;
mod tensors;
mod tokenizer;

pub use catalog::{
    build_catalog, build_catalog_for_snapshot, classification_summary_markdown, read_model_facts,
    read_safetensors_metadata, SafetensorsTensorMetadata,
};
pub use placement::{assignments_by_owner, build_load_plan};
pub use snapshot::{
    default_hf_home, empty_catalog_for_snapshot, model_cache_dir, resolve_snapshot,
    SnapshotResolution,
};
pub use tensors::{
    dtype_byte_width, load_tensor_bytes, load_tensor_bytes_with_options, load_tensor_rows,
    load_tensor_rows_with_options, read_tensor_bytes_into, read_tensor_bytes_into_with_options,
    read_tensor_row_prefix_into, read_tensor_row_prefix_into_with_options,
    read_tensor_row_window_into, read_tensor_row_window_into_with_options, read_tensor_rows_into,
    read_tensor_rows_into_with_options, LoadedTensor, LoadedTensorRows, LoadedTensorRowsSummary,
    LoadedTensorSummary, TensorLoadOptions,
};
pub use tokenizer::{
    decode_tokenizer_ids, encode_tokenizer_text, streaming_token_decoder, LoadedTokenizer,
    StreamingTokenDecoder, TokenizerDecodeSummary, TokenizerEncodingSummary,
};

#[cfg(test)]
mod tests;
