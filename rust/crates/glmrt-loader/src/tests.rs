use super::{
    build_load_plan, encode_tokenizer_text, load_tensor_bytes, load_tensor_bytes_with_options,
    load_tensor_rows, load_tensor_rows_with_options, model_cache_dir, read_safetensors_metadata,
    read_tensor_bytes_into, read_tensor_bytes_into_with_options, read_tensor_row_prefix_into,
    read_tensor_row_prefix_into_with_options, read_tensor_row_window_into, read_tensor_rows_into,
    read_tensor_rows_into_with_options, streaming_token_decoder, TensorLoadOptions,
};
use crate::catalog::{classify_tensor, is_quantization_tensor};
use glmrt_core::{
    owner_for_expert, DType, ModelFacts, PlacementPolicy, TensorCatalog, TensorInfo, TensorRole,
    COORDINATOR_HOST, GLM52_MTP_LAYER_ID,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[test]
fn model_cache_path_uses_hf_layout() {
    let p = model_cache_dir(Path::new("/tmp/hf"), "a/b");
    assert_eq!(p, PathBuf::from("/tmp/hf/hub/models--a--b"));
}

#[test]
fn tokenizer_is_reused_after_the_snapshot_file_changes() {
    let tempdir = tempfile::tempdir().unwrap();
    let tokenizer_path = tempdir.path().join("tokenizer.json");
    std::fs::write(
        &tokenizer_path,
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();

    let first = encode_tokenizer_text(tempdir.path(), "hello", false).unwrap();
    assert_eq!(first.token_ids, vec![1]);
    std::fs::write(&tokenizer_path, b"not valid tokenizer json").unwrap();
    let cached = encode_tokenizer_text(tempdir.path(), "hello", false).unwrap();
    assert_eq!(cached.token_ids, vec![1]);
}

#[test]
fn streaming_tokenizer_buffers_split_utf8_scalars() {
    let tempdir = tempfile::tempdir().unwrap();
    std::fs::write(
        tempdir.path().join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"WordLevel","vocab":{"[UNK]":0,"ðŁ":1,"¦":2,"ľ":3," ok":4},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();

    let mut decoder = streaming_token_decoder(tempdir.path(), false).unwrap();
    assert_eq!(decoder.step(1).unwrap(), None);
    assert_eq!(decoder.step(2).unwrap(), None);
    assert_eq!(decoder.step(3).unwrap().as_deref(), Some("🦜"));
}

#[test]
fn reads_single_file_safetensors_metadata_with_absolute_offsets() {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("model.safetensors");
    let header = serde_json::json!({
        "z": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
        "__metadata__": {"format": "pt"}
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let data_start = 8 + header_bytes.len();
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0_u8; 16]).unwrap();

    let metadata = read_safetensors_metadata(&path).unwrap();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata[0].name, "a");
    assert_eq!(metadata[0].dtype, DType::F32);
    assert_eq!(metadata[0].shape, [1]);
    assert_eq!(metadata[0].byte_offset, (data_start + 12) as u64);
    assert_eq!(metadata[0].byte_length, 4);
    assert_eq!(metadata[1].name, "z");
    assert_eq!(metadata[1].dtype, DType::Bf16);
    assert_eq!(metadata[1].shape, [2, 3]);
    assert_eq!(metadata[1].byte_offset, data_start as u64);
    assert_eq!(metadata[1].byte_length, 12);
}

#[test]
fn classifier_identifies_routed_expert_scale() {
    let facts = ModelFacts::default();
    let name = "model.layers.3.mlp.experts.17.down_proj.weight_scale";
    assert_eq!(
        classify_tensor(name, Some(3), Some(17), true, &facts),
        TensorRole::RoutedExpert
    );
    assert!(is_quantization_tensor(name));
}

#[test]
fn classifier_keeps_shared_expert_on_coordinator() {
    let facts = ModelFacts::default();
    let name = "model.layers.3.mlp.shared_experts.down_proj.weight";
    assert_eq!(
        classify_tensor(name, Some(3), None, false, &facts),
        TensorRole::SharedExpert
    );
}

#[test]
fn classifier_assigns_mtp_routed_experts_to_expert_placement() {
    let facts = ModelFacts::default();
    let name = "model.layers.78.mlp.experts.17.gate_proj.weight";
    assert_eq!(
        classify_tensor(
            name,
            Some(GLM52_MTP_LAYER_ID as u32),
            Some(17),
            false,
            &facts,
        ),
        TensorRole::RoutedExpert
    );
}

#[test]
fn classifier_keeps_mtp_non_expert_tensors_on_the_mtp_role() {
    let facts = ModelFacts::default();
    let name = "model.layers.78.eh_proj.weight";
    assert_eq!(
        classify_tensor(name, Some(GLM52_MTP_LAYER_ID as u32), None, false, &facts,),
        TensorRole::Mtp
    );
}

#[test]
fn load_plan_places_mtp_experts_on_sparks_and_mtp_envelope_on_coordinator() {
    let expert_hosts = vec!["ostrich".to_owned(), "dodo".to_owned()];
    let expert_name = "model.layers.78.mlp.experts.17.gate_proj.weight";
    let envelope_name = "model.layers.78.eh_proj.weight";
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: "/tmp/test-model".to_owned(),
        facts: ModelFacts::default(),
        tensors: vec![
            TensorInfo {
                name: expert_name.to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::U8,
                shape: vec![1],
                byte_offset: 0,
                byte_length: 1,
                role: TensorRole::RoutedExpert,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: Some(17),
                is_quantization_metadata: false,
            },
            TensorInfo {
                name: envelope_name.to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape: vec![1],
                byte_offset: 1,
                byte_length: 2,
                role: TensorRole::Mtp,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: None,
                is_quantization_metadata: false,
            },
        ],
    };

    let plan = build_load_plan(&catalog, PlacementPolicy::Range, expert_hosts.clone()).unwrap();
    let expert = plan
        .assignments
        .iter()
        .find(|assignment| assignment.tensor_name == expert_name)
        .unwrap();
    let envelope = plan
        .assignments
        .iter()
        .find(|assignment| assignment.tensor_name == envelope_name)
        .unwrap();

    assert_eq!(
        expert.owner,
        owner_for_expert(
            GLM52_MTP_LAYER_ID,
            17,
            &expert_hosts,
            PlacementPolicy::Range,
        )
        .unwrap()
    );
    assert_eq!(envelope.owner, COORDINATOR_HOST);
}

#[test]
fn load_tensor_bytes_reads_exact_range() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[9, 8, 7, 1, 2, 3, 4, 5]).unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 3,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let loaded = load_tensor_bytes(&catalog, "tensor").unwrap();
    assert_eq!(loaded.bytes, vec![1, 2, 3, 4]);
    assert_eq!(loaded.sha256, "");
    let summary = loaded.summary();
    assert_eq!(summary.bytes_requested, 4);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.tensor_name, "tensor");
    assert_eq!(summary.sha256, "");

    let hashed =
        load_tensor_bytes_with_options(&catalog, "tensor", TensorLoadOptions::verify_hashes())
            .unwrap();
    assert_eq!(hashed.bytes, vec![1, 2, 3, 4]);
    assert_eq!(hashed.sha256.len(), 64);
    assert_ne!(hashed.sha256, loaded.sha256);
}

#[test]
fn read_tensor_bytes_into_uses_caller_buffer_prefix() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[9, 8, 7, 1, 2, 3, 4, 5]).unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 3,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xee; 8];

    let summary = read_tensor_bytes_into(&catalog, "tensor", &mut dst).unwrap();

    assert_eq!(&dst[..4], &[1, 2, 3, 4]);
    assert_eq!(&dst[4..], &[0xee, 0xee, 0xee, 0xee]);
    assert_eq!(summary.bytes_requested, 4);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.byte_offset, 3);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_bytes_into_with_options(
        &catalog,
        "tensor",
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_bytes_into_rejects_small_destination() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[1, 2, 3, 4]).unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 0,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 3];

    let err = read_tensor_bytes_into(&catalog, "tensor", &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("destination buffer for tensor tensor has 3 bytes, needs 4"));
}

#[test]
fn load_tensor_rows_reads_exact_2d_window() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let rows = load_tensor_rows(&catalog, "matrix", 1, 2).unwrap();
    assert_eq!(rows.bytes, vec![20, 21, 22, 23, 30, 31, 32, 33]);
    assert_eq!(rows.sha256, "");
    let summary = rows.summary();
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 4);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 8);
    assert_eq!(summary.sha256, "");

    let hashed =
        load_tensor_rows_with_options(&catalog, "matrix", 1, 2, TensorLoadOptions::verify_hashes())
            .unwrap();
    assert_eq!(hashed.bytes, rows.bytes);
    assert_eq!(hashed.sha256.len(), 64);
    assert_ne!(hashed.sha256, rows.sha256);
}

#[test]
fn read_tensor_rows_into_uses_caller_buffer_prefix() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 12];

    let summary = read_tensor_rows_into(&catalog, "matrix", 1, 2, &mut dst).unwrap();

    assert_eq!(&dst[..8], &[20, 21, 22, 23, 30, 31, 32, 33]);
    assert_eq!(&dst[8..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 4);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 8);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_rows_into_with_options(
        &catalog,
        "matrix",
        1,
        2,
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_rows_into_rejects_small_destination() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 7];

    let err = read_tensor_rows_into(&catalog, "matrix", 1, 2, &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("destination buffer for tensor matrix rows 1..3 has 7 bytes, needs 8"));
}

#[test]
fn read_tensor_row_prefix_into_compacts_leading_columns() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 8];

    let summary = read_tensor_row_prefix_into(&catalog, "matrix", 1, 2, 2, &mut dst).unwrap();

    assert_eq!(&dst[..4], &[20, 21, 30, 31]);
    assert_eq!(&dst[4..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 2);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_row_prefix_into_with_options(
        &catalog,
        "matrix",
        1,
        2,
        2,
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_row_window_into_compacts_middle_columns() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 8];

    let summary = read_tensor_row_window_into(&catalog, "matrix", 1, 2, 1, 2, &mut dst).unwrap();

    assert_eq!(&dst[..4], &[21, 22, 31, 32]);
    assert_eq!(&dst[4..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 2);
    assert_eq!(summary.byte_offset, 8);
    assert_eq!(summary.bytes_read, 4);
}

#[test]
fn read_tensor_row_prefix_into_rejects_invalid_prefix_width() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 16];

    let err = read_tensor_row_prefix_into(&catalog, "matrix", 0, 1, 5, &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("row prefix width 5 exceeds tensor matrix row width 4"));
}

#[test]
fn load_tensor_rows_rejects_out_of_bounds_window() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let err = load_tensor_rows(&catalog, "matrix", 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds tensor matrix row count 3"));
}
