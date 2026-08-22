use super::*;

#[test]
fn kv_cache_config_tracks_compressed_bf16_bytes_per_token() {
    let config = KvCacheConfig::glm52_phase0(128);
    assert_eq!(config.layout, KvLayout::Glm52CompressedBf16);
    assert_eq!(config.dtype, KvCacheDType::Bf16);
    assert_eq!(config.key_value_width, 512 + 64);
    assert_eq!(config.dsa_indexer_layers, 21);
    assert_eq!(config.dsa_indexer_layer_ids(), &GLM52_DSA_INDEXER_LAYER_IDS);
    assert_eq!(config.dsa_index_head_dim, 128);
    assert_eq!(
        config.main_mla_bytes_per_token(),
        GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(
        config.dsa_indexer_bytes_per_token(),
        GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(config.bytes_per_token(), 95_232);
    assert_eq!(
        config.bytes_per_token(),
        GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(config.capacity_bytes(), config.bytes_per_token() * 128);
}

#[test]
fn kv_cache_layer_payload_bytes_sum_to_compressed_token_bytes() {
    let config = KvCacheConfig::glm52_phase0(128);
    assert_eq!(
        config.layer_bytes_per_token(LayerId(0)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(2)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(22)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(74)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(config.layer_bytes_per_token(LayerId(3)), (512 + 64) * 2);
    assert_eq!(config.layer_bytes_per_token(LayerId(20)), (512 + 64) * 2);
    assert_eq!(config.layer_bytes_per_token(LayerId(21)), (512 + 64) * 2);
    assert!(config.layer_has_dsa_indexer(LayerId(22)));
    assert!(!config.layer_has_dsa_indexer(LayerId(3)));

    let layer_sum = (0..GLM52_NUM_HIDDEN_LAYERS)
        .map(|layer_id| config.layer_bytes_per_token(LayerId(layer_id as u32)))
        .sum::<usize>();
    assert_eq!(layer_sum, config.bytes_per_token());
    assert_eq!(config.layer_payload_bytes(LayerId(0), 4), 5_632);
    assert_eq!(config.layer_payload_bytes(LayerId(3), 4), 4_608);
}

#[test]
fn kv_cache_compressed_fp8_uses_inline_b12x_mla_ds_layout() {
    let config = KvCacheConfig::glm52_compressed_fp8(128);

    assert_eq!(config.layout, KvLayout::Glm52CompressedFp8);
    assert_eq!(config.dtype, KvCacheDType::Fp8);
    assert_eq!(config.fp8_scale_metadata_bytes_per_token, 0);
    assert_eq!(config.layer_bytes_per_token(LayerId(0)), 656 + 128 * 2);
    assert_eq!(config.layer_bytes_per_token(LayerId(3)), 656);
    assert_eq!(config.main_mla_bytes_per_token(), 78 * 656);
    assert_eq!(config.dsa_indexer_bytes_per_token(), 21 * 128 * 2);
    assert_eq!(config.bytes_per_token(), 56_544);
}

#[test]
fn kv_cache_compressed_nvfp4_uses_native_nvfp4_mla_and_bf16_rope_layout() {
    let config = KvCacheConfig::glm52_compressed_nvfp4(128);

    assert_eq!(config.layout, KvLayout::Glm52CompressedNvfp4);
    assert_eq!(config.dtype, KvCacheDType::Nvfp4);
    assert_eq!(config.dtype.bits_per_element(), 4);
    assert_eq!(config.fp8_scale_metadata_bytes_per_token, 0);
    assert_eq!(
        config.layer_bytes_per_token(LayerId(0)),
        GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + 128 * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(3)),
        GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
    );
    assert_eq!(
        config.main_mla_bytes_per_token(),
        78 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
    );
    assert_eq!(config.dsa_indexer_bytes_per_token(), 21 * 128 * 2);
    assert_eq!(config.bytes_per_token(), 39_072);
    assert_eq!(config.main_mla_row_bytes(), Some(432));
    assert_eq!(config.main_mla_page_bytes(64), Some(64 * 432));
}

#[test]
fn compressed_main_mla_page_geometry_is_owned_by_each_format() {
    let bf16 = KvCacheConfig::glm52_compressed_bf16(128);
    let fp8 = KvCacheConfig::glm52_compressed_fp8(128);
    let nvfp4 = KvCacheConfig::glm52_compressed_nvfp4(128);

    assert_eq!(bf16.main_mla_row_bytes(), Some((512 + 64) * 2));
    assert_eq!(bf16.main_mla_page_bytes(64), Some(64 * (512 + 64) * 2));
    assert_eq!(fp8.main_mla_row_bytes(), Some(656));
    assert_eq!(fp8.main_mla_page_bytes(64), Some(64 * 656));
    assert_eq!(nvfp4.main_mla_row_bytes(), Some(432));
    assert_eq!(nvfp4.main_mla_page_bytes(64), Some(64 * 432));
}

#[test]
fn kv_cache_descriptor_offsets_use_layer_major_packed_layout() {
    let config = KvCacheConfig::glm52_phase0(8);
    let descriptor = KvBlockDescriptor {
        reservation_id: 1,
        sequence_id: "seq-a".to_owned(),
        layer_id: LayerId(3),
        token_start: PositionId(2),
        token_count: 3,
    };
    let dsa_layer_bytes = (512 + 64 + 128) * 2;
    let non_dsa_layer_bytes = (512 + 64) * 2;

    assert_eq!(
        config.layer_base_offset_bytes(LayerId(3)),
        Some(3 * dsa_layer_bytes * 8)
    );
    assert_eq!(
        config.descriptor_offset_bytes(&descriptor),
        Some(3 * dsa_layer_bytes * 8 + 2 * non_dsa_layer_bytes)
    );
    assert_eq!(
        config.descriptor_payload_bytes(&descriptor),
        Some(3 * non_dsa_layer_bytes)
    );
}

#[test]
fn kv_cache_descriptor_offsets_cover_fp8_and_nvfp4_packed_layouts() {
    let fp8 = KvCacheConfig::glm52_compressed_fp8(8);
    let nvfp4 = KvCacheConfig::glm52_compressed_nvfp4(8);
    let descriptor = KvBlockDescriptor {
        reservation_id: 1,
        sequence_id: "seq-a".to_owned(),
        layer_id: LayerId(3),
        token_start: PositionId(2),
        token_count: 3,
    };

    assert_eq!(fp8.descriptor_payload_bytes(&descriptor), Some(3 * 656));
    assert_eq!(
        fp8.descriptor_offset_bytes(&descriptor),
        Some(3 * (656 + 128 * 2) * 8 + 2 * 656)
    );
    assert_eq!(
        nvfp4.descriptor_payload_bytes(&descriptor),
        Some(3 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN)
    );
    assert_eq!(
        nvfp4.descriptor_offset_bytes(&descriptor),
        Some(
            3 * (GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + 128 * 2) * 8
                + 2 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
        )
    );
}

#[test]
fn kv_cache_descriptor_offsets_reject_invalid_ranges() {
    let config = KvCacheConfig::glm52_phase0(8);
    let invalid_layer = KvBlockDescriptor {
        reservation_id: 1,
        sequence_id: "seq-a".to_owned(),
        layer_id: LayerId(GLM52_NUM_HIDDEN_LAYERS as u32),
        token_start: PositionId(0),
        token_count: 1,
    };
    let invalid_tokens = KvBlockDescriptor {
        reservation_id: 1,
        sequence_id: "seq-a".to_owned(),
        layer_id: LayerId(3),
        token_start: PositionId(7),
        token_count: 2,
    };

    assert_eq!(config.layer_base_offset_bytes(invalid_layer.layer_id), None);
    assert_eq!(config.descriptor_offset_bytes(&invalid_layer), None);
    assert_eq!(config.descriptor_payload_bytes(&invalid_tokens), None);
    assert_eq!(config.descriptor_offset_bytes(&invalid_tokens), None);
}

#[test]
fn kv_cache_mtp_layer_adds_its_full_dsa_indexer() {
    let base = KvCacheConfig::glm52_phase0(8);
    let mtp = base.clone().with_mtp_layer();
    let mtp_layer = LayerId(GLM52_NUM_HIDDEN_LAYERS as u32);
    let invalid_layer = LayerId(GLM52_TOTAL_LAYERS_WITH_MTP as u32);
    let row_bytes =
        (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * std::mem::size_of::<u16>();

    assert_eq!(mtp.layers, GLM52_TOTAL_LAYERS_WITH_MTP);
    let dsa_bytes = GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>();
    assert_eq!(mtp.dsa_indexer_layers, base.dsa_indexer_layers + 1);
    assert!(mtp.layer_has_dsa_indexer(mtp_layer));
    assert_eq!(mtp.layer_bytes_per_token(mtp_layer), row_bytes + dsa_bytes);
    assert_eq!(
        mtp.bytes_per_token(),
        base.bytes_per_token() + row_bytes + dsa_bytes
    );
    assert!(mtp.layer_base_offset_bytes(mtp_layer).is_some());
    assert_eq!(mtp.layer_base_offset_bytes(invalid_layer), None);
}

#[test]
fn kv_cache_dtype_parser_accepts_runtime_cache_labels() {
    assert_eq!(
        KvCacheDType::parse_glm52_cache_dtype("bf16"),
        Some(KvCacheDType::Bf16)
    );
    assert_eq!(
        KvCacheDType::parse_glm52_cache_dtype("FP8"),
        Some(KvCacheDType::Fp8)
    );
    assert_eq!(
        KvCacheDType::parse_glm52_cache_dtype("nvfp4"),
        Some(KvCacheDType::Nvfp4)
    );
    assert_eq!(KvCacheDType::parse_glm52_cache_dtype("int4"), None);
}

#[test]
fn expanded_debug_layout_is_available_but_not_default() {
    let default = KvCacheConfig::glm52_phase0(128);
    let expanded = KvCacheConfig::glm52_expanded_debug_bf16(128);

    assert_ne!(default.layout, KvLayout::ExpandedDebugOnly);
    assert_eq!(expanded.layout, KvLayout::ExpandedDebugOnly);
    assert_eq!(
        expanded.bytes_per_token(),
        GLM52_EXPANDED_DEBUG_KV_BF16_BYTES_PER_TOKEN
    );
    assert!(expanded.bytes_per_token() > default.bytes_per_token());
}
