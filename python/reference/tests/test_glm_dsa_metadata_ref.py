from glmrt_reference.glm_dsa_metadata_ref import glm52_dsa_indexer_metadata


def test_glm52_dsa_indexer_metadata_tracks_noncontiguous_layers_and_bytes():
    metadata = glm52_dsa_indexer_metadata()

    assert metadata.indexer_layer_ids == (
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
    )
    assert metadata.main_mla_bytes_per_token == 89_856
    assert metadata.dsa_indexer_bytes_per_token == 5_376
    assert metadata.compressed_bf16_bytes_per_token == 95_232
    assert metadata.layer_has_dsa_indexer(22)
    assert not metadata.layer_has_dsa_indexer(3)
    assert metadata.phase0_attention_math_status.startswith("metadata-only:")
