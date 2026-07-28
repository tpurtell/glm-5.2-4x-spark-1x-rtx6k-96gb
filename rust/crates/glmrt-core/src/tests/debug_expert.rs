use super::*;

#[test]
fn expert_request_header_carries_layerwave_metadata() {
    let wave = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        16,
        99,
        Priority(0),
        GraphBucket::new(16),
        "placement-a",
    ));
    let request = ExpertRequest {
        protocol_version: 1,
        request_id: 77,
        placement_version: "placement-a".to_owned(),
        layer_id: wave.layer_id.0,
        hidden_dim: GLM52_HIDDEN_SIZE as u32,
        wave: Some(ExpertWaveMetadata::from_wave(&wave)),
        rows: Vec::new(),
    };
    let header = request.header();

    assert_eq!(header.wave_mode, Some(LayerWaveMode::Prefill));
    assert_eq!(header.graph_bucket_rows, Some(16));
    assert_eq!(
        header.logical_bf16_payload_bytes,
        Some(16 * GLM52_HIDDEN_BF16_BYTES)
    );
}
