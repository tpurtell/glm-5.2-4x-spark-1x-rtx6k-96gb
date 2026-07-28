use serde::{Deserialize, Serialize};

use super::policy::PrefillChunkPolicy;
use super::shape::LayerWaveMode;
use super::wave::LayerWave;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerWaveAdmission {
    pub selected: Vec<LayerWave>,
    pub deferred: Vec<LayerWave>,
    pub selected_decode_rows: usize,
    pub selected_mtp_rows: usize,
    pub selected_prefill_rows: usize,
    pub selected_prefill_chunks: usize,
}

pub fn admit_layerwaves_for_iteration(
    waves: Vec<LayerWave>,
    policy: &PrefillChunkPolicy,
) -> LayerWaveAdmission {
    let mut indexed = waves.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by_key(|(idx, wave)| {
        let mode_rank = if policy.decode_priority {
            match wave.mode {
                LayerWaveMode::Decode => 0,
                LayerWaveMode::MtpVerify => 1,
                LayerWaveMode::Benchmark => 2,
                LayerWaveMode::Prefill => 3,
            }
        } else {
            0
        };
        (mode_rank, wave.priority.0, *idx)
    });

    let mut selected = Vec::new();
    let mut deferred = Vec::new();
    let mut selected_decode_rows = 0_usize;
    let mut selected_mtp_rows = 0_usize;
    let mut selected_prefill_rows = 0_usize;
    let mut selected_prefill_chunks = 0_usize;

    for (_, wave) in indexed {
        match wave.mode {
            LayerWaveMode::Decode => {
                selected_decode_rows += wave.num_rows();
                selected.push(wave);
            }
            LayerWaveMode::MtpVerify => {
                selected_mtp_rows += wave.num_rows();
                selected.push(wave);
            }
            LayerWaveMode::Benchmark => selected.push(wave),
            LayerWaveMode::Prefill => {
                let would_exceed_tokens = selected_prefill_rows + wave.num_rows()
                    > policy.max_prefill_tokens_per_iteration;
                let would_exceed_chunks =
                    selected_prefill_chunks + 1 > policy.max_active_prefill_chunks;
                if would_exceed_tokens || would_exceed_chunks {
                    deferred.push(wave);
                } else {
                    selected_prefill_rows += wave.num_rows();
                    selected_prefill_chunks += 1;
                    selected.push(wave);
                }
            }
        }
    }

    LayerWaveAdmission {
        selected,
        deferred,
        selected_decode_rows,
        selected_mtp_rows,
        selected_prefill_rows,
        selected_prefill_chunks,
    }
}
