use glmrt_core::{LayerWave, RowSourceKind, GLM52_NUM_HIDDEN_LAYERS};

use crate::{runtime_error, ApiError};

const REQUEST_PROGRESS_HIDDEN_DIM: usize = 4;

pub(super) struct RealFullRequestNumericProgression {
    prefill_tokens: usize,
    decode_rows: usize,
    mtp_rows: usize,
    mtp_accepted_rows: usize,
    residual: Vec<f32>,
    selected_prefill_rows: usize,
    selected_decode_rows: usize,
    selected_mtp_rows: usize,
    attention_value_updates: usize,
    mlp_value_updates: usize,
}

pub(super) struct RealFullRequestNumericProgressionSummary {
    pub(super) passed: bool,
    pub(super) source_rows: usize,
    pub(super) hidden_dim: usize,
    pub(super) selected_prefill_rows: usize,
    pub(super) selected_decode_rows: usize,
    pub(super) selected_mtp_rows: usize,
    pub(super) attention_value_updates: usize,
    pub(super) mlp_value_updates: usize,
    pub(super) visible_checksum: f32,
    pub(super) rejected_mtp_checksum: f32,
}

impl RealFullRequestNumericProgression {
    pub(super) fn new(
        prefill_tokens: usize,
        decode_rows: usize,
        mtp_rows: usize,
        mtp_accepted_rows: usize,
    ) -> Self {
        let source_rows = prefill_tokens + decode_rows + mtp_rows;
        Self {
            prefill_tokens,
            decode_rows,
            mtp_rows,
            mtp_accepted_rows,
            residual: vec![0.0; source_rows * REQUEST_PROGRESS_HIDDEN_DIM],
            selected_prefill_rows: 0,
            selected_decode_rows: 0,
            selected_mtp_rows: 0,
            attention_value_updates: 0,
            mlp_value_updates: 0,
        }
    }

    pub(super) fn apply_selected(&mut self, selected: &[LayerWave]) -> Result<(), ApiError> {
        for wave in selected {
            for source in &wave.row_sources {
                for row_offset in 0..source.row_count {
                    let row_index = self
                        .row_index(source.kind, source.token_start.0 as usize, row_offset)
                        .map_err(runtime_error)?;
                    let (attention_delta, mlp_delta) = source_deltas(source.kind);
                    self.apply_row_delta(row_index, attention_delta, mlp_delta)
                        .map_err(runtime_error)?;
                }
                match source.kind {
                    RowSourceKind::PrefillChunk => self.selected_prefill_rows += source.row_count,
                    RowSourceKind::DecodeStep => self.selected_decode_rows += source.row_count,
                    RowSourceKind::MtpVerifyBlock => self.selected_mtp_rows += source.row_count,
                    RowSourceKind::Benchmark => {}
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish(self) -> RealFullRequestNumericProgressionSummary {
        let source_rows = self.prefill_tokens + self.decode_rows + self.mtp_rows;
        let prefill_values = self.prefill_tokens * REQUEST_PROGRESS_HIDDEN_DIM;
        let decode_values = self.decode_rows * REQUEST_PROGRESS_HIDDEN_DIM;
        let mtp_start = prefill_values + decode_values;
        let accepted_mtp_values = self.mtp_accepted_rows * REQUEST_PROGRESS_HIDDEN_DIM;
        let rejected_mtp_start = mtp_start + accepted_mtp_values;

        let visible_checksum = self.residual[..rejected_mtp_start].iter().sum::<f32>();
        let rejected_mtp_checksum = self.residual[rejected_mtp_start..].iter().sum::<f32>();
        let expected_visible_checksum =
            expected_source_checksum(self.prefill_tokens, RowSourceKind::PrefillChunk)
                + expected_source_checksum(self.decode_rows, RowSourceKind::DecodeStep)
                + expected_source_checksum(self.mtp_accepted_rows, RowSourceKind::MtpVerifyBlock);
        let expected_rejected_mtp_checksum = expected_source_checksum(
            self.mtp_rows - self.mtp_accepted_rows,
            RowSourceKind::MtpVerifyBlock,
        );
        let selected_rows =
            self.selected_prefill_rows + self.selected_decode_rows + self.selected_mtp_rows;
        let expected_selected_rows = source_rows * GLM52_NUM_HIDDEN_LAYERS;
        let expected_value_updates = expected_selected_rows * REQUEST_PROGRESS_HIDDEN_DIM;
        let passed = selected_rows == expected_selected_rows
            && self.selected_prefill_rows == self.prefill_tokens * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_decode_rows == self.decode_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_mtp_rows == self.mtp_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.attention_value_updates == expected_value_updates
            && self.mlp_value_updates == expected_value_updates
            && approx_eq(visible_checksum, expected_visible_checksum)
            && approx_eq(rejected_mtp_checksum, expected_rejected_mtp_checksum);

        RealFullRequestNumericProgressionSummary {
            passed,
            source_rows,
            hidden_dim: REQUEST_PROGRESS_HIDDEN_DIM,
            selected_prefill_rows: self.selected_prefill_rows,
            selected_decode_rows: self.selected_decode_rows,
            selected_mtp_rows: self.selected_mtp_rows,
            attention_value_updates: self.attention_value_updates,
            mlp_value_updates: self.mlp_value_updates,
            visible_checksum,
            rejected_mtp_checksum,
        }
    }

    fn row_index(
        &self,
        kind: RowSourceKind,
        token_start: usize,
        row_offset: usize,
    ) -> anyhow::Result<usize> {
        match kind {
            RowSourceKind::PrefillChunk => {
                let row_index = token_start + row_offset;
                if row_index >= self.prefill_tokens {
                    anyhow::bail!(
                        "request prefill row index {row_index} exceeds {} rows",
                        self.prefill_tokens
                    );
                }
                Ok(row_index)
            }
            RowSourceKind::DecodeStep => {
                let decode_start = self.prefill_tokens;
                let decode_end = decode_start + self.decode_rows;
                let row_index = token_start + row_offset;
                if row_index < decode_start || row_index >= decode_end {
                    anyhow::bail!(
                        "request decode row index {row_index} outside [{decode_start}, {decode_end})"
                    );
                }
                Ok(row_index)
            }
            RowSourceKind::MtpVerifyBlock => {
                let mtp_start = self.prefill_tokens + self.decode_rows;
                let mtp_end = mtp_start + self.mtp_rows;
                let row_index = token_start + row_offset;
                if row_index < mtp_start || row_index >= mtp_end {
                    anyhow::bail!(
                        "request MTP row index {row_index} outside [{mtp_start}, {mtp_end})"
                    );
                }
                Ok(row_index)
            }
            RowSourceKind::Benchmark => {
                anyhow::bail!("benchmark rows are not part of real-full request progression")
            }
        }
    }

    fn apply_row_delta(
        &mut self,
        row_index: usize,
        attention_delta: f32,
        mlp_delta: f32,
    ) -> anyhow::Result<()> {
        let start = row_index * REQUEST_PROGRESS_HIDDEN_DIM;
        let end = start + REQUEST_PROGRESS_HIDDEN_DIM;
        if end > self.residual.len() {
            anyhow::bail!(
                "request progression row index {row_index} exceeds residual rows {}",
                self.prefill_tokens + self.decode_rows + self.mtp_rows
            );
        }
        for value in &mut self.residual[start..end] {
            *value += attention_delta;
        }
        self.attention_value_updates += REQUEST_PROGRESS_HIDDEN_DIM;
        for value in &mut self.residual[start..end] {
            *value += mlp_delta;
        }
        self.mlp_value_updates += REQUEST_PROGRESS_HIDDEN_DIM;
        Ok(())
    }
}

fn source_deltas(kind: RowSourceKind) -> (f32, f32) {
    match kind {
        RowSourceKind::PrefillChunk => (1.0, 0.5),
        RowSourceKind::DecodeStep => (0.5, 0.25),
        RowSourceKind::MtpVerifyBlock => (1.5, 0.75),
        RowSourceKind::Benchmark => (0.0, 0.0),
    }
}

fn expected_source_checksum(rows: usize, kind: RowSourceKind) -> f32 {
    let (attention_delta, mlp_delta) = source_deltas(kind);
    rows as f32
        * REQUEST_PROGRESS_HIDDEN_DIM as f32
        * GLM52_NUM_HIDDEN_LAYERS as f32
        * (attention_delta + mlp_delta)
}

fn approx_eq(left: f32, right: f32) -> bool {
    (left - right).abs() < 1.0e-3
}
