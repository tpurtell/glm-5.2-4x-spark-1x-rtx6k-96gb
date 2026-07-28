use serde::{Deserialize, Serialize};

use crate::{
    DType, GlmrtError, GraphBucket, LayerId, LayerWave, ModelFacts, PlacementVersion, PositionId,
    RequestId, RowSourceKind,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertBatchRow {
    pub row_id: u64,
    pub source_kind: RowSourceKind,
    pub request_id: RequestId,
    pub sequence_id: String,
    pub token_position: PositionId,
    pub route_offset: usize,
    pub route_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertBatch {
    pub layer_id: LayerId,
    pub placement_version: PlacementVersion,
    pub hidden_dim: usize,
    pub hidden_bytes_per_row: usize,
    pub hidden_dtype: DType,
    pub graph_bucket: GraphBucket,
    pub quantization_recipe: String,
    pub rows: Vec<ExpertBatchRow>,
}

impl ExpertBatch {
    pub fn from_wave(
        wave: &LayerWave,
        hidden_dtype: DType,
        quantization_recipe: impl Into<String>,
    ) -> Result<Self, GlmrtError> {
        Self::from_wave_with_envelope(wave, hidden_dtype, quantization_recipe, wave.graph_bucket)
    }

    pub fn from_wave_with_envelope(
        wave: &LayerWave,
        hidden_dtype: DType,
        quantization_recipe: impl Into<String>,
        graph_bucket: GraphBucket,
    ) -> Result<Self, GlmrtError> {
        let source_rows = wave
            .row_sources
            .iter()
            .map(|source| source.row_count)
            .sum::<usize>();
        if source_rows != wave.num_rows() {
            return Err(GlmrtError::ExpertBatchRowCountMismatch {
                source_rows,
                hidden_rows: wave.num_rows(),
            });
        }
        if wave.num_rows() > graph_bucket.row_capacity {
            return Err(GlmrtError::ExpertBatchMixRejected {
                reason: format!(
                    "rows {} exceed graph bucket capacity {}",
                    wave.num_rows(),
                    graph_bucket.row_capacity
                ),
            });
        }

        let mut rows = Vec::with_capacity(wave.num_rows());
        let mut route_offset = 0_usize;
        for source in &wave.row_sources {
            for row_offset in 0..source.row_count {
                let route_count = wave.route_metadata.top_k;
                rows.push(ExpertBatchRow {
                    row_id: rows.len() as u64,
                    source_kind: source.kind,
                    request_id: source.request_id.clone(),
                    sequence_id: source.sequence_id.clone(),
                    token_position: PositionId(source.token_start.0 + row_offset as u64),
                    route_offset,
                    route_count,
                });
                route_offset += route_count;
            }
        }

        Ok(Self {
            layer_id: wave.layer_id,
            placement_version: wave.placement_version.clone(),
            hidden_dim: wave.hidden_shape.hidden_dim,
            hidden_bytes_per_row: wave.hidden_shape.bytes_per_row,
            hidden_dtype,
            graph_bucket,
            quantization_recipe: quantization_recipe.into(),
            rows,
        })
    }

    pub fn glm52_bf16_from_wave_with_envelope(
        wave: &LayerWave,
        graph_bucket: GraphBucket,
    ) -> Result<Self, GlmrtError> {
        Self::from_wave_with_envelope(
            wave,
            DType::Bf16,
            ModelFacts::default().quantization_recipe,
            graph_bucket,
        )
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn route_count(&self) -> usize {
        self.rows.iter().map(|row| row.route_count).sum()
    }

    pub fn can_mix_with(&self, other: &Self) -> bool {
        self.mix_rejection_reason(other).is_none()
    }

    pub fn try_merge(&self, other: &Self) -> Result<Self, GlmrtError> {
        if let Some(reason) = self.mix_rejection_reason(other) {
            return Err(GlmrtError::ExpertBatchMixRejected { reason });
        }
        let mut merged = self.clone();
        merged.append_rows_from(other);
        Ok(merged)
    }

    pub fn try_append_wave(
        &mut self,
        wave: &LayerWave,
        hidden_dtype: DType,
        quantization_recipe: impl Into<String>,
    ) -> Result<(), GlmrtError> {
        let other = Self::from_wave_with_envelope(
            wave,
            hidden_dtype,
            quantization_recipe,
            self.graph_bucket,
        )?;
        if let Some(reason) = self.mix_rejection_reason(&other) {
            return Err(GlmrtError::ExpertBatchMixRejected { reason });
        }
        self.append_rows_from(&other);
        Ok(())
    }

    pub fn reconstruct_partial_outputs<T: Clone>(
        &self,
        partial_outputs: &[T],
    ) -> Result<Vec<(ExpertBatchRow, T)>, GlmrtError> {
        if partial_outputs.len() != self.rows.len() {
            return Err(GlmrtError::ExpertBatchPartialRowCountMismatch {
                expected: self.rows.len(),
                actual: partial_outputs.len(),
            });
        }
        Ok(self
            .rows
            .iter()
            .cloned()
            .zip(partial_outputs.iter().cloned())
            .collect())
    }

    fn append_rows_from(&mut self, other: &Self) {
        let row_id_base = self.rows.len() as u64;
        let route_offset_base = self.route_count();
        self.rows.extend(other.rows.iter().cloned().map(|mut row| {
            row.row_id += row_id_base;
            row.route_offset += route_offset_base;
            row
        }));
    }

    fn mix_rejection_reason(&self, other: &Self) -> Option<String> {
        if self.layer_id != other.layer_id {
            return Some(format!(
                "different layers {} and {}",
                self.layer_id.0, other.layer_id.0
            ));
        }
        if self.placement_version != other.placement_version {
            return Some("different placement versions".to_owned());
        }
        if self.hidden_dim != other.hidden_dim
            || self.hidden_bytes_per_row != other.hidden_bytes_per_row
        {
            return Some("different hidden shapes".to_owned());
        }
        if self.hidden_dtype != other.hidden_dtype {
            return Some("different hidden dtypes".to_owned());
        }
        if self.graph_bucket != other.graph_bucket {
            return Some(format!(
                "different graph buckets {} and {}",
                self.graph_bucket.row_capacity, other.graph_bucket.row_capacity
            ));
        }
        if self.quantization_recipe != other.quantization_recipe {
            return Some("different quantization recipes".to_owned());
        }
        let merged_rows = self.num_rows() + other.num_rows();
        if merged_rows > self.graph_bucket.row_capacity {
            return Some(format!(
                "merged rows {merged_rows} exceed graph bucket capacity {}",
                self.graph_bucket.row_capacity
            ));
        }
        None
    }
}
