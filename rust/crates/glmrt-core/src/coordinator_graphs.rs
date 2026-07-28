use serde::{Deserialize, Serialize};

use crate::{
    DType, GlmrtError, GraphBucket, LayerId, LayerWaveMode, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_TOTAL_LAYERS_WITH_MTP,
};

pub const COORDINATOR_GRAPH_DECODE_BUCKET_ROWS: usize = 1;
pub const COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS: [usize; 13] = [
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
pub const COORDINATOR_GRAPH_SHAPES: [CoordinatorGraphShape; 5] = [
    CoordinatorGraphShape::CoordAttention,
    CoordinatorGraphShape::CoordCompressedAttention,
    CoordinatorGraphShape::CoordDense,
    CoordinatorGraphShape::CoordSparseA,
    CoordinatorGraphShape::CoordSparseB,
];
pub const COORDINATOR_GRAPH_INSTANCE_COUNT: usize =
    COORDINATOR_GRAPH_SHAPES.len() * (1 + COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len());

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoordinatorGraphShape {
    CoordAttention,
    CoordCompressedAttention,
    CoordDense,
    CoordSparseA,
    CoordSparseB,
}

impl CoordinatorGraphShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::CoordAttention => "Coord-Attention",
            Self::CoordCompressedAttention => "Coord-Compressed-Attention",
            Self::CoordDense => "Coord-Dense",
            Self::CoordSparseA => "Coord-Sparse-A",
            Self::CoordSparseB => "Coord-Sparse-B",
        }
    }

    pub fn op_count(self) -> usize {
        match self {
            Self::CoordAttention | Self::CoordCompressedAttention => 1,
            Self::CoordDense => 14,
            Self::CoordSparseA => 12,
            Self::CoordSparseB => 2,
        }
    }

    pub fn network_boundary(self) -> CoordinatorGraphNetworkBoundary {
        match self {
            Self::CoordAttention | Self::CoordCompressedAttention => {
                CoordinatorGraphNetworkBoundary::None
            }
            Self::CoordDense => CoordinatorGraphNetworkBoundary::None,
            Self::CoordSparseA => CoordinatorGraphNetworkBoundary::BeforeExpertSend,
            Self::CoordSparseB => CoordinatorGraphNetworkBoundary::AfterExpertRecv,
        }
    }

    pub fn accepts_layer(self, layer_id: LayerId) -> bool {
        let layer = layer_id.0 as usize;
        if layer >= GLM52_TOTAL_LAYERS_WITH_MTP {
            return false;
        }
        match self {
            Self::CoordAttention | Self::CoordCompressedAttention => true,
            Self::CoordDense => layer < GLM52_FIRST_K_DENSE_REPLACE,
            Self::CoordSparseA | Self::CoordSparseB => layer >= GLM52_FIRST_K_DENSE_REPLACE,
        }
    }

    pub fn validate_layer(self, layer_id: LayerId) -> Result<(), GlmrtError> {
        if self.accepts_layer(layer_id) {
            return Ok(());
        }
        Err(GlmrtError::GraphBufferContractInvalid {
            reason: format!(
                "{} cannot be used for GLM-5.2 layer {}",
                self.label(),
                layer_id.0
            ),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoordinatorGraphNetworkBoundary {
    None,
    BeforeExpertSend,
    AfterExpertRecv,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorGraphKey {
    pub shape: CoordinatorGraphShape,
    pub row_bucket: GraphBucket,
    pub dtype: DType,
}

impl CoordinatorGraphKey {
    pub fn glm52_bf16(
        shape: CoordinatorGraphShape,
        mode: LayerWaveMode,
        active_rows: usize,
    ) -> Result<Self, GlmrtError> {
        Self::new(shape, mode, active_rows, DType::Bf16)
    }

    pub fn new(
        shape: CoordinatorGraphShape,
        _mode: LayerWaveMode,
        active_rows: usize,
        dtype: DType,
    ) -> Result<Self, GlmrtError> {
        Ok(Self {
            shape,
            row_bucket: coordinator_graph_bucket_for_active_rows(active_rows)?,
            dtype,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorGraphInstancePlan {
    pub key: CoordinatorGraphKey,
    pub op_count: usize,
    pub network_boundary: CoordinatorGraphNetworkBoundary,
}

impl CoordinatorGraphInstancePlan {
    pub fn glm52_bf16_all() -> Vec<Self> {
        let mut plans = Vec::with_capacity(COORDINATOR_GRAPH_INSTANCE_COUNT);
        for shape in COORDINATOR_GRAPH_SHAPES {
            plans.push(Self::new(
                CoordinatorGraphKey {
                    shape,
                    row_bucket: GraphBucket::decode(),
                    dtype: DType::Bf16,
                },
                shape,
            ));
            for row_capacity in COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS {
                plans.push(Self::new(
                    CoordinatorGraphKey {
                        shape,
                        row_bucket: GraphBucket::new(row_capacity),
                        dtype: DType::Bf16,
                    },
                    shape,
                ));
            }
        }
        plans
    }

    fn new(key: CoordinatorGraphKey, shape: CoordinatorGraphShape) -> Self {
        Self {
            key,
            op_count: shape.op_count(),
            network_boundary: shape.network_boundary(),
        }
    }
}

pub fn coordinator_graph_bucket_for_active_rows(
    active_rows: usize,
) -> Result<GraphBucket, GlmrtError> {
    if active_rows == 0 {
        return Err(GlmrtError::GraphBufferContractInvalid {
            reason: "coordinator graph active rows must be nonzero".to_owned(),
        });
    }
    if active_rows == COORDINATOR_GRAPH_DECODE_BUCKET_ROWS {
        return Ok(GraphBucket::decode());
    }
    for row_capacity in COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS {
        if active_rows <= row_capacity {
            return Ok(GraphBucket::new(row_capacity));
        }
    }
    Err(GlmrtError::GraphBufferContractInvalid {
        reason: format!(
            "coordinator graph active rows {} exceed max prefill bucket {}",
            active_rows,
            COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS[COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len() - 1]
        ),
    })
}
