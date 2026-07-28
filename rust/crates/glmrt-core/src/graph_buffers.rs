use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::{
    DType, ExpertHostBatch, ExpertHostBatchSet, GlmrtError, GraphBucket, LayerId, LayerWaveMode,
    GLM52_HIDDEN_SIZE, GLM52_ROUTED_EXPERTS, GLM52_TOP_K,
};

pub const EXPERT_GRAPH_ROUTE_ENTRY_BYTES: usize = 12;
pub const EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES: usize = 4;
pub const EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES: usize = 4;
pub const EXPERT_GRAPH_ACTIVE_COUNTS_BYTES: usize = 16;
pub const EXPERT_GRAPH_TILE_METADATA_BYTES: usize = 16;
pub const EXPERT_GRAPH_PROTOCOL_V2_LAYOUT: &str = "expert-protocol-v2-strided";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExpertGraphExecutionEnvelope {
    SparseMoeMixedRows,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphKey {
    pub layer_id: LayerId,
    pub execution_envelope: ExpertGraphExecutionEnvelope,
    pub row_bucket: GraphBucket,
    pub dtype: DType,
    pub quantization_recipe: String,
    pub transport_layout: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenRowsBufferContract {
    pub max_rows: usize,
    pub hidden_dim: usize,
    pub dtype: DType,
    pub row_stride_bytes: usize,
}

impl HiddenRowsBufferContract {
    pub fn bytes(&self) -> usize {
        self.max_rows * self.row_stride_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialOutputBufferContract {
    pub max_rows: usize,
    pub output_dim: usize,
    pub dtype: DType,
    pub row_stride_bytes: usize,
}

impl PartialOutputBufferContract {
    pub fn bytes(&self) -> usize {
        self.max_rows * self.row_stride_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteMetadataBufferContract {
    pub max_rows: usize,
    pub max_local_routes_per_row: usize,
    pub route_entry_bytes: usize,
    pub row_route_count_bytes: usize,
    pub host_row_global_index_bytes: usize,
    pub active_counts_bytes: usize,
}

impl RouteMetadataBufferContract {
    pub fn route_capacity(self) -> usize {
        self.max_rows * self.max_local_routes_per_row
    }

    pub fn bytes(self) -> usize {
        self.route_capacity() * self.route_entry_bytes
            + self.max_rows * self.row_route_count_bytes
            + self.max_rows * self.host_row_global_index_bytes
            + self.active_counts_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertWorkspaceContract {
    pub max_rows_per_expert_bucket: usize,
    pub max_expert_tiles: usize,
    pub tile_metadata_bytes: usize,
    pub partial_accumulator_bytes: usize,
}

impl ExpertWorkspaceContract {
    pub fn bytes(self) -> usize {
        self.partial_accumulator_bytes + self.max_expert_tiles * self.tile_metadata_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphActiveCounts {
    pub rows: usize,
    pub routes: usize,
    pub expert_tiles: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphPoolLease {
    pub lease_id: u64,
    pub key: ExpertGraphKey,
    pub instance_index: usize,
    pub active_counts: ExpertGraphActiveCounts,
    pub fixed_buffer_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphHostBatchLease {
    pub host: String,
    pub lease: ExpertGraphPoolLease,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphHostBatchSetLease {
    pub host_leases: Vec<ExpertGraphHostBatchLease>,
    pub total_fixed_buffer_bytes: usize,
    pub active_counts: ExpertGraphActiveCounts,
}

impl ExpertGraphHostBatchSetLease {
    pub fn num_hosts(&self) -> usize {
        self.host_leases.len()
    }

    pub fn bucket_rows(&self) -> Vec<usize> {
        self.host_leases
            .iter()
            .map(|host_lease| host_lease.lease.key.row_bucket.row_capacity)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphPoolStats {
    pub graph_keys: usize,
    pub total_instances: usize,
    pub available_instances: usize,
    pub in_use_instances: usize,
    pub active_leases: usize,
    pub acquisitions: usize,
    pub reuses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphPoolEntry {
    pub contract: ExpertGraphBufferContract,
    pub total_instances: usize,
    pub acquisitions: usize,
    pub reuses: usize,
    available_instance_indices: Vec<usize>,
}

impl ExpertGraphPoolEntry {
    pub fn available_instances(&self) -> usize {
        self.available_instance_indices.len()
    }

    pub fn in_use_instances(&self) -> usize {
        self.total_instances - self.available_instances()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphInstancePool {
    entries: Vec<ExpertGraphPoolEntry>,
    active_leases: Vec<ExpertGraphPoolLease>,
    next_lease_id: u64,
}

impl Default for ExpertGraphInstancePool {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            active_leases: Vec::new(),
            next_lease_id: 1,
        }
    }
}

impl ExpertGraphInstancePool {
    pub fn new() -> Self {
        Self {
            next_lease_id: 1,
            ..Self::default()
        }
    }

    pub fn register_contract(
        &mut self,
        contract: ExpertGraphBufferContract,
        instances: usize,
    ) -> Result<(), GlmrtError> {
        if instances == 0 {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: "graph pool registration requires at least one instance".to_owned(),
            });
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.contract.key == contract.key)
        {
            if entry.contract != contract {
                return Err(GlmrtError::GraphBufferContractInvalid {
                    reason: format!(
                        "graph pool key {:?} registered with incompatible buffer contract",
                        entry.contract.key
                    ),
                });
            }
            let start = entry.total_instances;
            entry.total_instances += instances;
            entry
                .available_instance_indices
                .extend(start..entry.total_instances);
            return Ok(());
        }

        self.entries.push(ExpertGraphPoolEntry {
            contract,
            total_instances: instances,
            acquisitions: 0,
            reuses: 0,
            available_instance_indices: (0..instances).collect(),
        });
        Ok(())
    }

    pub fn register_glm52_bf16(
        &mut self,
        layer_id: LayerId,
        mode: LayerWaveMode,
        row_bucket: GraphBucket,
        quantization_recipe: impl Into<String>,
        instances: usize,
    ) -> Result<ExpertGraphKey, GlmrtError> {
        let contract =
            ExpertGraphBufferContract::glm52_bf16(layer_id, mode, row_bucket, quantization_recipe)?;
        let key = contract.key.clone();
        self.register_contract(contract, instances)?;
        Ok(key)
    }

    pub fn acquire_for_host_batch(
        &mut self,
        batch: &ExpertHostBatch,
    ) -> Result<ExpertGraphPoolLease, GlmrtError> {
        let Some((entry_index, counts)) = self.best_entry_for_host_batch(batch)? else {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "no registered graph pool entry accepts layer {} host {} rows {} dtype {:?} bucket {} recipe {}",
                    batch.layer_id.0,
                    batch.host,
                    batch.num_rows(),
                    batch.hidden_dtype,
                    batch.graph_bucket.row_capacity,
                    batch.quantization_recipe
                ),
            });
        };
        let entry = &mut self.entries[entry_index];
        let Some(instance_index) = entry.available_instance_indices.pop() else {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "graph pool exhausted for layer {} bucket {} instances {} active {}",
                    entry.contract.key.layer_id.0,
                    entry.contract.key.row_bucket.row_capacity,
                    entry.total_instances,
                    entry.in_use_instances()
                ),
            });
        };
        if entry.acquisitions > 0 {
            entry.reuses += 1;
        }
        entry.acquisitions += 1;
        let lease = ExpertGraphPoolLease {
            lease_id: self.next_lease_id,
            key: entry.contract.key.clone(),
            instance_index,
            active_counts: counts,
            fixed_buffer_bytes: entry.contract.fixed_buffer_bytes(),
        };
        self.next_lease_id += 1;
        self.active_leases.push(lease.clone());
        Ok(lease)
    }

    pub fn acquire_for_host_batch_set(
        &mut self,
        set: &ExpertHostBatchSet,
    ) -> Result<ExpertGraphHostBatchSetLease, GlmrtError> {
        let mut host_leases = Vec::with_capacity(set.batches.len());
        let mut total_fixed_buffer_bytes = 0_usize;
        let mut active_counts = ExpertGraphActiveCounts {
            rows: 0,
            routes: 0,
            expert_tiles: 0,
        };
        for batch in &set.batches {
            match self.acquire_for_host_batch(batch) {
                Ok(lease) => {
                    total_fixed_buffer_bytes += lease.fixed_buffer_bytes;
                    active_counts.rows += lease.active_counts.rows;
                    active_counts.routes += lease.active_counts.routes;
                    active_counts.expert_tiles += lease.active_counts.expert_tiles;
                    host_leases.push(ExpertGraphHostBatchLease {
                        host: batch.host.clone(),
                        lease,
                    });
                }
                Err(error) => {
                    for host_lease in host_leases.into_iter().rev() {
                        let _ = self.release(host_lease.lease);
                    }
                    return Err(error);
                }
            }
        }
        Ok(ExpertGraphHostBatchSetLease {
            host_leases,
            total_fixed_buffer_bytes,
            active_counts,
        })
    }

    pub fn release(&mut self, lease: ExpertGraphPoolLease) -> Result<(), GlmrtError> {
        let Some(active_index) = self
            .active_leases
            .iter()
            .position(|active| active.lease_id == lease.lease_id)
        else {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!("unknown graph pool lease {}", lease.lease_id),
            });
        };
        let active = self.active_leases.swap_remove(active_index);
        if active.key != lease.key || active.instance_index != lease.instance_index {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "graph pool lease {} does not match active lease",
                    lease.lease_id
                ),
            });
        }
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.contract.key == active.key)
        else {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!("graph pool entry disappeared for lease {}", lease.lease_id),
            });
        };
        if entry
            .available_instance_indices
            .contains(&active.instance_index)
        {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "graph pool instance {} for lease {} was already available",
                    active.instance_index, lease.lease_id
                ),
            });
        }
        entry.available_instance_indices.push(active.instance_index);
        Ok(())
    }

    pub fn release_host_batch_set(
        &mut self,
        lease: ExpertGraphHostBatchSetLease,
    ) -> Result<(), GlmrtError> {
        for host_lease in lease.host_leases.into_iter().rev() {
            self.release(host_lease.lease)?;
        }
        Ok(())
    }

    pub fn stats(&self) -> ExpertGraphPoolStats {
        let total_instances = self
            .entries
            .iter()
            .map(|entry| entry.total_instances)
            .sum::<usize>();
        let available_instances = self
            .entries
            .iter()
            .map(ExpertGraphPoolEntry::available_instances)
            .sum::<usize>();
        let acquisitions = self
            .entries
            .iter()
            .map(|entry| entry.acquisitions)
            .sum::<usize>();
        let reuses = self.entries.iter().map(|entry| entry.reuses).sum::<usize>();
        ExpertGraphPoolStats {
            graph_keys: self.entries.len(),
            total_instances,
            available_instances,
            in_use_instances: total_instances - available_instances,
            active_leases: self.active_leases.len(),
            acquisitions,
            reuses,
        }
    }

    pub fn entries(&self) -> &[ExpertGraphPoolEntry] {
        &self.entries
    }

    fn best_entry_for_host_batch(
        &self,
        batch: &ExpertHostBatch,
    ) -> Result<Option<(usize, ExpertGraphActiveCounts)>, GlmrtError> {
        let mut best: Option<(usize, ExpertGraphActiveCounts)> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let Ok(counts) = entry.contract.active_counts_for_host_batch(batch) else {
                continue;
            };
            let is_better = best
                .as_ref()
                .map(|(best_index, _)| {
                    entry.contract.key.row_bucket.row_capacity
                        < self.entries[*best_index]
                            .contract
                            .key
                            .row_bucket
                            .row_capacity
                })
                .unwrap_or(true);
            if is_better {
                best = Some((index, counts));
            }
        }
        Ok(best)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertGraphBufferContract {
    pub key: ExpertGraphKey,
    pub hidden_rows: HiddenRowsBufferContract,
    pub route_metadata: RouteMetadataBufferContract,
    pub workspace: ExpertWorkspaceContract,
    pub partial_outputs: PartialOutputBufferContract,
}

impl ExpertGraphBufferContract {
    pub fn glm52_bf16(
        layer_id: LayerId,
        mode: LayerWaveMode,
        row_bucket: GraphBucket,
        quantization_recipe: impl Into<String>,
    ) -> Result<Self, GlmrtError> {
        Self::glm52(layer_id, mode, row_bucket, DType::Bf16, quantization_recipe)
    }

    pub fn glm52(
        layer_id: LayerId,
        _mode: LayerWaveMode,
        row_bucket: GraphBucket,
        dtype: DType,
        quantization_recipe: impl Into<String>,
    ) -> Result<Self, GlmrtError> {
        let row_stride_bytes = hidden_exchange_row_stride_bytes(&dtype)?;
        let max_rows = row_bucket.row_capacity;
        let route_metadata = RouteMetadataBufferContract {
            max_rows,
            max_local_routes_per_row: GLM52_TOP_K,
            route_entry_bytes: EXPERT_GRAPH_ROUTE_ENTRY_BYTES,
            row_route_count_bytes: EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES,
            host_row_global_index_bytes: EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES,
            active_counts_bytes: EXPERT_GRAPH_ACTIVE_COUNTS_BYTES,
        };
        let max_expert_tiles = route_metadata.route_capacity().min(GLM52_ROUTED_EXPERTS);
        let partial_accumulator_bytes = max_rows * row_stride_bytes;
        Ok(Self {
            key: ExpertGraphKey {
                layer_id,
                execution_envelope: ExpertGraphExecutionEnvelope::SparseMoeMixedRows,
                row_bucket,
                dtype: dtype.clone(),
                quantization_recipe: quantization_recipe.into(),
                transport_layout: EXPERT_GRAPH_PROTOCOL_V2_LAYOUT.to_owned(),
            },
            hidden_rows: HiddenRowsBufferContract {
                max_rows,
                hidden_dim: GLM52_HIDDEN_SIZE,
                dtype: dtype.clone(),
                row_stride_bytes,
            },
            route_metadata,
            workspace: ExpertWorkspaceContract {
                max_rows_per_expert_bucket: max_rows,
                max_expert_tiles,
                tile_metadata_bytes: EXPERT_GRAPH_TILE_METADATA_BYTES,
                partial_accumulator_bytes,
            },
            partial_outputs: PartialOutputBufferContract {
                max_rows,
                output_dim: GLM52_HIDDEN_SIZE,
                dtype,
                row_stride_bytes,
            },
        })
    }

    pub fn request_payload_bytes(&self) -> usize {
        self.hidden_rows.bytes()
    }

    pub fn response_payload_bytes(&self) -> usize {
        self.partial_outputs.bytes()
    }

    pub fn fixed_buffer_bytes(&self) -> usize {
        self.hidden_rows.bytes()
            + self.route_metadata.bytes()
            + self.workspace.bytes()
            + self.partial_outputs.bytes()
    }

    pub fn validate_active_counts(
        &self,
        counts: ExpertGraphActiveCounts,
    ) -> Result<(), GlmrtError> {
        if counts.rows > self.hidden_rows.max_rows {
            return Err(GlmrtError::GraphBufferActiveCountOutOfBounds {
                field: "rows",
                actual: counts.rows,
                capacity: self.hidden_rows.max_rows,
            });
        }
        if counts.routes > self.route_metadata.route_capacity() {
            return Err(GlmrtError::GraphBufferActiveCountOutOfBounds {
                field: "routes",
                actual: counts.routes,
                capacity: self.route_metadata.route_capacity(),
            });
        }
        let route_limit_for_rows = counts.rows * self.route_metadata.max_local_routes_per_row;
        if counts.routes > route_limit_for_rows {
            return Err(GlmrtError::GraphBufferActiveCountOutOfBounds {
                field: "routes_per_active_rows",
                actual: counts.routes,
                capacity: route_limit_for_rows,
            });
        }
        if counts.expert_tiles > self.workspace.max_expert_tiles {
            return Err(GlmrtError::GraphBufferActiveCountOutOfBounds {
                field: "expert_tiles",
                actual: counts.expert_tiles,
                capacity: self.workspace.max_expert_tiles,
            });
        }
        Ok(())
    }

    pub fn active_counts_for_host_batch(
        &self,
        batch: &ExpertHostBatch,
    ) -> Result<ExpertGraphActiveCounts, GlmrtError> {
        if batch.layer_id != self.key.layer_id {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch layer {} does not match graph layer {}",
                    batch.layer_id.0, self.key.layer_id.0
                ),
            });
        }
        if batch.hidden_dtype != self.key.dtype {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch dtype {:?} does not match graph dtype {:?}",
                    batch.hidden_dtype, self.key.dtype
                ),
            });
        }
        if batch.hidden_dim != self.hidden_rows.hidden_dim {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch hidden dim {} does not match graph hidden dim {}",
                    batch.hidden_dim, self.hidden_rows.hidden_dim
                ),
            });
        }
        if batch.hidden_bytes_per_row != self.hidden_rows.row_stride_bytes {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch row stride {} does not match graph row stride {}",
                    batch.hidden_bytes_per_row, self.hidden_rows.row_stride_bytes
                ),
            });
        }
        if batch.graph_bucket.row_capacity > self.key.row_bucket.row_capacity {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch graph bucket {} exceeds graph row bucket {}",
                    batch.graph_bucket.row_capacity, self.key.row_bucket.row_capacity
                ),
            });
        }
        if batch.quantization_recipe != self.key.quantization_recipe {
            return Err(GlmrtError::GraphBufferContractInvalid {
                reason: format!(
                    "host batch quantization recipe {} does not match graph recipe {}",
                    batch.quantization_recipe, self.key.quantization_recipe
                ),
            });
        }

        let expert_tiles = batch
            .routes
            .iter()
            .map(|route| route.expert_id)
            .collect::<BTreeSet<_>>()
            .len();
        let counts = ExpertGraphActiveCounts {
            rows: batch.num_rows(),
            routes: batch.route_count(),
            expert_tiles,
        };
        self.validate_active_counts(counts)?;
        Ok(counts)
    }
}

fn hidden_exchange_row_stride_bytes(dtype: &DType) -> Result<usize, GlmrtError> {
    match dtype {
        DType::Bf16 | DType::F16 => Ok(GLM52_HIDDEN_SIZE * 2),
        other => Err(GlmrtError::GraphBufferContractInvalid {
            reason: format!("hidden exchange dtype {other:?} is not graphable for phase0"),
        }),
    }
}
