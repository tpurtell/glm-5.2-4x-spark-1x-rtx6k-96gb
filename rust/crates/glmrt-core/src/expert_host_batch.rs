use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    owner_for_expert, DType, ExpertBatch, ExpertOwnerLookup, GlmrtError, GraphBucket, LayerId,
    PlacementPolicy, PlacementVersion, PositionId, RequestId, RowSourceKind,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertBatchRoute {
    pub row_index: usize,
    pub expert_id: usize,
    pub gate_weight: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertHostBatchRow {
    pub global_row_index: usize,
    pub row_id: u64,
    pub source_kind: RowSourceKind,
    pub request_id: RequestId,
    pub sequence_id: String,
    pub token_position: PositionId,
    pub route_offset: usize,
    pub route_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertHostBatch {
    pub host: String,
    pub layer_id: LayerId,
    pub placement_version: PlacementVersion,
    pub hidden_dim: usize,
    pub hidden_bytes_per_row: usize,
    pub hidden_dtype: DType,
    pub graph_bucket: GraphBucket,
    pub quantization_recipe: String,
    pub rows: Vec<ExpertHostBatchRow>,
    pub routes: Vec<ExpertBatchRoute>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRowToGlobalRowMap {
    pub host: String,
    pub global_row_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialReconstructionPlan {
    pub global_row_count: usize,
    pub host_row_maps: Vec<HostRowToGlobalRowMap>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertHostBatchSet {
    pub global_row_count: usize,
    pub batches: Vec<ExpertHostBatch>,
    pub reconstruction_plan: PartialReconstructionPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHostBatchSetAccumulation {
    pub values: Vec<f32>,
    pub contribution_counts: Vec<usize>,
}

impl PartialReconstructionPlan {
    pub fn validate_for_batches(
        &self,
        batches: &[ExpertHostBatch],
        global_row_count: usize,
    ) -> Result<(), GlmrtError> {
        if self.global_row_count != global_row_count {
            return Err(
                GlmrtError::ExpertHostBatchSetReconstructionPlanGlobalRowCountMismatch {
                    expected: global_row_count,
                    actual: self.global_row_count,
                },
            );
        }
        if self.host_row_maps.len() != batches.len() {
            return Err(
                GlmrtError::ExpertHostBatchSetReconstructionPlanHostCountMismatch {
                    expected: batches.len(),
                    actual: self.host_row_maps.len(),
                },
            );
        }

        validate_unique_hosts(batches.iter().map(|batch| batch.host.as_str()))?;
        validate_unique_hosts(self.host_row_maps.iter().map(|map| map.host.as_str()))?;

        for (map_index, (batch, host_map)) in
            batches.iter().zip(self.host_row_maps.iter()).enumerate()
        {
            if host_map.host != batch.host {
                return Err(
                    GlmrtError::ExpertHostBatchSetReconstructionPlanHostMismatch {
                        map_index,
                        expected: batch.host.clone(),
                        actual: host_map.host.clone(),
                    },
                );
            }
            if host_map.global_row_indices.len() != batch.rows.len() {
                return Err(
                    GlmrtError::ExpertHostBatchSetReconstructionPlanRowCountMismatch {
                        host: host_map.host.clone(),
                        expected: batch.rows.len(),
                        actual: host_map.global_row_indices.len(),
                    },
                );
            }
            for (host_row_index, (actual, row)) in host_map
                .global_row_indices
                .iter()
                .zip(batch.rows.iter())
                .enumerate()
            {
                if *actual >= global_row_count {
                    return Err(GlmrtError::ExpertHostBatchGlobalRowOutOfBounds {
                        row_index: *actual,
                        row_count: global_row_count,
                    });
                }
                if *actual != row.global_row_index {
                    return Err(
                        GlmrtError::ExpertHostBatchSetReconstructionPlanGlobalRowMismatch {
                            host: host_map.host.clone(),
                            host_row_index,
                            expected: row.global_row_index,
                            actual: *actual,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    pub fn accumulate_partial_outputs_f32<T: AsRef<[f32]>>(
        &self,
        partial_outputs_by_host: &[Vec<T>],
        output_dim: usize,
    ) -> Result<ExpertHostBatchSetAccumulation, GlmrtError> {
        if partial_outputs_by_host.len() != self.host_row_maps.len() {
            return Err(GlmrtError::ExpertHostBatchSetPartialHostCountMismatch {
                expected: self.host_row_maps.len(),
                actual: partial_outputs_by_host.len(),
            });
        }
        validate_unique_hosts(self.host_row_maps.iter().map(|map| map.host.as_str()))?;

        let expected_values = self
            .global_row_count
            .checked_mul(output_dim)
            .ok_or_else(|| GlmrtError::GraphBufferContractInvalid {
                reason: "partial reconstruction accumulator value count overflow".to_owned(),
            })?;
        let mut values = vec![0.0_f32; expected_values];
        let mut contribution_counts = vec![0_usize; self.global_row_count];

        for (host_map, partial_outputs) in self
            .host_row_maps
            .iter()
            .zip(partial_outputs_by_host.iter())
        {
            if partial_outputs.len() != host_map.global_row_indices.len() {
                return Err(GlmrtError::ExpertHostBatchPartialRowCountMismatch {
                    expected: host_map.global_row_indices.len(),
                    actual: partial_outputs.len(),
                });
            }

            for (host_row_index, (global_row_index, partial)) in host_map
                .global_row_indices
                .iter()
                .zip(partial_outputs.iter())
                .enumerate()
            {
                if *global_row_index >= self.global_row_count {
                    return Err(GlmrtError::ExpertHostBatchGlobalRowOutOfBounds {
                        row_index: *global_row_index,
                        row_count: self.global_row_count,
                    });
                }
                let partial = partial.as_ref();
                if partial.len() != output_dim {
                    return Err(GlmrtError::ExpertHostBatchPartialOutputWidthMismatch {
                        row_index: host_row_index,
                        expected_width: output_dim,
                        actual_width: partial.len(),
                    });
                }

                let start = *global_row_index * output_dim;
                for (target, delta) in values[start..start + output_dim]
                    .iter_mut()
                    .zip(partial.iter())
                {
                    *target += *delta;
                }
                contribution_counts[*global_row_index] += 1;
            }
        }

        for (row_index, count) in contribution_counts.iter().enumerate() {
            if *count == 0 {
                return Err(
                    GlmrtError::ExpertHostBatchSetReconstructionPlanMissingGlobalRow { row_index },
                );
            }
        }

        Ok(ExpertHostBatchSetAccumulation {
            values,
            contribution_counts,
        })
    }
}

impl ExpertHostBatch {
    pub fn replicated_from_expert_batch(
        batch: &ExpertBatch,
        host: impl Into<String>,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
    ) -> Result<Self, GlmrtError> {
        let host = host.into();
        let route_owner = host.clone();
        Self::from_expert_batch_with_route_owner(batch, host, routes, expert_hosts, move |_, _| {
            Some(route_owner.clone())
        })
    }

    pub fn from_expert_batch(
        batch: &ExpertBatch,
        host: impl Into<String>,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
        placement_policy: PlacementPolicy,
    ) -> Result<Self, GlmrtError> {
        Self::from_expert_batch_with_route_owner(
            batch,
            host,
            routes,
            expert_hosts,
            |layer_id, expert_id| {
                owner_for_expert(layer_id, expert_id, expert_hosts, placement_policy)
            },
        )
    }

    pub fn from_expert_batch_with_owner_lookup(
        batch: &ExpertBatch,
        host: impl Into<String>,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
        owner_lookup: &ExpertOwnerLookup,
    ) -> Result<Self, GlmrtError> {
        Self::from_expert_batch_with_route_owner(
            batch,
            host,
            routes,
            expert_hosts,
            |layer_id, expert_id| {
                owner_lookup
                    .owner_for(layer_id, expert_id)
                    .map(str::to_owned)
            },
        )
    }

    fn from_expert_batch_with_route_owner(
        batch: &ExpertBatch,
        host: impl Into<String>,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
        mut route_owner: impl FnMut(usize, usize) -> Option<String>,
    ) -> Result<Self, GlmrtError> {
        let host = host.into();
        if !expert_hosts.iter().any(|candidate| candidate == &host) {
            return Err(GlmrtError::ExpertHostBatchUnknownHost { host });
        }
        if routes.len() != batch.route_count() {
            return Err(GlmrtError::ExpertHostBatchRouteCountMismatch {
                expected: batch.route_count(),
                actual: routes.len(),
            });
        }

        let mut rows = Vec::new();
        let mut local_routes = Vec::new();
        for (global_row_index, row) in batch.rows.iter().enumerate() {
            let end = row.route_offset + row.route_count;
            let row_routes = routes.get(row.route_offset..end).ok_or(
                GlmrtError::ExpertHostBatchRouteRangeOutOfBounds {
                    row_index: global_row_index,
                    start: row.route_offset,
                    end,
                    route_count: routes.len(),
                },
            )?;
            let host_row_index = rows.len();
            let route_offset = local_routes.len();
            for route in row_routes {
                if route.row_index != global_row_index {
                    return Err(GlmrtError::ExpertHostBatchRouteRowMismatch {
                        expected: global_row_index,
                        actual: route.row_index,
                    });
                }
                if route_owner(batch.layer_id.0 as usize, route.expert_id)
                    .as_deref()
                    .map(|owner| host_matches(owner, &host))
                    .unwrap_or(false)
                {
                    local_routes.push(ExpertBatchRoute {
                        row_index: host_row_index,
                        expert_id: route.expert_id,
                        gate_weight: route.gate_weight,
                    });
                }
            }

            let route_count = local_routes.len() - route_offset;
            if route_count > 0 {
                rows.push(ExpertHostBatchRow {
                    global_row_index,
                    row_id: row.row_id,
                    source_kind: row.source_kind,
                    request_id: row.request_id.clone(),
                    sequence_id: row.sequence_id.clone(),
                    token_position: row.token_position,
                    route_offset,
                    route_count,
                });
            }
        }

        Ok(Self {
            host,
            layer_id: batch.layer_id,
            placement_version: batch.placement_version.clone(),
            hidden_dim: batch.hidden_dim,
            hidden_bytes_per_row: batch.hidden_bytes_per_row,
            hidden_dtype: batch.hidden_dtype.clone(),
            graph_bucket: batch.graph_bucket,
            quantization_recipe: batch.quantization_recipe.clone(),
            rows,
            routes: local_routes,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    pub fn global_row_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows.iter().map(|row| row.global_row_index)
    }

    pub fn compact_hidden_payload(
        &self,
        global_hidden_payload: &[u8],
        global_row_count: usize,
    ) -> Result<Vec<u8>, GlmrtError> {
        let expected_bytes = global_row_count * self.hidden_bytes_per_row;
        if global_hidden_payload.len() != expected_bytes {
            return Err(GlmrtError::ExpertHostBatchHiddenPayloadSizeMismatch {
                expected_bytes,
                actual_bytes: global_hidden_payload.len(),
            });
        }

        let mut compact = Vec::with_capacity(self.num_rows() * self.hidden_bytes_per_row);
        for row in &self.rows {
            self.validate_global_row_index(row.global_row_index, global_row_count)?;
            let start = row.global_row_index * self.hidden_bytes_per_row;
            compact.extend_from_slice(
                &global_hidden_payload[start..start + self.hidden_bytes_per_row],
            );
        }
        Ok(compact)
    }

    pub fn scatter_partial_outputs<T: Clone>(
        &self,
        partial_outputs: &[T],
        global_row_count: usize,
    ) -> Result<Vec<Option<T>>, GlmrtError> {
        if partial_outputs.len() != self.rows.len() {
            return Err(GlmrtError::ExpertHostBatchPartialRowCountMismatch {
                expected: self.rows.len(),
                actual: partial_outputs.len(),
            });
        }

        let mut scattered = vec![None; global_row_count];
        for (row, output) in self.rows.iter().zip(partial_outputs.iter()) {
            self.validate_global_row_index(row.global_row_index, global_row_count)?;
            scattered[row.global_row_index] = Some(output.clone());
        }
        Ok(scattered)
    }

    pub fn accumulate_partial_outputs_f32<T: AsRef<[f32]>>(
        &self,
        partial_outputs: &[T],
        global_row_count: usize,
        output_dim: usize,
        global_accumulator: &mut [f32],
        contribution_counts: &mut [usize],
    ) -> Result<(), GlmrtError> {
        if partial_outputs.len() != self.rows.len() {
            return Err(GlmrtError::ExpertHostBatchPartialRowCountMismatch {
                expected: self.rows.len(),
                actual: partial_outputs.len(),
            });
        }
        let expected_values = global_row_count.checked_mul(output_dim).ok_or_else(|| {
            GlmrtError::GraphBufferContractInvalid {
                reason: "partial output accumulator value count overflow".to_owned(),
            }
        })?;
        if global_accumulator.len() != expected_values {
            return Err(GlmrtError::ExpertHostBatchPartialAccumulatorSizeMismatch {
                expected_values,
                actual_values: global_accumulator.len(),
            });
        }
        if contribution_counts.len() != global_row_count {
            return Err(GlmrtError::ExpertHostBatchContributionCountMismatch {
                expected: global_row_count,
                actual: contribution_counts.len(),
            });
        }

        for (host_row_index, (row, partial)) in
            self.rows.iter().zip(partial_outputs.iter()).enumerate()
        {
            let partial = partial.as_ref();
            self.validate_global_row_index(row.global_row_index, global_row_count)?;
            if partial.len() != output_dim {
                return Err(GlmrtError::ExpertHostBatchPartialOutputWidthMismatch {
                    row_index: host_row_index,
                    expected_width: output_dim,
                    actual_width: partial.len(),
                });
            }
            let start = row.global_row_index * output_dim;
            for (target, delta) in global_accumulator[start..start + output_dim]
                .iter_mut()
                .zip(partial.iter())
            {
                *target += *delta;
            }
            contribution_counts[row.global_row_index] += 1;
        }
        Ok(())
    }

    fn validate_global_row_index(
        &self,
        row_index: usize,
        global_row_count: usize,
    ) -> Result<(), GlmrtError> {
        if row_index >= global_row_count {
            return Err(GlmrtError::ExpertHostBatchGlobalRowOutOfBounds {
                row_index,
                row_count: global_row_count,
            });
        }
        Ok(())
    }
}

impl ExpertHostBatchSet {
    pub fn replicated_from_expert_batch(
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
    ) -> Result<Self, GlmrtError> {
        let batches = expert_hosts
            .iter()
            .map(|host| {
                ExpertHostBatch::replicated_from_expert_batch(
                    batch,
                    host.clone(),
                    routes,
                    expert_hosts,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_batches_with_route_replicas(batch, batches, expert_hosts.len())
    }

    pub fn from_expert_batch(
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
        placement_policy: PlacementPolicy,
    ) -> Result<Self, GlmrtError> {
        let mut batches = Vec::new();
        for host in expert_hosts {
            let host_batch = ExpertHostBatch::from_expert_batch(
                batch,
                host.clone(),
                routes,
                expert_hosts,
                placement_policy,
            )?;
            if host_batch.route_count() > 0 {
                batches.push(host_batch);
            }
        }

        Self::from_batches(batch, batches)
    }

    pub fn from_expert_batch_with_owner_lookup(
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        expert_hosts: &[String],
        owner_lookup: &ExpertOwnerLookup,
    ) -> Result<Self, GlmrtError> {
        let mut batches = Vec::new();
        for host in expert_hosts {
            let host_batch = ExpertHostBatch::from_expert_batch_with_owner_lookup(
                batch,
                host.clone(),
                routes,
                expert_hosts,
                owner_lookup,
            )?;
            if host_batch.route_count() > 0 {
                batches.push(host_batch);
            }
        }

        Self::from_batches(batch, batches)
    }

    fn from_batches(
        batch: &ExpertBatch,
        batches: Vec<ExpertHostBatch>,
    ) -> Result<Self, GlmrtError> {
        Self::from_batches_with_route_replicas(batch, batches, 1)
    }

    fn from_batches_with_route_replicas(
        batch: &ExpertBatch,
        batches: Vec<ExpertHostBatch>,
        route_replicas: usize,
    ) -> Result<Self, GlmrtError> {
        let total_routes = batches
            .iter()
            .map(ExpertHostBatch::route_count)
            .sum::<usize>();
        let expected_routes = batch
            .route_count()
            .checked_mul(route_replicas)
            .ok_or_else(|| GlmrtError::GraphBufferContractInvalid {
                reason: "replicated host-batch route count overflow".to_owned(),
            })?;
        if total_routes != expected_routes {
            return Err(GlmrtError::ExpertHostBatchSetRouteCountMismatch {
                expected: expected_routes,
                actual: total_routes,
            });
        }
        let reconstruction_plan = PartialReconstructionPlan {
            global_row_count: batch.num_rows(),
            host_row_maps: batches
                .iter()
                .map(|batch| HostRowToGlobalRowMap {
                    host: batch.host.clone(),
                    global_row_indices: batch.global_row_indices().collect(),
                })
                .collect(),
        };

        Ok(Self {
            global_row_count: batch.num_rows(),
            batches,
            reconstruction_plan,
        })
    }

    pub fn num_hosts(&self) -> usize {
        self.batches.len()
    }

    pub fn route_count(&self) -> usize {
        self.batches.iter().map(ExpertHostBatch::route_count).sum()
    }

    pub fn host_row_count(&self) -> usize {
        self.batches.iter().map(ExpertHostBatch::num_rows).sum()
    }

    pub fn touched_hosts(&self) -> impl Iterator<Item = &str> {
        self.batches.iter().map(|batch| batch.host.as_str())
    }

    pub fn compact_hidden_payloads(
        &self,
        global_hidden_payload: &[u8],
    ) -> Result<Vec<Vec<u8>>, GlmrtError> {
        self.batches
            .iter()
            .map(|batch| batch.compact_hidden_payload(global_hidden_payload, self.global_row_count))
            .collect()
    }

    pub fn accumulate_partial_outputs_f32<T: AsRef<[f32]>>(
        &self,
        partial_outputs_by_host: &[Vec<T>],
        output_dim: usize,
    ) -> Result<ExpertHostBatchSetAccumulation, GlmrtError> {
        self.reconstruction_plan
            .validate_for_batches(&self.batches, self.global_row_count)?;
        self.reconstruction_plan
            .accumulate_partial_outputs_f32(partial_outputs_by_host, output_dim)
    }
}

fn validate_unique_hosts<'a>(hosts: impl IntoIterator<Item = &'a str>) -> Result<(), GlmrtError> {
    let mut seen = BTreeSet::new();
    for host in hosts {
        if !seen.insert(host) {
            return Err(GlmrtError::ExpertHostBatchSetDuplicateHost {
                host: host.to_owned(),
            });
        }
    }
    Ok(())
}

fn host_matches(assignment_owner: &str, requested_owner: &str) -> bool {
    assignment_owner == requested_owner
        || assignment_owner.split('.').next() == Some(requested_owner)
        || requested_owner.split('.').next() == Some(assignment_owner)
}
