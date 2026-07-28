use std::collections::{BTreeMap, BTreeSet};

use crate::{GlmrtError, KvBlockDescriptor, LayerId, LayerWave, PositionId};

use super::{KvCacheAllocator, KvCacheConfig, KvCacheSnapshot, KvWriteState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBackedBlock {
    pub write_id: u64,
    pub descriptor: KvBlockDescriptor,
    pub state: KvWriteState,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct KvCacheBackingStore {
    allocator: KvCacheAllocator,
    blocks: BTreeMap<u64, Vec<u8>>,
    attention_metadata_runs: BTreeMap<(u64, LayerId), Vec<KvBlockDescriptor>>,
    attention_metadata_invalid_layers: BTreeSet<(u64, LayerId)>,
}

impl KvCacheBackingStore {
    pub fn new(config: KvCacheConfig) -> Self {
        Self {
            allocator: KvCacheAllocator::new(config),
            blocks: BTreeMap::new(),
            attention_metadata_runs: BTreeMap::new(),
            attention_metadata_invalid_layers: BTreeSet::new(),
        }
    }

    pub fn config(&self) -> &KvCacheConfig {
        self.allocator.config()
    }

    pub fn reserve(
        &mut self,
        sequence_id: impl Into<String>,
        tokens: usize,
    ) -> Result<u64, GlmrtError> {
        self.allocator.reserve(sequence_id, tokens)
    }

    pub fn snapshot(&self) -> KvCacheSnapshot {
        self.allocator.snapshot()
    }

    pub fn write_committed_block(
        &mut self,
        descriptor: KvBlockDescriptor,
        payload: Vec<u8>,
    ) -> Result<u64, GlmrtError> {
        self.validate_payload(&descriptor, payload.len())?;
        self.invalidate_attention_metadata_layer(descriptor.reservation_id, descriptor.layer_id);
        let write_id = self.allocator.record_prefill_write(descriptor)?;
        self.blocks.insert(write_id, payload);
        self.allocator.mark_write_written(write_id)?;
        Ok(write_id)
    }

    pub fn write_committed_block_metadata(
        &mut self,
        descriptor: KvBlockDescriptor,
    ) -> Result<u64, GlmrtError> {
        let write_id = self.allocator.record_prefill_write(descriptor.clone())?;
        self.blocks.insert(write_id, Vec::new());
        self.allocator.mark_write_written(write_id)?;
        self.record_attention_metadata_descriptor(descriptor);
        Ok(write_id)
    }

    pub fn write_committed_blocks_for_wave(
        &mut self,
        wave: &LayerWave,
        payloads: Vec<Vec<u8>>,
    ) -> Result<Vec<u64>, GlmrtError> {
        self.validate_payload_count(wave.kv_writes.len(), payloads.len())?;
        wave.kv_writes
            .iter()
            .cloned()
            .zip(payloads)
            .map(|(descriptor, payload)| self.write_committed_block(descriptor, payload))
            .collect()
    }

    pub fn write_committed_block_metadata_for_wave(
        &mut self,
        wave: &LayerWave,
    ) -> Result<Vec<u64>, GlmrtError> {
        wave.kv_writes
            .iter()
            .cloned()
            .map(|descriptor| self.write_committed_block_metadata(descriptor))
            .collect()
    }

    pub fn write_tentative_block(
        &mut self,
        descriptor: KvBlockDescriptor,
        payload: Vec<u8>,
    ) -> Result<u64, GlmrtError> {
        self.validate_payload(&descriptor, payload.len())?;
        self.invalidate_attention_metadata_layer(descriptor.reservation_id, descriptor.layer_id);
        let write_id = self.allocator.record_tentative_write(descriptor)?;
        self.blocks.insert(write_id, payload);
        Ok(write_id)
    }

    pub fn write_tentative_block_metadata(
        &mut self,
        descriptor: KvBlockDescriptor,
    ) -> Result<u64, GlmrtError> {
        let write_id = self.allocator.record_tentative_write(descriptor)?;
        self.blocks.insert(write_id, Vec::new());
        Ok(write_id)
    }

    pub fn write_tentative_blocks_for_wave(
        &mut self,
        wave: &LayerWave,
        payloads: Vec<Vec<u8>>,
    ) -> Result<Vec<u64>, GlmrtError> {
        self.validate_payload_count(wave.tentative_kv_writes.len(), payloads.len())?;
        wave.tentative_kv_writes
            .iter()
            .cloned()
            .zip(payloads)
            .map(|(descriptor, payload)| self.write_tentative_block(descriptor, payload))
            .collect()
    }

    pub fn write_tentative_block_metadata_for_wave(
        &mut self,
        wave: &LayerWave,
    ) -> Result<Vec<u64>, GlmrtError> {
        wave.tentative_kv_writes
            .iter()
            .cloned()
            .map(|descriptor| self.write_tentative_block_metadata(descriptor))
            .collect()
    }

    pub fn resolve_mtp_tentative_writes(
        &mut self,
        reservation_id: u64,
        layer_id: LayerId,
        token_start: PositionId,
        draft_tokens: usize,
        accepted_tokens: usize,
    ) -> Result<(), GlmrtError> {
        let (committed, discarded) = self.allocator.resolve_mtp_tentative_writes(
            reservation_id,
            layer_id,
            token_start,
            draft_tokens,
            accepted_tokens,
        )?;
        let committed_descriptors = committed
            .iter()
            .filter_map(|write_id| self.allocator.write(*write_id))
            .map(|write| KvBlockDescriptor {
                reservation_id: write.reservation_id,
                sequence_id: write.sequence_id.clone(),
                layer_id: write.layer_id,
                token_start: write.token_start,
                token_count: write.token_count,
            })
            .collect::<Vec<_>>();
        for descriptor in committed_descriptors {
            self.record_attention_metadata_descriptor(descriptor);
        }
        for write_id in discarded {
            self.blocks.remove(&write_id);
        }
        Ok(())
    }

    /// Candidate direct transaction path. Production range-based callers are
    /// intentionally unchanged until end-to-end MTP measurements are possible.
    pub fn resolve_mtp_tentative_write_ids(
        &mut self,
        write_ids: &[u64],
        accepted_tokens: usize,
    ) -> Result<(), GlmrtError> {
        let discarded = self
            .allocator
            .resolve_mtp_tentative_write_ids(write_ids, accepted_tokens)?;
        let committed_descriptors = write_ids[..accepted_tokens]
            .iter()
            .filter_map(|write_id| self.allocator.write(*write_id))
            .map(|write| KvBlockDescriptor {
                reservation_id: write.reservation_id,
                sequence_id: write.sequence_id.clone(),
                layer_id: write.layer_id,
                token_start: write.token_start,
                token_count: write.token_count,
            })
            .collect::<Vec<_>>();
        for descriptor in committed_descriptors {
            self.record_attention_metadata_descriptor(descriptor);
        }
        for write_id in discarded {
            self.blocks.remove(&write_id);
        }
        Ok(())
    }

    pub fn discard_writes_from(
        &mut self,
        reservation_id: u64,
        layer_id: LayerId,
        token_start: PositionId,
    ) -> usize {
        let discarded = self
            .allocator
            .discard_writes_from(reservation_id, layer_id, token_start);
        for write_id in &discarded {
            self.blocks.remove(write_id);
        }
        self.truncate_attention_metadata_runs(reservation_id, layer_id, token_start);
        discarded.len()
    }

    pub fn read_visible_blocks_for_decode(
        &self,
        reservation_id: u64,
        layer_id: LayerId,
        decode_position: PositionId,
    ) -> Vec<KvBackedBlock> {
        let mut blocks = self
            .allocator
            .visible_writes_for_decode(reservation_id, layer_id, decode_position)
            .into_iter()
            .filter_map(|write| {
                self.blocks.get(&write.id).map(|bytes| KvBackedBlock {
                    write_id: write.id,
                    descriptor: KvBlockDescriptor {
                        reservation_id: write.reservation_id,
                        sequence_id: write.sequence_id.clone(),
                        layer_id: write.layer_id,
                        token_start: write.token_start,
                        token_count: write.token_count,
                    },
                    state: write.state,
                    bytes: bytes.clone(),
                })
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| (block.descriptor.token_start, block.write_id));
        blocks
    }

    pub fn read_visible_blocks_for_descriptor(
        &self,
        descriptor: &KvBlockDescriptor,
    ) -> Vec<KvBackedBlock> {
        let read_start = descriptor.token_start.0;
        let read_end = read_start + descriptor.token_count as u64;
        let mut blocks = self
            .allocator
            .writes_for_reservation_layer(descriptor.reservation_id, descriptor.layer_id)
            .into_iter()
            .filter(|write| {
                write.is_visible_to_attention()
                    && write.token_start.0 >= read_start
                    && write.token_end() <= read_end
            })
            .filter_map(|write| {
                self.blocks.get(&write.id).map(|bytes| KvBackedBlock {
                    write_id: write.id,
                    descriptor: KvBlockDescriptor {
                        reservation_id: write.reservation_id,
                        sequence_id: write.sequence_id.clone(),
                        layer_id: write.layer_id,
                        token_start: write.token_start,
                        token_count: write.token_count,
                    },
                    state: write.state,
                    bytes: bytes.clone(),
                })
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| (block.descriptor.token_start, block.write_id));
        blocks
    }

    pub fn read_visible_blocks_for_wave(&self, wave: &LayerWave) -> Vec<KvBackedBlock> {
        let mut blocks = wave
            .kv_reads
            .iter()
            .flat_map(|descriptor| self.read_visible_blocks_for_descriptor(descriptor))
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| {
            (
                block.descriptor.layer_id,
                block.descriptor.token_start,
                block.write_id,
            )
        });
        blocks
    }

    pub fn read_attention_blocks_for_wave(&self, wave: &LayerWave) -> Vec<KvBackedBlock> {
        if wave.kv_reads.iter().any(|descriptor| {
            self.attention_metadata_invalid_layers
                .contains(&(descriptor.reservation_id, descriptor.layer_id))
        }) {
            return self.read_visible_blocks_for_wave(wave);
        }

        let mut blocks = wave
            .kv_reads
            .iter()
            .flat_map(|read| {
                let read_start = read.token_start.0;
                let read_end = read_start + read.token_count as u64;
                self.attention_metadata_runs
                    .get(&(read.reservation_id, read.layer_id))
                    .into_iter()
                    .flatten()
                    .filter_map(move |run| {
                        let run_start = run.token_start.0;
                        let run_end = run_start + run.token_count as u64;
                        let token_start = read_start.max(run_start);
                        let token_end = read_end.min(run_end);
                        (token_start < token_end).then(|| KvBackedBlock {
                            write_id: 0,
                            descriptor: KvBlockDescriptor {
                                reservation_id: read.reservation_id,
                                sequence_id: read.sequence_id.clone(),
                                layer_id: read.layer_id,
                                token_start: PositionId(token_start),
                                token_count: (token_end - token_start) as usize,
                            },
                            state: KvWriteState::Written,
                            bytes: Vec::new(),
                        })
                    })
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| {
            (
                block.descriptor.layer_id,
                block.descriptor.token_start,
                block.write_id,
            )
        });
        blocks
    }

    pub fn backed_write_bytes(&self) -> usize {
        self.blocks.values().map(Vec::len).sum()
    }

    pub fn backed_write_count(&self) -> usize {
        self.blocks.len()
    }

    fn validate_payload(
        &self,
        descriptor: &KvBlockDescriptor,
        actual_bytes: usize,
    ) -> Result<(), GlmrtError> {
        let expected_bytes = self
            .config()
            .layer_payload_bytes(descriptor.layer_id, descriptor.token_count);
        if actual_bytes != expected_bytes {
            return Err(GlmrtError::KvBackingPayloadSizeMismatch {
                expected_bytes,
                actual_bytes,
            });
        }
        Ok(())
    }

    fn validate_payload_count(
        &self,
        expected_blocks: usize,
        actual_blocks: usize,
    ) -> Result<(), GlmrtError> {
        if actual_blocks != expected_blocks {
            return Err(GlmrtError::KvBackingPayloadCountMismatch {
                expected_blocks,
                actual_blocks,
            });
        }
        Ok(())
    }

    fn invalidate_attention_metadata_layer(&mut self, reservation_id: u64, layer_id: LayerId) {
        let key = (reservation_id, layer_id);
        self.attention_metadata_invalid_layers.insert(key);
        self.attention_metadata_runs.remove(&key);
    }

    fn record_attention_metadata_descriptor(&mut self, descriptor: KvBlockDescriptor) {
        let key = (descriptor.reservation_id, descriptor.layer_id);
        if self.attention_metadata_invalid_layers.contains(&key) {
            return;
        }
        let runs = self.attention_metadata_runs.entry(key).or_default();
        runs.push(descriptor);
        runs.sort_by_key(|run| run.token_start);

        let mut merged: Vec<KvBlockDescriptor> = Vec::with_capacity(runs.len());
        for run in std::mem::take(runs) {
            let run_end = run.token_start.0 + run.token_count as u64;
            if let Some(previous) = merged.last_mut() {
                let previous_end = previous.token_start.0 + previous.token_count as u64;
                if previous.sequence_id == run.sequence_id && run.token_start.0 <= previous_end {
                    previous.token_count =
                        previous_end.max(run_end) as usize - previous.token_start.0 as usize;
                    continue;
                }
            }
            merged.push(run);
        }
        *runs = merged;
    }

    fn truncate_attention_metadata_runs(
        &mut self,
        reservation_id: u64,
        layer_id: LayerId,
        token_start: PositionId,
    ) {
        let Some(runs) = self
            .attention_metadata_runs
            .get_mut(&(reservation_id, layer_id))
        else {
            return;
        };
        runs.retain_mut(|run| {
            let run_start = run.token_start.0;
            if run_start >= token_start.0 {
                return false;
            }
            let run_end = run_start + run.token_count as u64;
            if run_end > token_start.0 {
                run.token_count = (token_start.0 - run_start) as usize;
            }
            run.token_count > 0
        });
    }
}
