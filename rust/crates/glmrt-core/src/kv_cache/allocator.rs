use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{GlmrtError, KvBlockDescriptor, LayerId, PositionId};

use super::KvCacheConfig;

mod snapshot;

pub use snapshot::KvCacheSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvReservationState {
    Active,
    Paused,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvWriteState {
    Pending,
    Written,
    Paused,
    Tentative,
    Committed,
    Discarded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvReservation {
    pub id: u64,
    pub sequence_id: String,
    pub tokens: usize,
    pub bytes: usize,
    pub state: KvReservationState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvWriteRecord {
    pub id: u64,
    pub reservation_id: u64,
    pub sequence_id: String,
    pub layer_id: LayerId,
    pub token_start: PositionId,
    pub token_count: usize,
    pub state: KvWriteState,
}

impl KvWriteRecord {
    pub fn token_end(&self) -> u64 {
        self.token_start.0 + self.token_count as u64
    }

    pub fn is_visible_to_attention(&self) -> bool {
        matches!(self.state, KvWriteState::Written | KvWriteState::Committed)
    }
}

#[derive(Debug, Clone)]
pub struct KvCacheAllocator {
    config: KvCacheConfig,
    next_id: u64,
    next_write_id: u64,
    reservations: BTreeMap<u64, KvReservation>,
    writes: BTreeMap<u64, KvWriteRecord>,
    writes_by_reservation_layer: BTreeMap<(u64, LayerId), Vec<u64>>,
    tentative_writes_by_reservation_layer: BTreeMap<(u64, LayerId), Vec<u64>>,
}

impl KvCacheAllocator {
    pub fn new(config: KvCacheConfig) -> Self {
        Self {
            config,
            next_id: 1,
            next_write_id: 1,
            reservations: BTreeMap::new(),
            writes: BTreeMap::new(),
            writes_by_reservation_layer: BTreeMap::new(),
            tentative_writes_by_reservation_layer: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    pub fn reserve(
        &mut self,
        sequence_id: impl Into<String>,
        tokens: usize,
    ) -> Result<u64, GlmrtError> {
        let resident_tokens = self.resident_tokens();
        if resident_tokens + tokens > self.config.max_tokens {
            return Err(GlmrtError::KvCapacityExceeded {
                requested_tokens: tokens,
                available_tokens: self.config.max_tokens.saturating_sub(resident_tokens),
            });
        }
        let id = self.next_id;
        self.next_id += 1;
        self.reservations.insert(
            id,
            KvReservation {
                id,
                sequence_id: sequence_id.into(),
                tokens,
                bytes: tokens * self.config.bytes_per_token(),
                state: KvReservationState::Active,
            },
        );
        Ok(id)
    }

    pub fn pause(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition(id, KvReservationState::Paused)
    }

    pub fn resume(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition(id, KvReservationState::Active)
    }

    pub fn release(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition(id, KvReservationState::Released)
    }

    pub fn reservation(&self, id: u64) -> Option<&KvReservation> {
        self.reservations.get(&id)
    }

    pub fn record_prefill_write(
        &mut self,
        descriptor: KvBlockDescriptor,
    ) -> Result<u64, GlmrtError> {
        self.record_write_with_state(descriptor, KvWriteState::Pending)
    }

    pub fn record_tentative_write(
        &mut self,
        descriptor: KvBlockDescriptor,
    ) -> Result<u64, GlmrtError> {
        self.record_write_with_state(descriptor, KvWriteState::Tentative)
    }

    fn record_write_with_state(
        &mut self,
        descriptor: KvBlockDescriptor,
        state: KvWriteState,
    ) -> Result<u64, GlmrtError> {
        let reservation = self
            .reservations
            .get(&descriptor.reservation_id)
            .ok_or(GlmrtError::UnknownKvReservation(descriptor.reservation_id))?;
        let token_start = descriptor.token_start.0 as usize;
        let token_end = token_start.saturating_add(descriptor.token_count);
        if token_end > reservation.tokens {
            return Err(GlmrtError::KvWriteOutOfBounds {
                token_start,
                token_count: descriptor.token_count,
                reservation_tokens: reservation.tokens,
            });
        }
        let id = self.next_write_id;
        self.next_write_id += 1;
        let reservation_layer = (descriptor.reservation_id, descriptor.layer_id);
        self.writes.insert(
            id,
            KvWriteRecord {
                id,
                reservation_id: descriptor.reservation_id,
                sequence_id: descriptor.sequence_id,
                layer_id: descriptor.layer_id,
                token_start: descriptor.token_start,
                token_count: descriptor.token_count,
                state,
            },
        );
        self.writes_by_reservation_layer
            .entry(reservation_layer)
            .or_default()
            .push(id);
        if state == KvWriteState::Tentative {
            self.tentative_writes_by_reservation_layer
                .entry(reservation_layer)
                .or_default()
                .push(id);
        }
        Ok(id)
    }

    pub fn mark_write_written(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition_write(id, KvWriteState::Written)
    }

    pub fn resolve_mtp_tentative_writes(
        &mut self,
        reservation_id: u64,
        layer_id: LayerId,
        token_start: PositionId,
        draft_tokens: usize,
        accepted_tokens: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), GlmrtError> {
        if accepted_tokens > draft_tokens {
            return Err(GlmrtError::MtpAcceptedTokensExceedDraft {
                accepted_tokens,
                draft_tokens,
            });
        }
        let draft_start = token_start.0;
        let accepted_end = draft_start + accepted_tokens as u64;
        let draft_end = draft_start + draft_tokens as u64;
        let reservation_layer = (reservation_id, layer_id);
        let write_ids = self
            .tentative_writes_by_reservation_layer
            .remove(&reservation_layer)
            .unwrap_or_default();
        let mut unresolved = Vec::new();
        let mut committed = Vec::new();
        let mut discarded = Vec::new();
        for write_id in write_ids {
            let Some(write) = self.writes.get_mut(&write_id) else {
                continue;
            };
            if write.state != KvWriteState::Tentative {
                continue;
            }
            let write_start = write.token_start.0;
            let write_end = write.token_end();
            if write_start < draft_start || write_end > draft_end {
                unresolved.push(write_id);
                continue;
            }
            write.state = if write_end <= accepted_end {
                committed.push(write_id);
                KvWriteState::Committed
            } else {
                discarded.push(write_id);
                KvWriteState::Discarded
            };
        }
        if !unresolved.is_empty() {
            self.tentative_writes_by_reservation_layer
                .insert(reservation_layer, unresolved);
        }
        Ok((committed, discarded))
    }

    /// Resolve one layer's ordered, single-token MTP writes without searching
    /// the reservation history. The caller already receives these IDs when it
    /// records a tentative wave, so retaining them makes commit/rollback O(k).
    pub fn resolve_mtp_tentative_write_ids(
        &mut self,
        write_ids: &[u64],
        accepted_tokens: usize,
    ) -> Result<Vec<u64>, GlmrtError> {
        if accepted_tokens > write_ids.len() {
            return Err(GlmrtError::MtpAcceptedTokensExceedDraft {
                accepted_tokens,
                draft_tokens: write_ids.len(),
            });
        }
        if write_ids.is_empty() {
            return Ok(Vec::new());
        }

        let first = self
            .writes
            .get(&write_ids[0])
            .ok_or(GlmrtError::UnknownKvWrite(write_ids[0]))?;
        let reservation_id = first.reservation_id;
        let layer_id = first.layer_id;
        let sequence_id = first.sequence_id.clone();
        let token_start = first.token_start.0;
        let reservation_layer = (reservation_id, layer_id);

        for (offset, write_id) in write_ids.iter().copied().enumerate() {
            let write = self
                .writes
                .get(&write_id)
                .ok_or(GlmrtError::UnknownKvWrite(write_id))?;
            let expected_token_start = token_start.checked_add(offset as u64).ok_or_else(|| {
                GlmrtError::InvalidMtpKvTransaction {
                    reason: "ordered write positions overflow u64".to_owned(),
                }
            })?;
            if write.reservation_id != reservation_id
                || write.layer_id != layer_id
                || write.sequence_id != sequence_id
                || write.token_start.0 != expected_token_start
                || write.token_count != 1
                || write.state != KvWriteState::Tentative
            {
                return Err(GlmrtError::InvalidMtpKvTransaction {
                    reason: format!(
                        "write {write_id} at offset {offset} is not the expected tentative single-token write for reservation {reservation_id}, layer {}, position {expected_token_start}",
                        layer_id.0
                    ),
                });
            }
        }

        let mut discarded = Vec::with_capacity(write_ids.len() - accepted_tokens);
        for (offset, write_id) in write_ids.iter().copied().enumerate() {
            let write = self
                .writes
                .get_mut(&write_id)
                .expect("direct MTP transaction was prevalidated");
            if offset < accepted_tokens {
                write.state = KvWriteState::Committed;
            } else {
                write.state = KvWriteState::Discarded;
                discarded.push(write_id);
            }
        }
        if let Some(tentative) = self
            .tentative_writes_by_reservation_layer
            .get_mut(&reservation_layer)
        {
            tentative.retain(|write_id| !write_ids.contains(write_id));
            if tentative.is_empty() {
                self.tentative_writes_by_reservation_layer
                    .remove(&reservation_layer);
            }
        }
        Ok(discarded)
    }

    pub fn discard_writes_from(
        &mut self,
        reservation_id: u64,
        layer_id: LayerId,
        token_start: PositionId,
    ) -> Vec<u64> {
        let write_ids = self
            .writes_by_reservation_layer
            .get(&(reservation_id, layer_id))
            .cloned()
            .unwrap_or_default();
        let mut discarded = Vec::new();
        for write_id in write_ids {
            let Some(write) = self.writes.get_mut(&write_id) else {
                continue;
            };
            if write.token_start.0 < token_start.0 || write.state == KvWriteState::Discarded {
                continue;
            }
            write.state = KvWriteState::Discarded;
            discarded.push(write_id);
        }
        if let Some(tentative) = self
            .tentative_writes_by_reservation_layer
            .get_mut(&(reservation_id, layer_id))
        {
            tentative.retain(|write_id| !discarded.contains(write_id));
            if tentative.is_empty() {
                self.tentative_writes_by_reservation_layer
                    .remove(&(reservation_id, layer_id));
            }
        }
        discarded
    }

    pub fn pause_write(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition_write(id, KvWriteState::Paused)
    }

    pub fn resume_write(&mut self, id: u64) -> Result<(), GlmrtError> {
        self.transition_write(id, KvWriteState::Pending)
    }

    pub fn write(&self, id: u64) -> Option<&KvWriteRecord> {
        self.writes.get(&id)
    }

    pub fn writes_for_reservation(&self, reservation_id: u64) -> Vec<&KvWriteRecord> {
        self.writes
            .values()
            .filter(|write| write.reservation_id == reservation_id)
            .collect()
    }

    pub fn writes_for_reservation_layer(
        &self,
        reservation_id: u64,
        layer_id: LayerId,
    ) -> Vec<&KvWriteRecord> {
        self.writes_by_reservation_layer
            .get(&(reservation_id, layer_id))
            .into_iter()
            .flatten()
            .filter_map(|write_id| self.writes.get(write_id))
            .collect()
    }

    pub fn visible_writes_for_decode(
        &self,
        reservation_id: u64,
        layer_id: LayerId,
        decode_position: PositionId,
    ) -> Vec<&KvWriteRecord> {
        self.writes_for_reservation_layer(reservation_id, layer_id)
            .into_iter()
            .filter(|write| {
                write.is_visible_to_attention() && write.token_end() <= decode_position.0
            })
            .collect()
    }

    fn transition(&mut self, id: u64, state: KvReservationState) -> Result<(), GlmrtError> {
        let reservation = self
            .reservations
            .get_mut(&id)
            .ok_or(GlmrtError::UnknownKvReservation(id))?;
        reservation.state = state;
        Ok(())
    }

    fn transition_write(&mut self, id: u64, state: KvWriteState) -> Result<(), GlmrtError> {
        let write = self
            .writes
            .get_mut(&id)
            .ok_or(GlmrtError::UnknownKvWrite(id))?;
        write.state = state;
        Ok(())
    }
}
