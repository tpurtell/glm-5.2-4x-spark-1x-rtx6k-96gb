use serde::{Deserialize, Serialize};

use super::{KvCacheAllocator, KvReservationState, KvWriteState};
use crate::kv_cache::config::KvCacheConfig;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheSnapshot {
    pub config: KvCacheConfig,
    pub bytes_per_token: usize,
    pub capacity_bytes: usize,
    pub resident_tokens: usize,
    pub resident_bytes: usize,
    pub active_reservations: usize,
    pub paused_reservations: usize,
    pub pending_writes: usize,
    pub written_writes: usize,
    pub paused_writes: usize,
    pub tentative_writes: usize,
    pub committed_writes: usize,
    pub discarded_writes: usize,
}

impl KvCacheAllocator {
    pub fn resident_tokens(&self) -> usize {
        self.reservations
            .values()
            .filter(|reservation| reservation.state != KvReservationState::Released)
            .map(|reservation| reservation.tokens)
            .sum()
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_tokens() * self.config.bytes_per_token()
    }

    pub fn snapshot(&self) -> KvCacheSnapshot {
        KvCacheSnapshot {
            config: self.config.clone(),
            bytes_per_token: self.config.bytes_per_token(),
            capacity_bytes: self.config.capacity_bytes(),
            resident_tokens: self.resident_tokens(),
            resident_bytes: self.resident_bytes(),
            active_reservations: self
                .reservations
                .values()
                .filter(|reservation| reservation.state == KvReservationState::Active)
                .count(),
            paused_reservations: self
                .reservations
                .values()
                .filter(|reservation| reservation.state == KvReservationState::Paused)
                .count(),
            pending_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Pending)
                .count(),
            written_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Written)
                .count(),
            paused_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Paused)
                .count(),
            tentative_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Tentative)
                .count(),
            committed_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Committed)
                .count(),
            discarded_writes: self
                .writes
                .values()
                .filter(|write| write.state == KvWriteState::Discarded)
                .count(),
        }
    }
}
