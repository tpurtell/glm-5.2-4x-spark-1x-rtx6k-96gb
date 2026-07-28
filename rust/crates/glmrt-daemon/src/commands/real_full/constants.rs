pub(super) const REAL_GLM_FULL_BLOCKER: &str = "real-glm-full is not runnable yet: full NVFP4 GLM transformer execution, live scheduler-driven real tensor execution, and real sampling path are not implemented in phase0";

pub(super) const REAL_FULL_PREFLIGHT_DECODE_ROWS: usize = 1;
pub(super) const REAL_FULL_PREFLIGHT_MTP_ROWS: usize = 8;
pub(super) const REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS: usize = 4;
pub(super) const REAL_FULL_PREFLIGHT_PREFILL_ROWS: usize = 512;
pub(super) const REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START: u64 = 512;
pub(super) const REAL_FULL_PREFLIGHT_DECODE_POSITION: u64 = 1024;
pub(super) const REAL_FULL_PREFLIGHT_MTP_TOKEN_START: u64 =
    REAL_FULL_PREFLIGHT_DECODE_POSITION + REAL_FULL_PREFLIGHT_DECODE_ROWS as u64;
pub(super) const REAL_FULL_PREFLIGHT_KV_RESERVATION_ID: u64 = 52;
pub(super) const REAL_FULL_PREFLIGHT_REQUEST_ID: &str = "real-full-preflight";
pub(super) const REAL_FULL_PREFLIGHT_SEQUENCE_ID: &str = "real-full-preflight-sequence";
