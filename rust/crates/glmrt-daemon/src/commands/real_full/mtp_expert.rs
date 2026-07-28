use anyhow::{bail, Result};
use std::env;

pub(crate) const MTP_BF16_EXPERTS_ENV: &str = "GLMRT_MTP_BF16_EXPERTS";

fn parse_bool(name: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{name} must be a boolean, got {value:?}"),
    }
}

/// Retain a checkpoint's BF16 layer-78 routed experts instead of quantizing
/// them to the normal packed NVFP4 serving representation at startup.
///
/// This is intentionally one process-wide switch shared by the coordinator
/// and every Spark. In retained-BF16 mode the coordinator must send BF16
/// activations and request BF16 partials, and the Sparks must not route those
/// partials through the global FP8 intermediate-reduction policy.
pub(crate) fn mtp_bf16_experts_enabled() -> Result<bool> {
    env::var(MTP_BF16_EXPERTS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or(Ok(false), |value| parse_bool(MTP_BF16_EXPERTS_ENV, &value))
}

#[cfg(test)]
mod tests {
    use super::parse_bool;

    #[test]
    fn mtp_bf16_expert_switch_accepts_explicit_booleans() {
        assert!(parse_bool("test", "true").unwrap());
        assert!(parse_bool("test", "ON").unwrap());
        assert!(!parse_bool("test", "0").unwrap());
        assert!(parse_bool("test", "bf16").is_err());
    }
}
