use crate::commands::real_full::types::RealFullBoundedAttentionOracleStepperEvidence;

pub(super) fn bounded_attention_oracle_stepper_evidence(
) -> RealFullBoundedAttentionOracleStepperEvidence {
    RealFullBoundedAttentionOracleStepperEvidence {
        status: "retired".to_owned(),
        source: "retired-stepper-validation-artifacts".to_owned(),
        stepper_stage: "retired".to_owned(),
        skipped_reason: Some(
            "stepper validation artifacts were removed after phase0 live token output completed"
                .to_owned(),
        ),
        ..RealFullBoundedAttentionOracleStepperEvidence::default()
    }
}
