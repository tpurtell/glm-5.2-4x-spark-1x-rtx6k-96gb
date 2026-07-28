"""Readable reference implementations for GLMRT phase0."""

from .glm_dsa_metadata_ref import GlmDsaIndexerMetadata, glm52_dsa_indexer_metadata
from .glm_attention_ref import (
    BoundedGlmAttentionResult,
    bounded_attention_oracle_status,
    bounded_attention_oracle_summary,
    bounded_dsa_indexer_attention_fixture,
    bounded_main_mla_attention_fixture,
)
from .glm52_config import ShapeConfig, glm_shape_config, tiny_config
from .glm_stepper_ref import (
    BoundedDenseMlpStepperFixture,
    BoundedEmbeddingLmHeadStepperFixture,
    BoundedSparseExpertWeights,
    BoundedSparseMoeStepperFixture,
    bounded_dense_mlp_stepper_fixture,
    bounded_embedding_lm_head_stepper_fixture,
    bounded_sparse_moe_stepper_fixture,
    bounded_stepper_oracle_status,
)

__all__ = [
    "BoundedDenseMlpStepperFixture",
    "BoundedEmbeddingLmHeadStepperFixture",
    "BoundedGlmAttentionResult",
    "BoundedSparseExpertWeights",
    "BoundedSparseMoeStepperFixture",
    "GlmDsaIndexerMetadata",
    "ShapeConfig",
    "bounded_attention_oracle_status",
    "bounded_attention_oracle_summary",
    "bounded_dense_mlp_stepper_fixture",
    "bounded_embedding_lm_head_stepper_fixture",
    "bounded_sparse_moe_stepper_fixture",
    "bounded_dsa_indexer_attention_fixture",
    "bounded_main_mla_attention_fixture",
    "bounded_stepper_oracle_status",
    "glm52_dsa_indexer_metadata",
    "glm_shape_config",
    "tiny_config",
]
