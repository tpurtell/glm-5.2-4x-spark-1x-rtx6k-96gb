from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlmDsaIndexerMetadata:
    indexer_layer_ids: tuple[int, ...]
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    index_head_dim: int = 128
    dtype_bytes: int = 2
    phase0_attention_math_status: str = (
        "metadata-only: DSA/indexer cache placement and byte accounting are covered; "
        "full DSA attention math golden is pending real attention reference integration"
    )

    @property
    def main_mla_bytes_per_token(self) -> int:
        return 78 * (self.kv_lora_rank + self.qk_rope_head_dim) * self.dtype_bytes

    @property
    def dsa_indexer_bytes_per_token(self) -> int:
        return len(self.indexer_layer_ids) * self.index_head_dim * self.dtype_bytes

    @property
    def compressed_bf16_bytes_per_token(self) -> int:
        return self.main_mla_bytes_per_token + self.dsa_indexer_bytes_per_token

    def layer_has_dsa_indexer(self, layer_id: int) -> bool:
        return layer_id in self.indexer_layer_ids


def glm52_dsa_indexer_metadata() -> GlmDsaIndexerMetadata:
    return GlmDsaIndexerMetadata(
        indexer_layer_ids=(
            0,
            1,
            2,
            6,
            10,
            14,
            18,
            22,
            26,
            30,
            34,
            38,
            42,
            46,
            50,
            54,
            58,
            62,
            66,
            70,
            74,
        )
    )
