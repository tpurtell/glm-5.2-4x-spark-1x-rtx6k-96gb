from __future__ import annotations

import math
from dataclasses import dataclass

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]

PHASE0_STEPPER_ORACLE_STATUS = (
    "executable-bounded: no-torch stepper math covers post-attention "
    "RMSNorm plus dense gated MLP, sparse routed/shared MoE, embedding, "
    "final RMSNorm, and LM-head argmax; real-checkpoint per-stage golden "
    "comparison pending"
)


@dataclass(frozen=True)
class BoundedDenseMlpStepperFixture:
    stage: str
    mode: str
    layer_id: int
    eps: float
    hidden: Matrix
    rms_weight: Vector
    rms_hidden: Matrix
    gate_weight: Matrix
    up_weight: Matrix
    down_weight: Matrix
    gate: Matrix
    up: Matrix
    activated: Matrix
    mlp_output: Matrix
    residual_output: Matrix
    rms_checksum: float
    mlp_output_checksum: float
    residual_output_checksum: float


@dataclass(frozen=True)
class BoundedEmbeddingLmHeadStepperFixture:
    stage: str
    mode: str
    eps: float
    token_ids: tuple[int, ...]
    embedding_weight: Matrix
    embedding_output: Matrix
    final_residual: Matrix
    final_norm_weight: Vector
    final_hidden: Matrix
    lm_head_weight: Matrix
    logits: Matrix
    top_token_ids: tuple[int, ...]
    sampled_token_id: int
    sampled_logit: float
    embedding_checksum: float
    final_hidden_checksum: float
    logits_checksum: float


@dataclass(frozen=True)
class BoundedSparseExpertWeights:
    expert_id: int
    gate_weight: Matrix
    up_weight: Matrix
    down_weight: Matrix


@dataclass(frozen=True)
class BoundedSparseMoeStepperFixture:
    stage: str
    mode: str
    layer_id: int
    eps: float
    top_k: int
    hidden: Matrix
    rms_weight: Vector
    rms_hidden: Matrix
    router_weight: Matrix
    router_logits: Matrix
    router_scores: Matrix
    route_indices: tuple[tuple[int, ...], ...]
    route_weights: Matrix
    expert_weights: tuple[BoundedSparseExpertWeights, ...]
    routed_expert_outputs: tuple[Matrix, ...]
    routed_output: Matrix
    shared_gate_weight: Matrix
    shared_up_weight: Matrix
    shared_down_weight: Matrix
    shared_gate: Matrix
    shared_up: Matrix
    shared_activated: Matrix
    shared_output: Matrix
    sparse_mlp_output: Matrix
    residual_output: Matrix
    router_logits_checksum: float
    routed_output_checksum: float
    shared_output_checksum: float
    sparse_mlp_output_checksum: float
    residual_output_checksum: float


def rmsnorm_rows(rows: Matrix, weight: Vector, eps: float = 1e-5) -> Matrix:
    if not rows:
        raise ValueError("rows must not be empty")
    if not weight:
        raise ValueError("weight must not be empty")
    width = len(weight)
    for row in rows:
        if len(row) != width:
            raise ValueError("row width must match weight width")
    normalized = []
    for row in rows:
        variance = sum(value * value for value in row) / width
        scale = 1.0 / math.sqrt(variance + eps)
        normalized.append(
            tuple(value * scale * weight_value for value, weight_value in zip(row, weight))
        )
    return tuple(normalized)


def embedding_lookup_rows(token_ids: tuple[int, ...], embedding_weight: Matrix) -> Matrix:
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if not embedding_weight:
        raise ValueError("embedding weight must not be empty")
    vocab_size = len(embedding_weight)
    width = len(embedding_weight[0])
    for row in embedding_weight:
        if len(row) != width:
            raise ValueError("embedding rows must have the same width")
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError("token id is outside embedding vocabulary")
    return tuple(embedding_weight[token_id] for token_id in token_ids)


def dense_gated_mlp_rows(
    rows: Matrix,
    gate_weight: Matrix,
    up_weight: Matrix,
    down_weight: Matrix,
) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    if not rows:
        raise ValueError("rows must not be empty")
    hidden = len(rows[0])
    intermediate = len(gate_weight)
    if intermediate == 0 or len(up_weight) != intermediate:
        raise ValueError("gate/up intermediate dimensions must match")
    if not down_weight:
        raise ValueError("down projection must not be empty")
    if any(len(weight_row) != intermediate for weight_row in down_weight):
        raise ValueError("down projection input width must match intermediate width")
    for row in rows:
        if len(row) != hidden:
            raise ValueError("all rows must have the same hidden width")
    for matrix_name, matrix in (("gate", gate_weight), ("up", up_weight)):
        for weight_row in matrix:
            if len(weight_row) != hidden:
                raise ValueError(f"{matrix_name} weight row width must match hidden width")

    gate = tuple(tuple(_dot(row, weight_row) for weight_row in gate_weight) for row in rows)
    up = tuple(tuple(_dot(row, weight_row) for weight_row in up_weight) for row in rows)
    activated = tuple(
        tuple(
            _silu(gate_value) * up_value
            for gate_value, up_value in zip(gate_row, up_row)
        )
        for gate_row, up_row in zip(gate, up)
    )
    output = tuple(
        tuple(_dot(activated_row, weight_row) for weight_row in down_weight)
        for activated_row in activated
    )
    return gate, up, activated, output


def router_topk_rows(
    rows: Matrix,
    router_weight: Matrix,
    top_k: int,
    normalize: bool = True,
) -> tuple[Matrix, Matrix, tuple[tuple[int, ...], ...], Matrix]:
    if not rows:
        raise ValueError("rows must not be empty")
    if not router_weight:
        raise ValueError("router weight must not be empty")
    if top_k <= 0 or top_k > len(router_weight):
        raise ValueError("top_k must be in 1..num_experts")
    hidden = len(rows[0])
    for row in rows:
        if len(row) != hidden:
            raise ValueError("all rows must have the same hidden width")
    for weight_row in router_weight:
        if len(weight_row) != hidden:
            raise ValueError("router weight row width must match hidden width")

    logits = tuple(
        tuple(_dot(row, weight_row) for weight_row in router_weight) for row in rows
    )
    scores = tuple(tuple(_sigmoid(logit) for logit in row) for row in logits)
    route_indices = []
    route_weights = []
    for score_row in scores:
        ordered = sorted(
            range(len(score_row)),
            key=lambda expert_id: (-score_row[expert_id], expert_id),
        )[:top_k]
        selected_scores = tuple(score_row[expert_id] for expert_id in ordered)
        if normalize:
            total = max(sum(selected_scores), 1e-12)
            selected_weights = tuple(score / total for score in selected_scores)
        else:
            selected_weights = selected_scores
        route_indices.append(tuple(ordered))
        route_weights.append(selected_weights)
    return logits, scores, tuple(route_indices), tuple(route_weights)


def sparse_routed_moe_rows(
    rows: Matrix,
    expert_weights: tuple[BoundedSparseExpertWeights, ...],
    route_indices: tuple[tuple[int, ...], ...],
    route_weights: Matrix,
) -> tuple[Matrix, tuple[Matrix, ...]]:
    if not rows:
        raise ValueError("rows must not be empty")
    if len(route_indices) != len(rows) or len(route_weights) != len(rows):
        raise ValueError("route rows must match hidden rows")
    experts = {expert.expert_id: expert for expert in expert_weights}
    hidden = len(rows[0])
    routed_output = tuple(tuple(0.0 for _ in range(hidden)) for _ in rows)
    per_route_outputs = []
    mutable_output = [list(row) for row in routed_output]
    for row_index, (expert_ids, weights) in enumerate(zip(route_indices, route_weights)):
        if len(expert_ids) != len(weights):
            raise ValueError("route index and weight widths must match")
        per_row_outputs = []
        for expert_id, route_weight in zip(expert_ids, weights):
            expert = experts.get(expert_id)
            if expert is None:
                raise ValueError(f"missing expert weights for expert {expert_id}")
            _, _, _, expert_output = dense_gated_mlp_rows(
                (rows[row_index],),
                expert.gate_weight,
                expert.up_weight,
                expert.down_weight,
            )
            output_row = expert_output[0]
            per_row_outputs.append(output_row)
            for column, value in enumerate(output_row):
                mutable_output[row_index][column] += route_weight * value
        per_route_outputs.append(tuple(per_row_outputs))
    return tuple(tuple(row) for row in mutable_output), tuple(per_route_outputs)


def lm_head_logits_rows(rows: Matrix, lm_head_weight: Matrix) -> Matrix:
    if not rows:
        raise ValueError("rows must not be empty")
    if not lm_head_weight:
        raise ValueError("lm_head weight must not be empty")
    hidden = len(rows[0])
    for row in rows:
        if len(row) != hidden:
            raise ValueError("all rows must have the same hidden width")
    for weight_row in lm_head_weight:
        if len(weight_row) != hidden:
            raise ValueError("lm_head weight row width must match hidden width")
    return tuple(
        tuple(_dot(row, weight_row) for weight_row in lm_head_weight)
        for row in rows
    )


def bounded_dense_mlp_stepper_fixture() -> BoundedDenseMlpStepperFixture:
    hidden = (
        (0.2, -0.4, 0.7, 0.1),
        (-0.3, 0.5, 0.25, -0.6),
    )
    rms_weight = (1.0, 0.75, 1.25, 0.5)
    gate_weight = (
        (0.1, -0.2, 0.3, 0.4),
        (-0.5, 0.25, 0.15, -0.1),
        (0.2, 0.1, -0.4, 0.05),
    )
    up_weight = (
        (0.35, -0.15, 0.2, 0.05),
        (0.1, 0.4, -0.25, 0.3),
        (-0.2, 0.05, 0.45, -0.35),
    )
    down_weight = (
        (0.2, -0.1, 0.05),
        (-0.3, 0.25, 0.1),
        (0.15, 0.05, -0.2),
        (0.4, -0.15, 0.3),
    )
    eps = 1e-5
    rms_hidden = rmsnorm_rows(hidden, rms_weight, eps)
    gate, up, activated, mlp_output = dense_gated_mlp_rows(
        rms_hidden,
        gate_weight,
        up_weight,
        down_weight,
    )
    residual_output = tuple(
        tuple(
            hidden_value + delta_value
            for hidden_value, delta_value in zip(hidden_row, delta_row)
        )
        for hidden_row, delta_row in zip(hidden, mlp_output)
    )
    return BoundedDenseMlpStepperFixture(
        stage="post_attention_rmsnorm_mlp",
        mode="dense_gated_mlp",
        layer_id=0,
        eps=eps,
        hidden=hidden,
        rms_weight=rms_weight,
        rms_hidden=rms_hidden,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        gate=gate,
        up=up,
        activated=activated,
        mlp_output=mlp_output,
        residual_output=residual_output,
        rms_checksum=_matrix_checksum(rms_hidden),
        mlp_output_checksum=_matrix_checksum(mlp_output),
        residual_output_checksum=_matrix_checksum(residual_output),
    )


def bounded_sparse_moe_stepper_fixture() -> BoundedSparseMoeStepperFixture:
    hidden = (
        (0.35, -0.25, 0.5, -0.15),
        (-0.45, 0.2, 0.15, 0.55),
    )
    rms_weight = (1.0, 0.85, 1.15, 0.7)
    router_weight = (
        (0.4, -0.1, 0.2, 0.05),
        (-0.2, 0.3, 0.1, 0.25),
        (0.15, 0.05, -0.35, 0.4),
        (-0.1, -0.25, 0.45, -0.2),
    )
    expert_weights = (
        BoundedSparseExpertWeights(
            expert_id=0,
            gate_weight=((0.12, -0.08, 0.22, 0.05), (-0.18, 0.14, 0.06, 0.2)),
            up_weight=((0.05, 0.18, -0.12, 0.16), (0.21, -0.04, 0.13, -0.09)),
            down_weight=((0.18, -0.11), (-0.07, 0.24), (0.13, 0.05), (-0.16, 0.09)),
        ),
        BoundedSparseExpertWeights(
            expert_id=1,
            gate_weight=((-0.09, 0.16, 0.11, -0.04), (0.2, 0.03, -0.15, 0.12)),
            up_weight=((0.14, -0.06, 0.19, 0.08), (-0.05, 0.22, 0.04, -0.17)),
            down_weight=((-0.12, 0.2), (0.17, -0.03), (0.06, 0.15), (0.23, -0.1)),
        ),
        BoundedSparseExpertWeights(
            expert_id=2,
            gate_weight=((0.07, 0.1, -0.21, 0.18), (0.16, -0.13, 0.09, 0.04)),
            up_weight=((-0.18, 0.12, 0.05, 0.2), (0.11, 0.06, -0.14, 0.17)),
            down_weight=((0.09, 0.12), (-0.2, 0.04), (0.15, -0.08), (0.05, 0.19)),
        ),
        BoundedSparseExpertWeights(
            expert_id=3,
            gate_weight=((0.19, -0.05, 0.08, -0.12), (-0.04, 0.21, 0.1, 0.03)),
            up_weight=((0.08, 0.15, -0.07, 0.11), (0.18, -0.1, 0.2, -0.06)),
            down_weight=((0.2, 0.02), (-0.11, 0.18), (0.04, 0.14), (-0.09, 0.21)),
        ),
    )
    shared_gate_weight = (
        (0.11, -0.09, 0.17, 0.06),
        (-0.13, 0.2, 0.04, -0.08),
        (0.07, 0.03, -0.16, 0.18),
    )
    shared_up_weight = (
        (0.18, -0.04, 0.12, 0.1),
        (0.05, 0.16, -0.07, 0.21),
        (-0.1, 0.09, 0.14, -0.05),
    )
    shared_down_weight = (
        (0.16, -0.08, 0.05),
        (-0.12, 0.19, 0.07),
        (0.1, 0.04, -0.14),
        (0.22, -0.06, 0.11),
    )
    eps = 1e-5
    top_k = 2
    rms_hidden = rmsnorm_rows(hidden, rms_weight, eps)
    router_logits, router_scores, route_indices, route_weights = router_topk_rows(
        rms_hidden,
        router_weight,
        top_k,
    )
    routed_output, routed_expert_outputs = sparse_routed_moe_rows(
        rms_hidden,
        expert_weights,
        route_indices,
        route_weights,
    )
    shared_gate, shared_up, shared_activated, shared_output = dense_gated_mlp_rows(
        rms_hidden,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
    )
    sparse_mlp_output = tuple(
        tuple(
            routed_value + shared_value
            for routed_value, shared_value in zip(routed_row, shared_row)
        )
        for routed_row, shared_row in zip(routed_output, shared_output)
    )
    residual_output = tuple(
        tuple(
            hidden_value + delta_value
            for hidden_value, delta_value in zip(hidden_row, delta_row)
        )
        for hidden_row, delta_row in zip(hidden, sparse_mlp_output)
    )
    return BoundedSparseMoeStepperFixture(
        stage="sparse_routed_shared_mlp",
        mode="router_topk_routed_shared",
        layer_id=3,
        eps=eps,
        top_k=top_k,
        hidden=hidden,
        rms_weight=rms_weight,
        rms_hidden=rms_hidden,
        router_weight=router_weight,
        router_logits=router_logits,
        router_scores=router_scores,
        route_indices=route_indices,
        route_weights=route_weights,
        expert_weights=expert_weights,
        routed_expert_outputs=routed_expert_outputs,
        routed_output=routed_output,
        shared_gate_weight=shared_gate_weight,
        shared_up_weight=shared_up_weight,
        shared_down_weight=shared_down_weight,
        shared_gate=shared_gate,
        shared_up=shared_up,
        shared_activated=shared_activated,
        shared_output=shared_output,
        sparse_mlp_output=sparse_mlp_output,
        residual_output=residual_output,
        router_logits_checksum=_matrix_checksum(router_logits),
        routed_output_checksum=_matrix_checksum(routed_output),
        shared_output_checksum=_matrix_checksum(shared_output),
        sparse_mlp_output_checksum=_matrix_checksum(sparse_mlp_output),
        residual_output_checksum=_matrix_checksum(residual_output),
    )


def bounded_embedding_lm_head_stepper_fixture() -> BoundedEmbeddingLmHeadStepperFixture:
    token_ids = (2, 0)
    embedding_weight = (
        (0.05, -0.2, 0.3, 0.1),
        (-0.4, 0.6, -0.1, 0.25),
        (0.7, 0.15, -0.35, 0.45),
        (-0.2, -0.3, 0.55, -0.05),
    )
    residual_delta = (
        (0.02, -0.01, 0.04, -0.03),
        (0.05, 0.02, -0.06, 0.01),
    )
    final_norm_weight = (1.1, 0.8, 1.25, 0.9)
    lm_head_weight = (
        (0.2, -0.1, 0.05, 0.3),
        (-0.15, 0.25, 0.4, -0.05),
        (0.35, 0.1, -0.2, 0.15),
        (-0.05, -0.3, 0.2, 0.25),
    )
    eps = 1e-5
    embedding_output = embedding_lookup_rows(token_ids, embedding_weight)
    final_residual = tuple(
        tuple(base + delta for base, delta in zip(base_row, delta_row))
        for base_row, delta_row in zip(embedding_output, residual_delta)
    )
    final_hidden = rmsnorm_rows(final_residual, final_norm_weight, eps)
    logits = lm_head_logits_rows(final_hidden, lm_head_weight)
    top_token_ids = tuple(
        max(range(len(logit_row)), key=lambda token_id: logit_row[token_id])
        for logit_row in logits
    )
    sampled_token_id = top_token_ids[-1]
    sampled_logit = logits[-1][sampled_token_id]
    return BoundedEmbeddingLmHeadStepperFixture(
        stage="embedding_final_norm_lm_head",
        mode="embedding_norm_argmax",
        eps=eps,
        token_ids=token_ids,
        embedding_weight=embedding_weight,
        embedding_output=embedding_output,
        final_residual=final_residual,
        final_norm_weight=final_norm_weight,
        final_hidden=final_hidden,
        lm_head_weight=lm_head_weight,
        logits=logits,
        top_token_ids=top_token_ids,
        sampled_token_id=sampled_token_id,
        sampled_logit=sampled_logit,
        embedding_checksum=_matrix_checksum(embedding_output),
        final_hidden_checksum=_matrix_checksum(final_hidden),
        logits_checksum=_matrix_checksum(logits),
    )


def bounded_stepper_oracle_status() -> str:
    return PHASE0_STEPPER_ORACLE_STATUS


def _dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("dot product widths must match")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _matrix_checksum(matrix: Matrix) -> float:
    return sum(sum(row) for row in matrix)
