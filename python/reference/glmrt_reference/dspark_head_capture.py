from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import triton
import triton.language as tl

from b12x_mla_capture import _bf16_tensor, _f32_tensor, _i32_tensor
from dspark_capture import _i64_tensor


_DSPARK_HEAD_STATES: dict[tuple[Any, ...], "_DsparkHeadState"] = {}
_ARGMAX_BLOCK = 512


@dataclass(frozen=True)
class _DsparkHeadState:
    device_id: int
    cuda_stream: int
    active_requests: int
    proposal_tokens: int
    hidden_rows_per_request: int
    hidden_start_row: int
    hidden_size: int
    markov_rank: int
    vocab_size: int
    hidden: Any
    hidden_position_major: Any
    base_logits: Any
    markov_logits: Any
    argmax_candidate_scores: Any
    argmax_candidate_tokens: Any
    embedding_steps: Any
    token_steps: Any
    confidence_features: Any
    confidence_logits: Any
    confidence_probabilities: Any
    anchor_tokens: Any
    output_tokens: Any
    output_confidence: Any
    reference_tokens: Any
    reference_confidence: Any
    eager_tokens: Any
    eager_confidence: Any
    lm_head_t: Any
    markov_w1: Any
    markov_w2_t: Any
    confidence_weight_t: Any
    confidence_bias: Any


def prepare_dspark_head(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Bind, initialize, and warm one fixed-address dSpark head."""

    import torch

    state = _head_state(ctx, kwargs, create=True)
    _run_reference(state)
    _run_head(state)
    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        state.eager_tokens.copy_(state.output_tokens)
        state.eager_confidence.copy_(state.output_confidence)


def capture_dspark_head(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch the allocation-free dSpark head during external capture."""

    _run_head(_head_state(ctx, kwargs, create=False))


def _head_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DsparkHeadState:
    active_requests = int(kwargs["active_requests"])
    proposal_tokens = int(kwargs["proposal_tokens"])
    hidden_rows_per_request = int(
        kwargs.get("hidden_rows_per_request", proposal_tokens)
    )
    hidden_start_row = int(kwargs.get("hidden_start_row", 0))
    hidden_size = int(kwargs["hidden_size"])
    markov_rank = int(kwargs["markov_rank"])
    vocab_size = int(kwargs["vocab_size"])
    seed = int(kwargs["seed"])
    initialize_hidden = bool(kwargs.get("initialize_hidden", True))
    if active_requests not in (1, 2, 4):
        raise ValueError(
            "dSpark head active_requests must be one of 1, 2, or 4, "
            f"got {active_requests}"
        )
    if proposal_tokens not in (7, 8, 15):
        raise ValueError(
            f"dSpark head requires 7, 8, or 15 native predictions, got {proposal_tokens}"
        )
    if (
        hidden_start_row < 0
        or hidden_rows_per_request < 1
        or hidden_start_row + proposal_tokens > hidden_rows_per_request
    ):
        raise ValueError(
            "dSpark head proposal rows exceed the hidden source: "
            f"start={hidden_start_row}, proposals={proposal_tokens}, "
            f"source_rows={hidden_rows_per_request}"
        )
    if hidden_size != 6144 or markov_rank != 256 or vocab_size != 154880:
        raise ValueError(
            "GLM-5.2 dSpark head requires hidden/rank/vocab 6144/256/154880, "
            f"got {hidden_size}/{markov_rank}/{vocab_size}"
        )

    buffers = ctx["buffers"]
    mutable_names = (
        "hidden",
        "hidden_position_major",
        "base_logits",
        "markov_logits",
        "argmax_candidate_scores",
        "argmax_candidate_tokens",
        "embedding_steps",
        "token_steps",
        "confidence_features",
        "confidence_logits",
        "confidence_probabilities",
        "anchor_tokens",
        "output_tokens",
        "output_confidence",
        "reference_tokens",
        "reference_confidence",
        "eager_tokens",
        "eager_confidence",
    )
    weight_names = (
        "lm_head",
        "markov_w1",
        "markov_w2",
        "confidence_weight",
        "confidence_bias",
    )
    required_names = (*mutable_names, *weight_names)
    missing = [name for name in required_names if name not in buffers]
    if missing:
        raise ValueError(f"dSpark head is missing buffers: {missing}")

    device_id = int(buffers["hidden"]["device_id"])
    for name in required_names:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(f"dSpark head buffer {name} is on another CUDA device")

    rows = active_requests * proposal_tokens
    feature_width = hidden_size + markov_rank
    argmax_blocks = triton.cdiv(vocab_size, _ARGMAX_BLOCK)
    shapes = {
        "hidden": (active_requests, hidden_rows_per_request, hidden_size),
        "hidden_position_major": (proposal_tokens, active_requests, hidden_size),
        "base_logits": (proposal_tokens, active_requests, vocab_size),
        "markov_logits": (active_requests, vocab_size),
        "argmax_candidate_scores": (active_requests, argmax_blocks),
        "argmax_candidate_tokens": (active_requests, argmax_blocks),
        "embedding_steps": (proposal_tokens, active_requests, markov_rank),
        "token_steps": (proposal_tokens, active_requests),
        "confidence_features": (rows, feature_width),
        "confidence_logits": (rows,),
        "confidence_probabilities": (rows,),
        "anchor_tokens": (active_requests,),
        "output_tokens": (active_requests, proposal_tokens),
        "output_confidence": (active_requests, proposal_tokens),
        "reference_tokens": (active_requests, proposal_tokens),
        "reference_confidence": (active_requests, proposal_tokens),
        "eager_tokens": (active_requests, proposal_tokens),
        "eager_confidence": (active_requests, proposal_tokens),
        "lm_head": (vocab_size, hidden_size),
        "markov_w1": (vocab_size, markov_rank),
        "markov_w2": (vocab_size, markov_rank),
        "confidence_weight": (1, feature_width),
        "confidence_bias": (1,),
    }
    for name, shape in shapes.items():
        element_bytes = 2
        if name in (
            "anchor_tokens",
            "token_steps",
            "output_tokens",
            "reference_tokens",
            "eager_tokens",
        ):
            element_bytes = 8
        elif name in (
            "argmax_candidate_scores",
            "argmax_candidate_tokens",
            "output_confidence",
            "reference_confidence",
            "eager_confidence",
        ):
            element_bytes = 4
        required_bytes = _tensor_bytes(shape, element_bytes)
        if int(buffers[name]["bytes"]) < required_bytes:
            raise ValueError(
                f"dSpark head buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires {required_bytes}"
            )

    key = (
        int(ctx["cuda_stream"]),
        active_requests,
        proposal_tokens,
        hidden_rows_per_request,
        hidden_start_row,
        hidden_size,
        markov_rank,
        vocab_size,
        initialize_hidden,
        *((name, int(buffers[name]["ptr"])) for name in required_names),
    )
    state = _DSPARK_HEAD_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError("dSpark head capture requires a matching startup prepare call")

    import torch

    cuda_stream = int(ctx["cuda_stream"])
    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream), torch.no_grad():
        tensors: dict[str, Any] = {}
        bf16_names = (
            "hidden",
            "hidden_position_major",
            "base_logits",
            "markov_logits",
            "embedding_steps",
            "confidence_features",
            "confidence_logits",
            "confidence_probabilities",
            "lm_head",
            "markov_w1",
            "markov_w2",
            "confidence_weight",
            "confidence_bias",
        )
        for name in bf16_names:
            tensors[name] = _bf16_tensor(buffers[name], shapes[name])
        i64_names = (
            "token_steps",
            "anchor_tokens",
            "output_tokens",
            "reference_tokens",
            "eager_tokens",
        )
        for name in i64_names:
            tensors[name] = _i64_tensor(buffers[name], shapes[name])
        tensors["argmax_candidate_tokens"] = _i32_tensor(
            buffers["argmax_candidate_tokens"],
            shapes["argmax_candidate_tokens"],
        )
        f32_names = (
            "argmax_candidate_scores",
            "output_confidence",
            "reference_confidence",
            "eager_confidence",
        )
        for name in f32_names:
            tensors[name] = _f32_tensor(buffers[name], shapes[name])

        if initialize_hidden:
            generator = torch.Generator(device=device_id)
            generator.manual_seed(seed)
            tensors["hidden"].normal_(generator=generator)

    hidden = tensors["hidden"][
        :, hidden_start_row : hidden_start_row + proposal_tokens, :
    ]

    state = _DsparkHeadState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        active_requests=active_requests,
        proposal_tokens=proposal_tokens,
        hidden_rows_per_request=hidden_rows_per_request,
        hidden_start_row=hidden_start_row,
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        vocab_size=vocab_size,
        hidden=hidden,
        hidden_position_major=tensors["hidden_position_major"],
        base_logits=tensors["base_logits"],
        markov_logits=tensors["markov_logits"],
        argmax_candidate_scores=tensors["argmax_candidate_scores"],
        argmax_candidate_tokens=tensors["argmax_candidate_tokens"],
        embedding_steps=tensors["embedding_steps"],
        token_steps=tensors["token_steps"],
        confidence_features=tensors["confidence_features"],
        confidence_logits=tensors["confidence_logits"],
        confidence_probabilities=tensors["confidence_probabilities"],
        anchor_tokens=tensors["anchor_tokens"],
        output_tokens=tensors["output_tokens"],
        output_confidence=tensors["output_confidence"],
        reference_tokens=tensors["reference_tokens"],
        reference_confidence=tensors["reference_confidence"],
        eager_tokens=tensors["eager_tokens"],
        eager_confidence=tensors["eager_confidence"],
        lm_head_t=tensors["lm_head"].t(),
        markov_w1=tensors["markov_w1"],
        markov_w2_t=tensors["markov_w2"].t(),
        confidence_weight_t=tensors["confidence_weight"].t(),
        confidence_bias=tensors["confidence_bias"],
    )
    _DSPARK_HEAD_STATES[key] = state
    return state


def _base_logits(state: _DsparkHeadState) -> None:
    import torch

    state.hidden_position_major.copy_(state.hidden.permute(1, 0, 2))
    torch.mm(
        state.hidden_position_major.view(-1, state.hidden_size),
        state.lm_head_t,
        out=state.base_logits.view(-1, state.vocab_size),
    )


def _run_reference(state: _DsparkHeadState) -> None:
    import torch
    import torch.nn.functional as functional

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        _base_logits(state)
        previous = state.anchor_tokens
        for position in range(state.proposal_tokens):
            previous_embedding = functional.embedding(previous, state.markov_w1)
            markov_bias = functional.linear(previous_embedding, state.markov_w2_t.t())
            next_token = torch.argmax(
                state.base_logits[position] + markov_bias, dim=-1
            )
            confidence_features = torch.cat(
                (state.hidden[:, position], previous_embedding), dim=-1
            )
            confidence = torch.sigmoid(
                functional.linear(
                    confidence_features,
                    state.confidence_weight_t.t(),
                    state.confidence_bias,
                ).squeeze(-1)
            )
            state.reference_tokens[:, position].copy_(next_token)
            state.reference_confidence[:, position].copy_(confidence)
            previous = next_token


def _run_head(state: _DsparkHeadState) -> None:
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        _base_logits(state)
        previous = state.anchor_tokens
        for position in range(state.proposal_tokens):
            current_embedding = state.embedding_steps[position]
            torch.index_select(
                state.markov_w1,
                0,
                previous,
                out=current_embedding,
            )
            torch.mm(
                current_embedding,
                state.markov_w2_t,
                out=state.markov_logits,
            )
            _fused_add_argmax(
                state.base_logits[position],
                state.markov_logits,
                state.argmax_candidate_scores,
                state.argmax_candidate_tokens,
                state.token_steps[position],
            )
            previous = state.token_steps[position]

        state.output_tokens.copy_(state.token_steps.t())
        features = state.confidence_features.view(
            state.active_requests,
            state.proposal_tokens,
            state.hidden_size + state.markov_rank,
        )
        features[:, :, : state.hidden_size].copy_(state.hidden)
        features[:, :, state.hidden_size :].copy_(
            state.embedding_steps.permute(1, 0, 2)
        )
        torch.mm(
            state.confidence_features,
            state.confidence_weight_t,
            out=state.confidence_logits.view(-1, 1),
        )
        torch.add(
            state.confidence_logits,
            state.confidence_bias,
            out=state.confidence_logits,
        )
        torch.sigmoid(
            state.confidence_logits,
            out=state.confidence_probabilities,
        )
        state.output_confidence.copy_(
            state.confidence_probabilities.view(
                state.active_requests, state.proposal_tokens
            )
        )


@triton.jit
def _combined_block_argmax(
    base_logits,
    markov_logits,
    candidate_scores,
    candidate_tokens,
    BASE_STRIDE_ROW: tl.constexpr,
    MARKOV_STRIDE_ROW: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    token_offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token_offsets < VOCAB_SIZE
    base = tl.load(
        base_logits + row * BASE_STRIDE_ROW + token_offsets,
        mask=valid,
        other=-float("inf"),
    )
    markov = tl.load(
        markov_logits + row * MARKOV_STRIDE_ROW + token_offsets,
        mask=valid,
        other=0.0,
    )
    # Match torch.add(..., out=BF16): round before argmax observes the value.
    combined = (base + markov).to(tl.bfloat16).to(tl.float32)
    best_score = tl.max(combined, axis=0)
    best_token = tl.min(
        tl.where(combined == best_score, token_offsets, VOCAB_SIZE),
        axis=0,
    )
    candidate_offset = row * NUM_BLOCKS + block
    tl.store(candidate_scores + candidate_offset, best_score)
    tl.store(candidate_tokens + candidate_offset, best_token)


@triton.jit
def _finalize_block_argmax(
    candidate_scores,
    candidate_tokens,
    output_tokens,
    OUTPUT_STRIDE_ROW: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block_offsets = tl.arange(0, REDUCTION_SIZE)
    valid = block_offsets < NUM_BLOCKS
    candidate_offset = row * NUM_BLOCKS + block_offsets
    scores = tl.load(
        candidate_scores + candidate_offset,
        mask=valid,
        other=-float("inf"),
    )
    tokens = tl.load(
        candidate_tokens + candidate_offset,
        mask=valid,
        other=VOCAB_SIZE,
    )
    best_score = tl.max(scores, axis=0)
    best_token = tl.min(
        tl.where((scores == best_score) & valid, tokens, VOCAB_SIZE),
        axis=0,
    )
    tl.store(output_tokens + row * OUTPUT_STRIDE_ROW, best_token.to(tl.int64))


def _fused_add_argmax(
    base_logits: Any,
    markov_logits: Any,
    candidate_scores: Any,
    candidate_tokens: Any,
    output_tokens: Any,
) -> None:
    rows, vocab_size = base_logits.shape
    num_blocks = triton.cdiv(vocab_size, _ARGMAX_BLOCK)
    _combined_block_argmax[(rows, num_blocks)](
        base_logits,
        markov_logits,
        candidate_scores,
        candidate_tokens,
        BASE_STRIDE_ROW=base_logits.stride(0),
        MARKOV_STRIDE_ROW=markov_logits.stride(0),
        VOCAB_SIZE=vocab_size,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=_ARGMAX_BLOCK,
    )
    _finalize_block_argmax[(rows,)](
        candidate_scores,
        candidate_tokens,
        output_tokens,
        OUTPUT_STRIDE_ROW=output_tokens.stride(0),
        VOCAB_SIZE=vocab_size,
        NUM_BLOCKS=num_blocks,
        REDUCTION_SIZE=triton.next_power_of_2(num_blocks),
    )


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dim in shape:
        values *= int(dim)
    return values * element_bytes
