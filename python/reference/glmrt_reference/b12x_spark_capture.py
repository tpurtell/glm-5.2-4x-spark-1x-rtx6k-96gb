from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any


_TARGET_ENV = "GLMRT_B12X_SPARK_CAPTURE_TARGET"
_STATE: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
_MLP_STATE: dict[tuple[int, int, int, int, int, int, int], dict[str, Any]] = {}
_MAX_B12X_SPARK_ROWS = 1024
_VALIDATION_RTOL = 1.0e-3
_VALIDATION_ATOL = 1.0e-2
_MLP_VALIDATION_ATOL = 2.0e-2


def _route_timing_enabled() -> bool:
    return os.environ.get("GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    } or os.environ.get("GLMRT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }


def _log_route_timing(stage: str, **fields: Any) -> None:
    if not _route_timing_enabled():
        return
    body = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"spark_b12x_route_stage stage={stage} {body}",
        file=sys.stderr,
        flush=True,
    )


def capture_dense_gemm(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch a graph-capturable synthetic SparkInfer NVFP4 dense GEMM.

    The Spark routed expert path uses source-native ModelOpt NVFP4 weights and
    BF16 activations. This adapter deliberately validates SparkInfer dense_gemm on a
    synthetic FP4 x FP4 contract first; production routed replacement still
    needs a resident packing bridge for checkpoint weights and route scatter.
    """

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch
    from b12x._lib.dense_gemm import dense_gemm

    rows = int(kwargs["rows"])
    n = int(kwargs["n"])
    k = int(kwargs["k"])
    seed = int(kwargs.get("seed", 0))
    validate = bool(kwargs.get("validate", True))
    if rows <= 0 or rows > _MAX_B12X_SPARK_ROWS:
        raise ValueError(
            f"SparkInfer dense GEMM rows must be in [1, {_MAX_B12X_SPARK_ROWS}], got {rows}"
        )
    if n <= 0 or k <= 0:
        raise ValueError(f"SparkInfer dense GEMM requires positive n and k, got n={n}, k={k}")
    if k % 16 != 0:
        raise ValueError(f"SparkInfer dense GEMM requires k divisible by 16, got {k}")

    output_buffer = ctx["buffers"].get("output")
    device_id = int(output_buffer["device_id"]) if output_buffer is not None else 0
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    state = _state_for(device_id, rows, n, k, seed, validate)
    output = (
        _bf16_tensor(output_buffer, (rows, n, 1))
        if output_buffer is not None
        else state["output"]
    )

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        dense_gemm(
            (state["a_packed"], state["a_scale"]),
            (state["b_packed"], state["b_scale"]),
            alpha=state["alpha"],
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            out=output,
        )


def capture_single_expert_mlp(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch a graph-capturable synthetic single-expert FP4 MLP."""

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch

    rows = int(kwargs["rows"])
    hidden = int(kwargs["hidden"])
    intermediate = int(kwargs["intermediate"])
    output_dim = int(kwargs["output"])
    seed = int(kwargs.get("seed", 0))
    validate = bool(kwargs.get("validate", True))
    if rows <= 0 or rows > _MAX_B12X_SPARK_ROWS:
        raise ValueError(
            f"SparkInfer single-expert MLP rows must be in [1, {_MAX_B12X_SPARK_ROWS}], got {rows}"
        )
    if hidden <= 0 or intermediate <= 0 or output_dim <= 0:
        raise ValueError(
            "SparkInfer single-expert MLP requires positive dimensions, "
            f"got hidden={hidden}, intermediate={intermediate}, output={output_dim}"
        )
    if hidden % 16 != 0 or intermediate % 16 != 0:
        raise ValueError(
            "SparkInfer single-expert MLP requires hidden and intermediate "
            f"divisible by 16, got hidden={hidden}, intermediate={intermediate}"
        )

    output_buffer = ctx["buffers"].get("output")
    input_buffer = ctx["buffers"].get("input")
    device_id = int(output_buffer["device_id"]) if output_buffer is not None else 0
    if input_buffer is not None:
        device_id = int(input_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    input_source = (
        _bf16_tensor(input_buffer, (1, rows, hidden))
        if input_buffer is not None
        else None
    )
    input_key = int(input_buffer["ptr"]) if input_buffer is not None else 0
    state = _mlp_state_for(
        device_id,
        rows,
        hidden,
        intermediate,
        output_dim,
        seed,
        validate,
        input_key,
        input_source,
    )
    output = (
        _bf16_tensor(output_buffer, (rows, output_dim, 1))
        if output_buffer is not None
        else state["output"]
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        _launch_single_expert_mlp(state, output)


def _state_for(
    device_id: int,
    rows: int,
    n: int,
    k: int,
    seed: int,
    validate: bool,
) -> dict[str, Any]:
    key = (device_id, rows, n, k, seed)
    cached = _STATE.get(key)
    if cached is not None:
        return cached

    import torch
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
    from b12x._lib.dense_gemm import dense_gemm

    with torch.cuda.device(device_id), torch.inference_mode():
        generator = torch.Generator(device=f"cuda:{device_id}")
        generator.manual_seed(seed)
        a_source = (
            torch.randn(
                (1, rows, k),
                generator=generator,
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()
        b_source = (
            torch.randn(
                (1, n, k),
                generator=generator,
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()
        a_packed, a_scale, a_global_scale = _quantize_operand(
            quantize_grouped_nvfp4_torch,
            a_source,
            rows,
        )
        b_packed, b_scale, b_global_scale = _quantize_operand(
            quantize_grouped_nvfp4_torch,
            b_source,
            n,
        )
        b_modelopt_scale = _grouped_scale_view_to_modelopt_scale_bytes(b_scale, n, k)
        b_scale = _modelopt_scale_bytes_to_grouped_scale_view(b_modelopt_scale, n, k)
        alpha = (1.0 / (a_global_scale[0] * b_global_scale[0])).view(1)
        output = torch.empty((rows, n, 1), device=f"cuda:{device_id}", dtype=torch.bfloat16)
        for _ in range(3):
            dense_gemm(
                (a_packed, a_scale),
                (b_packed, b_scale),
                alpha=alpha,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                out=output,
            )
        torch.cuda.synchronize(device_id)

        if validate:
            reference = _software_decode_dense_gemm_reference(
                a_packed,
                a_scale,
                a_global_scale,
                b_packed,
                b_scale,
                b_global_scale,
                rows,
                n,
                k,
            )
            candidate = output[:, :, 0].float()
            if not torch.isfinite(candidate).all().item() or not torch.isfinite(reference).all().item():
                raise RuntimeError("SparkInfer dense GEMM validation produced non-finite output")
            difference = (candidate - reference).abs()
            tolerance = _VALIDATION_ATOL + _VALIDATION_RTOL * reference.abs()
            if not (difference <= tolerance).all().item():
                max_abs = float(difference.max().item())
                max_rel = float((difference / reference.abs().clamp_min(1.0e-6)).max().item())
                numerator = torch.sum(candidate * reference)
                denominator = torch.linalg.vector_norm(candidate) * torch.linalg.vector_norm(reference)
                cosine = (
                    float((numerator / denominator).item())
                    if float(denominator.item()) != 0.0
                    else 1.0
                )
                raise RuntimeError(
                    "SparkInfer dense GEMM validation failed: "
                    f"max_abs={max_abs:.6f}, max_rel={max_rel:.6f}, "
                    f"cosine={cosine:.6f}, rtol={_VALIDATION_RTOL:.1e}, "
                    f"atol={_VALIDATION_ATOL:.1e}, rows={rows}, n={n}, k={k}"
                )

        state = {
            "a_packed": a_packed,
            "a_scale": a_scale,
            "b_packed": b_packed,
            "b_scale": b_scale,
            "alpha": alpha,
            "output": output,
        }
        _STATE[key] = state
        return state


def _mlp_state_for(
    device_id: int,
    rows: int,
    hidden: int,
    intermediate: int,
    output_dim: int,
    seed: int,
    validate: bool,
    input_key: int = 0,
    input_source: Any | None = None,
) -> dict[str, Any]:
    key = (device_id, rows, hidden, intermediate, output_dim, seed, input_key)
    cached = _MLP_STATE.get(key)
    if cached is not None:
        return cached

    import torch
    import torch.nn.functional as F
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch

    with torch.cuda.device(device_id):
        generator = torch.Generator(device=f"cuda:{device_id}")
        generator.manual_seed(seed)
        if input_source is not None:
            if input_source.shape != (1, rows, hidden) or input_source.dtype != torch.bfloat16:
                raise ValueError(
                    "SparkInfer single-expert MLP input buffer must be BF16 "
                    f"with shape {(1, rows, hidden)}, got {input_source.dtype} {tuple(input_source.shape)}"
                )
            hidden_source = input_source
        else:
            hidden_source = (
                torch.randn(
                    (1, rows, hidden),
                    generator=generator,
                    device=f"cuda:{device_id}",
                    dtype=torch.bfloat16,
                )
                / 4
            ).contiguous()
        if hidden_source.stride()[-1] != 1:
            raise ValueError(
                "SparkInfer single-expert MLP input tensor must have contiguous hidden dimension, "
                f"got strides {hidden_source.stride()}"
            )
        fc1_source = (
            torch.randn(
                (1, intermediate * 2, hidden),
                generator=generator,
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()
        fc2_source = (
            torch.randn(
                (1, output_dim, intermediate),
                generator=generator,
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()

        input_packed, input_scale, input_global_scale = _quantize_operand(
            quantize_grouped_nvfp4_torch,
            hidden_source,
            rows,
        )
        fc1_packed, fc1_scale, fc1_global_scale = _quantize_operand(
            quantize_grouped_nvfp4_torch,
            fc1_source,
            intermediate * 2,
        )
        fc2_packed, fc2_scale, fc2_global_scale = _quantize_operand(
            quantize_grouped_nvfp4_torch,
            fc2_source,
            output_dim,
        )
        fc1_modelopt_scale = _grouped_scale_view_to_modelopt_scale_bytes(
            fc1_scale,
            intermediate * 2,
            hidden,
        )
        fc2_modelopt_scale = _grouped_scale_view_to_modelopt_scale_bytes(
            fc2_scale,
            output_dim,
            intermediate,
        )
        fc1_scale = _modelopt_scale_bytes_to_grouped_scale_view(
            fc1_modelopt_scale,
            intermediate * 2,
            hidden,
        )
        fc2_scale = _modelopt_scale_bytes_to_grouped_scale_view(
            fc2_modelopt_scale,
            output_dim,
            intermediate,
        )

        fc1_reference = _software_decode_dense_gemm_reference(
            input_packed,
            input_scale,
            input_global_scale,
            fc1_packed,
            fc1_scale,
            fc1_global_scale,
            rows,
            intermediate * 2,
            hidden,
        )
        activated_reference = (
            F.silu(fc1_reference[:, :intermediate].float())
            * fc1_reference[:, intermediate:].float()
        ).to(torch.bfloat16)
        activation_global_scale = _global_scale_for(activated_reference)

        state = {
            "input_packed": input_packed,
            "input_scale": input_scale,
            "input_global_scale": input_global_scale,
            "input_source": input_source,
            "fc1_packed": fc1_packed,
            "fc1_scale": fc1_scale,
            "fc1_global_scale": fc1_global_scale,
            "fc2_packed": fc2_packed,
            "fc2_scale": fc2_scale,
            "fc2_global_scale": fc2_global_scale,
            "fc1_alpha": (1.0 / (input_global_scale[0] * fc1_global_scale[0])).view(1),
            "fc2_alpha": (1.0 / (activation_global_scale[0] * fc2_global_scale[0])).view(1),
            "activation_global_scale": activation_global_scale,
            "row_counts": torch.full((1,), rows, dtype=torch.int32),
            "fc1_output": torch.empty(
                (rows, intermediate * 2, 1),
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            ),
            "output": torch.empty(
                (rows, output_dim, 1),
                device=f"cuda:{device_id}",
                dtype=torch.bfloat16,
            ),
            "rows": rows,
            "hidden": hidden,
            "intermediate": intermediate,
            "output_dim": output_dim,
        }
        for _ in range(3):
            _launch_single_expert_mlp(state, state["output"])
        torch.cuda.synchronize(device_id)

        if validate:
            reference = _software_decode_single_expert_mlp_reference(
                state,
                quantize_grouped_nvfp4_torch,
            )
            candidate = state["output"][:, :, 0].float()
            _assert_close(
                candidate,
                reference,
                "SparkInfer single-expert MLP validation failed",
                atol=_MLP_VALIDATION_ATOL,
                rows=rows,
                n=output_dim,
                k=hidden,
            )

        _MLP_STATE[key] = state
        return state


def _launch_single_expert_mlp(state: dict[str, Any], output: Any) -> None:
    from b12x._lib.intrinsics import (
        quantize_grouped_nvfp4_torch,
        silu_mul_quantize_grouped_nvfp4_torch,
    )
    from b12x._lib.dense_gemm import dense_gemm

    if state["input_source"] is not None:
        input_packed, input_scale = quantize_grouped_nvfp4_torch(
            state["input_source"],
            state["row_counts"],
            state["input_global_scale"],
        )
    else:
        input_packed = state["input_packed"]
        input_scale = state["input_scale"]
    dense_gemm(
        (input_packed, input_scale),
        (state["fc1_packed"], state["fc1_scale"]),
        alpha=state["fc1_alpha"],
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        out=state["fc1_output"],
    )
    activation_packed, activation_scale = silu_mul_quantize_grouped_nvfp4_torch(
        state["fc1_output"][:, :, 0].unsqueeze(0),
        state["row_counts"],
        state["activation_global_scale"],
    )
    dense_gemm(
        (activation_packed, activation_scale),
        (state["fc2_packed"], state["fc2_scale"]),
        alpha=state["fc2_alpha"],
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        out=output,
    )


def _software_decode_single_expert_mlp_reference(
    state: dict[str, Any],
    quantize_grouped_nvfp4_torch: Any,
) -> Any:
    import torch
    import torch.nn.functional as F

    rows = state["rows"]
    hidden = state["hidden"]
    intermediate = state["intermediate"]
    output_dim = state["output_dim"]
    if state["input_source"] is not None:
        input_packed, input_scale = quantize_grouped_nvfp4_torch(
            state["input_source"],
            state["row_counts"],
            state["input_global_scale"],
        )
    else:
        input_packed = state["input_packed"]
        input_scale = state["input_scale"]
    fc1 = _software_decode_dense_gemm_reference(
        input_packed,
        input_scale,
        state["input_global_scale"],
        state["fc1_packed"],
        state["fc1_scale"],
        state["fc1_global_scale"],
        rows,
        intermediate * 2,
        hidden,
    )
    activated = (
        F.silu(fc1[:, :intermediate].float()) * fc1[:, intermediate:].float()
    ).to(torch.bfloat16)
    activation_packed, activation_scale = quantize_grouped_nvfp4_torch(
        activated.unsqueeze(0),
        state["row_counts"],
        state["activation_global_scale"],
    )
    activation = _decode_b12x_grouped_nvfp4_operand(
        activation_packed,
        activation_scale,
        state["activation_global_scale"],
        rows,
        intermediate,
    )
    fc2 = _decode_b12x_grouped_nvfp4_operand(
        state["fc2_packed"],
        state["fc2_scale"],
        state["fc2_global_scale"],
        output_dim,
        intermediate,
    )
    return (activation @ fc2.T).to(torch.bfloat16).float()


def _quantize_operand(quantize_grouped_nvfp4_torch: Any, source: Any, rows: int) -> tuple[Any, Any, Any]:
    import torch

    row_counts = torch.full((1,), rows, dtype=torch.int32, device=source.device)
    global_scale = _global_scale_for(source)
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def _global_scale_for(source: Any) -> Any:
    import torch

    tensor_amax = source.abs().max().to(torch.float32).clamp_min(1.0e-12)
    return torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )


def _software_decode_dense_gemm_reference(
    a_packed: Any,
    a_scale: Any,
    a_global_scale: Any,
    b_packed: Any,
    b_scale: Any,
    b_global_scale: Any,
    rows: int,
    n: int,
    k: int,
) -> Any:
    import torch

    a = _decode_b12x_grouped_nvfp4_operand(a_packed, a_scale, a_global_scale, rows, k)
    b = _decode_b12x_grouped_nvfp4_operand(b_packed, b_scale, b_global_scale, n, k)
    return (a @ b.T).to(torch.bfloat16).float()


def _assert_close(
    candidate: Any,
    reference: Any,
    message: str,
    *,
    rtol: float = _VALIDATION_RTOL,
    atol: float = _VALIDATION_ATOL,
    **dims: int,
) -> None:
    import torch

    if not torch.isfinite(candidate).all().item() or not torch.isfinite(reference).all().item():
        raise RuntimeError(f"{message}: produced non-finite output")
    difference = (candidate - reference).abs()
    tolerance = atol + rtol * reference.abs()
    if (difference <= tolerance).all().item():
        return
    max_abs = float(difference.max().item())
    max_rel = float((difference / reference.abs().clamp_min(1.0e-6)).max().item())
    numerator = torch.sum(candidate * reference)
    denominator = torch.linalg.vector_norm(candidate) * torch.linalg.vector_norm(reference)
    cosine = float((numerator / denominator).item()) if float(denominator.item()) != 0.0 else 1.0
    dim_text = ", ".join(f"{key}={value}" for key, value in dims.items())
    raise RuntimeError(
        f"{message}: max_abs={max_abs:.6f}, max_rel={max_rel:.6f}, "
        f"cosine={cosine:.6f}, rtol={rtol:.1e}, "
        f"atol={atol:.1e}, {dim_text}"
    )


def _decode_b12x_grouped_nvfp4_operand(
    packed: Any,
    scale: Any,
    global_scale: Any,
    rows: int,
    cols: int,
) -> Any:
    import torch

    if packed.ndim != 3 or packed.shape[2] != 1:
        raise ValueError(f"expected grouped FP4 packed shape [rows, cols/2, 1], got {tuple(packed.shape)}")
    if cols % 16 != 0:
        raise ValueError(f"NVFP4 decode requires cols divisible by 16, got {cols}")
    packed_2d = packed[:rows, : cols // 2, 0].contiguous().to(torch.uint8)
    low = (packed_2d & 0x0F).to(torch.int64)
    high = ((packed_2d >> 4) & 0x0F).to(torch.int64)
    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    raw = torch.stack((codebook[low], codebook[high]), dim=-1).reshape(rows, cols)
    block_scale = _grouped_scale_view_to_unswizzled(scale, rows, cols)
    expanded_scale = block_scale.unsqueeze(-1).expand(rows, cols // 16, 16).reshape(rows, cols)
    return raw * expanded_scale / global_scale.reshape(-1)[0].to(torch.float32)


def _grouped_scale_view_to_unswizzled(scale: Any, rows: int, cols: int) -> Any:
    import torch

    return _grouped_scale_view_storage(scale, rows, cols).to(torch.float32)


def _grouped_scale_view_to_modelopt_scale_bytes(scale: Any, rows: int, cols: int) -> Any:
    import torch

    return _grouped_scale_view_storage(scale, rows, cols).contiguous().view(torch.uint8)


def _grouped_scale_view_storage(scale: Any, rows: int, cols: int) -> Any:
    rows_padded = _align_up(rows, 128)
    cols_blocks = cols // 16
    cols_padded = _align_up(cols_blocks, 4)
    storage = scale.permute(5, 2, 1, 0, 4, 3).contiguous().reshape(-1, rows_padded, cols_padded)
    if storage.shape[0] != 1:
        raise ValueError(f"expected one grouped scale batch, got {storage.shape[0]}")
    return storage[0, :rows, :cols_blocks]


def _modelopt_scale_bytes_to_grouped_scale_view(modelopt_scale: Any, rows: int, cols: int) -> Any:
    import torch
    from b12x._lib.intrinsics import (
        as_grouped_scale_view,
        swizzle_block_scale,
    )

    if modelopt_scale.dtype != torch.uint8:
        raise ValueError(f"ModelOpt NVFP4 scales must be uint8, got {modelopt_scale.dtype}")
    cols_blocks = cols // 16
    if modelopt_scale.ndim != 2 or modelopt_scale.shape[0] < rows or modelopt_scale.shape[1] < cols_blocks:
        raise ValueError(
            "ModelOpt NVFP4 scale shape mismatch: "
            f"need at least {(rows, cols_blocks)}, got {tuple(modelopt_scale.shape)}"
        )
    unswizzled = modelopt_scale[:rows, :cols_blocks].contiguous().view(torch.float8_e4m3fn)
    swizzled = swizzle_block_scale(unswizzled.unsqueeze(0))
    return as_grouped_scale_view(swizzled.view(torch.uint8), rows, cols)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _bf16_tensor(buffer: dict[str, Any], shape: tuple[int, ...], *, name: str = ""):
    import torch

    return _raw_tensor(buffer, shape, torch.bfloat16, 2, name=name)


def _raw_tensor(
    buffer: dict[str, Any],
    shape: tuple[int, ...],
    dtype: Any,
    element_bytes: int,
    *,
    name: str = "",
):
    import torch

    required = element_bytes
    for dim in shape:
        required *= int(dim)
    if int(buffer["bytes"]) < required:
        label = f" {name}" if name else ""
        raise ValueError(
            f"raw tensor{label} buffer is too small for shape {shape}: "
            f"{buffer['bytes']} < {required}"
        )

    _log_route_timing(
        "python_raw_tensor_start",
        name=name,
        shape=shape,
        ptr=hex(int(buffer["ptr"])),
        device_id=int(buffer["device_id"]),
    )
    device = torch.device("cuda", int(buffer["device_id"]))
    strides = _contiguous_strides(shape)
    storage = torch._C._construct_storage_from_data_pointer(
        int(buffer["ptr"]),
        device,
        required,
    )
    metadata = {
        "nbytes": required,
        "data_ptr": int(buffer["ptr"]),
        "size": shape,
        "stride": strides,
        "dtype": dtype,
        "device": device,
        "storage_offset": 0,
    }
    tensor = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        metadata,
        storage,
    )
    _log_route_timing("python_raw_tensor_ready", name=name, shape=shape)
    return tensor


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= max(int(dim), 1)
    return tuple(reversed(strides))
