#!/usr/bin/env python3
"""Persistent MoonViT + PatchMerger worker for GLMRT vision requests.

The worker owns the roughly 900 MiB BF16 vision checkpoint in a separate CUDA
context.  Requests and replies are newline-delimited JSON on stdin/stdout.
Projected BF16 rows are returned through private files in /dev/shm so a
4096-token image does not become a 64 MiB base64 JSON message.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch import nn


HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 4304
TEXT_HIDDEN_SIZE = 6144
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
PATCH_SIZE = 14
MERGE_SIZE = 2
MAX_INPUT_PATCHES = 16_384
MAX_PATCHES_ONE_SIDE = 512
MAX_IMAGE_BYTES = 64 * 1024 * 1024


def navit_resize(width: int, height: int) -> dict[str, int]:
    scale_by_area = math.sqrt(
        MAX_INPUT_PATCHES
        / (
            max(1.0, width // PATCH_SIZE)
            * max(1.0, height // PATCH_SIZE)
        )
    )
    scale = min(
        1.0,
        scale_by_area,
        MAX_PATCHES_ONE_SIDE * PATCH_SIZE / width,
        MAX_PATCHES_ONE_SIDE * PATCH_SIZE / height,
    )
    new_width = min(
        max(1, int(width * scale)),
        MAX_PATCHES_ONE_SIDE * PATCH_SIZE,
    )
    new_height = min(
        max(1, int(height * scale)),
        MAX_PATCHES_ONE_SIDE * PATCH_SIZE,
    )
    factor = PATCH_SIZE * MERGE_SIZE
    pad_width = (factor - new_width % factor) % factor
    pad_height = (factor - new_height % factor) % factor
    grid_width = (new_width + pad_width) // PATCH_SIZE
    grid_height = (new_height + pad_height) // PATCH_SIZE
    return {
        "new_width": new_width,
        "new_height": new_height,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "num_tokens": (grid_width // MERGE_SIZE)
        * (grid_height // MERGE_SIZE),
    }


def read_image_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("image data URL must use base64 encoding")
        payload = base64.b64decode(encoded, validate=True)
    elif source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "glmrt-vision/1"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_IMAGE_BYTES + 1)
    elif source.startswith("file://"):
        if os.environ.get("GLMRT_VISION_ALLOW_FILE_URLS", "0") != "1":
            raise ValueError("file image URLs are disabled")
        payload = Path(source[7:]).read_bytes()
    else:
        raise ValueError("image URL must be data:, http://, or https://")
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"decoded image size must be in 1..={MAX_IMAGE_BYTES} bytes"
        )
    return payload


def preprocess_image(payload: bytes, device: torch.device):
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
    config = navit_resize(image.width, image.height)
    image = image.resize(
        (config["new_width"], config["new_height"]),
        resample=Image.Resampling.BICUBIC,
    )
    pixels = np.asarray(image)
    pixels = np.pad(
        pixels,
        (
            (0, config["pad_height"]),
            (0, config["pad_width"]),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )
    pixels = pixels.astype(np.float32) / 255.0
    pixels = (pixels - 0.5) * 2.0
    grid_height = config["grid_height"]
    grid_width = config["grid_width"]
    patches = pixels.reshape(
        grid_height,
        PATCH_SIZE,
        grid_width,
        PATCH_SIZE,
        3,
    )
    patches = (
        patches.transpose(0, 2, 4, 1, 3)
        .reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    )
    patch_tensor = torch.from_numpy(patches).to(
        device=device,
        dtype=torch.bfloat16,
    )
    return patch_tensor, grid_height, grid_width, config["num_tokens"]


def rope_frequencies(
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    flat = torch.arange(height * width, device=device, dtype=torch.float32)
    x_position = flat % width
    y_position = torch.div(flat, width, rounding_mode="floor")
    dimensions = torch.arange(0, HEAD_DIM, 4, device=device, dtype=torch.float32)
    frequencies = 1.0 / (10_000.0 ** (dimensions / HEAD_DIM))
    x_angles = torch.outer(x_position, frequencies)
    y_angles = torch.outer(y_position, frequencies)
    x_complex = torch.polar(torch.ones_like(x_angles), x_angles)
    y_complex = torch.polar(torch.ones_like(y_angles), y_angles)
    return torch.stack((x_complex, y_complex), dim=-1).reshape(
        height * width,
        HEAD_DIM // 2,
    )


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [tokens, heads, head_dim]
    frequency_rows = frequencies.unsqueeze(1)
    query_complex = torch.view_as_complex(
        query.float().reshape(*query.shape[:-1], -1, 2)
    )
    key_complex = torch.view_as_complex(
        key.float().reshape(*key.shape[:-1], -1, 2)
    )
    query = torch.view_as_real(query_complex * frequency_rows).flatten(-2)
    key = torch.view_as_real(key_complex * frequency_rows).flatten(-2)
    return query.to(dtype=torch.bfloat16), key.to(dtype=torch.bfloat16)


class VisionMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=True)
        self.fc1 = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(hidden), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm0 = nn.LayerNorm(HIDDEN_SIZE)
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE)
        self.wqkv = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 3, bias=True)
        self.wo = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        self.mlp = VisionMlp()

    def forward(
        self,
        hidden: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        qkv = self.wqkv(self.norm0(hidden)).reshape(
            hidden.shape[0],
            3,
            NUM_HEADS,
            HEAD_DIM,
        )
        query, key, value = qkv.unbind(dim=1)
        query, key = apply_rope(query, key, frequencies)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = (
            attended.squeeze(0)
            .transpose(0, 1)
            .contiguous()
            .reshape(hidden.shape[0], HIDDEN_SIZE)
        )
        hidden = residual + self.wo(attended)
        return hidden + self.mlp(self.norm1(hidden))


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            bias=True,
        )
        self.pos_emb = PositionEmbedding()

    def forward(
        self,
        patches: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        hidden = self.proj(patches).reshape(patches.shape[0], HIDDEN_SIZE)
        position_weight = self.pos_emb.weight
        if (height, width) == (64, 64):
            position = position_weight.reshape(-1, HIDDEN_SIZE)
        else:
            position = F.interpolate(
                position_weight.permute(2, 0, 1).unsqueeze(0),
                size=(height, width),
                mode="bicubic",
            ).squeeze(0).permute(1, 2, 0).reshape(-1, HIDDEN_SIZE)
        return hidden + position


class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(64, 64, HIDDEN_SIZE))


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([VisionBlock() for _ in range(27)])
        self.final_layernorm = nn.LayerNorm(HIDDEN_SIZE)

    def forward(
        self,
        hidden: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden, frequencies)
        return self.final_layernorm(hidden)


class VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.encoder = VisionEncoder()

    def forward(
        self,
        patches: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        hidden = self.patch_embed(patches, height, width)
        hidden = self.encoder(
            hidden,
            rope_frequencies(height, width, hidden.device),
        )
        hidden = hidden.reshape(
            height // MERGE_SIZE,
            MERGE_SIZE,
            width // MERGE_SIZE,
            MERGE_SIZE,
            HIDDEN_SIZE,
        )
        return hidden.permute(0, 2, 1, 3, 4).contiguous().reshape(
            -1,
            MERGE_SIZE * MERGE_SIZE,
            HIDDEN_SIZE,
        )


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        merged_hidden = HIDDEN_SIZE * MERGE_SIZE * MERGE_SIZE
        self.pre_norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-5)
        self.linear_1 = nn.Linear(merged_hidden, merged_hidden, bias=True)
        self.linear_2 = nn.Linear(
            merged_hidden,
            TEXT_HIDDEN_SIZE,
            bias=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.pre_norm(hidden).reshape(hidden.shape[0], -1)
        return self.linear_2(F.gelu(self.linear_1(hidden)))


class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = VisionTower()
        self.mm_projector = Projector()

    def forward(
        self,
        patches: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return self.mm_projector(self.vision_tower(patches, height, width))


def load_model(weights_dir: Path, device: torch.device) -> VisionModel:
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = VisionModel()
    finally:
        torch.set_default_dtype(original_dtype)
    model.to_empty(device=device)
    parameters = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for filename in ("vision_tower.safetensors", "mm_projector.safetensors"):
            with safe_open(
                weights_dir / filename,
                framework="pt",
                device="cpu",
            ) as tensors:
                for name in tensors.keys():
                    parameter = parameters.get(name)
                    if parameter is None:
                        raise RuntimeError(
                            f"vision checkpoint parameter {name} is unknown"
                        )
                    parameter.copy_(tensors.get_tensor(name).to(device=device))
                    loaded.add(name)
    missing = sorted(set(parameters) - loaded)
    if missing:
        raise RuntimeError(f"vision checkpoint omitted parameters: {missing[:8]}")
    return model.eval()


class EmbeddingCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        value = self.entries.pop(key, None)
        if value is not None:
            self.entries[key] = value
        return value

    def insert(self, key: str, value: torch.Tensor) -> None:
        value = value.contiguous().cpu()
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        self.entries[key] = value
        self.bytes += value.numel() * value.element_size()
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            _, evicted = self.entries.popitem(last=False)
            self.bytes -= evicted.numel() * evicted.element_size()


def encode_image(
    model: VisionModel,
    device: torch.device,
    payload: bytes,
) -> torch.Tensor:
    patches, height, width, expected_tokens = preprocess_image(payload, device)
    with torch.inference_mode():
        projected = model(patches, height, width)
    if projected.shape != (expected_tokens, TEXT_HIDDEN_SIZE):
        raise RuntimeError(
            "vision projection shape "
            f"{tuple(projected.shape)} != {(expected_tokens, TEXT_HIDDEN_SIZE)}"
        )
    return projected.to(dtype=torch.bfloat16).contiguous().cpu()


def write_embedding_file(embedding: torch.Tensor) -> tuple[str, int]:
    data = embedding.view(torch.uint16).numpy().tobytes()
    file_descriptor, path = tempfile.mkstemp(
        prefix="glmrt-vision-",
        suffix=".bf16",
        dir="/dev/shm",
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(data)
    except Exception:
        os.unlink(path)
        raise
    return path, len(data)


def run_worker(weights_dir: Path, device_index: int) -> None:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    model = load_model(weights_dir, device)
    cache_mebibytes = int(os.environ.get("GLMRT_VISION_CACHE_MIB", "512"))
    cache = EmbeddingCache(max(cache_mebibytes, 0) * 1024 * 1024)
    print(
        json.dumps(
            {
                "status": "ready",
                "device": device_index,
                "weights": str(weights_dir),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request_id = None
        paths: list[str] = []
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            sources = request["images"]
            outputs = []
            for source in sources:
                payload = read_image_bytes(source)
                digest = hashlib.sha256(payload).hexdigest()
                embedding = cache.get(digest)
                cache_hit = embedding is not None
                if embedding is None:
                    embedding = encode_image(model, device, payload)
                    cache.insert(digest, embedding)
                path, byte_count = write_embedding_file(embedding)
                paths.append(path)
                outputs.append(
                    {
                        "path": path,
                        "rows": embedding.shape[0],
                        "hidden_size": embedding.shape[1],
                        "bytes": byte_count,
                        "sha256": digest,
                        "cache_hit": cache_hit,
                    }
                )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "request_id": request_id,
                        "images": outputs,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            for path in paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            print(
                json.dumps(
                    {
                        "status": "error",
                        "request_id": request_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                ),
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run_worker(args.weights_dir.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
