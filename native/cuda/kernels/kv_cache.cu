#include "common.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace {

constexpr size_t kMlaFp8DsNopeValues = 512;
constexpr size_t kMlaFp8DsRopeValues = 64;
constexpr size_t kMlaFp8DsProjectedValues = kMlaFp8DsNopeValues + kMlaFp8DsRopeValues;
constexpr size_t kMlaFp8DsGroupSize = 128;
constexpr size_t kMlaFp8DsGroups = kMlaFp8DsNopeValues / kMlaFp8DsGroupSize;
constexpr size_t kMlaFp8DsNopeBytes = kMlaFp8DsNopeValues;
constexpr size_t kMlaFp8DsScaleBytes = kMlaFp8DsGroups * sizeof(float);
constexpr size_t kMlaFp8DsRopeBytes = kMlaFp8DsRopeValues * sizeof(uint16_t);
constexpr size_t kMlaFp8DsScaleOffsetBytes = kMlaFp8DsNopeBytes;
constexpr size_t kMlaFp8DsRopeOffsetBytes = kMlaFp8DsNopeBytes + kMlaFp8DsScaleBytes;
constexpr size_t kMlaFp8DsPackedBytes =
    kMlaFp8DsNopeBytes + kMlaFp8DsScaleBytes + kMlaFp8DsRopeBytes;
constexpr float kMlaFp8E4m3Max = 448.0f;
constexpr size_t kMlaMxfp4DsNopeValues = 512;
constexpr size_t kMlaMxfp4DsRopeValues = 64;
constexpr size_t kMlaMxfp4DsProjectedValues = kMlaMxfp4DsNopeValues + kMlaMxfp4DsRopeValues;
// Native Sparkinfer NVFP4 sparse-MLA ABI:
//   [0,256)   packed E2M1 latent
//   [256,288) E4M3 group-16 scales
//   [288,304) zero padding
//   [304,432) BF16 RoPE
constexpr size_t kMlaMxfp4DsBlockSize = 16;
constexpr size_t kMlaMxfp4DsGroups = kMlaMxfp4DsNopeValues / kMlaMxfp4DsBlockSize;
constexpr size_t kMlaMxfp4DsCodeBytes = kMlaMxfp4DsNopeValues / 2;
constexpr size_t kMlaMxfp4DsScaleBytes = kMlaMxfp4DsGroups;
constexpr size_t kMlaMxfp4DsPaddingBytes = 16;
constexpr size_t kMlaMxfp4DsRopeBytes = kMlaMxfp4DsRopeValues * sizeof(uint16_t);
constexpr size_t kMlaMxfp4DsCodeOffsetBytes = 0;
constexpr size_t kMlaMxfp4DsScaleOffsetBytes = kMlaMxfp4DsCodeBytes;
constexpr size_t kMlaMxfp4DsPaddingOffsetBytes =
    kMlaMxfp4DsCodeBytes + kMlaMxfp4DsScaleBytes;
constexpr size_t kMlaMxfp4DsRopeOffsetBytes =
    kMlaMxfp4DsPaddingOffsetBytes + kMlaMxfp4DsPaddingBytes;
constexpr size_t kMlaMxfp4DsPackedBytes =
    kMlaMxfp4DsRopeOffsetBytes + kMlaMxfp4DsRopeBytes;
constexpr float kMlaMxfp4E2m1Max = 6.0f;
constexpr int kMlaKvFinalizeBf16 = 0;
constexpr int kMlaKvFinalizeFp8 = 1;
constexpr int kMlaKvFinalizeMxfp4 = 2;

__global__ void kv_cache_write_bytes_kernel(const uint8_t* src, uint8_t* cache,
                                            size_t cache_offset_bytes, size_t bytes) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= bytes) {
    return;
  }
  cache[cache_offset_bytes + idx] = src[idx];
}

__global__ void kv_cache_read_bytes_kernel(const uint8_t* cache, uint8_t* dst,
                                           size_t cache_offset_bytes, size_t bytes) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= bytes) {
    return;
  }
  dst[idx] = cache[cache_offset_bytes + idx];
}

__global__ void kv_cache_write_blocks_kernel(const uint8_t* src, uint8_t* cache,
                                             const uint64_t* src_offsets,
                                             const uint64_t* cache_offsets,
                                             const uint64_t* block_bytes, size_t block_count) {
  const size_t block_idx = blockIdx.x;
  if (block_idx >= block_count) {
    return;
  }
  const uint64_t src_base = src_offsets[block_idx];
  const uint64_t cache_base = cache_offsets[block_idx];
  const uint64_t bytes = block_bytes[block_idx];
  for (uint64_t idx = threadIdx.x; idx < bytes; idx += blockDim.x) {
    cache[cache_base + idx] = src[src_base + idx];
  }
}

__global__ void kv_cache_read_blocks_kernel(const uint8_t* cache, uint8_t* dst,
                                            const uint64_t* cache_offsets,
                                            const uint64_t* dst_offsets,
                                            const uint64_t* block_bytes, size_t block_count) {
  const size_t block_idx = blockIdx.x;
  if (block_idx >= block_count) {
    return;
  }
  const uint64_t cache_base = cache_offsets[block_idx];
  const uint64_t dst_base = dst_offsets[block_idx];
  const uint64_t bytes = block_bytes[block_idx];
  for (uint64_t idx = threadIdx.x; idx < bytes; idx += blockDim.x) {
    dst[dst_base + idx] = cache[cache_base + idx];
  }
}

__global__ void mla_kv_cache_unpack_bf16_kernel(const uint8_t* payload, uint16_t* kv_latent,
                                                uint16_t* k_rope, uint16_t* dsa_key, size_t rows,
                                                size_t kv_lora_rank, size_t rope_dim,
                                                size_t dsa_dim, size_t payload_stride_bf16) {
  const size_t width = kv_lora_rank + rope_dim + dsa_dim;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * width;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / width;
  const size_t col = idx % width;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(payload) + row * payload_stride_bf16;
  if (col < kv_lora_rank) {
    kv_latent[row * kv_lora_rank + col] = src[col];
    return;
  }
  const size_t rope_col = col - kv_lora_rank;
  if (rope_col < rope_dim) {
    k_rope[row * rope_dim + rope_col] = src[col];
    return;
  }
  dsa_key[row * dsa_dim + (rope_col - rope_dim)] = src[col];
}

__global__ void mla_kv_projected_split_bf16_kernel(const uint16_t* projected, uint16_t* k_nope,
                                                   uint16_t* v, size_t rows, size_t heads,
                                                   size_t nope_dim, size_t v_dim) {
  const size_t head_width = nope_dim + v_dim;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * head_width;
  if (idx >= total) {
    return;
  }
  const size_t col = idx % head_width;
  const size_t head_index = idx / head_width;
  const uint16_t value = projected[idx];
  if (col < nope_dim) {
    k_nope[head_index * nope_dim + col] = value;
  } else {
    v[head_index * v_dim + (col - nope_dim)] = value;
  }
}

__global__ void mla_kv_prepare_bf16_kernel(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    uint16_t* prepared, size_t rows, size_t projected_stride_bf16,
    size_t prepared_stride_bf16, float eps, float theta) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr size_t threads = 256;
  constexpr size_t kv_lora_rank = 512;
  constexpr size_t rope_dim = 64;
  const size_t tid = threadIdx.x;
  const uint16_t* src = projected + row * projected_stride_bf16;
  uint16_t* dst = prepared + row * prepared_stride_bf16;
  const size_t col0 = tid;
  const size_t col1 = tid + threads;
  const float value0 = bf16_to_f32(src[col0]);
  const float value1 = bf16_to_f32(src[col1]);

  __shared__ float sums[threads];
  sums[tid] = value0 * value0 + value1 * value1;
  __syncthreads();
  for (size_t stride = threads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(sums[0] / static_cast<float>(kv_lora_rank) + eps);
  dst[col0] = f32_to_bf16(value0 * inverse_rms * bf16_to_f32(norm_weight[col0]));
  dst[col1] = f32_to_bf16(value1 * inverse_rms * bf16_to_f32(norm_weight[col1]));

  if (tid < rope_dim / 2) {
    const size_t pair = tid;
    const size_t rope_col = pair * 2;
    const float angle = static_cast<float>(positions[row]) *
                        powf(theta, -2.0f * static_cast<float>(pair) /
                                        static_cast<float>(rope_dim));
    const float cos_value = cosf(angle);
    const float sin_value = sinf(angle);
    const float even = bf16_to_f32(src[kv_lora_rank + rope_col]);
    const float odd = bf16_to_f32(src[kv_lora_rank + rope_col + 1]);
    dst[kv_lora_rank + rope_col] = f32_to_bf16(even * cos_value - odd * sin_value);
    dst[kv_lora_rank + rope_col + 1] = f32_to_bf16(even * sin_value + odd * cos_value);
  }
}

__global__ void mla_rope_factors_f32_candidate_kernel(
    const uint32_t* positions, float* factors, size_t rows, float theta) {
  constexpr size_t rope_pairs = 32;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * rope_pairs) {
    return;
  }
  const size_t row = index / rope_pairs;
  const size_t pair = index % rope_pairs;
  const float angle = static_cast<float>(positions[row]) *
                      powf(theta, -2.0f * static_cast<float>(pair) / 64.0f);
  factors[index * 2] = cosf(angle);
  factors[index * 2 + 1] = sinf(angle);
}

__global__ void mla_kv_prepare_bf16_precomputed_rope_candidate_kernel(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight, uint16_t* prepared, size_t rows,
    size_t projected_stride_bf16, size_t prepared_stride_bf16, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr size_t threads = 256;
  constexpr size_t kv_lora_rank = 512;
  constexpr size_t rope_dim = 64;
  const size_t tid = threadIdx.x;
  const uint16_t* src = projected + row * projected_stride_bf16;
  uint16_t* dst = prepared + row * prepared_stride_bf16;
  const size_t col0 = tid;
  const size_t col1 = tid + threads;
  const float value0 = bf16_to_f32(src[col0]);
  const float value1 = bf16_to_f32(src[col1]);

  __shared__ float sums[threads];
  sums[tid] = value0 * value0 + value1 * value1;
  __syncthreads();
  for (size_t stride = threads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(sums[0] / static_cast<float>(kv_lora_rank) + eps);
  dst[col0] = f32_to_bf16(value0 * inverse_rms * bf16_to_f32(norm_weight[col0]));
  dst[col1] = f32_to_bf16(value1 * inverse_rms * bf16_to_f32(norm_weight[col1]));

  if (tid < rope_dim / 2) {
    const size_t rope_col = tid * 2;
    const size_t factor_offset = (row * rope_dim) + rope_col;
    const float cos_value = rope_factors[factor_offset];
    const float sin_value = rope_factors[factor_offset + 1];
    const float even = bf16_to_f32(src[kv_lora_rank + rope_col]);
    const float odd = bf16_to_f32(src[kv_lora_rank + rope_col + 1]);
    dst[kv_lora_rank + rope_col] = f32_to_bf16(even * cos_value - odd * sin_value);
    dst[kv_lora_rank + rope_col + 1] = f32_to_bf16(even * sin_value + odd * cos_value);
  }
}

__global__ void mla_kv_pack_fp8_ds_mla_kernel(const uint16_t* projected, uint8_t* packed,
                                              size_t rows, size_t projected_stride_bf16,
                                              size_t packed_stride_bytes) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint16_t* src = projected + row * projected_stride_bf16;
  uint8_t* dst = packed + row * packed_stride_bytes;

  if (threadIdx.x < kMlaFp8DsGroups) {
    const size_t group = threadIdx.x;
    const size_t group_offset = group * kMlaFp8DsGroupSize;
    float max_abs = 0.0f;
    for (size_t idx = 0; idx < kMlaFp8DsGroupSize; ++idx) {
      const float value = bf16_to_f32(src[group_offset + idx]);
      const float abs_value = fabsf(value);
      if (abs_value > max_abs) {
        max_abs = abs_value;
      }
    }
    float scale = max_abs / kMlaFp8E4m3Max;
    if (!(scale > 0.0f)) {
      scale = 1.0f;
    }
    reinterpret_cast<float*>(dst + kMlaFp8DsScaleOffsetBytes)[group] = scale;
    for (size_t idx = 0; idx < kMlaFp8DsGroupSize; ++idx) {
      const float scaled = bf16_to_f32(src[group_offset + idx]) / scale;
      dst[group_offset + idx] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    }
  }

  for (size_t idx = threadIdx.x; idx < kMlaFp8DsRopeValues; idx += blockDim.x) {
    reinterpret_cast<uint16_t*>(dst + kMlaFp8DsRopeOffsetBytes)[idx] =
        src[kMlaFp8DsNopeValues + idx];
  }
}

__global__ void mla_kv_unpack_fp8_ds_mla_kernel(const uint8_t* packed, uint16_t* projected,
                                                size_t rows, size_t packed_stride_bytes,
                                                size_t projected_stride_bf16) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint8_t* src = packed + row * packed_stride_bytes;
  uint16_t* dst = projected + row * projected_stride_bf16;
  const float* scales = reinterpret_cast<const float*>(src + kMlaFp8DsScaleOffsetBytes);

  for (size_t idx = threadIdx.x; idx < kMlaFp8DsNopeValues; idx += blockDim.x) {
    const size_t group = idx / kMlaFp8DsGroupSize;
    const float value = f8e4m3_to_f32(src[idx]) * scales[group];
    dst[idx] = f32_to_bf16(value);
  }
  for (size_t idx = threadIdx.x; idx < kMlaFp8DsRopeValues; idx += blockDim.x) {
    dst[kMlaFp8DsNopeValues + idx] =
        reinterpret_cast<const uint16_t*>(src + kMlaFp8DsRopeOffsetBytes)[idx];
  }
}

__device__ uint8_t mxfp4_nearest_e2m1_code(float scaled) {
  return static_cast<uint8_t>(
             __nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest)) &
         0x0f;
}

__device__ uint8_t nvfp4_e4m3_scale_byte(float max_abs) {
  return __nv_cvt_float_to_fp8(
      max_abs / kMlaMxfp4E2m1Max, __NV_SATFINITE, __NV_E4M3);
}

__device__ float nvfp4_e4m3_scale_to_f32(uint8_t scale_byte) {
  return f8e4m3_to_f32(scale_byte);
}

__global__ void mla_kv_pack_mxfp4_ds_mla_kernel(const uint16_t* projected, uint8_t* packed,
                                                size_t rows, size_t projected_stride_bf16,
                                                size_t packed_stride_bytes) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint16_t* src = projected + row * projected_stride_bf16;
  uint8_t* dst = packed + row * packed_stride_bytes;

  if (threadIdx.x < kMlaMxfp4DsGroups) {
    const size_t group = threadIdx.x;
    const size_t group_offset = group * kMlaMxfp4DsBlockSize;
    float max_abs = 0.0f;
    for (size_t idx = 0; idx < kMlaMxfp4DsBlockSize; ++idx) {
      max_abs = fmaxf(max_abs, fabsf(bf16_to_f32(src[group_offset + idx])));
    }
    const uint8_t scale_byte = nvfp4_e4m3_scale_byte(max_abs);
    const float scale = nvfp4_e4m3_scale_to_f32(scale_byte);
    dst[kMlaMxfp4DsScaleOffsetBytes + group] = scale_byte;
    for (size_t local_idx = 0; local_idx < kMlaMxfp4DsBlockSize; ++local_idx) {
      const size_t value_idx = group_offset + local_idx;
      const uint8_t code =
          scale > 0.0f
              ? mxfp4_nearest_e2m1_code(bf16_to_f32(src[value_idx]) / scale)
              : 0;
      uint8_t* packed_byte = dst + kMlaMxfp4DsCodeOffsetBytes + value_idx / 2;
      if ((value_idx & 1) == 0) {
        *packed_byte = code;
      } else {
        *packed_byte = static_cast<uint8_t>((*packed_byte & 0x0f) | (code << 4));
      }
    }
  }

  for (size_t idx = threadIdx.x; idx < kMlaMxfp4DsPaddingBytes; idx += blockDim.x) {
    dst[kMlaMxfp4DsPaddingOffsetBytes + idx] = 0;
  }
  for (size_t idx = threadIdx.x; idx < kMlaMxfp4DsRopeValues; idx += blockDim.x) {
    reinterpret_cast<uint16_t*>(dst + kMlaMxfp4DsRopeOffsetBytes)[idx] =
        src[kMlaMxfp4DsNopeValues + idx];
  }
}

__global__ void mla_kv_unpack_mxfp4_ds_mla_kernel(const uint8_t* packed, uint16_t* projected,
                                                  size_t rows, size_t packed_stride_bytes,
                                                  size_t projected_stride_bf16) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint8_t* src = packed + row * packed_stride_bytes;
  uint16_t* dst = projected + row * projected_stride_bf16;

  for (size_t idx = threadIdx.x; idx < kMlaMxfp4DsNopeValues; idx += blockDim.x) {
    const uint8_t packed_byte = src[kMlaMxfp4DsCodeOffsetBytes + idx / 2];
    const uint8_t code = (idx & 1) == 0 ? (packed_byte & 0x0f) : (packed_byte >> 4);
    const uint8_t scale_byte = src[kMlaMxfp4DsScaleOffsetBytes + idx / kMlaMxfp4DsBlockSize];
    const float value =
        nvfp4_e2m1_code_value(code) * nvfp4_e4m3_scale_to_f32(scale_byte);
    dst[idx] = f32_to_bf16(value);
  }
  for (size_t idx = threadIdx.x; idx < kMlaMxfp4DsRopeValues; idx += blockDim.x) {
    dst[kMlaMxfp4DsNopeValues + idx] =
        reinterpret_cast<const uint16_t*>(src + kMlaMxfp4DsRopeOffsetBytes)[idx];
  }
}

template <int kCacheFormat>
__global__ void mla_kv_finalize_store_candidate_kernel(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight,
    uint8_t* cache_rows, uint16_t* attention_ready,
    const uint16_t* dsa_normalized, size_t rows,
    size_t projected_stride_bf16, size_t cache_stride_bytes,
    size_t attention_stride_bf16, size_t dsa_stride_bf16,
    size_t dsa_values, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr size_t threads = 256;
  constexpr size_t kv_lora_rank = 512;
  constexpr size_t rope_dim = 64;
  constexpr size_t kv_width = kv_lora_rank + rope_dim;
  constexpr size_t main_bytes =
      kCacheFormat == kMlaKvFinalizeBf16
          ? kv_width * sizeof(uint16_t)
          : (kCacheFormat == kMlaKvFinalizeFp8 ? kMlaFp8DsPackedBytes
                                               : kMlaMxfp4DsPackedBytes);
  const size_t tid = threadIdx.x;
  const uint16_t* src = projected + row * projected_stride_bf16;
  uint8_t* cache = cache_rows + row * cache_stride_bytes;
  __shared__ float sums[threads];
  __shared__ uint16_t prepared[kv_width];

  const size_t col0 = tid;
  const size_t col1 = tid + threads;
  const float value0 = bf16_to_f32(src[col0]);
  const float value1 = bf16_to_f32(src[col1]);
  sums[tid] = value0 * value0 + value1 * value1;
  __syncthreads();
  for (size_t stride = threads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(sums[0] / static_cast<float>(kv_lora_rank) + eps);
  prepared[col0] =
      f32_to_bf16(value0 * inverse_rms * bf16_to_f32(norm_weight[col0]));
  prepared[col1] =
      f32_to_bf16(value1 * inverse_rms * bf16_to_f32(norm_weight[col1]));

  if (tid < rope_dim / 2) {
    const size_t rope_col = tid * 2;
    const size_t factor_offset = row * rope_dim + rope_col;
    const float cos_value = rope_factors[factor_offset];
    const float sin_value = rope_factors[factor_offset + 1];
    const float even = bf16_to_f32(src[kv_lora_rank + rope_col]);
    const float odd = bf16_to_f32(src[kv_lora_rank + rope_col + 1]);
    prepared[kv_lora_rank + rope_col] =
        f32_to_bf16(even * cos_value - odd * sin_value);
    prepared[kv_lora_rank + rope_col + 1] =
        f32_to_bf16(even * sin_value + odd * cos_value);
  }
  __syncthreads();

  if (attention_ready != nullptr) {
    uint16_t* attention = attention_ready + row * attention_stride_bf16;
    for (size_t index = tid; index < kv_width; index += blockDim.x) {
      attention[index] = prepared[index];
    }
  }

  if constexpr (kCacheFormat == kMlaKvFinalizeBf16) {
    uint16_t* cache_bf16 = reinterpret_cast<uint16_t*>(cache);
    for (size_t index = tid; index < kv_width; index += blockDim.x) {
      cache_bf16[index] = prepared[index];
    }
  } else if constexpr (kCacheFormat == kMlaKvFinalizeFp8) {
    if (tid < kMlaFp8DsGroups) {
      const size_t group = tid;
      const size_t group_offset = group * kMlaFp8DsGroupSize;
      float max_abs = 0.0f;
      for (size_t index = 0; index < kMlaFp8DsGroupSize; ++index) {
        const float abs_value =
            fabsf(bf16_to_f32(prepared[group_offset + index]));
        if (abs_value > max_abs) {
          max_abs = abs_value;
        }
      }
      float scale = max_abs / kMlaFp8E4m3Max;
      if (!(scale > 0.0f)) {
        scale = 1.0f;
      }
      reinterpret_cast<float*>(cache + kMlaFp8DsScaleOffsetBytes)[group] =
          scale;
      for (size_t index = 0; index < kMlaFp8DsGroupSize; ++index) {
        const float scaled =
            bf16_to_f32(prepared[group_offset + index]) / scale;
        cache[group_offset + index] =
            __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
      }
    }
    for (size_t index = tid; index < kMlaFp8DsRopeValues;
         index += blockDim.x) {
      reinterpret_cast<uint16_t*>(cache + kMlaFp8DsRopeOffsetBytes)[index] =
          prepared[kMlaFp8DsNopeValues + index];
    }
  } else {
    if (tid < kMlaMxfp4DsGroups) {
      const size_t group = tid;
      const size_t group_offset = group * kMlaMxfp4DsBlockSize;
      float max_abs = 0.0f;
      for (size_t index = 0; index < kMlaMxfp4DsBlockSize; ++index) {
        max_abs = fmaxf(
            max_abs, fabsf(bf16_to_f32(prepared[group_offset + index])));
      }
      const uint8_t scale_byte = nvfp4_e4m3_scale_byte(max_abs);
      const float scale = nvfp4_e4m3_scale_to_f32(scale_byte);
      cache[kMlaMxfp4DsScaleOffsetBytes + group] = scale_byte;
      for (size_t local_index = 0; local_index < kMlaMxfp4DsBlockSize;
           ++local_index) {
        const size_t value_index = group_offset + local_index;
        const uint8_t code =
            scale > 0.0f
                ? mxfp4_nearest_e2m1_code(
                      bf16_to_f32(prepared[value_index]) / scale)
                : 0;
        uint8_t* packed_byte =
            cache + kMlaMxfp4DsCodeOffsetBytes + value_index / 2;
        if ((value_index & 1) == 0) {
          *packed_byte = code;
        } else {
          *packed_byte =
              static_cast<uint8_t>((*packed_byte & 0x0f) | (code << 4));
        }
      }
    }
    for (size_t index = tid; index < kMlaMxfp4DsPaddingBytes;
         index += blockDim.x) {
      cache[kMlaMxfp4DsPaddingOffsetBytes + index] = 0;
    }
    for (size_t index = tid; index < kMlaMxfp4DsRopeValues;
         index += blockDim.x) {
      reinterpret_cast<uint16_t*>(cache + kMlaMxfp4DsRopeOffsetBytes)[index] =
          prepared[kMlaMxfp4DsNopeValues + index];
    }
  }

  if (dsa_values > 0) {
    const uint16_t* dsa = dsa_normalized + row * dsa_stride_bf16;
    uint16_t* cache_dsa = reinterpret_cast<uint16_t*>(cache + main_bytes);
    for (size_t index = tid; index < dsa_values; index += blockDim.x) {
      cache_dsa[index] = dsa[index];
    }
  }
}

glmrt_status_t validate_kv_cache_write_args(const uint8_t* src, const uint8_t* cache,
                                            size_t cache_offset_bytes, size_t bytes) {
  if (bytes == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || cache == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (cache_offset_bytes > std::numeric_limits<size_t>::max() - bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_graph_kv_cache_write_buffers(glmrt_device_buffer_t src,
                                                     glmrt_device_buffer_t cache,
                                                     size_t cache_offset_bytes, size_t bytes) {
  if (bytes == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_kv_cache_write_args(
      static_cast<const uint8_t*>(src.ptr), static_cast<const uint8_t*>(cache.ptr),
      cache_offset_bytes, bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (src.bytes < bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  if (cache_offset_bytes > cache.bytes || bytes > cache.bytes - cache_offset_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_kv_cache_read_args(const uint8_t* cache, const uint8_t* dst,
                                           size_t cache_offset_bytes, size_t bytes) {
  if (bytes == 0) {
    return GLMRT_STATUS_OK;
  }
  if (cache == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (cache_offset_bytes > std::numeric_limits<size_t>::max() - bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_kv_cache_write_blocks_args(const uint8_t* src, const uint8_t* cache,
                                                   const uint64_t* src_offsets,
                                                   const uint64_t* cache_offsets,
                                                   const uint64_t* block_bytes,
                                                   size_t block_count) {
  if (block_count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || cache == nullptr || src_offsets == nullptr || cache_offsets == nullptr ||
      block_bytes == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_kv_cache_read_blocks_args(const uint8_t* cache, const uint8_t* dst,
                                                  const uint64_t* cache_offsets,
                                                  const uint64_t* dst_offsets,
                                                  const uint64_t* block_bytes,
                                                  size_t block_count) {
  if (block_count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (cache == nullptr || dst == nullptr || cache_offsets == nullptr || dst_offsets == nullptr ||
      block_bytes == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_cache_unpack_bf16_args(
    const uint8_t* payload, const uint16_t* kv_latent, const uint16_t* k_rope,
    const uint16_t* dsa_key, size_t rows, size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim,
    size_t payload_stride_bytes) {
  if (payload == nullptr || kv_latent == nullptr || k_rope == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (dsa_dim > 0 && dsa_key == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || kv_lora_rank == 0 || rope_dim == 0 || payload_stride_bytes == 0 ||
      payload_stride_bytes % sizeof(uint16_t) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t packed_width = 0;
  if (!checked_add(kv_lora_rank, rope_dim, &packed_width) ||
      !checked_add(packed_width, dsa_dim, &packed_width)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t payload_stride_bf16 = payload_stride_bytes / sizeof(uint16_t);
  if (payload_stride_bf16 < packed_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, packed_width, &ignored) ||
      !checked_mul(rows, kv_lora_rank, &ignored) || !checked_mul(rows, rope_dim, &ignored) ||
      (dsa_dim > 0 && !checked_mul(rows, dsa_dim, &ignored))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_projected_split_bf16_args(
    const uint16_t* projected, const uint16_t* k_nope, const uint16_t* v, size_t rows,
    size_t heads, size_t nope_dim, size_t v_dim) {
  if (projected == nullptr || k_nope == nullptr || v == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || nope_dim == 0 || v_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t head_width = 0;
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_add(nope_dim, v_dim, &head_width) || !checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, head_width, &ignored) ||
      !checked_mul(row_heads, nope_dim, &ignored) || !checked_mul(row_heads, v_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_prepare_bf16_args(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    const uint16_t* prepared, size_t rows, size_t projected_stride_bytes,
    size_t prepared_stride_bytes, float eps, float theta) {
  constexpr size_t projected_values = 512 + 64;
  constexpr size_t minimum_stride_bytes = projected_values * sizeof(uint16_t);
  if (projected == nullptr || positions == nullptr || norm_weight == nullptr ||
      prepared == nullptr || rows == 0 || projected_stride_bytes < minimum_stride_bytes ||
      prepared_stride_bytes < minimum_stride_bytes || projected_stride_bytes % sizeof(uint16_t) != 0 ||
      prepared_stride_bytes % sizeof(uint16_t) != 0 || !std::isfinite(eps) || eps <= 0.0f ||
      !std::isfinite(theta) || theta <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, projected_stride_bytes, &ignored) ||
      !checked_mul(rows, prepared_stride_bytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_pack_fp8_ds_mla_args(const uint16_t* projected,
                                                    const uint8_t* packed, size_t rows,
                                                    size_t projected_stride_bytes,
                                                    size_t packed_stride_bytes) {
  if (projected == nullptr || packed == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || projected_stride_bytes == 0 || packed_stride_bytes == 0 ||
      projected_stride_bytes % sizeof(uint16_t) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t projected_stride_bf16 = projected_stride_bytes / sizeof(uint16_t);
  if (projected_stride_bf16 < kMlaFp8DsProjectedValues ||
      packed_stride_bytes < kMlaFp8DsPackedBytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, projected_stride_bytes, &ignored) ||
      !checked_mul(rows, packed_stride_bytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_unpack_fp8_ds_mla_args(const uint8_t* packed,
                                                      const uint16_t* projected, size_t rows,
                                                      size_t packed_stride_bytes,
                                                      size_t projected_stride_bytes) {
  return validate_mla_kv_pack_fp8_ds_mla_args(projected, packed, rows, projected_stride_bytes,
                                             packed_stride_bytes);
}

glmrt_status_t validate_mla_kv_pack_mxfp4_ds_mla_args(const uint16_t* projected,
                                                      const uint8_t* packed, size_t rows,
                                                      size_t projected_stride_bytes,
                                                      size_t packed_stride_bytes) {
  if (projected == nullptr || packed == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || projected_stride_bytes == 0 || packed_stride_bytes == 0 ||
      projected_stride_bytes % sizeof(uint16_t) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t projected_stride_bf16 = projected_stride_bytes / sizeof(uint16_t);
  if (projected_stride_bf16 < kMlaMxfp4DsProjectedValues ||
      packed_stride_bytes < kMlaMxfp4DsPackedBytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, projected_stride_bytes, &ignored) ||
      !checked_mul(rows, packed_stride_bytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_kv_unpack_mxfp4_ds_mla_args(const uint8_t* packed,
                                                        const uint16_t* projected, size_t rows,
                                                        size_t packed_stride_bytes,
                                                        size_t projected_stride_bytes) {
  return validate_mla_kv_pack_mxfp4_ds_mla_args(projected, packed, rows, projected_stride_bytes,
                                               packed_stride_bytes);
}

glmrt_status_t validate_mla_kv_finalize_store_candidate_args(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight,
    const uint8_t* cache_rows, const uint16_t* attention_ready,
    const uint16_t* dsa_normalized, size_t rows,
    size_t projected_stride_bytes, size_t cache_stride_bytes,
    size_t attention_stride_bytes, size_t dsa_stride_bytes,
    size_t dsa_values, int cache_format, float eps) {
  constexpr size_t kv_bytes = kMlaFp8DsProjectedValues * sizeof(uint16_t);
  if (projected == nullptr || rope_factors == nullptr ||
      norm_weight == nullptr || cache_rows == nullptr ||
      rows == 0 || projected_stride_bytes < kv_bytes ||
      projected_stride_bytes % sizeof(uint16_t) != 0 ||
      !std::isfinite(eps) || eps <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t main_bytes = 0;
  switch (cache_format) {
    case kMlaKvFinalizeBf16:
      main_bytes = kv_bytes;
      break;
    case kMlaKvFinalizeFp8:
      main_bytes = kMlaFp8DsPackedBytes;
      break;
    case kMlaKvFinalizeMxfp4:
      main_bytes = kMlaMxfp4DsPackedBytes;
      break;
    default:
      return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t dsa_bytes = 0;
  size_t required_cache_bytes = 0;
  if (!checked_mul(dsa_values, sizeof(uint16_t), &dsa_bytes) ||
      !checked_add(main_bytes, dsa_bytes, &required_cache_bytes) ||
      cache_stride_bytes < required_cache_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if ((attention_ready != nullptr &&
       (attention_stride_bytes < kv_bytes ||
        attention_stride_bytes % sizeof(uint16_t) != 0)) ||
      (dsa_values > 0 &&
       (dsa_normalized == nullptr || dsa_stride_bytes < dsa_bytes ||
        dsa_stride_bytes % sizeof(uint16_t) != 0))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, projected_stride_bytes, &ignored) ||
      !checked_mul(rows, cache_stride_bytes, &ignored) ||
      (attention_ready != nullptr &&
       !checked_mul(rows, attention_stride_bytes, &ignored)) ||
      (dsa_values > 0 && !checked_mul(rows, dsa_stride_bytes, &ignored))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_kv_cache_write_bytes_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t cache, size_t cache_offset_bytes,
    size_t bytes) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_graph_kv_cache_write_buffers(src, cache, cache_offset_bytes, bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const glmrt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != GLMRT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(kv_cache_write_bytes_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint8_t* src_ptr = static_cast<const uint8_t*>(src.ptr);
  uint8_t* cache_ptr = static_cast<uint8_t*>(cache.ptr);
  void* args[] = {
      &src_ptr,
      &cache_ptr,
      &cache_offset_bytes,
      &bytes,
  };
  const int threads = 256;
  const size_t block_count = (bytes - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(kv_cache_write_bytes_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(block_count), 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_graph_update_mla_kv_cache_unpack_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t payload, glmrt_device_buffer_t kv_latent,
    glmrt_device_buffer_t k_rope, glmrt_device_buffer_t dsa_key, size_t rows,
    size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim, size_t payload_stride_bytes) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const uint8_t* payload_ptr = static_cast<const uint8_t*>(payload.ptr);
  uint16_t* kv_latent_ptr = static_cast<uint16_t*>(kv_latent.ptr);
  uint16_t* k_rope_ptr = static_cast<uint16_t*>(k_rope.ptr);
  uint16_t* dsa_key_ptr = static_cast<uint16_t*>(dsa_key.ptr);
  const glmrt_status_t valid = validate_mla_kv_cache_unpack_bf16_args(
      payload_ptr, kv_latent_ptr, k_rope_ptr, dsa_key_ptr, rows, kv_lora_rank, rope_dim, dsa_dim,
      payload_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const glmrt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != GLMRT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(mla_kv_cache_unpack_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  size_t width = 0;
  size_t total = 0;
  if (!checked_add(kv_lora_rank, rope_dim, &width) || !checked_add(width, dsa_dim, &width) ||
      !checked_mul(rows, width, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t payload_stride_bf16 = payload_stride_bytes / sizeof(uint16_t);
  void* args[] = {
      &payload_ptr,
      &kv_latent_ptr,
      &k_rope_ptr,
      &dsa_key_ptr,
      &rows,
      &kv_lora_rank,
      &rope_dim,
      &dsa_dim,
      &payload_stride_bf16,
  };

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(mla_kv_cache_unpack_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(block_count), 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_graph_update_mla_kv_projected_split_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t projected, glmrt_device_buffer_t k_nope, glmrt_device_buffer_t v,
    size_t rows, size_t heads, size_t nope_dim, size_t v_dim) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const uint16_t* projected_ptr = static_cast<const uint16_t*>(projected.ptr);
  uint16_t* k_nope_ptr = static_cast<uint16_t*>(k_nope.ptr);
  uint16_t* v_ptr = static_cast<uint16_t*>(v.ptr);
  const glmrt_status_t valid = validate_mla_kv_projected_split_bf16_args(
      projected_ptr, k_nope_ptr, v_ptr, rows, heads, nope_dim, v_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const glmrt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != GLMRT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(mla_kv_projected_split_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  size_t head_width = 0;
  size_t row_heads = 0;
  size_t total = 0;
  if (!checked_add(nope_dim, v_dim, &head_width) || !checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, head_width, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  void* args[] = {
      &projected_ptr,
      &k_nope_ptr,
      &v_ptr,
      &rows,
      &heads,
      &nope_dim,
      &v_dim,
  };

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(mla_kv_projected_split_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(block_count), 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_write_bytes_async(const uint8_t* src,
                                                                uint8_t* cache,
                                                                size_t cache_offset_bytes,
                                                                size_t bytes,
                                                                void* cuda_stream) {
  const glmrt_status_t valid =
      validate_kv_cache_write_args(src, cache, cache_offset_bytes, bytes);
  if (valid != GLMRT_STATUS_OK || bytes == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t block_count = (bytes - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  kv_cache_write_bytes_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      src, cache, cache_offset_bytes, bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_write_bytes(const uint8_t* src, uint8_t* cache,
                                                          size_t cache_offset_bytes,
                                                          size_t bytes) {
  const glmrt_status_t status =
      glmrt_cuda_kv_cache_write_bytes_async(src, cache, cache_offset_bytes, bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_read_bytes_async(const uint8_t* cache,
                                                               uint8_t* dst,
                                                               size_t cache_offset_bytes,
                                                               size_t bytes,
                                                               void* cuda_stream) {
  const glmrt_status_t valid = validate_kv_cache_read_args(cache, dst, cache_offset_bytes, bytes);
  if (valid != GLMRT_STATUS_OK || bytes == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t block_count = (bytes - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  kv_cache_read_bytes_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      cache, dst, cache_offset_bytes, bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_read_bytes(const uint8_t* cache, uint8_t* dst,
                                                         size_t cache_offset_bytes,
                                                         size_t bytes) {
  const glmrt_status_t status =
      glmrt_cuda_kv_cache_read_bytes_async(cache, dst, cache_offset_bytes, bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_write_blocks_async(
    const uint8_t* src, uint8_t* cache, const uint64_t* src_offsets,
    const uint64_t* cache_offsets, const uint64_t* block_bytes, size_t block_count,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_kv_cache_write_blocks_args(
      src, cache, src_offsets, cache_offsets, block_bytes, block_count);
  if (valid != GLMRT_STATUS_OK || block_count == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  kv_cache_write_blocks_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      src, cache, src_offsets, cache_offsets, block_bytes, block_count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_write_blocks(
    const uint8_t* src, uint8_t* cache, const uint64_t* src_offsets,
    const uint64_t* cache_offsets, const uint64_t* block_bytes, size_t block_count) {
  const glmrt_status_t status = glmrt_cuda_kv_cache_write_blocks_async(
      src, cache, src_offsets, cache_offsets, block_bytes, block_count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_read_blocks_async(
    const uint8_t* cache, uint8_t* dst, const uint64_t* cache_offsets,
    const uint64_t* dst_offsets, const uint64_t* block_bytes, size_t block_count,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_kv_cache_read_blocks_args(
      cache, dst, cache_offsets, dst_offsets, block_bytes, block_count);
  if (valid != GLMRT_STATUS_OK || block_count == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  kv_cache_read_blocks_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      cache, dst, cache_offsets, dst_offsets, block_bytes, block_count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_kv_cache_read_blocks(
    const uint8_t* cache, uint8_t* dst, const uint64_t* cache_offsets,
    const uint64_t* dst_offsets, const uint64_t* block_bytes, size_t block_count) {
  const glmrt_status_t status = glmrt_cuda_kv_cache_read_blocks_async(
      cache, dst, cache_offsets, dst_offsets, block_bytes, block_count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_cache_unpack_bf16_async(
    const uint8_t* payload, uint16_t* kv_latent, uint16_t* k_rope, uint16_t* dsa_key, size_t rows,
    size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim, size_t payload_stride_bytes,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_cache_unpack_bf16_args(
      payload, kv_latent, k_rope, dsa_key, rows, kv_lora_rank, rope_dim, dsa_dim,
      payload_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t width = kv_lora_rank + rope_dim + dsa_dim;
  const size_t total = rows * width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  mla_kv_cache_unpack_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      payload, kv_latent, k_rope, dsa_key, rows, kv_lora_rank, rope_dim, dsa_dim,
      payload_stride_bytes / sizeof(uint16_t));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_cache_unpack_bf16(
    const uint8_t* payload, uint16_t* kv_latent, uint16_t* k_rope, uint16_t* dsa_key, size_t rows,
    size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim, size_t payload_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_cache_unpack_bf16_async(
      payload, kv_latent, k_rope, dsa_key, rows, kv_lora_rank, rope_dim, dsa_dim,
      payload_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_projected_split_bf16_async(
    const uint16_t* projected, uint16_t* k_nope, uint16_t* v, size_t rows, size_t heads,
    size_t nope_dim, size_t v_dim, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_projected_split_bf16_args(
      projected, k_nope, v, rows, heads, nope_dim, v_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * (nope_dim + v_dim);
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  // FlashInfer graph replay can leave a consumed launch error in CUDA's
  // thread-local last-error slot. Report only this kernel's launch status.
  (void)cudaGetLastError();
  mla_kv_projected_split_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      projected, k_nope, v, rows, heads, nope_dim, v_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_projected_split_bf16(
    const uint16_t* projected, uint16_t* k_nope, uint16_t* v, size_t rows, size_t heads,
    size_t nope_dim, size_t v_dim) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_projected_split_bf16_async(
      projected, k_nope, v, rows, heads, nope_dim, v_dim, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16_async(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    uint16_t* prepared, size_t rows, size_t projected_stride_bytes,
    size_t prepared_stride_bytes, float eps, float theta, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_prepare_bf16_args(
      projected, positions, norm_weight, prepared, rows, projected_stride_bytes,
      prepared_stride_bytes, eps, theta);
  if (valid != GLMRT_STATUS_OK ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  // Capability probes in other backends may intentionally observe and handle
  // a CUDA launch error. Do not attribute that sticky status to this launch.
  (void)cudaGetLastError();
  mla_kv_prepare_bf16_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      projected, positions, norm_weight, prepared, rows,
      projected_stride_bytes / sizeof(uint16_t), prepared_stride_bytes / sizeof(uint16_t), eps,
      theta);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_rope_factors_f32_candidate_async(
    const uint32_t* positions, float* factors, size_t rows, float theta,
    void* cuda_stream) {
  if (positions == nullptr || factors == nullptr || rows == 0 ||
      !std::isfinite(theta) || theta <= 0.0f ||
      rows > std::numeric_limits<size_t>::max() / 32) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t factor_pairs = rows * 32;
  constexpr size_t threads = 32;
  const size_t blocks = (factor_pairs + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_rope_factors_f32_candidate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      positions, factors, rows, theta);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16_precomputed_rope_candidate_async(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight, uint16_t* prepared, size_t rows,
    size_t projected_stride_bytes, size_t prepared_stride_bytes, float eps,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_prepare_bf16_args(
      projected, reinterpret_cast<const uint32_t*>(rope_factors), norm_weight,
      prepared, rows, projected_stride_bytes, prepared_stride_bytes, eps, 1.0f);
  if (valid != GLMRT_STATUS_OK ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_kv_prepare_bf16_precomputed_rope_candidate_kernel<<<
      static_cast<int>(rows), 256, 0, stream>>>(
      projected, rope_factors, norm_weight, prepared, rows,
      projected_stride_bytes / sizeof(uint16_t),
      prepared_stride_bytes / sizeof(uint16_t), eps);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_finalize_store_candidate_async(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight,
    uint8_t* cache_rows, uint16_t* attention_ready,
    const uint16_t* dsa_normalized, size_t rows,
    size_t projected_stride_bytes, size_t cache_stride_bytes,
    size_t attention_stride_bytes, size_t dsa_stride_bytes,
    size_t dsa_values, int cache_format, float eps, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_finalize_store_candidate_args(
      projected, rope_factors, norm_weight, cache_rows,
      attention_ready, dsa_normalized, rows, projected_stride_bytes,
      cache_stride_bytes, attention_stride_bytes, dsa_stride_bytes, dsa_values,
      cache_format, eps);
  if (valid != GLMRT_STATUS_OK ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t projected_stride_bf16 =
      projected_stride_bytes / sizeof(uint16_t);
  const size_t attention_stride_bf16 =
      attention_stride_bytes / sizeof(uint16_t);
  const size_t dsa_stride_bf16 = dsa_stride_bytes / sizeof(uint16_t);
  (void)cudaGetLastError();
  switch (cache_format) {
    case kMlaKvFinalizeBf16:
      mla_kv_finalize_store_candidate_kernel<kMlaKvFinalizeBf16>
          <<<static_cast<int>(rows), 256, 0, stream>>>(
              projected, rope_factors, norm_weight, cache_rows,
              attention_ready, dsa_normalized, rows, projected_stride_bf16,
              cache_stride_bytes, attention_stride_bf16, dsa_stride_bf16,
              dsa_values, eps);
      break;
    case kMlaKvFinalizeFp8:
      mla_kv_finalize_store_candidate_kernel<kMlaKvFinalizeFp8>
          <<<static_cast<int>(rows), 256, 0, stream>>>(
              projected, rope_factors, norm_weight, cache_rows,
              attention_ready, dsa_normalized, rows, projected_stride_bf16,
              cache_stride_bytes, attention_stride_bf16, dsa_stride_bf16,
              dsa_values, eps);
      break;
    case kMlaKvFinalizeMxfp4:
      mla_kv_finalize_store_candidate_kernel<kMlaKvFinalizeMxfp4>
          <<<static_cast<int>(rows), 256, 0, stream>>>(
              projected, rope_factors, norm_weight, cache_rows,
              attention_ready, dsa_normalized, rows, projected_stride_bf16,
              cache_stride_bytes, attention_stride_bf16, dsa_stride_bf16,
              dsa_values, eps);
      break;
    default:
      return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    uint16_t* prepared, size_t rows, size_t projected_stride_bytes,
    size_t prepared_stride_bytes, float eps, float theta) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_prepare_bf16_async(
      projected, positions, norm_weight, prepared, rows, projected_stride_bytes,
      prepared_stride_bytes, eps, theta, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_pack_fp8_ds_mla_async(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_pack_fp8_ds_mla_args(
      projected, packed, rows, projected_stride_bytes, packed_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_kv_pack_fp8_ds_mla_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      projected, packed, rows, projected_stride_bytes / sizeof(uint16_t), packed_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_pack_fp8_ds_mla(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_pack_fp8_ds_mla_async(
      projected, packed, rows, projected_stride_bytes, packed_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_unpack_fp8_ds_mla_async(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_unpack_fp8_ds_mla_args(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_kv_unpack_fp8_ds_mla_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes / sizeof(uint16_t));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_unpack_fp8_ds_mla(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_unpack_fp8_ds_mla_async(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_pack_mxfp4_ds_mla_async(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_pack_mxfp4_ds_mla_args(
      projected, packed, rows, projected_stride_bytes, packed_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_kv_pack_mxfp4_ds_mla_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      projected, packed, rows, projected_stride_bytes / sizeof(uint16_t), packed_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_pack_mxfp4_ds_mla(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_pack_mxfp4_ds_mla_async(
      projected, packed, rows, projected_stride_bytes, packed_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla_async(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_kv_unpack_mxfp4_ds_mla_args(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_kv_unpack_mxfp4_ds_mla_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes / sizeof(uint16_t));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla_async(
      packed, projected, rows, packed_stride_bytes, projected_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
