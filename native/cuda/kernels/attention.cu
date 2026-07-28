#include "common.h"

#include <cuda_fp8.h>

namespace {

__global__ void causal_attention_f32_kernel(const float* q, const float* k, const float* v,
                                            float* out, size_t rows, size_t heads,
                                            size_t qk_dim, size_t v_dim, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * v_dim;
  if (idx >= total) {
    return;
  }
  const size_t v_col = idx % v_dim;
  const size_t head_index = idx / v_dim;
  const size_t head = head_index % heads;
  const size_t row = head_index / heads;

  const float* q_vec = q + (row * heads + head) * qk_dim;
  float max_score = -CUDART_INF_F;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const float* k_vec = k + (key_row * heads + head) * qk_dim;
    float dot = 0.0f;
    for (size_t col = 0; col < qk_dim; ++col) {
      dot += q_vec[col] * k_vec[col];
    }
    const float score = dot * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const float* k_vec = k + (key_row * heads + head) * qk_dim;
    float dot = 0.0f;
    for (size_t col = 0; col < qk_dim; ++col) {
      dot += q_vec[col] * k_vec[col];
    }
    const float weight = expf(dot * scale - max_score);
    denom += weight;
    acc += weight * v[(key_row * heads + head) * v_dim + v_col];
  }
  out[idx] = acc / fmaxf(denom, 1.0e-12f);
}

__global__ void causal_attention_bf16_kernel(const uint16_t* q, const uint16_t* k,
                                             const uint16_t* v, uint16_t* out, size_t rows,
                                             size_t heads, size_t qk_dim, size_t v_dim,
                                             float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * v_dim;
  if (idx >= total) {
    return;
  }
  const size_t v_col = idx % v_dim;
  const size_t head_index = idx / v_dim;
  const size_t head = head_index % heads;
  const size_t row = head_index / heads;

  const uint16_t* q_vec = q + (row * heads + head) * qk_dim;
  float max_score = -CUDART_INF_F;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_vec = k + (key_row * heads + head) * qk_dim;
    float dot = 0.0f;
    for (size_t col = 0; col < qk_dim; ++col) {
      dot += bf16_to_f32(q_vec[col]) * bf16_to_f32(k_vec[col]);
    }
    const float score = dot * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_vec = k + (key_row * heads + head) * qk_dim;
    float dot = 0.0f;
    for (size_t col = 0; col < qk_dim; ++col) {
      dot += bf16_to_f32(q_vec[col]) * bf16_to_f32(k_vec[col]);
    }
    const float weight = expf(dot * scale - max_score);
    denom += weight;
    acc += weight * bf16_to_f32(v[(key_row * heads + head) * v_dim + v_col]);
  }
  out[idx] = f32_to_bf16(acc / fmaxf(denom, 1.0e-12f));
}

__global__ void rope_f32_kernel(const float* input, const uint32_t* positions, float* out,
                                size_t rows, size_t heads, size_t rotary_dim, float theta) {
  const size_t pair_count = rotary_dim / 2;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * pair_count;
  if (idx >= total) {
    return;
  }
  const size_t pair = idx % pair_count;
  const size_t head_index = idx / pair_count;
  const size_t head = head_index % heads;
  const size_t row = head_index / heads;
  const size_t offset = (row * heads + head) * rotary_dim + pair * 2;
  const float angle =
      static_cast<float>(positions[row]) * powf(theta, -2.0f * static_cast<float>(pair) /
                                                          static_cast<float>(rotary_dim));
  const float cos_value = cosf(angle);
  const float sin_value = sinf(angle);
  const float even = input[offset];
  const float odd = input[offset + 1];
  out[offset] = even * cos_value - odd * sin_value;
  out[offset + 1] = even * sin_value + odd * cos_value;
}

__global__ void rope_bf16_kernel(const uint16_t* input, const uint32_t* positions, uint16_t* out,
                                 size_t rows, size_t heads, size_t rotary_dim, float theta) {
  const size_t pair_count = rotary_dim / 2;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * pair_count;
  if (idx >= total) {
    return;
  }
  const size_t pair = idx % pair_count;
  const size_t head_index = idx / pair_count;
  const size_t head = head_index % heads;
  const size_t row = head_index / heads;
  const size_t offset = (row * heads + head) * rotary_dim + pair * 2;
  const float angle =
      static_cast<float>(positions[row]) * powf(theta, -2.0f * static_cast<float>(pair) /
                                                          static_cast<float>(rotary_dim));
  const float cos_value = cosf(angle);
  const float sin_value = sinf(angle);
  const float even = bf16_to_f32(input[offset]);
  const float odd = bf16_to_f32(input[offset + 1]);
  out[offset] = f32_to_bf16(even * cos_value - odd * sin_value);
  out[offset + 1] = f32_to_bf16(even * sin_value + odd * cos_value);
}

__global__ void mla_rope_attention_bf16_kernel(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * v_dim;
  if (idx >= total) {
    return;
  }
  const size_t v_col = idx % v_dim;
  const size_t head_index = idx / v_dim;
  const size_t head = head_index % heads;
  const size_t row = head_index / heads;

  const uint16_t* q_nope_vec = q_nope + (row * heads + head) * nope_dim;
  const uint16_t* q_rope_vec = q_rope + (row * heads + head) * rope_dim;
  float max_score = -CUDART_INF_F;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_nope_vec = k_nope + (key_row * heads + head) * nope_dim;
    const uint16_t* k_rope_vec = k_rope + key_row * rope_dim;
    float nope_dot = 0.0f;
    for (size_t col = 0; col < nope_dim; ++col) {
      nope_dot += bf16_to_f32(q_nope_vec[col]) * bf16_to_f32(k_nope_vec[col]);
    }
    float rope_dot = 0.0f;
    for (size_t col = 0; col < rope_dim; ++col) {
      rope_dot += bf16_to_f32(q_rope_vec[col]) * bf16_to_f32(k_rope_vec[col]);
    }
    const float score = (nope_dot + rope_dot) * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_nope_vec = k_nope + (key_row * heads + head) * nope_dim;
    const uint16_t* k_rope_vec = k_rope + key_row * rope_dim;
    float nope_dot = 0.0f;
    for (size_t col = 0; col < nope_dim; ++col) {
      nope_dot += bf16_to_f32(q_nope_vec[col]) * bf16_to_f32(k_nope_vec[col]);
    }
    float rope_dot = 0.0f;
    for (size_t col = 0; col < rope_dim; ++col) {
      rope_dot += bf16_to_f32(q_rope_vec[col]) * bf16_to_f32(k_rope_vec[col]);
    }
    const float weight = expf((nope_dot + rope_dot) * scale - max_score);
    denom += weight;
    acc += weight * bf16_to_f32(v[(key_row * heads + head) * v_dim + v_col]);
  }
  out[idx] = f32_to_bf16(acc / fmaxf(denom, 1.0e-12f));
}

__global__ void mla_rope_attention_bf16_suffix_kernel(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * heads * v_dim;
  if (idx >= total) {
    return;
  }
  const size_t v_col = idx % v_dim;
  const size_t head_index = idx / v_dim;
  const size_t head = head_index % heads;
  const size_t query_row = head_index / heads;
  const size_t row = query_row_offset + query_row;

  const uint16_t* q_nope_vec = q_nope + (row * heads + head) * nope_dim;
  const uint16_t* q_rope_vec = q_rope + (row * heads + head) * rope_dim;
  float max_score = -CUDART_INF_F;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_nope_vec = k_nope + (key_row * heads + head) * nope_dim;
    const uint16_t* k_rope_vec = k_rope + key_row * rope_dim;
    float nope_dot = 0.0f;
    for (size_t col = 0; col < nope_dim; ++col) {
      nope_dot += bf16_to_f32(q_nope_vec[col]) * bf16_to_f32(k_nope_vec[col]);
    }
    float rope_dot = 0.0f;
    for (size_t col = 0; col < rope_dim; ++col) {
      rope_dot += bf16_to_f32(q_rope_vec[col]) * bf16_to_f32(k_rope_vec[col]);
    }
    const float score = (nope_dot + rope_dot) * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (size_t key_row = 0; key_row <= row; ++key_row) {
    const uint16_t* k_nope_vec = k_nope + (key_row * heads + head) * nope_dim;
    const uint16_t* k_rope_vec = k_rope + key_row * rope_dim;
    float nope_dot = 0.0f;
    for (size_t col = 0; col < nope_dim; ++col) {
      nope_dot += bf16_to_f32(q_nope_vec[col]) * bf16_to_f32(k_nope_vec[col]);
    }
    float rope_dot = 0.0f;
    for (size_t col = 0; col < rope_dim; ++col) {
      rope_dot += bf16_to_f32(q_rope_vec[col]) * bf16_to_f32(k_rope_vec[col]);
    }
    const float weight = expf((nope_dot + rope_dot) * scale - max_score);
    denom += weight;
    acc += weight * bf16_to_f32(v[(key_row * heads + head) * v_dim + v_col]);
  }
  out[(query_row * heads + head) * v_dim + v_col] =
      f32_to_bf16(acc / fmaxf(denom, 1.0e-12f));
}

constexpr int kMlaKvSplitBf16 = 0;
constexpr int kMlaKvInterleavedBf16 = 1;
constexpr int kMlaKvInterleavedFp8 = 2;
constexpr int kMlaKvInterleavedMxfp4 = 3;

template <int kKvFormat>
__device__ float mla_compressed_kv_latent_value(const uint16_t* split_latent,
                                                const uint8_t* row_bytes,
                                                size_t row, size_t col,
                                                size_t kv_lora_rank) {
  if constexpr (kKvFormat == kMlaKvSplitBf16) {
    return bf16_to_f32(split_latent[row * kv_lora_rank + col]);
  } else if constexpr (kKvFormat == kMlaKvInterleavedBf16) {
    return bf16_to_f32(reinterpret_cast<const uint16_t*>(row_bytes)[col]);
  } else if constexpr (kKvFormat == kMlaKvInterleavedFp8) {
    constexpr size_t kGroupSize = 128;
    const float* scales = reinterpret_cast<const float*>(row_bytes + kv_lora_rank);
    return f8e4m3_to_f32(row_bytes[col]) * scales[col / kGroupSize];
  } else {
    constexpr size_t kCodeBytes = 512 / 2;
    constexpr size_t kBlockSize = 16;
    const uint8_t packed = row_bytes[col / 2];
    const uint8_t code = (col & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
    const uint8_t scale_byte = row_bytes[kCodeBytes + col / kBlockSize];
    return nvfp4_e2m1_code_value(code) * f8e4m3_to_f32(scale_byte);
  }
}

template <int kKvFormat>
__global__ void mla_compressed_attention_bf16_kernel(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope, uint16_t* out_latent,
    size_t rows, size_t rope_dim, size_t kv_lora_rank, size_t kv_row_stride_bytes,
    size_t rope_offset_bytes, float scale) {
  const size_t head = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t latent_col0 = tid;
  const size_t latent_col1 = tid + blockDim.x;
  const uint16_t* q_absorbed_head = q_absorbed + head * kv_lora_rank;
  const uint16_t* q_rope_head = q_rope + head * rope_dim;
  const float q_absorbed0 = bf16_to_f32(q_absorbed_head[latent_col0]);
  const float q_absorbed1 = bf16_to_f32(q_absorbed_head[latent_col1]);

  __shared__ float warp_sums[8];
  __shared__ float score_shared;
  float running_max = -CUDART_INF_F;
  float running_denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (size_t row = 0; row < rows; ++row) {
    const uint8_t* row_bytes = reinterpret_cast<const uint8_t*>(kv_latent) +
                               row * kv_row_stride_bytes;
    const float latent0 = mla_compressed_kv_latent_value<kKvFormat>(
        kv_latent, row_bytes, row, latent_col0, kv_lora_rank);
    const float latent1 = mla_compressed_kv_latent_value<kKvFormat>(
        kv_latent, row_bytes, row, latent_col1, kv_lora_rank);
    float partial = q_absorbed0 * latent0 + q_absorbed1 * latent1;
    if (tid < rope_dim / 2) {
      const size_t rope_col = tid * 2;
      const uint16_t* rope_row = kKvFormat == kMlaKvSplitBf16
                                     ? k_rope + row * rope_dim
                                     : reinterpret_cast<const uint16_t*>(
                                           row_bytes + rope_offset_bytes);
      partial += bf16_to_f32(q_rope_head[rope_col]) * bf16_to_f32(rope_row[rope_col]);
      partial +=
          bf16_to_f32(q_rope_head[rope_col + 1]) * bf16_to_f32(rope_row[rope_col + 1]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    const size_t lane = tid & 31;
    const size_t warp = tid >> 5;
    if (lane == 0) {
      warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < 8 ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
      }
      if (lane == 0) {
        score_shared = block_sum * scale;
      }
    }
    __syncthreads();

    const float score = score_shared;
    const float next_max = fmaxf(running_max, score);
    const float old_weight = expf(running_max - next_max);
    const float new_weight = expf(score - next_max);
    running_denom = running_denom * old_weight + new_weight;
    acc0 = acc0 * old_weight + latent0 * new_weight;
    acc1 = acc1 * old_weight + latent1 * new_weight;
    running_max = next_max;
    __syncthreads();
  }

  const float inverse_denom = 1.0f / fmaxf(running_denom, 1.0e-12f);
  uint16_t* output_head = out_latent + head * kv_lora_rank;
  output_head[latent_col0] = f32_to_bf16(acc0 * inverse_denom);
  output_head[latent_col1] = f32_to_bf16(acc1 * inverse_denom);
}

constexpr size_t kSparseMlaNvfp4Rank = 512;
constexpr size_t kSparseMlaNvfp4RopeDim = 64;
constexpr size_t kSparseMlaNvfp4CodeBytes = kSparseMlaNvfp4Rank / 2;
constexpr size_t kSparseMlaNvfp4ScaleBytes = kSparseMlaNvfp4Rank / 16;
constexpr size_t kSparseMlaNvfp4PaddingBytes = 16;
constexpr size_t kSparseMlaNvfp4RopeOffset =
    kSparseMlaNvfp4CodeBytes + kSparseMlaNvfp4ScaleBytes +
    kSparseMlaNvfp4PaddingBytes;
constexpr size_t kSparseMlaNvfp4MinRowBytes =
    kSparseMlaNvfp4RopeOffset + kSparseMlaNvfp4RopeDim * sizeof(uint16_t);
constexpr size_t kSparseMlaKeysPerSplit = 64;
constexpr size_t kSparseMlaMaxTopk = 2048;
constexpr size_t kSparseMlaMaxSplits =
    kSparseMlaMaxTopk / kSparseMlaKeysPerSplit;
constexpr size_t kSparseMlaSplitQueryLimit = 64;
constexpr size_t kSparseMlaFp8RowBytes = 656;
constexpr size_t kSparseMlaFp8ScaleOffset = 512;
constexpr size_t kSparseMlaFp8RopeOffset = 528;

__device__ __forceinline__ float sparse_mla_nvfp4_value(
    const uint8_t* row_bytes, size_t col);

__global__ void sparse_mla_nvfp4_gather_fp8_kernel(
    const uint8_t* nvfp4_kv, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint8_t* fp8_kv, int32_t* fp8_indices,
    size_t query_rows, size_t selected_index_stride, size_t staged_topk,
    size_t nvfp4_row_stride_bytes) {
  const size_t staged_row = blockIdx.x;
  const size_t query_row = staged_row / staged_topk;
  const size_t key = staged_row - query_row * staged_topk;
  if (query_row >= query_rows ||
      key >= static_cast<size_t>(max(0, topk_lengths[query_row]))) {
    return;
  }
  const int32_t physical_row =
      selected_indices[query_row * selected_index_stride + key];
  const uint8_t* src =
      nvfp4_kv + static_cast<size_t>(physical_row) * nvfp4_row_stride_bytes;
  uint8_t* dst = fp8_kv + staged_row * kSparseMlaFp8RowBytes;
  fp8_indices[staged_row] = static_cast<int32_t>(staged_row);

  const size_t warp = threadIdx.x >> 5;
  const size_t lane = threadIdx.x & 31;
  __shared__ float group_scales[4];
  if (warp < 4) {
    const size_t group_offset = warp * 128;
    float max_abs = 0.0f;
#pragma unroll
    for (size_t item = 0; item < 4; ++item) {
      max_abs = fmaxf(
          max_abs,
          fabsf(sparse_mla_nvfp4_value(src, group_offset + lane + item * 32)));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }
    if (lane == 0) {
      float group_scale = max_abs / 448.0f;
      if (!(group_scale > 0.0f)) {
        group_scale = 1.0f;
      }
      group_scales[warp] = group_scale;
      reinterpret_cast<float*>(dst + kSparseMlaFp8ScaleOffset)[warp] =
          group_scale;
    }
  }
  __syncthreads();
  if (warp < 4) {
    const size_t group_offset = warp * 128;
    const float group_scale = group_scales[warp];
#pragma unroll
    for (size_t item = 0; item < 4; ++item) {
      const size_t col = group_offset + lane + item * 32;
      dst[col] = __nv_cvt_float_to_fp8(
          sparse_mla_nvfp4_value(src, col) / group_scale, __NV_SATFINITE,
          __NV_E4M3);
    }
  }
  if (threadIdx.x < kSparseMlaNvfp4RopeDim) {
    reinterpret_cast<uint16_t*>(dst + kSparseMlaFp8RopeOffset)[threadIdx.x] =
        reinterpret_cast<const uint16_t*>(src + kSparseMlaNvfp4RopeOffset)
            [threadIdx.x];
  }
}

__global__ void mla_nvfp4_expand_fp8_paged_kernel(
    const uint8_t* nvfp4_kv, const uint32_t* physical_pages,
    const int32_t* active_rows, uint8_t* fp8_kv, size_t max_tokens,
    size_t page_size, size_t nvfp4_row_stride_bytes) {
  const int32_t active = max(0, min(*active_rows, static_cast<int32_t>(max_tokens)));
  const size_t warp = threadIdx.x >> 5;
  const size_t lane = threadIdx.x & 31;
  __shared__ float group_scales[4];

  for (size_t logical_row = blockIdx.x;
       logical_row < static_cast<size_t>(active); logical_row += gridDim.x) {
    const size_t logical_page = logical_row / page_size;
    const size_t page_offset = logical_row - logical_page * page_size;
    const size_t physical_row =
        static_cast<size_t>(physical_pages[logical_page]) * page_size + page_offset;
    if (physical_row >= max_tokens) {
      continue;
    }
    const uint8_t* src = nvfp4_kv + physical_row * nvfp4_row_stride_bytes;
    uint8_t* dst = fp8_kv + physical_row * kSparseMlaFp8RowBytes;

    if (warp < 4) {
      const size_t source_scale = warp * 4;
      uint8_t max_scale = src[kSparseMlaNvfp4CodeBytes + source_scale];
#pragma unroll
      for (size_t item = 1; item < 4; ++item) {
        max_scale =
            max(max_scale, src[kSparseMlaNvfp4CodeBytes + source_scale + item]);
      }
      // A 2^-6 factor maps the largest source E2M1 value (6) below the
      // E4M3 maximum (448), while retaining an exact power-of-two scale.
      const float group_scale =
          ldexpf(1.0f, static_cast<int>(max_scale) - 127 - 6);
      if (lane == 0) {
        group_scales[warp] = group_scale;
        reinterpret_cast<float*>(dst + kSparseMlaFp8ScaleOffset)[warp] =
            group_scale;
      }
    }
    __syncthreads();

    if (warp < 4) {
      const size_t group_offset = warp * 128;
      const float group_scale = group_scales[warp];
#pragma unroll
      for (size_t item = 0; item < 4; ++item) {
        const size_t col = group_offset + lane + item * 32;
        dst[col] = __nv_cvt_float_to_fp8(
            sparse_mla_nvfp4_value(src, col) / group_scale, __NV_SATFINITE,
            __NV_E4M3);
      }
    }
    if (threadIdx.x < kSparseMlaNvfp4RopeDim) {
      reinterpret_cast<uint16_t*>(dst + kSparseMlaFp8RopeOffset)[threadIdx.x] =
          reinterpret_cast<const uint16_t*>(src + kSparseMlaNvfp4RopeOffset)
              [threadIdx.x];
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float sparse_mla_nvfp4_value(
    const uint8_t* row_bytes, size_t col) {
  const uint8_t packed = row_bytes[col / 2];
  const uint8_t code = (col & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
  const uint8_t scale_byte =
      row_bytes[kSparseMlaNvfp4CodeBytes + col / 16];
  return nvfp4_e2m1_code_value(code) * f8e4m3_to_f32(scale_byte);
}

template <bool kSplit>
__global__ void sparse_mla_nvfp4_attention_kernel(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* output, float* output_lse, size_t query_rows,
    size_t heads, size_t topk, size_t kv_row_stride_bytes, float scale) {
  const size_t query_head = blockIdx.x;
  const size_t query_row = query_head / heads;
  const size_t head = query_head - query_row * heads;
  if (query_row >= query_rows) {
    return;
  }
  const size_t split = kSplit ? blockIdx.y : 0;
  const size_t split_start = kSplit ? split * kSparseMlaKeysPerSplit : 0;
  const size_t valid = static_cast<size_t>(
      max(0, min(topk_lengths[query_row], static_cast<int32_t>(topk))));
  const size_t split_end =
      kSplit ? min(valid, split_start + kSparseMlaKeysPerSplit) : valid;
  const size_t tid = threadIdx.x;
  const size_t col0 = tid;
  const size_t col1 = tid + blockDim.x;
  const uint16_t* query_head_ptr =
      query + (query_row * heads + head) *
                  (kSparseMlaNvfp4Rank + kSparseMlaNvfp4RopeDim);
  const float query0 = bf16_to_f32(query_head_ptr[col0]);
  const float query1 = bf16_to_f32(query_head_ptr[col1]);

  __shared__ float warp_sums[8];
  __shared__ float score_shared;
  float running_max = -CUDART_INF_F;
  float running_denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (size_t key = split_start; key < split_end; ++key) {
    const int32_t physical_row = selected_indices[query_row * topk + key];
    const uint8_t* row_bytes =
        kv_payload + static_cast<size_t>(physical_row) * kv_row_stride_bytes;
    const float latent0 = sparse_mla_nvfp4_value(row_bytes, col0);
    const float latent1 = sparse_mla_nvfp4_value(row_bytes, col1);
    float partial = query0 * latent0 + query1 * latent1;
    if (tid < kSparseMlaNvfp4RopeDim / 2) {
      const size_t rope_col = tid * 2;
      const uint16_t* rope =
          reinterpret_cast<const uint16_t*>(row_bytes + kSparseMlaNvfp4RopeOffset);
      partial +=
          bf16_to_f32(query_head_ptr[kSparseMlaNvfp4Rank + rope_col]) *
          bf16_to_f32(rope[rope_col]);
      partial +=
          bf16_to_f32(query_head_ptr[kSparseMlaNvfp4Rank + rope_col + 1]) *
          bf16_to_f32(rope[rope_col + 1]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    const size_t lane = tid & 31;
    const size_t warp = tid >> 5;
    if (lane == 0) {
      warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < 8 ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
      }
      if (lane == 0) {
        score_shared = block_sum * scale;
      }
    }
    __syncthreads();
    const float score = score_shared;
    const float next_max = fmaxf(running_max, score);
    const float old_weight = expf(running_max - next_max);
    const float new_weight = expf(score - next_max);
    running_denom = running_denom * old_weight + new_weight;
    acc0 = acc0 * old_weight + latent0 * new_weight;
    acc1 = acc1 * old_weight + latent1 * new_weight;
    running_max = next_max;
    __syncthreads();
  }

  const size_t output_head =
      kSplit ? (query_head * kSparseMlaMaxSplits + split) : query_head;
  uint16_t* output_ptr = output + output_head * kSparseMlaNvfp4Rank;
  if (split_start < split_end) {
    const float inverse_denom = 1.0f / fmaxf(running_denom, 1.0e-12f);
    output_ptr[col0] = f32_to_bf16(acc0 * inverse_denom);
    output_ptr[col1] = f32_to_bf16(acc1 * inverse_denom);
    if (tid == 0) {
      output_lse[output_head] = running_max + logf(running_denom);
    }
  } else {
    output_ptr[col0] = 0;
    output_ptr[col1] = 0;
    if (tid == 0) {
      output_lse[output_head] = -CUDART_INF_F;
    }
  }
}

template <bool kSplit>
__global__ void sparse_mla_bf16_attention_kernel(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* output, float* output_lse, size_t query_rows,
    size_t heads, size_t topk, size_t kv_row_stride_bytes, float scale) {
  constexpr size_t kRank = 512;
  constexpr size_t kRopeDim = 64;
  constexpr size_t kRopeOffsetBytes = kRank * sizeof(uint16_t);
  const size_t query_head = blockIdx.x;
  const size_t query_row = query_head / heads;
  const size_t head = query_head - query_row * heads;
  if (query_row >= query_rows) {
    return;
  }
  const size_t split = kSplit ? blockIdx.y : 0;
  const size_t split_start = kSplit ? split * kSparseMlaKeysPerSplit : 0;
  const size_t valid = static_cast<size_t>(
      max(0, min(topk_lengths[query_row], static_cast<int32_t>(topk))));
  const size_t split_end =
      kSplit ? min(valid, split_start + kSparseMlaKeysPerSplit) : valid;
  const size_t tid = threadIdx.x;
  const size_t col0 = tid;
  const size_t col1 = tid + blockDim.x;
  const uint16_t* query_head_ptr =
      query + (query_row * heads + head) * (kRank + kRopeDim);
  const float query0 = bf16_to_f32(query_head_ptr[col0]);
  const float query1 = bf16_to_f32(query_head_ptr[col1]);

  __shared__ float warp_sums[8];
  __shared__ float score_shared;
  float running_max = -CUDART_INF_F;
  float running_denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (size_t key = split_start; key < split_end; ++key) {
    const int32_t physical_row = selected_indices[query_row * topk + key];
    const uint8_t* row_bytes =
        kv_payload + static_cast<size_t>(physical_row) * kv_row_stride_bytes;
    const uint16_t* latent = reinterpret_cast<const uint16_t*>(row_bytes);
    const float latent0 = bf16_to_f32(latent[col0]);
    const float latent1 = bf16_to_f32(latent[col1]);
    float partial = query0 * latent0 + query1 * latent1;
    if (tid < kRopeDim / 2) {
      const size_t rope_col = tid * 2;
      const uint16_t* rope =
          reinterpret_cast<const uint16_t*>(row_bytes + kRopeOffsetBytes);
      partial += bf16_to_f32(query_head_ptr[kRank + rope_col]) *
                 bf16_to_f32(rope[rope_col]);
      partial += bf16_to_f32(query_head_ptr[kRank + rope_col + 1]) *
                 bf16_to_f32(rope[rope_col + 1]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    const size_t lane = tid & 31;
    const size_t warp = tid >> 5;
    if (lane == 0) {
      warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < 8 ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
      }
      if (lane == 0) {
        score_shared = block_sum * scale;
      }
    }
    __syncthreads();
    const float score = score_shared;
    const float next_max = fmaxf(running_max, score);
    const float old_weight = expf(running_max - next_max);
    const float new_weight = expf(score - next_max);
    running_denom = running_denom * old_weight + new_weight;
    acc0 = acc0 * old_weight + latent0 * new_weight;
    acc1 = acc1 * old_weight + latent1 * new_weight;
    running_max = next_max;
    __syncthreads();
  }

  const size_t output_head =
      kSplit ? (query_head * kSparseMlaMaxSplits + split) : query_head;
  uint16_t* output_ptr = output + output_head * kRank;
  if (split_start < split_end) {
    const float inverse_denom = 1.0f / fmaxf(running_denom, 1.0e-12f);
    output_ptr[col0] = f32_to_bf16(acc0 * inverse_denom);
    output_ptr[col1] = f32_to_bf16(acc1 * inverse_denom);
    if (tid == 0) {
      output_lse[output_head] = running_max + logf(running_denom);
    }
  } else {
    output_ptr[col0] = 0;
    output_ptr[col1] = 0;
    if (tid == 0) {
      output_lse[output_head] = -CUDART_INF_F;
    }
  }
}

__global__ void sparse_mla_bf16_gather_kv_kernel(
    const uint8_t* kv_payload, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint16_t* gathered_k,
    uint16_t* gathered_v, size_t query_rows, size_t topk,
    size_t kv_row_stride_bytes) {
  constexpr size_t kRank = 512;
  constexpr size_t kRopeDim = 64;
  constexpr size_t kHeadDim = kRank + kRopeDim;
  const size_t selected_row = blockIdx.x;
  const size_t query_row = selected_row / topk;
  const size_t key = selected_row - query_row * topk;
  if (query_row >= query_rows) {
    return;
  }
  const int32_t valid =
      max(0, min(topk_lengths[query_row], static_cast<int32_t>(topk)));
  const int32_t physical_row =
      key < static_cast<size_t>(valid)
          ? selected_indices[query_row * topk + key]
          : -1;
  const uint16_t* source =
      physical_row >= 0
          ? reinterpret_cast<const uint16_t*>(
                kv_payload +
                static_cast<size_t>(physical_row) * kv_row_stride_bytes)
          : nullptr;
  uint16_t* key_output = gathered_k + selected_row * kHeadDim;
  uint16_t* value_output = gathered_v + selected_row * kRank;
  for (size_t col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    const uint16_t value = source == nullptr ? 0 : source[col];
    key_output[col] = value;
    if (col < kRank) {
      value_output[col] = value;
    }
  }
}

__device__ float sparse_mla_block_reduce_max(float value, float* warp_values) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  float block_value =
      threadIdx.x < static_cast<unsigned int>(blockDim.x / 32)
          ? warp_values[lane]
          : -CUDART_INF_F;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset /= 2) {
      block_value =
          fmaxf(block_value,
                __shfl_down_sync(0xffffffff, block_value, offset));
    }
    if (lane == 0) {
      warp_values[0] = block_value;
    }
  }
  __syncthreads();
  return warp_values[0];
}

__device__ float sparse_mla_block_reduce_sum(float value, float* warp_values) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  float block_value =
      threadIdx.x < static_cast<unsigned int>(blockDim.x / 32)
          ? warp_values[lane]
          : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset /= 2) {
      block_value +=
          __shfl_down_sync(0xffffffff, block_value, offset);
    }
    if (lane == 0) {
      warp_values[0] = block_value;
    }
  }
  __syncthreads();
  return warp_values[0];
}

__global__ void sparse_mla_bf16_softmax_kernel(
    uint16_t* scores, const int32_t* topk_lengths, float* output_lse,
    size_t query_rows, size_t heads, size_t topk, float scale) {
  const size_t query_head = blockIdx.x;
  const size_t query_row = query_head / heads;
  if (query_row >= query_rows) {
    return;
  }
  const size_t valid = static_cast<size_t>(
      max(0, min(topk_lengths[query_row], static_cast<int32_t>(topk))));
  uint16_t* row = scores + query_head * topk;
  __shared__ float warp_values[8];
  float local_max = -CUDART_INF_F;
  for (size_t key = threadIdx.x; key < valid; key += blockDim.x) {
    local_max = fmaxf(local_max, bf16_to_f32(row[key]) * scale);
  }
  const float row_max = sparse_mla_block_reduce_max(local_max, warp_values);
  float local_sum = 0.0f;
  if (valid > 0) {
    for (size_t key = threadIdx.x; key < valid; key += blockDim.x) {
      local_sum += expf(bf16_to_f32(row[key]) * scale - row_max);
    }
  }
  const float row_sum = sparse_mla_block_reduce_sum(local_sum, warp_values);
  const float inverse_sum = valid > 0 ? 1.0f / fmaxf(row_sum, 1.0e-12f) : 0.0f;
  for (size_t key = threadIdx.x; key < topk; key += blockDim.x) {
    const float weight =
        key < valid
            ? expf(bf16_to_f32(row[key]) * scale - row_max) * inverse_sum
            : 0.0f;
    row[key] = f32_to_bf16(weight);
  }
  if (threadIdx.x == 0) {
    output_lse[query_head] =
        valid > 0 ? row_max + logf(fmaxf(row_sum, 1.0e-12f))
                  : -CUDART_INF_F;
  }
}

__global__ void sparse_mla_nvfp4_merge_kernel(
    const uint16_t* partial, const float* partial_lse, uint16_t* output,
    float* output_lse, size_t query_rows, size_t heads) {
  const size_t query_head = blockIdx.x;
  if (query_head >= query_rows * heads) {
    return;
  }
  const size_t tid = threadIdx.x;
  const size_t col0 = tid;
  const size_t col1 = tid + blockDim.x;
  float running_max = -CUDART_INF_F;
  float running_denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (size_t split = 0; split < kSparseMlaMaxSplits; ++split) {
    const size_t partial_head = query_head * kSparseMlaMaxSplits + split;
    const float lse = partial_lse[partial_head];
    if (!isfinite(lse)) {
      continue;
    }
    const float next_max = fmaxf(running_max, lse);
    const float old_weight = expf(running_max - next_max);
    const float new_weight = expf(lse - next_max);
    const uint16_t* partial_ptr =
        partial + partial_head * kSparseMlaNvfp4Rank;
    acc0 = acc0 * old_weight + bf16_to_f32(partial_ptr[col0]) * new_weight;
    acc1 = acc1 * old_weight + bf16_to_f32(partial_ptr[col1]) * new_weight;
    running_denom = running_denom * old_weight + new_weight;
    running_max = next_max;
  }
  uint16_t* output_ptr = output + query_head * kSparseMlaNvfp4Rank;
  const float inverse_denom = 1.0f / fmaxf(running_denom, 1.0e-12f);
  output_ptr[col0] = f32_to_bf16(acc0 * inverse_denom);
  output_ptr[col1] = f32_to_bf16(acc1 * inverse_denom);
  if (tid == 0) {
    output_lse[query_head] = running_max + logf(running_denom);
  }
}

__global__ void mla_merge_state_bf16_kernel(
    uint16_t* accumulator, float* accumulator_lse, const uint16_t* partial,
    const float* partial_lse, size_t kv_lora_rank) {
  const size_t head = blockIdx.x;
  const size_t tid = threadIdx.x;
  const float lhs_lse = accumulator_lse[head];
  const float rhs_lse = partial_lse[head];
  const float merged_max = fmaxf(lhs_lse, rhs_lse);
  const float lhs_weight = expf(lhs_lse - merged_max);
  const float rhs_weight = expf(rhs_lse - merged_max);
  const float denominator = lhs_weight + rhs_weight;
  const size_t head_offset = head * kv_lora_rank;

  for (size_t col = tid; col < kv_lora_rank; col += blockDim.x) {
    const float lhs = bf16_to_f32(accumulator[head_offset + col]);
    const float rhs = bf16_to_f32(partial[head_offset + col]);
    accumulator[head_offset + col] =
        f32_to_bf16((lhs * lhs_weight + rhs * rhs_weight) / denominator);
  }
  if (tid == 0) {
    accumulator_lse[head] = merged_max + logf(denominator);
  }
}

glmrt_status_t validate_causal_attention_args(const float* q, const float* k, const float* v,
                                              const float* out, size_t rows, size_t heads,
                                              size_t qk_dim, size_t v_dim) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || qk_dim == 0 || v_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, qk_dim, &ignored) ||
      !checked_mul(row_heads, v_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_causal_attention_bf16_args(const uint16_t* q, const uint16_t* k,
                                                   const uint16_t* v, const uint16_t* out,
                                                   size_t rows, size_t heads, size_t qk_dim,
                                                   size_t v_dim) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || qk_dim == 0 || v_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, qk_dim, &ignored) ||
      !checked_mul(row_heads, v_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_rope_args(const float* input, const uint32_t* positions, const float* out,
                                  size_t rows, size_t heads, size_t rotary_dim, float theta) {
  if (input == nullptr || positions == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 ||
      theta <= 0.0f || !std::isfinite(theta)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, rotary_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_rope_bf16_args(const uint16_t* input, const uint32_t* positions,
                                       const uint16_t* out, size_t rows, size_t heads,
                                       size_t rotary_dim, float theta) {
  if (input == nullptr || positions == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 ||
      theta <= 0.0f || !std::isfinite(theta)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, rotary_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_rope_attention_bf16_args(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, const uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale) {
  if (q_nope == nullptr || q_rope == nullptr || k_nope == nullptr || k_rope == nullptr ||
      v == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || nope_dim == 0 || rope_dim == 0 || rope_dim % 2 != 0 ||
      v_dim == 0 || !std::isfinite(scale)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_heads = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, nope_dim, &ignored) ||
      !checked_mul(row_heads, rope_dim, &ignored) ||
      !checked_mul(rows, rope_dim, &ignored) ||
      !checked_mul(row_heads, v_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_rope_attention_bf16_suffix_args(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, const uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale) {
  const glmrt_status_t valid = validate_mla_rope_attention_bf16_args(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, heads, nope_dim, rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (query_rows == 0 || query_row_offset > rows || query_rows > rows - query_row_offset) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_compressed_attention_bf16_args(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope,
    const uint16_t* out_latent, size_t rows, size_t heads, size_t rope_dim,
    size_t kv_lora_rank, float scale) {
  if (q_absorbed == nullptr || q_rope == nullptr || kv_latent == nullptr ||
      k_rope == nullptr || out_latent == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || rope_dim == 0 || kv_lora_rank != 512 ||
      rope_dim > 64 || rope_dim % 2 != 0 || !std::isfinite(scale) ||
      scale <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, kv_lora_rank, &ignored) || !checked_mul(rows, rope_dim, &ignored) ||
      !checked_mul(heads, kv_lora_rank, &ignored) ||
      !checked_mul(heads, rope_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mla_compressed_attention_interleaved_bf16_args(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_payload, const uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, size_t rope_offset_bytes,
    size_t minimum_rope_offset_bytes, float scale) {
  if (q_absorbed == nullptr || q_rope == nullptr || kv_payload == nullptr ||
      out_latent == nullptr || kv_row_stride_bytes % sizeof(uint16_t) != 0 ||
      rope_offset_bytes % sizeof(uint16_t) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || heads == 0 || rope_dim == 0 || kv_lora_rank != 512 ||
      rope_dim > 64 || rope_dim % 2 != 0 ||
      rope_offset_bytes < minimum_rope_offset_bytes ||
      rope_offset_bytes > kv_row_stride_bytes ||
      rope_dim * sizeof(uint16_t) > kv_row_stride_bytes - rope_offset_bytes ||
      !std::isfinite(scale) || scale <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, kv_row_stride_bytes, &ignored) ||
      !checked_mul(heads, kv_lora_rank, &ignored) ||
      !checked_mul(heads, rope_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_causal_attention_buffers(
    glmrt_device_buffer_t q, glmrt_device_buffer_t k, glmrt_device_buffer_t v,
    glmrt_device_buffer_t out, size_t rows, size_t heads, size_t qk_dim, size_t v_dim) {
  const glmrt_status_t valid = validate_causal_attention_bf16_args(
      static_cast<const uint16_t*>(q.ptr), static_cast<const uint16_t*>(k.ptr),
      static_cast<const uint16_t*>(v.ptr), static_cast<const uint16_t*>(out.ptr), rows, heads,
      qk_dim, v_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  size_t row_heads = 0;
  size_t qk_values = 0;
  size_t v_values = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, qk_dim, &qk_values) ||
      !checked_mul(row_heads, v_dim, &v_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t qk_bytes = 0;
  size_t v_bytes = 0;
  if (!checked_mul(qk_values, sizeof(uint16_t), &qk_bytes) ||
      !checked_mul(v_values, sizeof(uint16_t), &v_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (q.bytes < qk_bytes || k.bytes < qk_bytes || v.bytes < v_bytes || out.bytes < v_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_rope_buffers(glmrt_device_buffer_t input,
                                                glmrt_device_buffer_t positions,
                                                glmrt_device_buffer_t out, size_t rows,
                                                size_t heads, size_t rotary_dim, float theta) {
  const glmrt_status_t valid = validate_rope_bf16_args(
      static_cast<const uint16_t*>(input.ptr), static_cast<const uint32_t*>(positions.ptr),
      static_cast<const uint16_t*>(out.ptr), rows, heads, rotary_dim, theta);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t row_heads = 0;
  size_t values = 0;
  if (!checked_mul(rows, heads, &row_heads) || !checked_mul(row_heads, rotary_dim, &values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t value_bytes = 0;
  size_t position_bytes = 0;
  if (!checked_mul(values, sizeof(uint16_t), &value_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &position_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (input.bytes < value_bytes || positions.bytes < position_bytes || out.bytes < value_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_mla_rope_attention_buffers(
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope, glmrt_device_buffer_t k_nope,
    glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v, glmrt_device_buffer_t out,
    size_t rows, size_t heads, size_t nope_dim, size_t rope_dim, size_t v_dim, float scale) {
  const glmrt_status_t valid = validate_mla_rope_attention_bf16_args(
      static_cast<const uint16_t*>(q_nope.ptr), static_cast<const uint16_t*>(q_rope.ptr),
      static_cast<const uint16_t*>(k_nope.ptr), static_cast<const uint16_t*>(k_rope.ptr),
      static_cast<const uint16_t*>(v.ptr), static_cast<const uint16_t*>(out.ptr), rows, heads,
      nope_dim, rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  size_t row_heads = 0;
  size_t nope_values = 0;
  size_t q_rope_values = 0;
  size_t k_rope_values = 0;
  size_t v_values = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(row_heads, nope_dim, &nope_values) ||
      !checked_mul(row_heads, rope_dim, &q_rope_values) ||
      !checked_mul(rows, rope_dim, &k_rope_values) ||
      !checked_mul(row_heads, v_dim, &v_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  size_t nope_bytes = 0;
  size_t q_rope_bytes = 0;
  size_t k_rope_bytes = 0;
  size_t v_bytes = 0;
  if (!checked_mul(nope_values, sizeof(uint16_t), &nope_bytes) ||
      !checked_mul(q_rope_values, sizeof(uint16_t), &q_rope_bytes) ||
      !checked_mul(k_rope_values, sizeof(uint16_t), &k_rope_bytes) ||
      !checked_mul(v_values, sizeof(uint16_t), &v_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (q_nope.bytes < nope_bytes || k_nope.bytes < nope_bytes ||
      q_rope.bytes < q_rope_bytes || k_rope.bytes < k_rope_bytes ||
      v.bytes < v_bytes || out.bytes < v_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_mla_rope_attention_suffix_buffers(
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope, glmrt_device_buffer_t k_nope,
    glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v, glmrt_device_buffer_t out,
    size_t rows, size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale) {
  const glmrt_status_t valid = validate_mla_rope_attention_bf16_suffix_args(
      static_cast<const uint16_t*>(q_nope.ptr), static_cast<const uint16_t*>(q_rope.ptr),
      static_cast<const uint16_t*>(k_nope.ptr), static_cast<const uint16_t*>(k_rope.ptr),
      static_cast<const uint16_t*>(v.ptr), static_cast<const uint16_t*>(out.ptr), rows,
      query_row_offset, query_rows, heads, nope_dim, rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  size_t row_heads = 0;
  size_t query_row_heads = 0;
  size_t nope_values = 0;
  size_t q_rope_values = 0;
  size_t k_rope_values = 0;
  size_t v_values = 0;
  size_t out_values = 0;
  if (!checked_mul(rows, heads, &row_heads) ||
      !checked_mul(query_rows, heads, &query_row_heads) ||
      !checked_mul(row_heads, nope_dim, &nope_values) ||
      !checked_mul(row_heads, rope_dim, &q_rope_values) ||
      !checked_mul(rows, rope_dim, &k_rope_values) ||
      !checked_mul(row_heads, v_dim, &v_values) ||
      !checked_mul(query_row_heads, v_dim, &out_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  size_t nope_bytes = 0;
  size_t q_rope_bytes = 0;
  size_t k_rope_bytes = 0;
  size_t v_bytes = 0;
  size_t out_bytes = 0;
  if (!checked_mul(nope_values, sizeof(uint16_t), &nope_bytes) ||
      !checked_mul(q_rope_values, sizeof(uint16_t), &q_rope_bytes) ||
      !checked_mul(k_rope_values, sizeof(uint16_t), &k_rope_bytes) ||
      !checked_mul(v_values, sizeof(uint16_t), &v_bytes) ||
      !checked_mul(out_values, sizeof(uint16_t), &out_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (q_nope.bytes < nope_bytes || k_nope.bytes < nope_bytes ||
      q_rope.bytes < q_rope_bytes || k_rope.bytes < k_rope_bytes ||
      v.bytes < v_bytes || out.bytes < out_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_causal_attention_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t q,
    glmrt_device_buffer_t k, glmrt_device_buffer_t v, glmrt_device_buffer_t out, size_t rows,
    size_t heads, size_t qk_dim, size_t v_dim, float scale) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_causal_attention_buffers(q, k, v, out, rows, heads, qk_dim, v_dim);
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
  if (existing.func != reinterpret_cast<void*>(causal_attention_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* q_ptr = static_cast<const uint16_t*>(q.ptr);
  const uint16_t* k_ptr = static_cast<const uint16_t*>(k.ptr);
  const uint16_t* v_ptr = static_cast<const uint16_t*>(v.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &q_ptr,
      &k_ptr,
      &v_ptr,
      &out_ptr,
      &rows,
      &heads,
      &qk_dim,
      &v_dim,
      &scale,
  };
  const int threads = 256;
  size_t row_heads = 0;
  size_t total = 0;
  if (!checked_mul(rows, heads, &row_heads) || !checked_mul(row_heads, v_dim, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(causal_attention_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_rope_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t input, glmrt_device_buffer_t positions, glmrt_device_buffer_t out,
    size_t rows, size_t heads, size_t rotary_dim, float theta) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_rope_buffers(input, positions, out, rows, heads, rotary_dim, theta);
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
  if (existing.func != reinterpret_cast<void*>(rope_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* input_ptr = static_cast<const uint16_t*>(input.ptr);
  const uint32_t* positions_ptr = static_cast<const uint32_t*>(positions.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &input_ptr,
      &positions_ptr,
      &out_ptr,
      &rows,
      &heads,
      &rotary_dim,
      &theta,
  };
  const int threads = 256;
  const size_t total = rows * heads * (rotary_dim / 2);
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(rope_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_mla_rope_attention_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope,
    glmrt_device_buffer_t k_nope, glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v,
    glmrt_device_buffer_t out, size_t rows, size_t heads, size_t nope_dim, size_t rope_dim,
    size_t v_dim, float scale) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_mla_rope_attention_buffers(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, heads, nope_dim, rope_dim, v_dim, scale);
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
  if (existing.func != reinterpret_cast<void*>(mla_rope_attention_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* q_nope_ptr = static_cast<const uint16_t*>(q_nope.ptr);
  const uint16_t* q_rope_ptr = static_cast<const uint16_t*>(q_rope.ptr);
  const uint16_t* k_nope_ptr = static_cast<const uint16_t*>(k_nope.ptr);
  const uint16_t* k_rope_ptr = static_cast<const uint16_t*>(k_rope.ptr);
  const uint16_t* v_ptr = static_cast<const uint16_t*>(v.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &q_nope_ptr,
      &q_rope_ptr,
      &k_nope_ptr,
      &k_rope_ptr,
      &v_ptr,
      &out_ptr,
      &rows,
      &heads,
      &nope_dim,
      &rope_dim,
      &v_dim,
      &scale,
  };
  const int threads = 256;
  size_t row_heads = 0;
  size_t total = 0;
  if (!checked_mul(rows, heads, &row_heads) || !checked_mul(row_heads, v_dim, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(mla_rope_attention_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_mla_rope_attention_bf16_suffix_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope,
    glmrt_device_buffer_t k_nope, glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v,
    glmrt_device_buffer_t out, size_t rows, size_t query_row_offset, size_t query_rows,
    size_t heads, size_t nope_dim, size_t rope_dim, size_t v_dim, float scale) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_mla_rope_attention_suffix_buffers(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, query_row_offset, query_rows, heads, nope_dim,
      rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (query_rows == 0 || query_row_offset > rows || query_rows > rows - query_row_offset) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
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
  if (existing.func != reinterpret_cast<void*>(mla_rope_attention_bf16_suffix_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* q_nope_ptr = static_cast<const uint16_t*>(q_nope.ptr);
  const uint16_t* q_rope_ptr = static_cast<const uint16_t*>(q_rope.ptr);
  const uint16_t* k_nope_ptr = static_cast<const uint16_t*>(k_nope.ptr);
  const uint16_t* k_rope_ptr = static_cast<const uint16_t*>(k_rope.ptr);
  const uint16_t* v_ptr = static_cast<const uint16_t*>(v.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &q_nope_ptr,
      &q_rope_ptr,
      &k_nope_ptr,
      &k_rope_ptr,
      &v_ptr,
      &out_ptr,
      &rows,
      &query_row_offset,
      &query_rows,
      &heads,
      &nope_dim,
      &rope_dim,
      &v_dim,
      &scale,
  };
  const int threads = 256;
  size_t query_row_heads = 0;
  size_t total = 0;
  if (!checked_mul(query_rows, heads, &query_row_heads) ||
      !checked_mul(query_row_heads, v_dim, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(mla_rope_attention_bf16_suffix_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_causal_attention_f32_async(
    const float* q, const float* k, const float* v, float* out, size_t rows, size_t heads,
    size_t qk_dim, size_t v_dim, float scale, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_causal_attention_args(q, k, v, out, rows, heads, qk_dim, v_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * v_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  causal_attention_f32_kernel<<<blocks, threads, 0, stream>>>(q, k, v, out, rows, heads, qk_dim,
                                                              v_dim, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_causal_attention_f32(const float* q, const float* k,
                                                          const float* v, float* out,
                                                          size_t rows, size_t heads,
                                                          size_t qk_dim, size_t v_dim,
                                                          float scale) {
  const glmrt_status_t status =
      glmrt_cuda_causal_attention_f32_async(q, k, v, out, rows, heads, qk_dim, v_dim, scale,
                                            nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_causal_attention_bf16_async(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, size_t rows,
    size_t heads, size_t qk_dim, size_t v_dim, float scale, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_causal_attention_bf16_args(q, k, v, out, rows, heads, qk_dim, v_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * v_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  causal_attention_bf16_kernel<<<blocks, threads, 0, stream>>>(q, k, v, out, rows, heads, qk_dim,
                                                               v_dim, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_causal_attention_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, size_t rows,
    size_t heads, size_t qk_dim, size_t v_dim, float scale) {
  const glmrt_status_t status =
      glmrt_cuda_causal_attention_bf16_async(q, k, v, out, rows, heads, qk_dim, v_dim, scale,
                                             nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_rope_f32_async(const float* input,
                                                    const uint32_t* positions, float* out,
                                                    size_t rows, size_t heads,
                                                    size_t rotary_dim, float theta,
                                                    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_rope_args(input, positions, out, rows, heads, rotary_dim, theta);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * (rotary_dim / 2);
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  rope_f32_kernel<<<blocks, threads, 0, stream>>>(input, positions, out, rows, heads, rotary_dim,
                                                  theta);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_rope_f32(const float* input, const uint32_t* positions,
                                              float* out, size_t rows, size_t heads,
                                              size_t rotary_dim, float theta) {
  const glmrt_status_t status =
      glmrt_cuda_rope_f32_async(input, positions, out, rows, heads, rotary_dim, theta, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_rope_bf16_async(const uint16_t* input,
                                                     const uint32_t* positions, uint16_t* out,
                                                     size_t rows, size_t heads,
                                                     size_t rotary_dim, float theta,
                                                     void* cuda_stream) {
  const glmrt_status_t valid =
      validate_rope_bf16_args(input, positions, out, rows, heads, rotary_dim, theta);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * (rotary_dim / 2);
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  rope_bf16_kernel<<<blocks, threads, 0, stream>>>(input, positions, out, rows, heads, rotary_dim,
                                                   theta);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_rope_bf16(const uint16_t* input,
                                               const uint32_t* positions, uint16_t* out,
                                               size_t rows, size_t heads, size_t rotary_dim,
                                               float theta) {
  const glmrt_status_t status =
      glmrt_cuda_rope_bf16_async(input, positions, out, rows, heads, rotary_dim, theta, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_async(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_rope_attention_bf16_args(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, heads, nope_dim, rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * heads * v_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  mla_rope_attention_bf16_kernel<<<blocks, threads, 0, stream>>>(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, heads, nope_dim, rope_dim, v_dim, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_suffix_async(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_rope_attention_bf16_suffix_args(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, query_row_offset, query_rows, heads,
      nope_dim, rope_dim, v_dim, scale);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  size_t query_row_heads = 0;
  size_t total = 0;
  if (!checked_mul(query_rows, heads, &query_row_heads) ||
      !checked_mul(query_row_heads, v_dim, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  mla_rope_attention_bf16_suffix_kernel<<<blocks, threads, 0, stream>>>(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, query_row_offset, query_rows, heads,
      nope_dim, rope_dim, v_dim, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_bf16_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope, uint16_t* out_latent,
    size_t rows, size_t heads, size_t rope_dim, size_t kv_lora_rank, float scale,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_compressed_attention_bf16_args(
      q_absorbed, q_rope, kv_latent, k_rope, out_latent, rows, heads,
      rope_dim, kv_lora_rank, scale);
  if (valid != GLMRT_STATUS_OK || heads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_compressed_attention_bf16_kernel<kMlaKvSplitBf16>
      <<<static_cast<int>(heads), 256, 0, stream>>>(
      q_absorbed, q_rope, kv_latent, k_rope, out_latent, rows, rope_dim,
      kv_lora_rank, kv_lora_rank * sizeof(uint16_t), 0, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_bf16(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope, uint16_t* out_latent,
    size_t rows, size_t heads, size_t rope_dim, size_t kv_lora_rank, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_compressed_attention_bf16_async(
      q_absorbed, q_rope, kv_latent, k_rope, out_latent, rows, heads,
      rope_dim, kv_lora_rank, scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_bf16_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, size_t rope_offset_bytes, float scale,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_compressed_attention_interleaved_bf16_args(
      q_absorbed, q_rope, kv_payload, out_latent, rows, heads, rope_dim,
      kv_lora_rank, kv_row_stride_bytes, rope_offset_bytes,
      kv_lora_rank * sizeof(uint16_t), scale);
  if (valid != GLMRT_STATUS_OK || heads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_compressed_attention_bf16_kernel<kMlaKvInterleavedBf16>
      <<<static_cast<int>(heads), 256, 0, stream>>>(
      q_absorbed, q_rope, kv_payload, nullptr, out_latent, rows, rope_dim,
      kv_lora_rank, kv_row_stride_bytes, rope_offset_bytes, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_bf16(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, size_t rope_offset_bytes, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_compressed_attention_interleaved_bf16_async(
      q_absorbed, q_rope, kv_payload, out_latent, rows, heads, rope_dim,
      kv_lora_rank, kv_row_stride_bytes, rope_offset_bytes, scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_fp8_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream) {
  constexpr size_t kFp8RopeOffsetBytes = 512 + 4 * sizeof(float);
  const glmrt_status_t valid = validate_mla_compressed_attention_interleaved_bf16_args(
      q_absorbed, q_rope, reinterpret_cast<const uint16_t*>(kv_payload), out_latent,
      rows, heads, rope_dim, kv_lora_rank, kv_row_stride_bytes,
      kFp8RopeOffsetBytes, kFp8RopeOffsetBytes, scale);
  if (valid != GLMRT_STATUS_OK || heads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_compressed_attention_bf16_kernel<kMlaKvInterleavedFp8>
      <<<static_cast<int>(heads), 256, 0, stream>>>(
          q_absorbed, q_rope, reinterpret_cast<const uint16_t*>(kv_payload), nullptr,
          out_latent, rows, rope_dim, kv_lora_rank, kv_row_stride_bytes,
          kFp8RopeOffsetBytes, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_fp8(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_compressed_attention_interleaved_fp8_async(
      q_absorbed, q_rope, kv_payload, out_latent, rows, heads, rope_dim,
      kv_lora_rank, kv_row_stride_bytes, scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_mxfp4_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream) {
  constexpr size_t kMxfp4RopeOffsetBytes = 512 / 2 + 512 / 16 + 16;
  const glmrt_status_t valid = validate_mla_compressed_attention_interleaved_bf16_args(
      q_absorbed, q_rope, reinterpret_cast<const uint16_t*>(kv_payload), out_latent,
      rows, heads, rope_dim, kv_lora_rank, kv_row_stride_bytes,
      kMxfp4RopeOffsetBytes, kMxfp4RopeOffsetBytes, scale);
  if (valid != GLMRT_STATUS_OK || heads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_compressed_attention_bf16_kernel<kMlaKvInterleavedMxfp4>
      <<<static_cast<int>(heads), 256, 0, stream>>>(
          q_absorbed, q_rope, reinterpret_cast<const uint16_t*>(kv_payload), nullptr,
          out_latent, rows, rope_dim, kv_lora_rank, kv_row_stride_bytes,
          kMxfp4RopeOffsetBytes, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_mxfp4(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_compressed_attention_interleaved_mxfp4_async(
      q_absorbed, q_rope, kv_payload, out_latent, rows, heads, rope_dim,
      kv_lora_rank, kv_row_stride_bytes, scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_sparse_mla_nvfp4_async(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* partial, float* partial_lse, uint16_t* output,
    float* output_lse, size_t query_rows, size_t heads, size_t topk,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream) {
  if (query == nullptr || kv_payload == nullptr || selected_indices == nullptr ||
      topk_lengths == nullptr || output == nullptr || output_lse == nullptr ||
      query_rows == 0 || heads != 64 || topk != kSparseMlaMaxTopk ||
      kv_row_stride_bytes < kSparseMlaNvfp4MinRowBytes ||
      !isfinite(scale) || scale <= 0.0f ||
      query_rows * heads >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int query_heads = static_cast<int>(query_rows * heads);
  if (query_rows <= kSparseMlaSplitQueryLimit) {
    if (partial == nullptr || partial_lse == nullptr) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    dim3 grid(query_heads, static_cast<unsigned int>(kSparseMlaMaxSplits));
    sparse_mla_nvfp4_attention_kernel<true><<<grid, 256, 0, stream>>>(
        query, kv_payload, selected_indices, topk_lengths, partial,
        partial_lse, query_rows, heads, topk, kv_row_stride_bytes, scale);
    glmrt_status_t status = status_from_cuda(cudaGetLastError());
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
    sparse_mla_nvfp4_merge_kernel<<<query_heads, 256, 0, stream>>>(
        partial, partial_lse, output, output_lse, query_rows, heads);
  } else {
    sparse_mla_nvfp4_attention_kernel<false><<<query_heads, 256, 0, stream>>>(
        query, kv_payload, selected_indices, topk_lengths, output, output_lse,
        query_rows, heads, topk, kv_row_stride_bytes, scale);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_sparse_mla_bf16_async(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* partial, float* partial_lse, uint16_t* output,
    float* output_lse, size_t query_rows, size_t heads, size_t topk,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream) {
  constexpr size_t kBf16RowBytes = (512 + 64) * sizeof(uint16_t);
  if (query == nullptr || kv_payload == nullptr || selected_indices == nullptr ||
      topk_lengths == nullptr || output == nullptr || output_lse == nullptr ||
      query_rows == 0 || heads != 64 || topk != kSparseMlaMaxTopk ||
      kv_row_stride_bytes < kBf16RowBytes ||
      !isfinite(scale) || scale <= 0.0f ||
      query_rows * heads >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int query_heads = static_cast<int>(query_rows * heads);
  if (query_rows <= kSparseMlaSplitQueryLimit) {
    if (partial == nullptr || partial_lse == nullptr) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    dim3 grid(query_heads, static_cast<unsigned int>(kSparseMlaMaxSplits));
    sparse_mla_bf16_attention_kernel<true><<<grid, 256, 0, stream>>>(
        query, kv_payload, selected_indices, topk_lengths, partial,
        partial_lse, query_rows, heads, topk, kv_row_stride_bytes, scale);
    glmrt_status_t status = status_from_cuda(cudaGetLastError());
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
    sparse_mla_nvfp4_merge_kernel<<<query_heads, 256, 0, stream>>>(
        partial, partial_lse, output, output_lse, query_rows, heads);
  } else {
    sparse_mla_bf16_attention_kernel<false><<<query_heads, 256, 0, stream>>>(
        query, kv_payload, selected_indices, topk_lengths, output, output_lse,
        query_rows, heads, topk, kv_row_stride_bytes, scale);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_sparse_mla_bf16_gather_kv_async(
    const uint8_t* kv_payload, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint16_t* gathered_k,
    uint16_t* gathered_v, size_t query_rows, size_t topk,
    size_t kv_row_stride_bytes, void* cuda_stream) {
  constexpr size_t kBf16RowBytes = (512 + 64) * sizeof(uint16_t);
  if (kv_payload == nullptr || selected_indices == nullptr ||
      topk_lengths == nullptr || gathered_k == nullptr ||
      gathered_v == nullptr || query_rows == 0 ||
      topk != kSparseMlaMaxTopk ||
      kv_row_stride_bytes < kBf16RowBytes ||
      query_rows * topk >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sparse_mla_bf16_gather_kv_kernel<<<
      static_cast<int>(query_rows * topk), 256, 0, stream>>>(
      kv_payload, selected_indices, topk_lengths, gathered_k, gathered_v,
      query_rows, topk, kv_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_sparse_mla_bf16_softmax_async(
    uint16_t* scores, const int32_t* topk_lengths, float* output_lse,
    size_t query_rows, size_t heads, size_t topk, float scale,
    void* cuda_stream) {
  if (scores == nullptr || topk_lengths == nullptr || output_lse == nullptr ||
      query_rows == 0 || heads != 64 || topk != kSparseMlaMaxTopk ||
      !isfinite(scale) || scale <= 0.0f ||
      query_rows * heads >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sparse_mla_bf16_softmax_kernel<<<
      static_cast<int>(query_rows * heads), 256, 0, stream>>>(
      scores, topk_lengths, output_lse, query_rows, heads, topk, scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_sparse_mla_nvfp4_gather_fp8_async(
    const uint8_t* nvfp4_kv, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint8_t* fp8_kv, int32_t* fp8_indices,
    size_t query_rows, size_t selected_index_stride, size_t staged_topk,
    size_t nvfp4_row_stride_bytes, void* cuda_stream) {
  const size_t rows = query_rows * staged_topk;
  if (nvfp4_kv == nullptr || selected_indices == nullptr ||
      topk_lengths == nullptr || fp8_kv == nullptr || fp8_indices == nullptr ||
      query_rows == 0 || query_rows > kSparseMlaSplitQueryLimit ||
      staged_topk == 0 || staged_topk > kSparseMlaMaxTopk ||
      staged_topk % kSparseMlaKeysPerSplit != 0 ||
      selected_index_stride < staged_topk ||
      selected_index_stride > kSparseMlaMaxTopk ||
      nvfp4_row_stride_bytes < kSparseMlaNvfp4MinRowBytes ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sparse_mla_nvfp4_gather_fp8_kernel<<<static_cast<int>(rows), 128, 0, stream>>>(
      nvfp4_kv, selected_indices, topk_lengths, fp8_kv, fp8_indices,
      query_rows, selected_index_stride, staged_topk,
      nvfp4_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_nvfp4_expand_fp8_paged_async(
    const uint8_t* nvfp4_kv, const uint32_t* physical_pages,
    const int32_t* active_rows, uint8_t* fp8_kv, size_t max_tokens,
    size_t page_size, size_t nvfp4_row_stride_bytes, void* cuda_stream) {
  if (nvfp4_kv == nullptr || physical_pages == nullptr ||
      active_rows == nullptr || fp8_kv == nullptr || max_tokens == 0 ||
      max_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      page_size == 0 || max_tokens % page_size != 0 ||
      nvfp4_row_stride_bytes < kSparseMlaNvfp4MinRowBytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr size_t kExpansionBlocks = 4096;
  const size_t blocks = min(max_tokens, kExpansionBlocks);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_nvfp4_expand_fp8_paged_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
      nvfp4_kv, physical_pages, active_rows, fp8_kv, max_tokens, page_size,
      nvfp4_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_merge_state_bf16_async(
    uint16_t* accumulator, float* accumulator_lse, const uint16_t* partial,
    const float* partial_lse, size_t heads, size_t kv_lora_rank, void* cuda_stream) {
  if (accumulator == nullptr || accumulator_lse == nullptr || partial == nullptr ||
      partial_lse == nullptr || heads == 0 || kv_lora_rank != 512 ||
      heads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  mla_merge_state_bf16_kernel<<<static_cast<int>(heads), 256, 0, stream>>>(
      accumulator, accumulator_lse, partial, partial_lse, kv_lora_rank);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_merge_state_bf16(
    uint16_t* accumulator, float* accumulator_lse, const uint16_t* partial,
    const float* partial_lse, size_t heads, size_t kv_lora_rank) {
  const glmrt_status_t status = glmrt_cuda_mla_merge_state_bf16_async(
      accumulator, accumulator_lse, partial, partial_lse, heads, kv_lora_rank, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_rope_attention_bf16(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_rope_attention_bf16_async(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, heads, nope_dim, rope_dim, v_dim, scale,
      nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_suffix(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale) {
  const glmrt_status_t status = glmrt_cuda_mla_rope_attention_bf16_suffix_async(
      q_nope, q_rope, k_nope, k_rope, v, out, rows, query_row_offset, query_rows, heads,
      nope_dim, rope_dim, v_dim, scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
