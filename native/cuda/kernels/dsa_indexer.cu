#include "common.h"

#include <cuda_fp8.h>

namespace {

constexpr size_t kDsaIndexHeads = 32;
constexpr size_t kDsaIndexHeadDim = 128;
constexpr size_t kDsaRopeDim = 64;
constexpr size_t kDsaRopePairs = kDsaRopeDim / 2;
constexpr size_t kDsaPageSize = 64;
constexpr size_t kDsaFp8RowBytes = kDsaIndexHeadDim;
constexpr size_t kDsaFp8PageDataBytes = kDsaPageSize * kDsaFp8RowBytes;
constexpr size_t kDsaFp8PageScaleBytes = kDsaPageSize * sizeof(float);
constexpr size_t kDsaFp8PageBytes = kDsaFp8PageDataBytes + kDsaFp8PageScaleBytes;
constexpr float kDsaFp8E4m3Max = 448.0f;
constexpr size_t kBf16ValuesPerVector = sizeof(uint4) / sizeof(uint16_t);
constexpr size_t kDsaMaxSortedTopK = 2048;

__device__ float dsa_rotated_value(const uint16_t* row, size_t output_col,
                                   const float* rope_cos, const float* rope_sin) {
  if (output_col >= kDsaRopeDim) {
    return bf16_to_f32(row[output_col]);
  }

  // GlmMoeDsa's interleaved RoPE pairs adjacent source values. Its reference
  // helper concatenates all rotated even values followed by all rotated odd
  // values, so preserve that exact output ordering for both Q and K.
  const bool odd_output = output_col >= kDsaRopePairs;
  const size_t pair = odd_output ? output_col - kDsaRopePairs : output_col;
  const float even = bf16_to_f32(row[pair * 2]);
  const float odd = bf16_to_f32(row[pair * 2 + 1]);
  return odd_output ? odd * rope_cos[pair] + even * rope_sin[pair]
                    : even * rope_cos[pair] - odd * rope_sin[pair];
}

__global__ void glm_dsa_index_k_pack_b12x_kernel(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bf16, float theta) {
  const size_t row_index = blockIdx.x;
  if (row_index >= rows) {
    return;
  }
  const size_t col = threadIdx.x;
  const uint32_t cache_slot = cache_slots[row_index];
  if (cache_slot >= cache_tokens) {
    return;
  }

  __shared__ float rope_cos[kDsaRopePairs];
  __shared__ float rope_sin[kDsaRopePairs];
  __shared__ float values[kDsaIndexHeadDim];
  __shared__ float maxima[kDsaIndexHeadDim];
  __shared__ float row_scale;

  if (col < kDsaRopePairs) {
    const float angle = static_cast<float>(positions[row_index]) *
                        powf(theta, -2.0f * static_cast<float>(col) /
                                        static_cast<float>(kDsaRopeDim));
    sincosf(angle, &rope_sin[col], &rope_cos[col]);
  }
  __syncthreads();

  const uint16_t* source = normalized_k + row_index * normalized_stride_bf16;
  const float value = dsa_rotated_value(source, col, rope_cos, rope_sin);
  values[col] = value;
  maxima[col] = fabsf(value);
  __syncthreads();

  for (size_t stride = kDsaIndexHeadDim / 2; stride > 0; stride >>= 1) {
    if (col < stride) {
      maxima[col] = fmaxf(maxima[col], maxima[col + stride]);
    }
    __syncthreads();
  }

  const size_t page = cache_slot / kDsaPageSize;
  const size_t page_slot = cache_slot % kDsaPageSize;
  uint8_t* page_base = index_k_cache + page * kDsaFp8PageBytes;
  if (col == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / kDsaFp8E4m3Max : 1.0f;
    reinterpret_cast<float*>(page_base + kDsaFp8PageDataBytes)[page_slot] = row_scale;
  }
  __syncthreads();

  page_base[page_slot * kDsaFp8RowBytes + col] =
      static_cast<uint8_t>(__nv_cvt_float_to_fp8(
          values[col] / row_scale, __NV_SATFINITE, __NV_E4M3));
}

__global__ void glm_dsa_query_prepare_b12x_kernel(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, uint8_t* query_fp8, float* adjusted_weights,
    size_t rows, size_t query_stride_bf16, size_t raw_weights_stride_bf16,
    size_t query_fp8_stride_bytes, size_t adjusted_weights_stride_f32,
    float theta, float score_scale) {
  const size_t row_index = blockIdx.x;
  if (row_index >= rows) {
    return;
  }

  constexpr size_t kWarpsPerBlock = 8;
  const size_t lane = threadIdx.x & 31;
  const size_t warp = threadIdx.x / 32;
  __shared__ float rope_cos[kDsaRopePairs];
  __shared__ float rope_sin[kDsaRopePairs];

  if (threadIdx.x < kDsaRopePairs) {
    const float angle = static_cast<float>(positions[row_index]) *
                        powf(theta, -2.0f * static_cast<float>(threadIdx.x) /
                                        static_cast<float>(kDsaRopeDim));
    sincosf(angle, &rope_sin[threadIdx.x], &rope_cos[threadIdx.x]);
  }
  __syncthreads();

  const uint16_t* query_row = query + row_index * query_stride_bf16;
  const uint16_t* weight_row =
      raw_weights + row_index * raw_weights_stride_bf16;
  uint8_t* query_output = query_fp8 + row_index * query_fp8_stride_bytes;
  float* weight_output =
      adjusted_weights + row_index * adjusted_weights_stride_f32;

  for (size_t head = warp; head < kDsaIndexHeads; head += kWarpsPerBlock) {
    const uint16_t* source = query_row + head * kDsaIndexHeadDim;
    float values[4];
    float maximum = 0.0f;
#pragma unroll
    for (size_t item = 0; item < 4; ++item) {
      const size_t col = lane + item * 32;
      values[item] = dsa_rotated_value(source, col, rope_cos, rope_sin);
      maximum = fmaxf(maximum, fabsf(values[item]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    const float scale = __shfl_sync(
        0xffffffffu, maximum > 0.0f ? maximum / kDsaFp8E4m3Max : 1.0f, 0);
    uint8_t* head_output = query_output + head * kDsaIndexHeadDim;
#pragma unroll
    for (size_t item = 0; item < 4; ++item) {
      const size_t col = lane + item * 32;
      head_output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
          values[item] / scale, __NV_SATFINITE, __NV_E4M3));
    }
    if (lane == 0) {
      weight_output[head] =
          bf16_to_f32(weight_row[head]) * scale * score_scale;
    }
  }
}

__global__ void transpose_rows_heads_bf16_kernel(
    const uint4* input, uint4* output, size_t rows, size_t heads,
    size_t width_vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * width_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % width_vectors;
  const size_t row_head = index / width_vectors;
  const size_t head = row_head % heads;
  const size_t row = row_head / heads;
  output[(head * rows + row) * width_vectors + vector_col] = input[index];
}

__global__ void transpose_heads_rows_bf16_kernel(
    const uint4* input, uint4* output, size_t rows, size_t heads,
    size_t width_vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * width_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % width_vectors;
  const size_t head_row = index / width_vectors;
  const size_t row = head_row % rows;
  const size_t head = head_row / rows;
  output[(row * heads + head) * width_vectors + vector_col] = input[index];
}

__global__ void mla_compose_absorbed_query_bf16_kernel(
    const uint4* latent_heads_rows, const uint4* rope_rows_heads,
    uint4* output_rows_heads, size_t rows, size_t heads,
    size_t latent_vectors, size_t rope_vectors) {
  const size_t output_vectors = latent_vectors + rope_vectors;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * output_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % output_vectors;
  const size_t row_head = index / output_vectors;
  const size_t head = row_head % heads;
  const size_t row = row_head / heads;
  if (vector_col < latent_vectors) {
    output_rows_heads[index] =
        latent_heads_rows[(head * rows + row) * latent_vectors + vector_col];
  } else {
    output_rows_heads[index] =
        rope_rows_heads[(row * heads + head) * rope_vectors +
                        vector_col - latent_vectors];
  }
}

__global__ void glm_dsa_page_table_init_kernel(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    int32_t base_offset) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * page_table_width;
  if (index < total) {
    page_table[index] =
        base_offset + static_cast<int32_t>(index % page_table_width);
  }
}

__global__ void glm_dsa_page_table_init_offsets_kernel(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * page_table_width;
  if (index < total) {
    const size_t row = index / page_table_width;
    const size_t column = index % page_table_width;
    page_table[index] = row_offsets[row] + static_cast<int32_t>(column);
  }
}

__global__ void target_kv_page_table_expand_indices_kernel(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * output_width;
  if (index >= total) {
    return;
  }
  const size_t logical_token = index % output_width;
  if (logical_token >= active_tokens) {
    output_indices[index] = 0;
    return;
  }
  const size_t page_index = logical_token / kDsaPageSize;
  const size_t page_token = logical_token % kDsaPageSize;
  output_indices[index] = static_cast<int32_t>(
      static_cast<size_t>(physical_pages[page_index]) * kDsaPageSize +
      page_token);
}

__global__ void glm_dsa_prefill_metadata_kernel(
    int32_t* cache_seqlens, int32_t* topk_lengths, int32_t* active_width,
    size_t bucket_rows, size_t active_rows, size_t prefix_rows,
    size_t total_rows, size_t topk) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < bucket_rows) {
    const size_t length = row < active_rows ? prefix_rows + row + 1 : 1;
    cache_seqlens[row] = static_cast<int32_t>(length);
    topk_lengths[row] = static_cast<int32_t>(min(length, topk));
  }
  if (row == 0) {
    active_width[0] = static_cast<int32_t>(total_rows);
  }
}

__global__ void glm_dsa_sort_selected_indices_kernel(
    int32_t* selected_indices, size_t rows, size_t width) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ int32_t values[];
  int32_t* input = selected_indices + row * width;
  for (size_t index = threadIdx.x; index < width; index += blockDim.x) {
    values[index] = input[index];
  }
  __syncthreads();

  // The selector's top-k set is mathematically unordered, but sparse MLA
  // reduces in the supplied order. Canonical physical-slot order makes q=1
  // and grouped q=2..8 consume the same sequence without changing membership.
  for (size_t span = 2; span <= width; span <<= 1) {
    for (size_t stride = span >> 1; stride > 0; stride >>= 1) {
      for (size_t index = threadIdx.x; index < width;
           index += blockDim.x) {
        const size_t peer = index ^ stride;
        if (peer > index) {
          const bool ascending = (index & span) == 0;
          const int32_t lhs = values[index];
          const int32_t rhs = values[peer];
          if ((lhs > rhs) == ascending) {
            values[index] = rhs;
            values[peer] = lhs;
          }
        }
      }
      __syncthreads();
    }
  }
  for (size_t index = threadIdx.x; index < width; index += blockDim.x) {
    input[index] = values[index];
  }
}

glmrt_status_t validate_transpose_rows_heads_bf16_args(
    const uint16_t* input, const uint16_t* output, size_t rows,
    size_t heads, size_t width) {
  if (input == nullptr || output == nullptr || rows == 0 || heads == 0 ||
      width == 0 || width % kBf16ValuesPerVector != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint4) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  return checked_mul(rows, heads, &values) && checked_mul(values, width, &values)
             ? GLMRT_STATUS_OK
             : GLMRT_STATUS_INVALID_ARGUMENT;
}

glmrt_status_t validate_mla_compose_absorbed_query_bf16_args(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    const uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width) {
  if (latent_heads_rows == nullptr || rope_rows_heads == nullptr ||
      output_rows_heads == nullptr || rows == 0 || heads == 0 ||
      latent_width == 0 || rope_width == 0 ||
      latent_width % kBf16ValuesPerVector != 0 ||
      rope_width % kBf16ValuesPerVector != 0 ||
      reinterpret_cast<uintptr_t>(latent_heads_rows) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(rope_rows_heads) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(output_rows_heads) % alignof(uint4) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  size_t output_width = 0;
  return checked_add(latent_width, rope_width, &output_width) &&
                 checked_mul(rows, heads, &values) &&
                 checked_mul(values, output_width, &values)
             ? GLMRT_STATUS_OK
             : GLMRT_STATUS_INVALID_ARGUMENT;
}

glmrt_status_t validate_glm_dsa_page_table_init_args(
    const int32_t* page_table, size_t query_rows, size_t page_table_width) {
  size_t entries = 0;
  if (page_table == nullptr || query_rows == 0 || page_table_width == 0 ||
      page_table_width > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      !checked_mul(query_rows, page_table_width, &entries)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_glm_dsa_page_table_base_offset(
    size_t page_table_width, size_t base_offset) {
  size_t end = 0;
  if (!checked_add(base_offset, page_table_width, &end) ||
      end > static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_target_kv_page_table_expand_indices_args(
    const int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  size_t entries = 0;
  if (output_indices == nullptr || physical_pages == nullptr ||
      query_rows == 0 || output_width == 0 || active_tokens == 0 ||
      active_tokens > output_width ||
      !checked_mul(query_rows, output_width, &entries) ||
      active_tokens >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_glm_dsa_prefill_metadata_args(
    const int32_t* cache_seqlens, const int32_t* topk_lengths,
    const int32_t* active_width, size_t bucket_rows, size_t active_rows,
    size_t prefix_rows, size_t total_rows, size_t topk) {
  size_t expected_total = 0;
  if (cache_seqlens == nullptr || topk_lengths == nullptr ||
      active_width == nullptr || bucket_rows == 0 || active_rows == 0 ||
      active_rows > bucket_rows || topk == 0 ||
      !checked_add(prefix_rows, active_rows, &expected_total) ||
      expected_total != total_rows ||
      total_rows > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      topk > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_glm_dsa_sort_selected_indices_args(
    const int32_t* selected_indices, size_t rows, size_t width) {
  size_t entries = 0;
  if (selected_indices == nullptr || rows == 0 || width == 0 ||
      width > kDsaMaxSortedTopK || (width & (width - 1)) != 0 ||
      !checked_mul(rows, width, &entries)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_glm_dsa_index_k_pack_b12x_args(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, const uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bytes, float theta) {
  const size_t minimum_stride = kDsaIndexHeadDim * sizeof(uint16_t);
  if (normalized_k == nullptr || positions == nullptr || cache_slots == nullptr ||
      index_k_cache == nullptr || rows == 0 || cache_tokens == 0 ||
      normalized_stride_bytes < minimum_stride ||
      normalized_stride_bytes % sizeof(uint16_t) != 0 ||
      !std::isfinite(theta) || theta <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t pages = 0;
  size_t ignored = 0;
  if (!checked_add(cache_tokens, kDsaPageSize - 1, &pages)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  pages /= kDsaPageSize;
  if (!checked_mul(rows, normalized_stride_bytes, &ignored) ||
      !checked_mul(pages, kDsaFp8PageBytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_glm_dsa_query_prepare_b12x_args(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, const uint8_t* query_fp8,
    const float* adjusted_weights, size_t rows, size_t query_stride_bytes,
    size_t raw_weights_stride_bytes, size_t query_fp8_stride_bytes,
    size_t adjusted_weights_stride_bytes, float theta, float score_scale) {
  constexpr size_t query_values = kDsaIndexHeads * kDsaIndexHeadDim;
  if (query == nullptr || raw_weights == nullptr || positions == nullptr ||
      query_fp8 == nullptr || adjusted_weights == nullptr || rows == 0 ||
      query_stride_bytes < query_values * sizeof(uint16_t) ||
      raw_weights_stride_bytes < kDsaIndexHeads * sizeof(uint16_t) ||
      query_fp8_stride_bytes < query_values ||
      adjusted_weights_stride_bytes < kDsaIndexHeads * sizeof(float) ||
      query_stride_bytes % sizeof(uint16_t) != 0 ||
      raw_weights_stride_bytes % sizeof(uint16_t) != 0 ||
      adjusted_weights_stride_bytes % sizeof(float) != 0 ||
      !std::isfinite(theta) || theta <= 0.0f ||
      !std::isfinite(score_scale)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, query_stride_bytes, &ignored) ||
      !checked_mul(rows, raw_weights_stride_bytes, &ignored) ||
      !checked_mul(rows, query_fp8_stride_bytes, &ignored) ||
      !checked_mul(rows, adjusted_weights_stride_bytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_index_k_pack_b12x_async(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bytes, float theta,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_index_k_pack_b12x_args(
      normalized_k, positions, cache_slots, index_k_cache, rows, cache_tokens,
      normalized_stride_bytes, theta);
  if (valid != GLMRT_STATUS_OK ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_index_k_pack_b12x_kernel<<<static_cast<int>(rows), 128, 0, stream>>>(
      normalized_k, positions, cache_slots, index_k_cache, rows, cache_tokens,
      normalized_stride_bytes / sizeof(uint16_t), theta);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_index_k_pack_b12x(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bytes, float theta) {
  const glmrt_status_t status = glmrt_cuda_glm_dsa_index_k_pack_b12x_async(
      normalized_k, positions, cache_slots, index_k_cache, rows, cache_tokens,
      normalized_stride_bytes, theta, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_query_prepare_b12x_async(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, uint8_t* query_fp8, float* adjusted_weights,
    size_t rows, size_t query_stride_bytes, size_t raw_weights_stride_bytes,
    size_t query_fp8_stride_bytes, size_t adjusted_weights_stride_bytes,
    float theta, float score_scale, void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_query_prepare_b12x_args(
      query, raw_weights, positions, query_fp8, adjusted_weights, rows,
      query_stride_bytes, raw_weights_stride_bytes, query_fp8_stride_bytes,
      adjusted_weights_stride_bytes, theta, score_scale);
  if (valid != GLMRT_STATUS_OK ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_query_prepare_b12x_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
      query, raw_weights, positions, query_fp8, adjusted_weights, rows,
      query_stride_bytes / sizeof(uint16_t),
      raw_weights_stride_bytes / sizeof(uint16_t), query_fp8_stride_bytes,
      adjusted_weights_stride_bytes / sizeof(float), theta, score_scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_query_prepare_b12x(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, uint8_t* query_fp8, float* adjusted_weights,
    size_t rows, size_t query_stride_bytes, size_t raw_weights_stride_bytes,
    size_t query_fp8_stride_bytes, size_t adjusted_weights_stride_bytes,
    float theta, float score_scale) {
  const glmrt_status_t status = glmrt_cuda_glm_dsa_query_prepare_b12x_async(
      query, raw_weights, positions, query_fp8, adjusted_weights, rows,
      query_stride_bytes, raw_weights_stride_bytes, query_fp8_stride_bytes,
      adjusted_weights_stride_bytes, theta, score_scale, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_transpose_rows_heads_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream) {
  const glmrt_status_t valid = validate_transpose_rows_heads_bf16_args(
      input, output, rows, heads, width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t total_vectors = rows * heads * (width / kBf16ValuesPerVector);
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  transpose_rows_heads_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(input), reinterpret_cast<uint4*>(output),
      rows, heads, width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_transpose_rows_heads_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width) {
  const glmrt_status_t status = glmrt_cuda_transpose_rows_heads_bf16_async(
      input, output, rows, heads, width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_transpose_heads_rows_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream) {
  const glmrt_status_t valid = validate_transpose_rows_heads_bf16_args(
      input, output, rows, heads, width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t total_vectors = rows * heads * (width / kBf16ValuesPerVector);
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  transpose_heads_rows_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(input), reinterpret_cast<uint4*>(output),
      rows, heads, width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_transpose_heads_rows_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width) {
  const glmrt_status_t status = glmrt_cuda_transpose_heads_rows_bf16_async(
      input, output, rows, heads, width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_mla_compose_absorbed_query_bf16_async(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width, void* cuda_stream) {
  const glmrt_status_t valid = validate_mla_compose_absorbed_query_bf16_args(
      latent_heads_rows, rope_rows_heads, output_rows_heads, rows, heads,
      latent_width, rope_width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t output_vectors =
      (latent_width + rope_width) / kBf16ValuesPerVector;
  const size_t total_vectors = rows * heads * output_vectors;
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_compose_absorbed_query_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(latent_heads_rows),
      reinterpret_cast<const uint4*>(rope_rows_heads),
      reinterpret_cast<uint4*>(output_rows_heads), rows, heads,
      latent_width / kBf16ValuesPerVector,
      rope_width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_compose_absorbed_query_bf16(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width) {
  const glmrt_status_t status = glmrt_cuda_mla_compose_absorbed_query_bf16_async(
      latent_heads_rows, rope_rows_heads, output_rows_heads, rows, heads,
      latent_width, rope_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    void* cuda_stream) {
  return glmrt_cuda_glm_dsa_page_table_init_base_async(
      page_table, query_rows, page_table_width, 0, cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_base_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset, void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_page_table_init_args(
      page_table, query_rows, page_table_width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t valid_base =
      validate_glm_dsa_page_table_base_offset(page_table_width, base_offset);
  if (valid_base != GLMRT_STATUS_OK) {
    return valid_base;
  }
  const size_t entries = query_rows * page_table_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_page_table_init_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      page_table, query_rows, page_table_width,
      static_cast<int32_t>(base_offset));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init(
    int32_t* page_table, size_t query_rows, size_t page_table_width) {
  const glmrt_status_t status = glmrt_cuda_glm_dsa_page_table_init_async(
      page_table, query_rows, page_table_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_base(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset) {
  const glmrt_status_t status = glmrt_cuda_glm_dsa_page_table_init_base_async(
      page_table, query_rows, page_table_width, base_offset, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_offsets_async(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width, void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_page_table_init_args(
      page_table, query_rows, page_table_width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (row_offsets == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t entries = query_rows * page_table_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_page_table_init_offsets_kernel<<<static_cast<int>(blocks), threads, 0,
                                            stream>>>(
      page_table, row_offsets, query_rows, page_table_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_offsets(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width) {
  const glmrt_status_t status =
      glmrt_cuda_glm_dsa_page_table_init_offsets_async(
          page_table, row_offsets, query_rows, page_table_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t
glmrt_cuda_target_kv_page_table_expand_indices_async(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens,
    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_target_kv_page_table_expand_indices_args(
          output_indices, physical_pages, query_rows, output_width,
          active_tokens);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t entries = query_rows * output_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  target_kv_page_table_expand_indices_kernel<<<
      static_cast<int>(blocks), threads, 0, stream>>>(
      output_indices, physical_pages, query_rows, output_width, active_tokens);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_target_kv_page_table_expand_indices(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  const glmrt_status_t status =
      glmrt_cuda_target_kv_page_table_expand_indices_async(
          output_indices, physical_pages, query_rows, output_width,
          active_tokens, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_prefill_metadata_async(
    int32_t* cache_seqlens, int32_t* topk_lengths, int32_t* active_width,
    size_t bucket_rows, size_t active_rows, size_t prefix_rows,
    size_t total_rows, size_t topk, void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_prefill_metadata_args(
      cache_seqlens, topk_lengths, active_width, bucket_rows, active_rows,
      prefix_rows, total_rows, topk);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  constexpr int threads = 256;
  const size_t blocks = (bucket_rows + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_prefill_metadata_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      cache_seqlens, topk_lengths, active_width, bucket_rows, active_rows,
      prefix_rows, total_rows, topk);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_prefill_metadata(
    int32_t* cache_seqlens, int32_t* topk_lengths, int32_t* active_width,
    size_t bucket_rows, size_t active_rows, size_t prefix_rows,
    size_t total_rows, size_t topk) {
  const glmrt_status_t status = glmrt_cuda_glm_dsa_prefill_metadata_async(
      cache_seqlens, topk_lengths, active_width, bucket_rows, active_rows,
      prefix_rows, total_rows, topk, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_glm_dsa_sort_selected_indices_async(
    int32_t* selected_indices, size_t rows, size_t width,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_glm_dsa_sort_selected_indices_args(
      selected_indices, rows, width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  constexpr int threads = 1024;
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  glm_dsa_sort_selected_indices_kernel<<<static_cast<int>(rows), threads,
                                         width * sizeof(int32_t), stream>>>(
      selected_indices, rows, width);
  return status_from_cuda(cudaGetLastError());
}
