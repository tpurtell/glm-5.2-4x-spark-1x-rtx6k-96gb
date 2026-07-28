#include "common.h"

#include <cuda_fp8.h>

namespace {

__global__ void residual_add_f32_kernel(const float* residual, const float* delta, float* out,
                                        size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  out[idx] = residual[idx] + delta[idx];
}

__global__ void residual_add_bf16_kernel(const uint16_t* residual, const uint16_t* delta,
                                         uint16_t* out, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  out[idx] = f32_to_bf16(bf16_to_f32(residual[idx]) + bf16_to_f32(delta[idx]));
}

__global__ void residual_add_f32_delta_bf16_kernel(const uint16_t* residual,
                                                   const float* delta_f32, uint16_t* out,
                                                   size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float delta = bf16_to_f32(f32_to_bf16(delta_f32[idx]));
  out[idx] = f32_to_bf16(bf16_to_f32(residual[idx]) + delta);
}

__global__ void residual_add_shared_f32_delta_bf16_kernel(
    const uint16_t* residual, const uint16_t* shared_delta, const float* routed_delta_f32,
    uint16_t* out, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float routed_delta = bf16_to_f32(f32_to_bf16(routed_delta_f32[idx]));
  const float mlp_delta =
      bf16_to_f32(f32_to_bf16(bf16_to_f32(shared_delta[idx]) + routed_delta));
  out[idx] = f32_to_bf16(bf16_to_f32(residual[idx]) + mlp_delta);
}

__global__ void residual_add_shared_fp8_e4m3_row_scaled_bf16_kernel(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* routed_delta_fp8,
    uint16_t* out, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float scale = *reinterpret_cast<const float*>(routed_delta_fp8 + count);
  const float routed_delta = bf16_to_f32(
      f32_to_bf16(f8e4m3_to_f32(routed_delta_fp8[idx]) * scale));
  const float mlp_delta =
      bf16_to_f32(f32_to_bf16(bf16_to_f32(shared_delta[idx]) + routed_delta));
  out[idx] = f32_to_bf16(bf16_to_f32(residual[idx]) + mlp_delta);
}

__global__ void scheduler_mlp_delta_bf16_kernel(const uint16_t* hidden,
                                                const uint16_t* gate_weight,
                                                const uint16_t* up_weight,
                                                const uint16_t* down_weight, uint16_t* out,
                                                size_t rows, size_t hidden_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden_dim;
  if (idx >= total) {
    return;
  }
  const size_t col = idx % hidden_dim;
  const float value = bf16_to_f32(hidden[idx]);
  const float gate = value * bf16_to_f32(gate_weight[col]);
  const float up = value * bf16_to_f32(up_weight[col]);
  const float silu = gate / (1.0f + expf(-gate));
  out[idx] = f32_to_bf16(silu * up * bf16_to_f32(down_weight[col]));
}

__global__ void summarize_bf16_kernel(const uint16_t* input, size_t count,
                                      glmrt_bf16_summary_t* out) {
  __shared__ double sums[kBlock];
  __shared__ unsigned int finite_counts[kBlock];
  __shared__ unsigned int nonzero_counts[kBlock];

  const int lane = threadIdx.x;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  unsigned int finite = 0;
  unsigned int nonzero = 0;
  if (idx < count) {
    const float value = bf16_to_f32(input[idx]);
    sum = static_cast<double>(value);
    finite = isfinite(value) ? 1U : 0U;
    nonzero = value != 0.0f ? 1U : 0U;
  }
  sums[lane] = sum;
  finite_counts[lane] = finite;
  nonzero_counts[lane] = nonzero;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      finite_counts[lane] += finite_counts[lane + stride];
      nonzero_counts[lane] += nonzero_counts[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    if (blockIdx.x == 0) {
      out->values = static_cast<uint64_t>(count);
    }
    atomicAdd(&out->checksum, sums[0]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->finite_values),
              static_cast<unsigned long long>(finite_counts[0]));
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->nonzero_values),
              static_cast<unsigned long long>(nonzero_counts[0]));
  }
}

__global__ void f32_to_bf16_kernel(const float* src, uint16_t* dst, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  dst[idx] = f32_to_bf16(src[idx]);
}

__global__ void gather_rows_f32_kernel(const float* src, const uint32_t* row_indices, float* dst,
                                       size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t source_row = static_cast<size_t>(row_indices[compact_row]);
  dst[idx] = src[source_row * row_width + col];
}

__global__ void gather_rows_f32_to_bf16_candidate_kernel(
    const float* src, const uint32_t* row_indices, uint16_t* dst, size_t rows,
    size_t row_width) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t source_row = static_cast<size_t>(row_indices[compact_row]);
  dst[idx] = f32_to_bf16(src[source_row * row_width + col]);
}

__global__ void gather_rows_f32_to_fp8_e4m3_row_scaled_kernel(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t compact_row = blockIdx.x;
  if (compact_row >= rows) {
    return;
  }
  const size_t source_row = static_cast<size_t>(row_indices[compact_row]);
  const float* source = src + source_row * row_width;
  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    const float value = source[col];
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(dst + compact_row * dst_row_stride_bytes + row_width) = row_scale;
  }
  __syncthreads();
  uint8_t* output = dst + compact_row * dst_row_stride_bytes;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    const float value = source[col];
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void gather_rows_f32_to_fp8_e4m3_row_scaled_register_kernel(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t dst_row_stride_bytes) {
  constexpr int row_width = 6144;
  constexpr int values_per_thread = row_width / kBlock;
  __shared__ float warp_maxima[kBlock / 32];
  __shared__ float row_scale;
  const size_t compact_row = blockIdx.x;
  if (compact_row >= rows) {
    return;
  }
  const size_t source_row = static_cast<size_t>(row_indices[compact_row]);
  const float* source = src + source_row * row_width;
  float values[values_per_thread];
  float maximum = 0.0f;
#pragma unroll
  for (int item = 0; item < values_per_thread; ++item) {
    const float value = source[threadIdx.x + item * kBlock];
    values[item] = value;
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }

  constexpr unsigned int mask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_maxima[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kBlock / 32 ? warp_maxima[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
    }
    if (lane == 0) {
      row_scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
      *reinterpret_cast<float*>(dst + compact_row * dst_row_stride_bytes + row_width) =
          row_scale;
    }
  }
  __syncthreads();
  uint8_t* output = dst + compact_row * dst_row_stride_bytes;
#pragma unroll
  for (int item = 0; item < values_per_thread; ++item) {
    const size_t col = threadIdx.x + item * kBlock;
    const float value = values[item];
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void bf16_rows_to_fp8_e4m3_row_scaled_kernel(
    const uint16_t* src, uint8_t* dst, size_t rows, size_t row_width,
    size_t dst_row_stride_bytes) {
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint16_t* source = src + row * row_width;
  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    const float value = bf16_to_f32(source[col]);
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(dst + row * dst_row_stride_bytes + row_width) = row_scale;
  }
  __syncthreads();
  uint8_t* output = dst + row * dst_row_stride_bytes;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    const float value = bf16_to_f32(source[col]);
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void combine_fp8_e4m3_row_scaled_to_fp8_kernel(
    const float* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    float value = local[row * row_width + col];
    for (size_t peer = 0; peer < peer_count; ++peer) {
      const uint8_t* source = peers + peer * peer_payload_stride_bytes +
                              row * peer_row_stride_bytes;
      const float scale = *reinterpret_cast<const float*>(source + row_width);
      value += f8e4m3_to_f32(source[col]) * scale;
    }
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(dst + row * dst_row_stride_bytes + row_width) = row_scale;
  }
  __syncthreads();
  uint8_t* output = dst + row * dst_row_stride_bytes;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    float value = local[row * row_width + col];
    for (size_t peer = 0; peer < peer_count; ++peer) {
      const uint8_t* source = peers + peer * peer_payload_stride_bytes +
                              row * peer_row_stride_bytes;
      const float scale = *reinterpret_cast<const float*>(source + row_width);
      value += f8e4m3_to_f32(source[col]) * scale;
    }
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void combine_bf16_fp8_e4m3_row_scaled_to_fp8_kernel(
    const uint16_t* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    float value = bf16_to_f32(local[row * row_width + col]);
    for (size_t peer = 0; peer < peer_count; ++peer) {
      const uint8_t* source = peers + peer * peer_payload_stride_bytes +
                              row * peer_row_stride_bytes;
      const float scale = *reinterpret_cast<const float*>(source + row_width);
      value += f8e4m3_to_f32(source[col]) * scale;
    }
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(dst + row * dst_row_stride_bytes + row_width) = row_scale;
  }
  __syncthreads();
  uint8_t* output = dst + row * dst_row_stride_bytes;
  for (size_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    float value = bf16_to_f32(local[row * row_width + col]);
    for (size_t peer = 0; peer < peer_count; ++peer) {
      const uint8_t* source = peers + peer * peer_payload_stride_bytes +
                              row * peer_row_stride_bytes;
      const float scale = *reinterpret_cast<const float*>(source + row_width);
      value += f8e4m3_to_f32(source[col]) * scale;
    }
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void gather_rows_bf16_kernel(const uint16_t* src, const uint32_t* row_indices,
                                        uint16_t* dst, size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t source_row = static_cast<size_t>(row_indices[compact_row]);
  dst[idx] = src[source_row * row_width + col];
}

__global__ void copy_row_prefix_bf16_kernel(const uint16_t* src, uint16_t* dst, size_t rows,
                                            size_t src_row_width, size_t dst_row_width,
                                            size_t prefix_width, size_t src_row_offset) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * prefix_width;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / prefix_width;
  const size_t col = idx % prefix_width;
  dst[row * dst_row_width + col] = src[(src_row_offset + row) * src_row_width + col];
}

__global__ void scatter_add_rows_f32_kernel(const float* src, const uint32_t* row_indices,
                                            float* dst, size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
  atomicAdd(&dst[dest_row * row_width + col], src[idx]);
}

constexpr size_t kOrderedScatterMaxRows = 64;

__global__ void scatter_add_rows_bf16_to_f32_kernel(const uint16_t* src,
                                                    const uint32_t* row_indices, float* dst,
                                                    size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
  atomicAdd(&dst[dest_row * row_width + col], bf16_to_f32(src[idx]));
}

// Small decode/verify batches need a fixed accumulation order. Assigning one
// thread to each output column lets it visit compact contributions in wire
// order without atomics or contribution-sized scratch storage.
__global__ void scatter_add_rows_bf16_to_f32_ordered_kernel(
    const uint16_t* src, const uint32_t* row_indices, float* dst, size_t rows,
    size_t row_width) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= row_width) {
    return;
  }
  for (size_t compact_row = 0; compact_row < rows; ++compact_row) {
    const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
    float* value = dst + dest_row * row_width + col;
    *value += bf16_to_f32(src[compact_row * row_width + col]);
  }
}

__global__ void scatter_add_rows_fp8_e4m3_row_scaled_to_f32_kernel(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const uint8_t* source = src + compact_row * src_row_stride_bytes;
  const float scale = *reinterpret_cast<const float*>(source + row_width);
  const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
  atomicAdd(&dst[dest_row * row_width + col], f8e4m3_to_f32(source[col]) * scale);
}

__global__ void scatter_add_rows_fp8_e4m3_row_scaled_to_f32_ordered_kernel(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices,
    float* dst, size_t rows, size_t row_width) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= row_width) {
    return;
  }
  for (size_t compact_row = 0; compact_row < rows; ++compact_row) {
    const uint8_t* source = src + compact_row * src_row_stride_bytes;
    const float scale = *reinterpret_cast<const float*>(source + row_width);
    const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
    float* value = dst + dest_row * row_width + col;
    *value += f8e4m3_to_f32(source[col]) * scale;
  }
}

template <int kPartialRows>
__global__ void fp8_decode_combine_residual_kernel(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* partials,
    size_t partial_row_stride_bytes, uint16_t* output, size_t row_width) {
  const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= row_width) {
    return;
  }
  float routed_delta = 0.0f;
#pragma unroll
  for (int row = 0; row < kPartialRows; ++row) {
    const uint8_t* source = partials + row * partial_row_stride_bytes;
    const float scale = *reinterpret_cast<const float*>(source + row_width);
    routed_delta += f8e4m3_to_f32(source[col]) * scale;
  }
  routed_delta = bf16_to_f32(f32_to_bf16(routed_delta));
  const float mlp_delta =
      bf16_to_f32(f32_to_bf16(bf16_to_f32(shared_delta[col]) + routed_delta));
  output[col] = f32_to_bf16(bf16_to_f32(residual[col]) + mlp_delta);
}

__global__ void scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_kernel(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const uint8_t* source = src + compact_row * src_row_stride_bytes;
  const uint8_t packed = source[col / 2];
  const uint8_t code = col % 2 == 0 ? (packed & 0x0f) : (packed >> 4);
  const float scale = f8e4m3_to_f32(source[row_width / 2 + col / 16]);
  const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
  atomicAdd(&dst[dest_row * row_width + col], nvfp4_e2m1_code_value(code) * scale);
}

__global__ void scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_ordered_kernel(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices,
    float* dst, size_t rows, size_t row_width) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= row_width) {
    return;
  }
  for (size_t compact_row = 0; compact_row < rows; ++compact_row) {
    const uint8_t* source = src + compact_row * src_row_stride_bytes;
    const uint8_t packed = source[col / 2];
    const uint8_t code = col % 2 == 0 ? (packed & 0x0f) : (packed >> 4);
    const float scale = f8e4m3_to_f32(source[row_width / 2 + col / 16]);
    const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
    float* value = dst + dest_row * row_width + col;
    *value += nvfp4_e2m1_code_value(code) * scale;
  }
}

__device__ float route_shard_wire_value(const uint8_t* source, size_t row, size_t col,
                                        size_t row_width, size_t row_stride_bytes,
                                        uint32_t peer_dtype) {
  source += row * row_stride_bytes;
  if (peer_dtype == GLMRT_ROUTE_SHARD_WIRE_BF16) {
    return bf16_to_f32(reinterpret_cast<const uint16_t*>(source)[col]);
  }
  if (peer_dtype == GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED) {
    const float scale = *reinterpret_cast<const float*>(source + row_width);
    return f8e4m3_to_f32(source[col]) * scale;
  }
  const uint8_t packed = source[col / 2];
  const uint8_t code = col % 2 == 0 ? (packed & 0x0f) : (packed >> 4);
  const float scale = f8e4m3_to_f32(source[row_width / 2 + col / 16]);
  return nvfp4_e2m1_code_value(code) * scale;
}

__global__ void reduce_route_shards_to_f32_kernel(
    const uint8_t* local, const uint8_t* peer_0, const uint8_t* peer_1,
    const uint8_t* peer_2, float* output_f32, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / row_width;
  const size_t col = idx % row_width;
  float sum = local_dtype == GLMRT_ROUTE_SHARD_LOCAL_F32
                  ? reinterpret_cast<const float*>(local)[idx]
                  : bf16_to_f32(reinterpret_cast<const uint16_t*>(local)[idx]);
  if (peer_count > 0) {
    sum += route_shard_wire_value(peer_0, row, col, row_width, peer_row_stride_bytes,
                                  peer_dtype);
  }
  if (peer_count > 1) {
    sum += route_shard_wire_value(peer_1, row, col, row_width, peer_row_stride_bytes,
                                  peer_dtype);
  }
  if (peer_count > 2) {
    sum += route_shard_wire_value(peer_2, row, col, row_width, peer_row_stride_bytes,
                                  peer_dtype);
  }
  output_f32[idx] = sum;
}

template <int kThreads>
__global__ void reduce_route_shards_bf16_fp8_to_fp8_rail_candidate_kernel(
    const uint16_t* local_bf16, const uint8_t* peer_rail0_0,
    const uint8_t* peer_rail0_1, const uint8_t* peer_rail0_2,
    const uint8_t* peer_rail1_0, const uint8_t* peer_rail1_1,
    const uint8_t* peer_rail1_2, uint8_t* output_fp8, size_t rows,
    size_t rail0_rows, size_t peer_row_stride_bytes,
    size_t output_row_stride_bytes) {
  constexpr size_t row_width = 6144;
  constexpr int values_per_thread = row_width / kThreads;
  __shared__ float warp_maxima[kThreads / 32];
  __shared__ float output_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const bool first_rail = row < rail0_rows;
  const size_t rail_row = first_rail ? row : row - rail0_rows;
  const uint8_t* peer_0 =
      (first_rail ? peer_rail0_0 : peer_rail1_0) + rail_row * peer_row_stride_bytes;
  const uint8_t* peer_1 =
      (first_rail ? peer_rail0_1 : peer_rail1_1) + rail_row * peer_row_stride_bytes;
  const uint8_t* peer_2 =
      (first_rail ? peer_rail0_2 : peer_rail1_2) + rail_row * peer_row_stride_bytes;
  const float peer_scale_0 = *reinterpret_cast<const float*>(peer_0 + row_width);
  const float peer_scale_1 = *reinterpret_cast<const float*>(peer_1 + row_width);
  const float peer_scale_2 = *reinterpret_cast<const float*>(peer_2 + row_width);

  float values[values_per_thread];
  float maximum = 0.0f;
#pragma unroll
  for (int item = 0; item < values_per_thread; ++item) {
    const size_t col = threadIdx.x + item * kThreads;
    float value = bf16_to_f32(local_bf16[row * row_width + col]);
    value += f8e4m3_to_f32(peer_0[col]) * peer_scale_0;
    value += f8e4m3_to_f32(peer_1[col]) * peer_scale_1;
    value += f8e4m3_to_f32(peer_2[col]) * peer_scale_2;
    values[item] = value;
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }

  constexpr unsigned int mask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_maxima[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kThreads / 32 ? warp_maxima[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
    }
    if (lane == 0) {
      output_scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
      *reinterpret_cast<float*>(output_fp8 + row * output_row_stride_bytes + row_width) =
          output_scale;
    }
  }
  __syncthreads();

  uint8_t* output = output_fp8 + row * output_row_stride_bytes;
#pragma unroll
  for (int item = 0; item < values_per_thread; ++item) {
    const size_t col = threadIdx.x + item * kThreads;
    const float value = values[item];
    output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / output_scale : 0.0f, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void scatter_add_rows_bf16_weighted_to_f32_kernel(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, float* dst,
    size_t rows, size_t row_width) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * row_width;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / row_width;
  const size_t col = idx % row_width;
  const size_t dest_row = static_cast<size_t>(row_indices[compact_row]);
  atomicAdd(&dst[dest_row * row_width + col], row_weights[compact_row] * bf16_to_f32(src[idx]));
}

glmrt_status_t validate_row_kernel_args(const float* src, const uint32_t* row_indices,
                                        const float* dst, size_t rows, size_t row_width) {
  if (rows == 0 || row_width == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || row_indices == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_gather_bf16_args(const uint16_t* src, const uint32_t* row_indices,
                                             const uint16_t* dst, size_t rows,
                                             size_t row_width) {
  if (src == nullptr || row_indices == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_gather_fp8_row_scaled_args(
    const float* src, const uint32_t* row_indices, const uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 || row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_stride = 0;
  if (!checked_add(row_width, sizeof(float), &minimum_stride) ||
      dst_row_stride_bytes < minimum_stride || dst_row_stride_bytes % alignof(float) != 0 ||
      rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_rows_to_fp8_row_scaled_args(
    const uint16_t* src, const uint8_t* dst, size_t rows, size_t row_width,
    size_t dst_row_stride_bytes) {
  if (src == nullptr || dst == nullptr || rows == 0 || row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_stride = 0;
  if (!checked_add(row_width, sizeof(float), &minimum_stride) ||
      dst_row_stride_bytes < minimum_stride || dst_row_stride_bytes % alignof(float) != 0 ||
      rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_combine_fp8_row_scaled_args(
    const float* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, const uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  if (local == nullptr || peers == nullptr || dst == nullptr || peer_count == 0 || rows == 0 ||
      row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_row_stride = 0;
  size_t minimum_peer_payload = 0;
  if (!checked_add(row_width, sizeof(float), &minimum_row_stride) ||
      peer_row_stride_bytes < minimum_row_stride ||
      peer_row_stride_bytes % alignof(float) != 0 ||
      dst_row_stride_bytes < minimum_row_stride || dst_row_stride_bytes % alignof(float) != 0 ||
      !checked_mul(rows, peer_row_stride_bytes, &minimum_peer_payload) ||
      peer_payload_stride_bytes < minimum_peer_payload ||
      peer_count > std::numeric_limits<size_t>::max() / peer_payload_stride_bytes ||
      rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_combine_bf16_fp8_row_scaled_args(
    const uint16_t* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, const uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  if (local == nullptr || peers == nullptr || dst == nullptr || peer_count == 0 || rows == 0 ||
      row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_row_stride = 0;
  size_t minimum_peer_payload = 0;
  if (!checked_add(row_width, sizeof(float), &minimum_row_stride) ||
      peer_row_stride_bytes < minimum_row_stride ||
      peer_row_stride_bytes % alignof(float) != 0 ||
      dst_row_stride_bytes < minimum_row_stride || dst_row_stride_bytes % alignof(float) != 0 ||
      !checked_mul(rows, peer_row_stride_bytes, &minimum_peer_payload) ||
      peer_payload_stride_bytes < minimum_peer_payload ||
      peer_count > std::numeric_limits<size_t>::max() / peer_payload_stride_bytes ||
      rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_scatter_fp8_row_scaled_args(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices,
    const float* dst, size_t rows, size_t row_width) {
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 || row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_stride = 0;
  if (!checked_add(row_width, sizeof(float), &minimum_stride) ||
      src_row_stride_bytes < minimum_stride || src_row_stride_bytes % alignof(float) != 0 ||
      rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_prefix_copy_bf16_args(const uint16_t* src, const uint16_t* dst,
                                                  size_t rows, size_t src_row_width,
                                                  size_t dst_row_width, size_t prefix_width,
                                                  size_t src_row_offset) {
  if (rows == 0 || prefix_width == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (src_row_width == 0 || dst_row_width == 0 || prefix_width > src_row_width ||
      prefix_width > dst_row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, prefix_width, &ignored) ||
      !checked_add(src_row_offset, rows, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_scatter_bf16_to_f32_args(const uint16_t* src,
                                                     const uint32_t* row_indices,
                                                     const float* dst, size_t rows,
                                                     size_t row_width) {
  if (src == nullptr || row_indices == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || row_width == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_row_scatter_bf16_weighted_to_f32_args(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, const float* dst,
    size_t rows, size_t row_width) {
  if (rows == 0 || row_width == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || row_indices == nullptr || row_weights == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_route_shard_reduction_args(
    const glmrt_route_shard_reduction_buffers_t* buffers, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count) {
  if (buffers == nullptr || buffers->local.ptr == nullptr ||
      buffers->output_f32.ptr == nullptr || rows == 0 || row_width == 0 ||
      peer_count == 0 || peer_count > 3) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  size_t output_bytes = 0;
  size_t local_bytes = 0;
  if (!checked_mul(rows, row_width, &values) ||
      !checked_mul(values, sizeof(float), &output_bytes) ||
      (local_dtype == GLMRT_ROUTE_SHARD_LOCAL_F32 &&
       !checked_mul(values, sizeof(float), &local_bytes)) ||
      (local_dtype == GLMRT_ROUTE_SHARD_LOCAL_BF16 &&
       !checked_mul(values, sizeof(uint16_t), &local_bytes)) ||
      (local_dtype != GLMRT_ROUTE_SHARD_LOCAL_F32 &&
       local_dtype != GLMRT_ROUTE_SHARD_LOCAL_BF16) ||
      buffers->local.bytes < local_bytes || buffers->output_f32.bytes < output_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t minimum_peer_stride = 0;
  if (peer_dtype == GLMRT_ROUTE_SHARD_WIRE_BF16) {
    if (!checked_mul(row_width, sizeof(uint16_t), &minimum_peer_stride)) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
  } else if (peer_dtype == GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED) {
    if (!checked_add(row_width, sizeof(float), &minimum_peer_stride) ||
        peer_row_stride_bytes % alignof(float) != 0) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
  } else if (peer_dtype == GLMRT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3) {
    if (row_width % 16 != 0 ||
        !checked_add(row_width / 2, row_width / 16, &minimum_peer_stride)) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
  } else {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t peer_bytes = 0;
  if (peer_row_stride_bytes < minimum_peer_stride ||
      !checked_mul(rows, peer_row_stride_bytes, &peer_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  for (uint32_t peer = 0; peer < peer_count; ++peer) {
    if (buffers->peers[peer].ptr == nullptr || buffers->peers[peer].bytes < peer_bytes) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_route_shard_fp8_rail_reduction_args(
    const glmrt_route_shard_fp8_rail_reduction_buffers_t* buffers, size_t rows,
    size_t rail0_rows, size_t row_width, size_t peer_row_stride_bytes,
    size_t output_row_stride_bytes) {
  constexpr size_t supported_row_width = 6144;
  if (buffers == nullptr || rows == 0 || rail0_rows == 0 || rail0_rows > rows ||
      row_width != supported_row_width ||
      peer_row_stride_bytes < row_width + sizeof(float) ||
      output_row_stride_bytes < row_width + sizeof(float) ||
      peer_row_stride_bytes % alignof(float) != 0 ||
      output_row_stride_bytes % alignof(float) != 0 ||
      buffers->local_bf16.ptr == nullptr || buffers->output_fp8.ptr == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  size_t local_bytes = 0;
  size_t output_bytes = 0;
  size_t rail0_bytes = 0;
  size_t rail1_bytes = 0;
  const size_t rail1_rows = rows - rail0_rows;
  if (!checked_mul(rows, row_width, &values) ||
      !checked_mul(values, sizeof(uint16_t), &local_bytes) ||
      !checked_mul(rows, output_row_stride_bytes, &output_bytes) ||
      !checked_mul(rail0_rows, peer_row_stride_bytes, &rail0_bytes) ||
      !checked_mul(rail1_rows, peer_row_stride_bytes, &rail1_bytes) ||
      buffers->local_bf16.bytes < local_bytes ||
      buffers->output_fp8.bytes < output_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  for (size_t peer = 0; peer < 3; ++peer) {
    if (buffers->peer_rail0[peer].ptr == nullptr ||
        buffers->peer_rail0[peer].bytes < rail0_bytes ||
        (rail1_rows > 0 &&
         (buffers->peer_rail1[peer].ptr == nullptr ||
          buffers->peer_rail1[peer].bytes < rail1_bytes))) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_residual_add_buffers(glmrt_device_buffer_t residual,
                                                        glmrt_device_buffer_t delta,
                                                        glmrt_device_buffer_t out,
                                                        size_t count) {
  if (count == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t bytes = 0;
  if (!checked_mul(count, sizeof(uint16_t), &bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (residual.ptr == nullptr || delta.ptr == nullptr || out.ptr == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (residual.bytes < bytes || delta.bytes < bytes || out.bytes < bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_residual_add_f32_delta_buffers(
    glmrt_device_buffer_t residual, glmrt_device_buffer_t delta_f32, glmrt_device_buffer_t out,
    size_t count) {
  if (count == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t bf16_bytes = 0;
  size_t f32_bytes = 0;
  if (!checked_mul(count, sizeof(uint16_t), &bf16_bytes) ||
      !checked_mul(count, sizeof(float), &f32_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (residual.ptr == nullptr || delta_f32.ptr == nullptr || out.ptr == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (residual.bytes < bf16_bytes || delta_f32.bytes < f32_bytes || out.bytes < bf16_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_residual_add_shared_f32_delta_buffers(
    glmrt_device_buffer_t residual, glmrt_device_buffer_t shared_delta,
    glmrt_device_buffer_t routed_delta_f32, glmrt_device_buffer_t out, size_t count) {
  const glmrt_status_t valid =
      validate_bf16_graph_residual_add_f32_delta_buffers(residual, routed_delta_f32, out, count);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t bf16_bytes = 0;
  if (!checked_mul(count, sizeof(uint16_t), &bf16_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (shared_delta.ptr == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (shared_delta.bytes < bf16_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_f32_to_bf16_buffers(glmrt_device_buffer_t src,
                                                       glmrt_device_buffer_t dst,
                                                       size_t count) {
  if (count == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (src.ptr == nullptr || dst.ptr == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t src_bytes = 0;
  size_t dst_bytes = 0;
  if (!checked_mul(count, sizeof(float), &src_bytes) ||
      !checked_mul(count, sizeof(uint16_t), &dst_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (src.bytes < src_bytes || dst.bytes < dst_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_scatter_add_rows_bf16_to_f32_buffers(
    glmrt_device_buffer_t src, glmrt_device_buffer_t row_indices, glmrt_device_buffer_t dst,
    size_t dst_rows, size_t rows, size_t row_width) {
  const glmrt_status_t valid = validate_row_scatter_bf16_to_f32_args(
      static_cast<const uint16_t*>(src.ptr), static_cast<const uint32_t*>(row_indices.ptr),
      static_cast<const float*>(dst.ptr), rows, row_width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (dst_rows == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t src_values = 0;
  size_t dst_values = 0;
  if (!checked_mul(rows, row_width, &src_values) ||
      !checked_mul(dst_rows, row_width, &dst_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t src_bytes = 0;
  size_t index_bytes = 0;
  size_t dst_bytes = 0;
  if (!checked_mul(src_values, sizeof(uint16_t), &src_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &index_bytes) ||
      !checked_mul(dst_values, sizeof(float), &dst_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (src.bytes < src_bytes || row_indices.bytes < index_bytes || dst.bytes < dst_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_residual_add_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t delta, glmrt_device_buffer_t out,
    size_t count) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_residual_add_buffers(residual, delta, out, count);
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
  if (existing.func != reinterpret_cast<void*>(residual_add_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* residual_ptr = static_cast<const uint16_t*>(residual.ptr);
  const uint16_t* delta_ptr = static_cast<const uint16_t*>(delta.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &residual_ptr,
      &delta_ptr,
      &out_ptr,
      &count,
  };
  const int threads = 256;
  const size_t block_count = (count - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(residual_add_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_residual_add_f32_delta_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t delta_f32, glmrt_device_buffer_t out,
    size_t count) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_residual_add_f32_delta_buffers(residual, delta_f32, out, count);
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
  if (existing.func != reinterpret_cast<void*>(residual_add_f32_delta_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* residual_ptr = static_cast<const uint16_t*>(residual.ptr);
  const float* delta_f32_ptr = static_cast<const float*>(delta_f32.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &residual_ptr,
      &delta_f32_ptr,
      &out_ptr,
      &count,
  };
  const int threads = 256;
  const size_t block_count = (count - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(residual_add_f32_delta_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t shared_delta,
    glmrt_device_buffer_t routed_delta_f32, glmrt_device_buffer_t out, size_t count) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_residual_add_shared_f32_delta_buffers(
      residual, shared_delta, routed_delta_f32, out, count);
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
  if (existing.func != reinterpret_cast<void*>(residual_add_shared_f32_delta_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* residual_ptr = static_cast<const uint16_t*>(residual.ptr);
  const uint16_t* shared_delta_ptr = static_cast<const uint16_t*>(shared_delta.ptr);
  const float* routed_delta_f32_ptr = static_cast<const float*>(routed_delta_f32.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &residual_ptr,
      &shared_delta_ptr,
      &routed_delta_f32_ptr,
      &out_ptr,
      &count,
  };
  const int threads = 256;
  const size_t block_count = (count - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(residual_add_shared_f32_delta_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_f32_to_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t dst, size_t count) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_f32_to_bf16_buffers(src, dst, count);
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
  if (existing.func != reinterpret_cast<void*>(f32_to_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const float* src_ptr = static_cast<const float*>(src.ptr);
  uint16_t* dst_ptr = static_cast<uint16_t*>(dst.ptr);
  void* args[] = {
      &src_ptr,
      &dst_ptr,
      &count,
  };
  const int threads = 256;
  const size_t block_count = (count - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(f32_to_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t row_indices, glmrt_device_buffer_t dst,
    size_t dst_rows, size_t rows, size_t row_width) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_scatter_add_rows_bf16_to_f32_buffers(
      src, row_indices, dst, dst_rows, rows, row_width);
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
  if (existing.func != reinterpret_cast<void*>(scatter_add_rows_bf16_to_f32_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* src_ptr = static_cast<const uint16_t*>(src.ptr);
  const uint32_t* row_indices_ptr = static_cast<const uint32_t*>(row_indices.ptr);
  float* dst_ptr = static_cast<float*>(dst.ptr);
  void* args[] = {
      &src_ptr,
      &row_indices_ptr,
      &dst_ptr,
      &rows,
      &row_width,
  };
  const int threads = 256;
  size_t total = 0;
  if (!checked_mul(rows, row_width, &total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(scatter_add_rows_bf16_to_f32_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_residual_add_f32_async(const float* residual,
                                                            const float* delta, float* out,
                                                            size_t count, void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (residual == nullptr || delta == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  residual_add_f32_kernel<<<blocks, threads, 0, stream>>>(residual, delta, out, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_f32(const float* residual, const float* delta,
                                                      float* out, size_t count) {
  const glmrt_status_t status =
      glmrt_cuda_residual_add_f32_async(residual, delta, out, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_bf16_async(const uint16_t* residual,
                                                             const uint16_t* delta,
                                                             uint16_t* out, size_t count,
                                                             void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (residual == nullptr || delta == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  residual_add_bf16_kernel<<<blocks, threads, 0, stream>>>(residual, delta, out, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_bf16(const uint16_t* residual,
                                                       const uint16_t* delta, uint16_t* out,
                                                       size_t count) {
  const glmrt_status_t status =
      glmrt_cuda_residual_add_bf16_async(residual, delta, out, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_f32_delta_bf16_async(
    const uint16_t* residual, const float* delta_f32, uint16_t* out, size_t count,
    void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (residual == nullptr || delta_f32 == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  residual_add_f32_delta_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      residual, delta_f32, out, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_f32_delta_bf16(
    const uint16_t* residual, const float* delta_f32, uint16_t* out, size_t count) {
  const glmrt_status_t status =
      glmrt_cuda_residual_add_f32_delta_bf16_async(residual, delta_f32, out, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_shared_f32_delta_bf16_async(
    const uint16_t* residual, const uint16_t* shared_delta, const float* routed_delta_f32,
    uint16_t* out, size_t count, void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (residual == nullptr || shared_delta == nullptr || routed_delta_f32 == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  residual_add_shared_f32_delta_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      residual, shared_delta, routed_delta_f32, out, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_shared_f32_delta_bf16(
    const uint16_t* residual, const uint16_t* shared_delta, const float* routed_delta_f32,
    uint16_t* out, size_t count) {
  const glmrt_status_t status = glmrt_cuda_residual_add_shared_f32_delta_bf16_async(
      residual, shared_delta, routed_delta_f32, out, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* routed_delta_fp8,
    uint16_t* out, size_t count, void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (residual == nullptr || shared_delta == nullptr || routed_delta_fp8 == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  residual_add_shared_fp8_e4m3_row_scaled_bf16_kernel<<<
      static_cast<int>(block_count), threads, 0, stream>>>(
      residual, shared_delta, routed_delta_fp8, out, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scheduler_mlp_delta_bf16_async(
    const uint16_t* hidden, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden_dim,
    void* cuda_stream) {
  if (rows == 0 || hidden_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (hidden == nullptr || gate_weight == nullptr || up_weight == nullptr ||
      down_weight == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<size_t>::max() / hidden_dim) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden_dim;
  const size_t block_count = (total + threads - 1) / threads;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  scheduler_mlp_delta_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      hidden, gate_weight, up_weight, down_weight, out, rows, hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scheduler_mlp_delta_bf16(
    const uint16_t* hidden, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden_dim) {
  const glmrt_status_t status = glmrt_cuda_scheduler_mlp_delta_bf16_async(
      hidden, gate_weight, up_weight, down_weight, out, rows, hidden_dim, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_summarize_bf16_async(const uint16_t* input, size_t count,
                                                          glmrt_bf16_summary_t* out_device,
                                                          void* cuda_stream) {
  if (out_device == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (count > 0 && input == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (count > static_cast<size_t>(std::numeric_limits<int>::max()) * kBlock) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t err = cudaMemsetAsync(out_device, 0, sizeof(glmrt_bf16_summary_t), stream);
  if (err != cudaSuccess || count == 0) {
    return status_from_cuda(err);
  }

  const int threads = kBlock;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  summarize_bf16_kernel<<<blocks, threads, 0, stream>>>(input, count, out_device);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_summarize_bf16(const uint16_t* input, size_t count,
                                                    glmrt_bf16_summary_t* out) {
  if (out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  out->checksum = 0.0;
  out->values = static_cast<uint64_t>(count);
  out->finite_values = 0;
  out->nonzero_values = 0;
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (input == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  glmrt_bf16_summary_t* device_out = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void**>(&device_out), sizeof(glmrt_bf16_summary_t));
  if (err != cudaSuccess) {
    return GLMRT_STATUS_ALLOCATION_FAILED;
  }
  const glmrt_status_t launch_status =
      glmrt_cuda_summarize_bf16_async(input, count, device_out, nullptr);
  if (launch_status != GLMRT_STATUS_OK) {
    cudaFree(device_out);
    return launch_status;
  }
  err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    cudaFree(device_out);
    return status_from_cuda(err);
  }
  err = cudaMemcpy(out, device_out, sizeof(glmrt_bf16_summary_t), cudaMemcpyDeviceToHost);
  const cudaError_t free_err = cudaFree(device_out);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  if (free_err != cudaSuccess) {
    return status_from_cuda(free_err);
  }
  out->values = static_cast<uint64_t>(count);
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_zero_f32_async(float* dst, size_t count,
                                                    void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const cudaError_t err = cudaMemsetAsync(dst, 0, count * sizeof(float), stream);
  return status_from_cuda(err);
}

extern "C" glmrt_status_t glmrt_cuda_zero_f32(float* dst, size_t count) {
  const glmrt_status_t status = glmrt_cuda_zero_f32_async(dst, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_zero_bytes_async(void* dst, size_t bytes,
                                                      void* cuda_stream) {
  if (bytes == 0) {
    return GLMRT_STATUS_OK;
  }
  if (dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const cudaError_t err = cudaMemsetAsync(dst, 0, bytes, stream);
  return status_from_cuda(err);
}

extern "C" glmrt_status_t glmrt_cuda_zero_bytes(void* dst, size_t bytes) {
  const glmrt_status_t status = glmrt_cuda_zero_bytes_async(dst, bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_f32_to_bf16_async(const float* src, uint16_t* dst,
                                                       size_t count, void* cuda_stream) {
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  if (src == nullptr || dst == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(src, dst, count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_f32_to_bf16(const float* src, uint16_t* dst, size_t count) {
  const glmrt_status_t status = glmrt_cuda_f32_to_bf16_async(src, dst, count, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_async(const float* src,
                                                           const uint32_t* row_indices,
                                                           float* dst, size_t rows,
                                                           size_t row_width,
                                                           void* cuda_stream) {
  const glmrt_status_t valid = validate_row_kernel_args(src, row_indices, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK || rows == 0 || row_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  gather_rows_f32_kernel<<<blocks, threads, 0, stream>>>(src, row_indices, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32(const float* src,
                                                     const uint32_t* row_indices, float* dst,
                                                     size_t rows, size_t row_width) {
  const glmrt_status_t status =
      glmrt_cuda_gather_rows_f32_async(src, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_to_bf16_candidate_async(
    const float* src, const uint32_t* row_indices, uint16_t* dst, size_t rows,
    size_t row_width, void* cuda_stream) {
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 ||
      row_width == 0 || rows > std::numeric_limits<size_t>::max() / row_width) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t total = rows * row_width;
  constexpr size_t threads = 256;
  const size_t blocks = (total + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  gather_rows_f32_to_bf16_candidate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      src, row_indices, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_row_gather_fp8_row_scaled_args(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (row_width == 6144) {
    gather_rows_f32_to_fp8_e4m3_row_scaled_register_kernel<<<
        static_cast<int>(rows), kBlock, 0, stream>>>(
        src, row_indices, dst, rows, dst_row_stride_bytes);
  } else {
    gather_rows_f32_to_fp8_e4m3_row_scaled_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
        src, row_indices, dst, rows, row_width, dst_row_stride_bytes);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_register_candidate_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_row_gather_fp8_row_scaled_args(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK || row_width != 6144) {
    return valid != GLMRT_STATUS_OK ? valid : GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  gather_rows_f32_to_fp8_e4m3_row_scaled_register_kernel<<<
      static_cast<int>(rows), kBlock, 0, stream>>>(
      src, row_indices, dst, rows, dst_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
    const uint16_t* src, uint8_t* dst, size_t rows, size_t row_width,
    size_t dst_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_bf16_rows_to_fp8_row_scaled_args(
      src, dst, rows, row_width, dst_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  bf16_rows_to_fp8_e4m3_row_scaled_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
      src, dst, rows, row_width, dst_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_combine_fp8_e4m3_row_scaled_to_fp8_async(
    const float* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_combine_fp8_row_scaled_args(
      local, peers, peer_payload_stride_bytes, peer_count, peer_row_stride_bytes, dst, rows,
      row_width, dst_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  combine_fp8_e4m3_row_scaled_to_fp8_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
      local, peers, peer_payload_stride_bytes, peer_count, peer_row_stride_bytes, dst, rows,
      row_width, dst_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async(
    const uint16_t* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_combine_bf16_fp8_row_scaled_args(
      local, peers, peer_payload_stride_bytes, peer_count, peer_row_stride_bytes, dst, rows,
      row_width, dst_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  combine_bf16_fp8_e4m3_row_scaled_to_fp8_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
      local, peers, peer_payload_stride_bytes, peer_count, peer_row_stride_bytes, dst, rows,
      row_width, dst_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_bf16_async(const uint16_t* src,
                                                            const uint32_t* row_indices,
                                                            uint16_t* dst, size_t rows,
                                                            size_t row_width,
                                                            void* cuda_stream) {
  const glmrt_status_t valid =
      validate_row_gather_bf16_args(src, row_indices, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK || rows == 0 || row_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  gather_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(src, row_indices, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_bf16(const uint16_t* src,
                                                      const uint32_t* row_indices, uint16_t* dst,
                                                      size_t rows, size_t row_width) {
  const glmrt_status_t status =
      glmrt_cuda_gather_rows_bf16_async(src, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_copy_row_prefix_bf16_async(
    const uint16_t* src, uint16_t* dst, size_t rows, size_t src_row_width,
    size_t dst_row_width, size_t prefix_width, size_t src_row_offset, void* cuda_stream) {
  const glmrt_status_t valid = validate_row_prefix_copy_bf16_args(
      src, dst, rows, src_row_width, dst_row_width, prefix_width, src_row_offset);
  if (valid != GLMRT_STATUS_OK || rows == 0 || prefix_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * prefix_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  copy_row_prefix_bf16_kernel<<<blocks, threads, 0, stream>>>(
      src, dst, rows, src_row_width, dst_row_width, prefix_width, src_row_offset);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_copy_row_prefix_bf16(
    const uint16_t* src, uint16_t* dst, size_t rows, size_t src_row_width,
    size_t dst_row_width, size_t prefix_width, size_t src_row_offset) {
  const glmrt_status_t status = glmrt_cuda_copy_row_prefix_bf16_async(
      src, dst, rows, src_row_width, dst_row_width, prefix_width, src_row_offset, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_f32_async(const float* src,
                                                                const uint32_t* row_indices,
                                                                float* dst, size_t rows,
                                                                size_t row_width,
                                                                void* cuda_stream) {
  const glmrt_status_t valid = validate_row_kernel_args(src, row_indices, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK || rows == 0 || row_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  scatter_add_rows_f32_kernel<<<blocks, threads, 0, stream>>>(src, row_indices, dst, rows,
                                                              row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_f32(const float* src,
                                                          const uint32_t* row_indices, float* dst,
                                                          size_t rows, size_t row_width) {
  const glmrt_status_t status =
      glmrt_cuda_scatter_add_rows_f32_async(src, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_to_f32_async(
    const uint16_t* src, const uint32_t* row_indices, float* dst, size_t rows, size_t row_width,
    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_row_scatter_bf16_to_f32_args(src, row_indices, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK || rows == 0 || row_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  if (rows <= kOrderedScatterMaxRows) {
    const size_t ordered_block_count = (row_width - 1) / threads + 1;
    scatter_add_rows_bf16_to_f32_ordered_kernel<<<static_cast<int>(ordered_block_count), threads,
                                                   0, stream>>>(src, row_indices, dst, rows,
                                                               row_width);
    return status_from_cuda(cudaGetLastError());
  }
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  // Report this launch's status rather than inheriting a stale CUDA error from
  // a preceding graph-backed coordinator operation on the same host thread.
  (void)cudaGetLastError();
  scatter_add_rows_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(src, row_indices, dst, rows,
                                                                      row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_to_f32(
    const uint16_t* src, const uint32_t* row_indices, float* dst, size_t rows, size_t row_width) {
  const glmrt_status_t status = glmrt_cuda_scatter_add_rows_bf16_to_f32_async(
      src, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width, void* cuda_stream) {
  const glmrt_status_t valid = validate_row_scatter_fp8_row_scaled_args(
      src, src_row_stride_bytes, row_indices, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  if (rows <= kOrderedScatterMaxRows) {
    const size_t ordered_block_count = (row_width - 1) / threads + 1;
    scatter_add_rows_fp8_e4m3_row_scaled_to_f32_ordered_kernel
        <<<static_cast<int>(ordered_block_count), threads, 0, stream>>>(
            src, src_row_stride_bytes, row_indices, dst, rows, row_width);
    return status_from_cuda(cudaGetLastError());
  }
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  scatter_add_rows_fp8_e4m3_row_scaled_to_f32_kernel<<<static_cast<int>(block_count), threads, 0,
                                                       stream>>>(
      src, src_row_stride_bytes, row_indices, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width) {
  const glmrt_status_t status = glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
      src, src_row_stride_bytes, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_fp8_decode_combine_residual_async(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* partials,
    size_t partial_row_stride_bytes, uint16_t* output, size_t partial_rows,
    size_t row_width, void* cuda_stream) {
  size_t minimum_stride = 0;
  if (residual == nullptr || shared_delta == nullptr || partials == nullptr ||
      output == nullptr || partial_rows == 0 || partial_rows > 4 || row_width == 0 ||
      !checked_add(row_width, sizeof(float), &minimum_stride) ||
      partial_row_stride_bytes < minimum_stride ||
      partial_row_stride_bytes % alignof(float) != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (row_width + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  switch (partial_rows) {
    case 1:
      fp8_decode_combine_residual_kernel<1><<<static_cast<int>(blocks), threads, 0, stream>>>(
          residual, shared_delta, partials, partial_row_stride_bytes, output, row_width);
      break;
    case 2:
      fp8_decode_combine_residual_kernel<2><<<static_cast<int>(blocks), threads, 0, stream>>>(
          residual, shared_delta, partials, partial_row_stride_bytes, output, row_width);
      break;
    case 3:
      fp8_decode_combine_residual_kernel<3><<<static_cast<int>(blocks), threads, 0, stream>>>(
          residual, shared_delta, partials, partial_row_stride_bytes, output, row_width);
      break;
    case 4:
      fp8_decode_combine_residual_kernel<4><<<static_cast<int>(blocks), threads, 0, stream>>>(
          residual, shared_delta, partials, partial_row_stride_bytes, output, row_width);
      break;
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width, void* cuda_stream) {
  size_t logical_row_bytes = 0;
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 ||
      row_width == 0 || row_width % 16 != 0 ||
      !checked_add(row_width / 2, row_width / 16, &logical_row_bytes) ||
      src_row_stride_bytes < logical_row_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t total = rows * row_width;
  const int threads = 256;
  if (rows <= kOrderedScatterMaxRows) {
    const size_t ordered_block_count = (row_width - 1) / threads + 1;
    scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_ordered_kernel
        <<<static_cast<int>(ordered_block_count), threads, 0,
           reinterpret_cast<cudaStream_t>(cuda_stream)>>>(src, src_row_stride_bytes, row_indices,
                                                          dst, rows, row_width);
    return status_from_cuda(cudaGetLastError());
  }
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_kernel<<<static_cast<int>(block_count), threads, 0,
                                                        reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      src, src_row_stride_bytes, row_indices, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width) {
  const glmrt_status_t status = glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
      src, src_row_stride_bytes, row_indices, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_reduce_route_shards_to_f32_async(
    const glmrt_route_shard_reduction_buffers_t* buffers, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count, void* cuda_stream) {
  const glmrt_status_t valid = validate_route_shard_reduction_args(
      buffers, rows, row_width, peer_row_stride_bytes, local_dtype, peer_dtype, peer_count);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t total = rows * row_width;
  const int threads = 256;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  reduce_route_shards_to_f32_kernel<<<static_cast<int>(block_count), threads, 0,
                                      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(buffers->local.ptr),
      static_cast<const uint8_t*>(buffers->peers[0].ptr),
      static_cast<const uint8_t*>(buffers->peers[1].ptr),
      static_cast<const uint8_t*>(buffers->peers[2].ptr),
      static_cast<float*>(buffers->output_f32.ptr), rows, row_width,
      peer_row_stride_bytes, local_dtype, peer_dtype, peer_count);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_reduce_route_shards_bf16_fp8_to_fp8_rail_candidate_async(
    const glmrt_route_shard_fp8_rail_reduction_buffers_t* buffers, size_t rows,
    size_t rail0_rows, size_t row_width, size_t peer_row_stride_bytes,
    size_t output_row_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid = validate_route_shard_fp8_rail_reduction_args(
      buffers, rows, rail0_rows, row_width, peer_row_stride_bytes,
      output_row_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  reduce_route_shards_bf16_fp8_to_fp8_rail_candidate_kernel<384>
      <<<static_cast<int>(rows), 384, 0, stream>>>(
          static_cast<const uint16_t*>(buffers->local_bf16.ptr),
          static_cast<const uint8_t*>(buffers->peer_rail0[0].ptr),
          static_cast<const uint8_t*>(buffers->peer_rail0[1].ptr),
          static_cast<const uint8_t*>(buffers->peer_rail0[2].ptr),
          static_cast<const uint8_t*>(buffers->peer_rail1[0].ptr),
          static_cast<const uint8_t*>(buffers->peer_rail1[1].ptr),
          static_cast<const uint8_t*>(buffers->peer_rail1[2].ptr),
          static_cast<uint8_t*>(buffers->output_fp8.ptr), rows, rail0_rows,
          peer_row_stride_bytes, output_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_reduce_route_shards_to_f32(
    const glmrt_route_shard_reduction_buffers_t* buffers, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count) {
  const glmrt_status_t status = glmrt_cuda_reduce_route_shards_to_f32_async(
      buffers, rows, row_width, peer_row_stride_bytes, local_dtype, peer_dtype, peer_count,
      nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32_async(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, float* dst,
    size_t rows, size_t row_width, void* cuda_stream) {
  const glmrt_status_t valid = validate_row_scatter_bf16_weighted_to_f32_args(
      src, row_indices, row_weights, dst, rows, row_width);
  if (valid != GLMRT_STATUS_OK || rows == 0 || row_width == 0) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * row_width;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  scatter_add_rows_bf16_weighted_to_f32_kernel<<<blocks, threads, 0, stream>>>(
      src, row_indices, row_weights, dst, rows, row_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, float* dst,
    size_t rows, size_t row_width) {
  const glmrt_status_t status = glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32_async(
      src, row_indices, row_weights, dst, rows, row_width, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
