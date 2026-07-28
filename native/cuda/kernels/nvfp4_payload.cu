#include "common.h"

#include <cuda_fp8.h>

namespace {

constexpr int kQuantThreads = 32;
constexpr int kGroupedQuantThreads = 256;
constexpr int kWarpsPerGroupedBlock = kGroupedQuantThreads / 32;

__device__ uint8_t fp4_e2m1_payload_code(float value) {
  const bool negative = signbit(value);
  const float magnitude = fabsf(value);
  uint8_t code = 0;
  if (magnitude <= 0.25f) {
    code = 0;
  } else if (magnitude < 0.75f) {
    code = 1;
  } else if (magnitude <= 1.25f) {
    code = 2;
  } else if (magnitude < 1.75f) {
    code = 3;
  } else if (magnitude <= 2.5f) {
    code = 4;
  } else if (magnitude < 3.5f) {
    code = 5;
  } else if (magnitude <= 5.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return static_cast<uint8_t>(code | (negative ? 8 : 0));
}

__global__ void quantize_bf16_nvfp4_row_payload_kernel(const uint16_t* input,
                                                        uint8_t* payload, size_t rows,
                                                        size_t cols) {
  __shared__ float values[16];
  __shared__ uint8_t codes[16];
  __shared__ float inverse_scale;
  const size_t row = blockIdx.y;
  const size_t col_block = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane < 16) {
    values[lane] = bf16_to_f32(input[row * cols + col_block * 16 + lane]);
  }
  __syncthreads();

  float maximum = lane < 16 ? fabsf(values[lane]) : 0.0f;
  for (int offset = 8; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffu, maximum, offset));
  }
  const size_t packed_row_bytes = cols / 2;
  const size_t scale_cols = cols / 16;
  uint8_t* row_payload = payload + row * (packed_row_bytes + scale_cols);
  if (lane == 0) {
    const uint8_t scale_byte =
        static_cast<uint8_t>(__nv_cvt_float_to_fp8(maximum / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float decoded_scale = f8e4m3_to_f32(scale_byte);
    inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    row_payload[packed_row_bytes + col_block] = scale_byte;
  }
  __syncthreads();
  if (lane < 16) {
    const float quantized = fminf(fmaxf(values[lane] * inverse_scale, -6.0f), 6.0f);
    codes[lane] = fp4_e2m1_payload_code(quantized);
  }
  __syncthreads();
  if (lane < 8) {
    row_payload[col_block * 8 + lane] =
        static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
  }
}

__global__ void gather_f32_nvfp4_row_payload_kernel(const float* input,
                                                     const uint32_t* row_indices,
                                                     uint8_t* payload, size_t rows,
                                                     size_t cols,
                                                     size_t row_stride_bytes) {
  __shared__ float values[16];
  __shared__ uint8_t codes[16];
  __shared__ float inverse_scale;
  const size_t row = blockIdx.y;
  const size_t source_row = static_cast<size_t>(row_indices[row]);
  const size_t col_block = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane < 16) {
    values[lane] = input[source_row * cols + col_block * 16 + lane];
  }
  __syncthreads();

  float maximum = lane < 16 ? fabsf(values[lane]) : 0.0f;
  for (int offset = 8; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffu, maximum, offset));
  }
  const size_t packed_row_bytes = cols / 2;
  uint8_t* row_payload = payload + row * row_stride_bytes;
  if (lane == 0) {
    const uint8_t scale_byte =
        static_cast<uint8_t>(__nv_cvt_float_to_fp8(maximum / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float decoded_scale = f8e4m3_to_f32(scale_byte);
    inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    row_payload[packed_row_bytes + col_block] = scale_byte;
  }
  __syncthreads();
  if (lane < 16) {
    const float quantized = fminf(fmaxf(values[lane] * inverse_scale, -6.0f), 6.0f);
    codes[lane] = fp4_e2m1_payload_code(quantized);
  }
  __syncthreads();
  if (lane < 8) {
    row_payload[col_block * 8 + lane] =
        static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
  }
}

__global__ void gather_f32_nvfp4_row_payload_grouped_candidate_kernel(
    const float* input, const uint32_t* row_indices, uint8_t* payload,
    size_t rows, size_t cols, size_t row_stride_bytes,
    size_t blocks_per_row) {
  const size_t row = static_cast<size_t>(blockIdx.x) / blocks_per_row;
  if (row >= rows) {
    return;
  }
  const size_t row_block = static_cast<size_t>(blockIdx.x) % blocks_per_row;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const size_t group_count = cols / 16;
  const size_t group_stride = blocks_per_row * kWarpsPerGroupedBlock;
  const size_t source_row = static_cast<size_t>(row_indices[row]);
  const float* source = input + source_row * cols;
  uint8_t* row_payload = payload + row * row_stride_bytes;
  constexpr unsigned int mask = 0xffffffffU;

  for (size_t group = row_block * kWarpsPerGroupedBlock + warp;
       group < group_count; group += group_stride) {
    const float value = lane < 16 ? source[group * 16 + lane] : 0.0f;
    float maximum = lane < 16 ? fabsf(value) : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
    }
    const uint8_t scale_byte = static_cast<uint8_t>(__shfl_sync(
        mask,
        lane == 0
            ? static_cast<unsigned int>(__nv_cvt_float_to_fp8(
                  maximum / 6.0f, __NV_SATFINITE, __NV_E4M3))
            : 0U,
        0));
    if (lane == 0) {
      row_payload[cols / 2 + group] = scale_byte;
    }
    const float decoded_scale = f8e4m3_to_f32(scale_byte);
    const float inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    const uint8_t code = lane < 16
                             ? fp4_e2m1_payload_code(
                                   fminf(fmaxf(value * inverse_scale, -6.0f), 6.0f))
                             : 0;
    const int pair_lane = (lane & 7) * 2;
    const uint8_t low = static_cast<uint8_t>(__shfl_sync(
        mask, static_cast<unsigned int>(code), pair_lane));
    const uint8_t high = static_cast<uint8_t>(__shfl_sync(
        mask, static_cast<unsigned int>(code), pair_lane + 1));
    if (lane < 8) {
      row_payload[group * 8 + lane] =
          static_cast<uint8_t>(low | (high << 4));
    }
  }
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
    glmrt_device_buffer_t input, glmrt_device_buffer_t payload, size_t rows, size_t hidden_dim,
    void* cuda_stream) {
  size_t input_values = 0;
  size_t input_bytes = 0;
  size_t row_stride_bytes = 0;
  size_t payload_bytes = 0;
  if (rows == 0 || hidden_dim == 0 || hidden_dim % 16 != 0 ||
      !checked_mul(rows, hidden_dim, &input_values) ||
      !checked_mul(input_values, sizeof(uint16_t), &input_bytes) ||
      !checked_add(hidden_dim / 2, hidden_dim / 16, &row_stride_bytes) ||
      !checked_mul(rows, row_stride_bytes, &payload_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (input.ptr == nullptr || input.bytes < input_bytes || payload.ptr == nullptr ||
      payload.bytes < payload_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const dim3 grid(static_cast<unsigned int>(hidden_dim / 16),
                  static_cast<unsigned int>(rows));
  // Python attention fallbacks can leave a consumed launch error in CUDA's
  // thread-local last-error slot. Report only this kernel's launch status.
  (void)cudaGetLastError();
  quantize_bf16_nvfp4_row_payload_kernel<<<grid, kQuantThreads, 0,
                                          reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint16_t*>(input.ptr), static_cast<uint8_t*>(payload.ptr), rows,
      hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  size_t logical_row_bytes = 0;
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 ||
      row_width == 0 || row_width % 16 != 0 ||
      !checked_add(row_width / 2, row_width / 16, &logical_row_bytes) ||
      dst_row_stride_bytes < logical_row_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const dim3 grid(static_cast<unsigned int>(row_width / 16),
                  static_cast<unsigned int>(rows));
  (void)cudaGetLastError();
  gather_f32_nvfp4_row_payload_kernel<<<grid, kQuantThreads, 0,
                                        reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_grouped_candidate_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, size_t blocks_per_row,
    void* cuda_stream) {
  size_t logical_row_bytes = 0;
  size_t grid_blocks = 0;
  if (src == nullptr || row_indices == nullptr || dst == nullptr || rows == 0 ||
      row_width == 0 || row_width % 16 != 0 || blocks_per_row == 0 ||
      blocks_per_row > row_width / 16 ||
      !checked_add(row_width / 2, row_width / 16, &logical_row_bytes) ||
      dst_row_stride_bytes < logical_row_bytes ||
      !checked_mul(rows, blocks_per_row, &grid_blocks) ||
      grid_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  (void)cudaGetLastError();
  gather_f32_nvfp4_row_payload_grouped_candidate_kernel<<<
      static_cast<unsigned int>(grid_blocks), kGroupedQuantThreads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes,
      blocks_per_row);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_policy_candidate_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream) {
  if (rows <= 4) {
    return glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
        src, row_indices, dst, rows, row_width, dst_row_stride_bytes,
        cuda_stream);
  }
  size_t blocks_per_row = 8;
  if (rows <= 8) {
    blocks_per_row = 24;
  } else if (rows <= 16) {
    blocks_per_row = 16;
  } else if (rows > 128) {
    blocks_per_row = 4;
  }
  return glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_grouped_candidate_async(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes,
      blocks_per_row, cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes) {
  const glmrt_status_t status = glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
      src, row_indices, dst, rows, row_width, dst_row_stride_bytes, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
