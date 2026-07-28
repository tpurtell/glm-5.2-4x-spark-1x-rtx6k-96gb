#include "common.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace {

__global__ void silu_gated_mlp_f32_kernel(const float* x, const float* gate_weight,
                                          const float* up_weight, const float* down_weight,
                                          float* out, int hidden, int intermediate) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int out_col = 0; out_col < hidden; ++out_col) {
    float acc = 0.0f;
    for (int mid = 0; mid < intermediate; ++mid) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int col = 0; col < hidden; ++col) {
        gate += x[col] * gate_weight[mid * hidden + col];
        up += x[col] * up_weight[mid * hidden + col];
      }
      const float silu = gate / (1.0f + expf(-gate));
      acc += silu * up * down_weight[out_col * intermediate + mid];
    }
    out[out_col] = acc;
  }
}

__global__ void silu_gated_mlp_rows_f32_kernel(const float* x, const float* gate_weight,
                                               const float* up_weight,
                                               const float* down_weight, float* out,
                                               size_t rows, size_t hidden, size_t intermediate) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t out_col = idx % hidden;
  const float* row_x = x + row * hidden;
  float acc = 0.0f;
  for (size_t mid = 0; mid < intermediate; ++mid) {
    float gate = 0.0f;
    float up = 0.0f;
    for (size_t col = 0; col < hidden; ++col) {
      gate += row_x[col] * gate_weight[mid * hidden + col];
      up += row_x[col] * up_weight[mid * hidden + col];
    }
    const float silu = gate / (1.0f + expf(-gate));
    acc += silu * up * down_weight[out_col * intermediate + mid];
  }
  out[idx] = acc;
}

__global__ void silu_gated_mlp_rows_bf16_kernel(const uint16_t* x, const uint16_t* gate_weight,
                                                const uint16_t* up_weight,
                                                const uint16_t* down_weight, uint16_t* out,
                                                size_t rows, size_t hidden,
                                                size_t intermediate) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t out_col = idx % hidden;
  const uint16_t* row_x = x + row * hidden;
  float acc = 0.0f;
  for (size_t mid = 0; mid < intermediate; ++mid) {
    float gate = 0.0f;
    float up = 0.0f;
    for (size_t col = 0; col < hidden; ++col) {
      gate += bf16_to_f32(row_x[col]) * bf16_to_f32(gate_weight[mid * hidden + col]);
      up += bf16_to_f32(row_x[col]) * bf16_to_f32(up_weight[mid * hidden + col]);
    }
    const float silu = gate / (1.0f + expf(-gate));
    acc += silu * up * bf16_to_f32(down_weight[out_col * intermediate + mid]);
  }
  out[idx] = f32_to_bf16(acc);
}

__global__ void silu_mul_bf16_kernel(const uint16_t* gate_up, uint16_t* out,
                                     size_t rows, size_t intermediate) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * intermediate;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / intermediate;
  const size_t col = idx % intermediate;
  const uint16_t* row_gate_up = gate_up + row * intermediate * 2;
  const float gate = bf16_to_f32(row_gate_up[col]);
  const float up = bf16_to_f32(row_gate_up[intermediate + col]);
  out[idx] = f32_to_bf16(gate * sigmoid_f32(gate) * up);
}

__global__ void quantize_bf16_weight_nvfp4_kernel(
    const uint16_t* input, uint8_t* packed, uint8_t* scales,
    size_t cols, float global_scale) {
  __shared__ float block_values[16];
  __shared__ uint8_t block_codes[16];
  __shared__ uint8_t block_scale;
  const size_t group = blockIdx.x;
  const size_t lane = threadIdx.x;
  const size_t value_index = group * 16 + lane;
  const float value = bf16_to_f32(input[value_index]);
  block_values[lane] = value;
  __syncthreads();
  if (lane == 0) {
    float max_abs = 0.0f;
    for (size_t index = 0; index < 16; ++index) {
      max_abs = fmaxf(max_abs, fabsf(block_values[index]));
    }
    block_scale = __nv_cvt_float_to_fp8(
        max_abs * global_scale / 6.0f, __NV_SATFINITE, __NV_E4M3);
    scales[group] = block_scale;
  }
  __syncthreads();
  const float decoded_scale = f8e4m3_to_f32(block_scale);
  const float scaled =
      decoded_scale == 0.0f ? 0.0f : value * global_scale / decoded_scale;
  block_codes[lane] =
      static_cast<uint8_t>(
          __nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest)) &
      0x0f;
  __syncthreads();
  if (lane < 8) {
    packed[group * 8 + lane] =
        block_codes[lane * 2] | (block_codes[lane * 2 + 1] << 4);
  }
}

__global__ void silu_gated_mlp_rows_bf16_down_stride_kernel(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    size_t down_stride) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t out_col = idx % hidden;
  const uint16_t* row_x = x + row * hidden;
  const uint16_t* down_row = down_weight + out_col * down_stride;
  float acc = 0.0f;
  for (size_t mid = 0; mid < intermediate; ++mid) {
    float gate = 0.0f;
    float up = 0.0f;
    for (size_t col = 0; col < hidden; ++col) {
      gate += bf16_to_f32(row_x[col]) * bf16_to_f32(gate_weight[mid * hidden + col]);
      up += bf16_to_f32(row_x[col]) * bf16_to_f32(up_weight[mid * hidden + col]);
    }
    const float silu = gate / (1.0f + expf(-gate));
    acc += silu * up * bf16_to_f32(down_row[mid]);
  }
  out[idx] = f32_to_bf16(acc);
}

__global__ void silu_gated_mlp_rows_bf16_activation_f32_reduce_kernel(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    float* activations, size_t rows, size_t hidden, size_t intermediate) {
  __shared__ float gate_scratch[kNvfp4RouteReduceBlock];
  __shared__ float up_scratch[kNvfp4RouteReduceBlock];
  const size_t mid = blockIdx.x;
  const size_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows || mid >= intermediate) {
    return;
  }
  const uint16_t* row_x = x + row * hidden;
  const uint16_t* gate_row = gate_weight + mid * hidden;
  const uint16_t* up_row = up_weight + mid * hidden;
  float gate = 0.0f;
  float up = 0.0f;
  for (size_t col = tid; col < hidden; col += blockDim.x) {
    const float value = bf16_to_f32(row_x[col]);
    gate = fmaf(value, bf16_to_f32(gate_row[col]), gate);
    up = fmaf(value, bf16_to_f32(up_row[col]), up);
  }
  gate_scratch[tid] = gate;
  up_scratch[tid] = up;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      gate_scratch[tid] += gate_scratch[tid + stride];
      up_scratch[tid] += up_scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float reduced_gate = gate_scratch[0];
    activations[row * intermediate + mid] =
        reduced_gate * sigmoid_f32(reduced_gate) * up_scratch[0];
  }
}

__global__ void silu_gated_mlp_rows_bf16_down_from_activation_kernel(
    const uint16_t* down_weight, const float* activations, uint16_t* out, size_t rows,
    size_t hidden, size_t intermediate, size_t down_stride) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t out_col = idx % hidden;
  const float* row_activations = activations + row * intermediate;
  const uint16_t* down_row = down_weight + out_col * down_stride;
  float acc = 0.0f;
  for (size_t mid = 0; mid < intermediate; ++mid) {
    acc = fmaf(row_activations[mid], bf16_to_f32(down_row[mid]), acc);
  }
  out[idx] = f32_to_bf16(acc);
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_grouped_activation_f32_reduce_kernel(
    const uint16_t* hidden, const uint32_t* row_indices, const uint8_t* gate_weight,
    const uint8_t* gate_scale, const uint8_t* up_weight, const uint8_t* up_scale,
    float* activations, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t intermediate, float gate_scale_2, float up_scale_2) {
  __shared__ float gate_scratch[kNvfp4RouteReduceBlock];
  __shared__ float up_scratch[kNvfp4RouteReduceBlock];
  const size_t route_idx = blockIdx.y;
  const size_t mid = blockIdx.x;
  if (route_idx >= routes || mid >= intermediate) {
    return;
  }

  const int tid = threadIdx.x;
  const size_t idx = route_idx * intermediate + mid;
  const size_t dest_row = static_cast<size_t>(row_indices[route_idx]);
  if (dest_row >= rows) {
    if (tid == 0) {
      activations[idx] = 0.0f;
    }
    return;
  }
  const uint16_t* route_hidden = hidden + dest_row * hidden_row_stride;

  const size_t packed_hidden_bytes = (hidden_dim + 1) / 2;
  const size_t hidden_scale_bytes = (hidden_dim + 15) / 16;
  const uint8_t* gate_row = gate_weight + mid * packed_hidden_bytes;
  const uint8_t* gate_scale_row = gate_scale + mid * hidden_scale_bytes;
  const uint8_t* up_row = up_weight + mid * packed_hidden_bytes;
  const uint8_t* up_scale_row = up_scale + mid * hidden_scale_bytes;

  float gate = 0.0f;
  float up = 0.0f;
  for (size_t col = static_cast<size_t>(tid); col < hidden_dim; col += blockDim.x) {
    const float hidden_value = bf16_to_f32(route_hidden[col]);
    gate = fmaf(hidden_value, packed_nvfp4_value(gate_row, gate_scale_row, col, gate_scale_2),
                gate);
    up = fmaf(hidden_value, packed_nvfp4_value(up_row, up_scale_row, col, up_scale_2), up);
  }
  gate_scratch[tid] = gate;
  up_scratch[tid] = up;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      gate_scratch[tid] += gate_scratch[tid + stride];
      up_scratch[tid] += up_scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float reduced_gate = gate_scratch[0];
    activations[idx] = reduced_gate * sigmoid_f32(reduced_gate) * up_scratch[0];
  }
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_grouped_down_accumulate_f32_reduce_kernel(
    const uint32_t* row_indices, const float* route_weights, const uint8_t* down_weight,
    const uint8_t* down_scale, const float* activations, float* accumulator, size_t rows,
    size_t routes, size_t intermediate, size_t output_dim, size_t down_weight_row_stride_bytes,
    size_t down_scale_row_stride_bytes, float down_scale_2) {
  __shared__ float scratch[kNvfp4RouteReduceBlock];
  const size_t route_idx = blockIdx.y;
  const size_t out_col = blockIdx.x;
  if (route_idx >= routes || out_col >= output_dim) {
    return;
  }

  const int tid = threadIdx.x;
  const size_t dest_row = static_cast<size_t>(row_indices[route_idx]);
  if (dest_row >= rows) {
    return;
  }

  const uint8_t* down_row = down_weight + out_col * down_weight_row_stride_bytes;
  const uint8_t* down_scale_row = down_scale + out_col * down_scale_row_stride_bytes;
  const float* route_activations = activations + route_idx * intermediate;
  float acc = 0.0f;
  for (size_t mid = static_cast<size_t>(tid); mid < intermediate; mid += blockDim.x) {
    const float down = packed_nvfp4_value(down_row, down_scale_row, mid, down_scale_2);
    acc = fmaf(route_activations[mid], down, acc);
  }
  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&accumulator[dest_row * output_dim + out_col], route_weights[route_idx] * scratch[0]);
  }
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_batched_activation_f32_reduce_kernel(
    const uint16_t* hidden, const uint32_t* row_indices,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activations, size_t rows,
    size_t routes, size_t hidden_dim, size_t hidden_row_stride, size_t max_intermediate) {
  __shared__ float gate_scratch[kNvfp4RouteReduceBlock];
  __shared__ float up_scratch[kNvfp4RouteReduceBlock];
  const size_t route_idx = blockIdx.y;
  const size_t mid = blockIdx.x;
  if (route_idx >= routes || mid >= max_intermediate) {
    return;
  }

  const int tid = threadIdx.x;
  const glmrt_nvfp4_route_batched_metadata_t metadata = route_metadata[route_idx];
  const size_t idx = route_idx * max_intermediate + mid;
  if (mid >= metadata.intermediate) {
    if (tid == 0) {
      activations[idx] = 0.0f;
    }
    return;
  }
  const size_t dest_row = static_cast<size_t>(row_indices[route_idx]);
  if (dest_row >= rows) {
    if (tid == 0) {
      activations[idx] = 0.0f;
    }
    return;
  }
  const uint16_t* route_hidden = hidden + dest_row * hidden_row_stride;

  const size_t packed_hidden_bytes = (hidden_dim + 1) / 2;
  const size_t hidden_scale_bytes = (hidden_dim + 15) / 16;
  const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(metadata.gate_weight);
  const uint8_t* gate_scale = reinterpret_cast<const uint8_t*>(metadata.gate_scale);
  const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(metadata.up_weight);
  const uint8_t* up_scale = reinterpret_cast<const uint8_t*>(metadata.up_scale);
  const uint8_t* gate_row = gate_weight + mid * packed_hidden_bytes;
  const uint8_t* gate_scale_row = gate_scale + mid * hidden_scale_bytes;
  const uint8_t* up_row = up_weight + mid * packed_hidden_bytes;
  const uint8_t* up_scale_row = up_scale + mid * hidden_scale_bytes;

  float gate = 0.0f;
  float up = 0.0f;
  for (size_t col = static_cast<size_t>(tid); col < hidden_dim; col += blockDim.x) {
    const float hidden_value = bf16_to_f32(route_hidden[col]);
    gate = fmaf(hidden_value,
                packed_nvfp4_value(gate_row, gate_scale_row, col, metadata.gate_scale_2), gate);
    up = fmaf(hidden_value, packed_nvfp4_value(up_row, up_scale_row, col, metadata.up_scale_2),
              up);
  }
  gate_scratch[tid] = gate;
  up_scratch[tid] = up;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      gate_scratch[tid] += gate_scratch[tid + stride];
      up_scratch[tid] += up_scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float reduced_gate = gate_scratch[0];
    activations[idx] = reduced_gate * sigmoid_f32(reduced_gate) * up_scratch[0];
  }
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_batched_down_accumulate_f32_reduce_kernel(
    const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, const float* activations,
    float* accumulator, size_t rows, size_t routes, size_t max_intermediate,
    size_t output_dim) {
  __shared__ float scratch[kNvfp4RouteReduceBlock];
  const size_t route_idx = blockIdx.y;
  const size_t out_col = blockIdx.x;
  if (route_idx >= routes || out_col >= output_dim) {
    return;
  }

  const int tid = threadIdx.x;
  const glmrt_nvfp4_route_batched_metadata_t metadata = route_metadata[route_idx];
  const size_t dest_row = static_cast<size_t>(row_indices[route_idx]);
  if (dest_row >= rows) {
    return;
  }

  const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(metadata.down_weight);
  const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(metadata.down_scale);
  const uint8_t* down_row = down_weight + out_col * metadata.down_weight_row_stride_bytes;
  const uint8_t* down_scale_row = down_scale + out_col * metadata.down_scale_row_stride_bytes;
  const float* route_activations = activations + route_idx * max_intermediate;
  float acc = 0.0f;
  for (size_t mid = static_cast<size_t>(tid); mid < metadata.intermediate; mid += blockDim.x) {
    const float down = packed_nvfp4_value(down_row, down_scale_row, mid, metadata.down_scale_2);
    acc = fmaf(route_activations[mid], down, acc);
  }
  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&accumulator[dest_row * output_dim + out_col], route_weights[route_idx] * scratch[0]);
  }
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_batched_down_accumulate_f32_single_row_kernel(
    const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, const float* activations,
    float* accumulator, size_t routes, size_t max_intermediate, size_t output_dim) {
  __shared__ float scratch[kNvfp4RouteReduceBlock];
  const size_t out_col = blockIdx.x;
  if (out_col >= output_dim) {
    return;
  }

  const int tid = threadIdx.x;
  float acc = 0.0f;
  for (size_t route_idx = 0; route_idx < routes; ++route_idx) {
    if (row_indices[route_idx] != 0) {
      continue;
    }
    const glmrt_nvfp4_route_batched_metadata_t metadata = route_metadata[route_idx];
    const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(metadata.down_weight);
    const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(metadata.down_scale);
    const uint8_t* down_row = down_weight + out_col * metadata.down_weight_row_stride_bytes;
    const uint8_t* down_scale_row = down_scale + out_col * metadata.down_scale_row_stride_bytes;
    const float* route_activations = activations + route_idx * max_intermediate;
    float route_acc = 0.0f;
    for (size_t mid = static_cast<size_t>(tid); mid < metadata.intermediate;
         mid += blockDim.x) {
      const float down = packed_nvfp4_value(down_row, down_scale_row, mid, metadata.down_scale_2);
      route_acc = fmaf(route_activations[mid], down, route_acc);
    }
    acc = fmaf(route_weights[route_idx], route_acc, acc);
  }

  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    accumulator[out_col] += scratch[0];
  }
}

__global__ void nvfp4_silu_gated_mlp_route_bf16_batched_down_output_bf16_single_row_kernel(
    const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, const float* activations,
    uint16_t* out, size_t routes, size_t max_intermediate, size_t output_dim) {
  __shared__ float scratch[kNvfp4RouteReduceBlock];
  const size_t out_col = blockIdx.x;
  if (out_col >= output_dim) {
    return;
  }

  const int tid = threadIdx.x;
  float acc = 0.0f;
  for (size_t route_idx = 0; route_idx < routes; ++route_idx) {
    if (row_indices[route_idx] != 0) {
      continue;
    }
    const glmrt_nvfp4_route_batched_metadata_t metadata = route_metadata[route_idx];
    const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(metadata.down_weight);
    const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(metadata.down_scale);
    const uint8_t* down_row = down_weight + out_col * metadata.down_weight_row_stride_bytes;
    const uint8_t* down_scale_row = down_scale + out_col * metadata.down_scale_row_stride_bytes;
    const float* route_activations = activations + route_idx * max_intermediate;
    float route_acc = 0.0f;
    for (size_t mid = static_cast<size_t>(tid); mid < metadata.intermediate;
         mid += blockDim.x) {
      const float down = packed_nvfp4_value(down_row, down_scale_row, mid, metadata.down_scale_2);
      route_acc = fmaf(route_activations[mid], down, route_acc);
    }
    acc = fmaf(route_weights[route_idx], route_acc, acc);
  }

  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[out_col] = f32_to_bf16(scratch[0]);
  }
}

glmrt_status_t validate_mlp_rows_args(const float* x, const float* gate_weight,
                                      const float* up_weight, const float* down_weight,
                                      const float* out, size_t rows, size_t hidden,
                                      size_t intermediate) {
  if (x == nullptr || gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden == 0 || intermediate == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden, &ignored) ||
      !checked_mul(intermediate, hidden, &ignored) ||
      !checked_mul(hidden, intermediate, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mlp_rows_bf16_args(const uint16_t* x, const uint16_t* gate_weight,
                                           const uint16_t* up_weight,
                                           const uint16_t* down_weight, const uint16_t* out,
                                           size_t rows, size_t hidden, size_t intermediate) {
  if (x == nullptr || gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden == 0 || intermediate == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden, &ignored) ||
      !checked_mul(intermediate, hidden, &ignored) ||
      !checked_mul(hidden, intermediate, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mlp_rows_bf16_down_stride_args(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, const uint16_t* out, size_t rows, size_t hidden,
    size_t intermediate, size_t down_stride) {
  if (x == nullptr || gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden == 0 || intermediate == 0 || down_stride < intermediate) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden, &ignored) ||
      !checked_mul(intermediate, hidden, &ignored) ||
      !checked_mul(hidden, down_stride, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_mlp_rows_bf16_down_stride_staged_args(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, const float* activation_workspace, const uint16_t* out,
    size_t rows, size_t hidden, size_t intermediate, size_t down_stride) {
  if (activation_workspace == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_mlp_rows_bf16_down_stride_args(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, down_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, intermediate, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_nvfp4_route_bf16_grouped_args(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    const uint16_t* out, size_t routes, size_t hidden_dim, size_t hidden_row_stride,
    size_t intermediate, size_t output_dim, size_t down_weight_row_stride_bytes,
    size_t down_scale_row_stride_bytes, float gate_scale_2, float up_scale_2,
    float down_scale_2) {
  if (hidden == nullptr || row_indices == nullptr || route_weights == nullptr ||
      gate_weight == nullptr || gate_scale == nullptr || up_weight == nullptr ||
      up_scale == nullptr || down_weight == nullptr || down_scale == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (routes == 0 || hidden_dim == 0 || hidden_row_stride < hidden_dim || intermediate == 0 ||
      output_dim == 0 || !std::isfinite(gate_scale_2) || !std::isfinite(up_scale_2) ||
      !std::isfinite(down_scale_2)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (hidden_dim > std::numeric_limits<size_t>::max() - 15 ||
      intermediate > std::numeric_limits<size_t>::max() - 15 ||
      !checked_mul(routes, output_dim, &ignored) ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t packed_hidden_bytes = (hidden_dim + 1) / 2;
  const size_t hidden_scale_bytes = (hidden_dim + 15) / 16;
  const size_t packed_intermediate_bytes = (intermediate + 1) / 2;
  const size_t intermediate_scale_bytes = (intermediate + 15) / 16;
  if (down_weight_row_stride_bytes < packed_intermediate_bytes ||
      down_scale_row_stride_bytes < intermediate_scale_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (!checked_mul(intermediate, packed_hidden_bytes, &ignored) ||
      !checked_mul(intermediate, hidden_scale_bytes, &ignored) ||
      !checked_mul(output_dim, down_weight_row_stride_bytes, &ignored) ||
      !checked_mul(output_dim, down_scale_row_stride_bytes, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_nvfp4_route_bf16_grouped_accumulate_f32_args(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    const float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2) {
  if (accumulator == nullptr || rows == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t grouped_valid = validate_nvfp4_route_bf16_grouped_args(
      hidden, row_indices, route_weights, gate_weight, gate_scale, up_weight, up_scale,
      down_weight, down_scale, reinterpret_cast<const uint16_t*>(accumulator), routes,
      hidden_dim, hidden_row_stride, intermediate, output_dim, down_weight_row_stride_bytes,
      down_scale_row_stride_bytes, gate_scale_2, up_scale_2, down_scale_2);
  if (grouped_valid != GLMRT_STATUS_OK) {
    return grouped_valid;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, output_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_nvfp4_route_bf16_grouped_staged_accumulate_f32_args(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    float* activation_workspace, const float* accumulator, size_t rows, size_t routes,
    size_t hidden_dim, size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2) {
  if (activation_workspace == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_nvfp4_route_bf16_grouped_accumulate_f32_args(
      hidden, row_indices, route_weights, gate_weight, gate_scale, up_weight, up_scale,
      down_weight, down_scale, accumulator, rows, routes, hidden_dim, hidden_row_stride,
      intermediate, output_dim, down_weight_row_stride_bytes, down_scale_row_stride_bytes,
      gate_scale_2, up_scale_2, down_scale_2);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t ignored = 0;
  if (!checked_mul(routes, intermediate, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_nvfp4_route_bf16_batched_staged_accumulate_f32_args(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    const float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim) {
  if (hidden == nullptr || row_indices == nullptr || route_weights == nullptr ||
      route_metadata == nullptr || activation_workspace == nullptr || accumulator == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || routes == 0 || hidden_dim == 0 || hidden_row_stride < hidden_dim ||
      max_intermediate == 0 || output_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_row_stride, &ignored) ||
      !checked_mul(routes, max_intermediate, &ignored) ||
      !checked_mul(rows, output_dim, &ignored) ||
      routes > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      max_intermediate > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      output_dim > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_nvfp4_route_bf16_batched_staged_single_row_bf16_args(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    const uint16_t* out, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim) {
  const glmrt_status_t valid = validate_nvfp4_route_bf16_batched_staged_accumulate_f32_args(
      hidden, row_indices, route_weights, route_metadata, activation_workspace,
      reinterpret_cast<const float*>(out), rows, routes, hidden_dim, hidden_row_stride,
      max_intermediate, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows != 1) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_mlp_down_stride_buffers(
    glmrt_device_buffer_t x, glmrt_device_buffer_t gate_weight, glmrt_device_buffer_t up_weight,
    glmrt_device_buffer_t down_weight, glmrt_device_buffer_t out, size_t rows, size_t hidden,
    size_t intermediate, size_t down_stride) {
  const glmrt_status_t valid = validate_mlp_rows_bf16_down_stride_args(
      static_cast<const uint16_t*>(x.ptr), static_cast<const uint16_t*>(gate_weight.ptr),
      static_cast<const uint16_t*>(up_weight.ptr), static_cast<const uint16_t*>(down_weight.ptr),
      static_cast<const uint16_t*>(out.ptr), rows, hidden, intermediate, down_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t row_values = 0;
  size_t gate_values = 0;
  size_t down_values = 0;
  if (!checked_mul(rows, hidden, &row_values) ||
      !checked_mul(intermediate, hidden, &gate_values) ||
      !checked_mul(hidden, down_stride, &down_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_bytes = 0;
  size_t gate_bytes = 0;
  size_t down_bytes = 0;
  if (!checked_mul(row_values, sizeof(uint16_t), &row_bytes) ||
      !checked_mul(gate_values, sizeof(uint16_t), &gate_bytes) ||
      !checked_mul(down_values, sizeof(uint16_t), &down_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (x.bytes < row_bytes || gate_weight.bytes < gate_bytes || up_weight.bytes < gate_bytes ||
      down_weight.bytes < down_bytes || out.bytes < row_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t gate_weight, glmrt_device_buffer_t up_weight,
    glmrt_device_buffer_t down_weight, glmrt_device_buffer_t out, size_t rows, size_t hidden,
    size_t intermediate, size_t down_stride) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_mlp_down_stride_buffers(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, down_stride);
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
  if (existing.func != reinterpret_cast<void*>(silu_gated_mlp_rows_bf16_down_stride_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* x_ptr = static_cast<const uint16_t*>(x.ptr);
  const uint16_t* gate_weight_ptr = static_cast<const uint16_t*>(gate_weight.ptr);
  const uint16_t* up_weight_ptr = static_cast<const uint16_t*>(up_weight.ptr);
  const uint16_t* down_weight_ptr = static_cast<const uint16_t*>(down_weight.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &x_ptr,
      &gate_weight_ptr,
      &up_weight_ptr,
      &down_weight_ptr,
      &out_ptr,
      &rows,
      &hidden,
      &intermediate,
      &down_stride,
  };
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(silu_gated_mlp_rows_bf16_down_stride_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_f32(const float* x, const float* gate_weight,
                                                        const float* up_weight,
                                                        const float* down_weight, float* out,
                                                        int hidden, int intermediate) {
  if (x == nullptr || gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr ||
      out == nullptr || hidden <= 0 || intermediate <= 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  silu_gated_mlp_f32_kernel<<<1, 1>>>(x, gate_weight, up_weight, down_weight, out, hidden,
                                      intermediate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_f32_async(
    const float* x, const float* gate_weight, const float* up_weight, const float* down_weight,
    float* out, size_t rows, size_t hidden, size_t intermediate, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_mlp_rows_args(x, gate_weight, up_weight, down_weight, out, rows, hidden,
                             intermediate);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  silu_gated_mlp_rows_f32_kernel<<<blocks, threads, 0, stream>>>(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_f32(
    const float* x, const float* gate_weight, const float* up_weight, const float* down_weight,
    float* out, size_t rows, size_t hidden, size_t intermediate) {
  const glmrt_status_t status = glmrt_cuda_silu_gated_mlp_rows_f32_async(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_mlp_rows_bf16_args(x, gate_weight, up_weight, down_weight, out, rows, hidden,
                                  intermediate);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  silu_gated_mlp_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate) {
  const glmrt_status_t status = glmrt_cuda_silu_gated_mlp_rows_bf16_async(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_silu_mul_bf16_async(
    const uint16_t* gate_up, uint16_t* out, size_t rows, size_t intermediate,
    void* cuda_stream) {
  if (gate_up == nullptr || out == nullptr || rows == 0 || intermediate == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t total = rows * intermediate;
  if (rows != 0 && total / rows != intermediate) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr int threads = 256;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  silu_mul_bf16_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
      gate_up, out, rows, intermediate);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_quantize_bf16_weight_nvfp4_async(
    glmrt_device_buffer_t input, glmrt_device_buffer_t packed,
    glmrt_device_buffer_t scales, size_t rows, size_t cols,
    float global_scale, void* cuda_stream) {
  if (rows == 0 || cols == 0 || cols % 16 != 0 ||
      !isfinite(global_scale) || global_scale <= 0.0f ||
      rows > std::numeric_limits<size_t>::max() / cols) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t values = rows * cols;
  if (input.ptr == nullptr || packed.ptr == nullptr || scales.ptr == nullptr ||
      input.bytes < values * sizeof(uint16_t) ||
      packed.bytes < values / 2 || scales.bytes < values / 16) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  if (values / 16 > std::numeric_limits<unsigned int>::max()) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  quantize_bf16_weight_nvfp4_kernel<<<
      static_cast<unsigned int>(values / 16), 16, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint16_t*>(input.ptr),
      static_cast<uint8_t*>(packed.ptr),
      static_cast<uint8_t*>(scales.ptr), cols, global_scale);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    size_t down_stride, void* cuda_stream) {
  const glmrt_status_t valid = validate_mlp_rows_bf16_down_stride_args(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, down_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  silu_gated_mlp_rows_bf16_down_stride_kernel<<<blocks, threads, 0, stream>>>(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, down_stride);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    size_t down_stride) {
  const glmrt_status_t status = glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_async(
      x, gate_weight, up_weight, down_weight, out, rows, hidden, intermediate, down_stride,
      nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, float* activation_workspace, uint16_t* out, size_t rows,
    size_t hidden, size_t intermediate, size_t down_stride, void* cuda_stream) {
  const glmrt_status_t valid = validate_mlp_rows_bf16_down_stride_staged_args(
      x, gate_weight, up_weight, down_weight, activation_workspace, out, rows, hidden,
      intermediate, down_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      intermediate > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = kNvfp4RouteReduceBlock;
  const dim3 activation_grid(static_cast<unsigned int>(intermediate),
                             static_cast<unsigned int>(rows));
  silu_gated_mlp_rows_bf16_activation_f32_reduce_kernel<<<activation_grid, threads, 0, stream>>>(
      x, gate_weight, up_weight, activation_workspace, rows, hidden, intermediate);
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  silu_gated_mlp_rows_bf16_down_from_activation_kernel<<<blocks, threads, 0, stream>>>(
      down_weight, activation_workspace, out, rows, hidden, intermediate, down_stride);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, float* activation_workspace, uint16_t* out, size_t rows,
    size_t hidden, size_t intermediate, size_t down_stride) {
  const glmrt_status_t status = glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
      x, gate_weight, up_weight, down_weight, activation_workspace, out, rows, hidden,
      intermediate, down_stride, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t
glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    float* activation_workspace, float* accumulator, size_t rows, size_t routes,
    size_t hidden_dim, size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2, void* cuda_stream) {
  const glmrt_status_t valid = validate_nvfp4_route_bf16_grouped_staged_accumulate_f32_args(
      hidden, row_indices, route_weights, gate_weight, gate_scale, up_weight, up_scale,
      down_weight, down_scale, activation_workspace, accumulator, rows, routes, hidden_dim,
      hidden_row_stride, intermediate, output_dim, down_weight_row_stride_bytes,
      down_scale_row_stride_bytes, gate_scale_2, up_scale_2, down_scale_2);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const int threads = kNvfp4RouteReduceBlock;
  size_t activation_total = 0;
  size_t output_total = 0;
  if (!checked_mul(routes, intermediate, &activation_total) ||
      !checked_mul(routes, output_dim, &output_total)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (routes > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      intermediate > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      output_dim > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const dim3 activation_grid(static_cast<unsigned int>(intermediate),
                             static_cast<unsigned int>(routes));
  const dim3 output_grid(static_cast<unsigned int>(output_dim), static_cast<unsigned int>(routes));
  nvfp4_silu_gated_mlp_route_bf16_grouped_activation_f32_reduce_kernel<<<
      activation_grid, threads, 0, stream>>>(
      hidden, row_indices, gate_weight, gate_scale, up_weight, up_scale, activation_workspace,
      rows, routes, hidden_dim, hidden_row_stride, intermediate, gate_scale_2, up_scale_2);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  nvfp4_silu_gated_mlp_route_bf16_grouped_down_accumulate_f32_reduce_kernel<<<
      output_grid, threads, 0, stream>>>(
      row_indices, route_weights, down_weight, down_scale, activation_workspace, accumulator, rows,
      routes, intermediate, output_dim, down_weight_row_stride_bytes,
      down_scale_row_stride_bytes, down_scale_2);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    float* activation_workspace, float* accumulator, size_t rows, size_t routes,
    size_t hidden_dim, size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2) {
  const glmrt_status_t status =
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
          hidden, row_indices, route_weights, gate_weight, gate_scale, up_weight, up_scale,
          down_weight, down_scale, activation_workspace, accumulator, rows, routes, hidden_dim,
          hidden_row_stride, intermediate, output_dim, down_weight_row_stride_bytes,
          down_scale_row_stride_bytes, gate_scale_2, up_scale_2, down_scale_2, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t
glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim, void* cuda_stream) {
  const glmrt_status_t valid = validate_nvfp4_route_bf16_batched_staged_accumulate_f32_args(
      hidden, row_indices, route_weights, route_metadata, activation_workspace, accumulator, rows,
      routes, hidden_dim, hidden_row_stride, max_intermediate, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const int threads = kNvfp4RouteReduceBlock;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const dim3 activation_grid(static_cast<unsigned int>(max_intermediate),
                             static_cast<unsigned int>(routes));
  nvfp4_silu_gated_mlp_route_bf16_batched_activation_f32_reduce_kernel<<<
      activation_grid, threads, 0, stream>>>(
      hidden, row_indices, route_metadata, activation_workspace, rows, routes, hidden_dim,
      hidden_row_stride, max_intermediate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  if (rows == 1) {
    const dim3 output_grid(static_cast<unsigned int>(output_dim));
    nvfp4_silu_gated_mlp_route_bf16_batched_down_accumulate_f32_single_row_kernel<<<
        output_grid, threads, 0, stream>>>(
        row_indices, route_weights, route_metadata, activation_workspace, accumulator, routes,
        max_intermediate, output_dim);
  } else {
    const dim3 output_grid(static_cast<unsigned int>(output_dim),
                           static_cast<unsigned int>(routes));
    nvfp4_silu_gated_mlp_route_bf16_batched_down_accumulate_f32_reduce_kernel<<<
        output_grid, threads, 0, stream>>>(
        row_indices, route_weights, route_metadata, activation_workspace, accumulator, rows,
        routes, max_intermediate, output_dim);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim) {
  const glmrt_status_t status =
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
          hidden, row_indices, route_weights, route_metadata, activation_workspace, accumulator,
          rows, routes, hidden_dim, hidden_row_stride, max_intermediate, output_dim, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t
glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    uint16_t* out, size_t rows, size_t routes, size_t hidden_dim, size_t hidden_row_stride,
    size_t max_intermediate, size_t output_dim, void* cuda_stream) {
  const glmrt_status_t valid = validate_nvfp4_route_bf16_batched_staged_single_row_bf16_args(
      hidden, row_indices, route_weights, route_metadata, activation_workspace, out, rows, routes,
      hidden_dim, hidden_row_stride, max_intermediate, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const int threads = kNvfp4RouteReduceBlock;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const dim3 activation_grid(static_cast<unsigned int>(max_intermediate),
                             static_cast<unsigned int>(routes));
  nvfp4_silu_gated_mlp_route_bf16_batched_activation_f32_reduce_kernel<<<
      activation_grid, threads, 0, stream>>>(
      hidden, row_indices, route_metadata, activation_workspace, rows, routes, hidden_dim,
      hidden_row_stride, max_intermediate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  const dim3 output_grid(static_cast<unsigned int>(output_dim));
  nvfp4_silu_gated_mlp_route_bf16_batched_down_output_bf16_single_row_kernel<<<
      output_grid, threads, 0, stream>>>(
      row_indices, route_weights, route_metadata, activation_workspace, out, routes,
      max_intermediate, output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    uint16_t* out, size_t rows, size_t routes, size_t hidden_dim, size_t hidden_row_stride,
    size_t max_intermediate, size_t output_dim) {
  const glmrt_status_t status =
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
          hidden, row_indices, route_weights, route_metadata, activation_workspace, out, rows,
          routes, hidden_dim, hidden_row_stride, max_intermediate, output_dim, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
