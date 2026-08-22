#include "common.h"

#include "b12x_coordinator_aot_config.h"
#include "coordinator_w4a16_o_proj_m1.h"
#include "coordinator_w4a16_o_proj_m16_candidate.h"
#include "coordinator_w4a16_o_proj_m1_tn64_candidate.h"
#include "coordinator_w4a16_q_b_m16_candidate.h"
#include "coordinator_w4a16_q_b_m8.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace {

constexpr size_t kRouteSlots = 8;
constexpr size_t kMaxQueryRows = 8;
constexpr size_t kM16CandidateRouteSlots = 16;
constexpr size_t kM16CandidateRouteBlocks = 2;
constexpr size_t kScratchElements = 2'097'152;
constexpr size_t kLockElements = 1'024;
constexpr int kQuantThreads = 32;

glmrt_b12x_coordinator_w4a16_q_b_m8_Kernel_Module_t q_b_module;
glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Kernel_Module_t
    q_b_m16_candidate_module;
glmrt_b12x_coordinator_w4a16_o_proj_m1_Kernel_Module_t o_proj_module;
glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Kernel_Module_t
    o_proj_m16_candidate_module;
glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Kernel_Module_t
    o_proj_tn64_candidate_module;
std::once_flag module_init_once;
glmrt_status_t module_init_status = GLMRT_STATUS_OK;
std::once_flag q_b_m16_candidate_init_once;
glmrt_status_t q_b_m16_candidate_init_status = GLMRT_STATUS_OK;
std::once_flag o_proj_m16_candidate_init_once;
glmrt_status_t o_proj_m16_candidate_init_status = GLMRT_STATUS_OK;

bool buffer_has_bytes(glmrt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

__device__ uint8_t fp4_e2m1_code(float value) {
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

__global__ void reduce_bf16_max_abs_kernel(const uint16_t* input, size_t values,
                                           float* maximum) {
  float local_maximum = 0.0f;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < values; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    local_maximum = fmaxf(local_maximum, fabsf(bf16_to_f32(input[index])));
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_maximum = fmaxf(local_maximum, __shfl_down_sync(0xffffffffU, local_maximum, offset));
  }
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicMax(reinterpret_cast<unsigned int*>(maximum), __float_as_uint(local_maximum));
  }
}

__global__ void prepare_tensor_scale_kernel(float* tensor_scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float maximum = tensor_scale[0];
    tensor_scale[0] = maximum > 0.0f ? maximum / (6.0f * 448.0f) : 1.0f;
  }
}

__global__ void quantize_bf16_nvfp4_normalized_payload_kernel(
    const uint16_t* input, uint8_t* payload, const float* tensor_scale,
    size_t rows, size_t cols) {
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
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffU, maximum, offset));
  }
  const size_t packed_row_bytes = cols / 2;
  const size_t scale_cols = cols / 16;
  uint8_t* row_payload = payload + row * (packed_row_bytes + scale_cols);
  if (lane == 0) {
    const float global = tensor_scale[0];
    const uint8_t scale_byte = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        maximum / (6.0f * global), __NV_SATFINITE, __NV_E4M3));
    const float decoded_scale = f8e4m3_to_f32(scale_byte) * global;
    inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    row_payload[packed_row_bytes + col_block] = scale_byte;
  }
  __syncthreads();
  if (lane < 16) {
    const float quantized = fminf(fmaxf(values[lane] * inverse_scale, -6.0f), 6.0f);
    codes[lane] = fp4_e2m1_code(quantized);
  }
  __syncthreads();
  if (lane < 8) {
    row_payload[col_block * 8 + lane] =
        static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
  }
}

__global__ void finalize_tensor_scale_kernel(float* tensor_scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    tensor_scale[0] = ldexpf(tensor_scale[0], 119);
  }
}

__global__ void pack_payload_weight_kernel(const uint8_t* payload, uint32_t* destination,
                                           size_t size_k, size_t size_n) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t k_tiles = size_k / 16;
  const size_t n_tiles = size_n / 64;
  const size_t output_words = k_tiles * n_tiles * 128;
  if (output_index >= output_words) {
    return;
  }

  const size_t packed_position = output_index % 128;
  const size_t tile_index = output_index / 128;
  const size_t n_tile = tile_index % n_tiles;
  const size_t k_tile = tile_index / n_tiles;
  const size_t thread_group = packed_position / 4;
  const size_t warp_column = packed_position % 4;
  const size_t tensor_column = thread_group / 4;
  const size_t tensor_row = (thread_group % 4) * 2;
  constexpr int element_offsets[4] = {0, 1, 8, 9};
  constexpr int pack_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const size_t payload_stride = size_k / 2 + size_k / 16;
  uint32_t result = 0;
  for (int slot = 0; slot < 8; ++slot) {
    const int source_slot = pack_order[slot];
    const int element_slot = source_slot & 3;
    const size_t element = tensor_row + element_offsets[element_slot];
    const size_t k_half = element / 8;
    const size_t nibble = element % 8;
    const size_t column_base = warp_column * 16 + tensor_column;
    const size_t source_row = n_tile * 64 + column_base + (source_slot >= 4 ? 8 : 0);
    const size_t source_word = k_tile * 2 + k_half;
    const uint32_t word = reinterpret_cast<const uint32_t*>(
        payload + source_row * payload_stride)[source_word];
    result |= ((word >> (nibble * 4)) & 0x0fU) << (slot * 4);
  }
  destination[output_index] = result;
}

__global__ void pack_payload_scale_kernel(const uint8_t* payload, uint8_t* destination,
                                          size_t size_k, size_t size_n) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = (size_k / 16) * size_n;
  if (output_index >= values) {
    return;
  }

  const size_t k_block = output_index / size_n;
  const size_t output_row = output_index % size_n;
  constexpr int swap_four[4] = {0, 2, 1, 3};
  const size_t swapped = (output_row & ~size_t{3}) + swap_four[output_row & 3];
  const size_t group_base = (swapped / 64) * 64;
  const size_t group_offset = swapped % 64;
  const size_t source_row = group_base + group_offset / 8 + 8 * (group_offset % 8);
  const size_t packed_row_bytes = size_k / 2;
  const size_t payload_stride = packed_row_bytes + size_k / 16;
  const uint8_t source_scale =
      payload[source_row * payload_stride + packed_row_bytes + k_block];
  const float adjusted = f8e4m3_to_f32(source_scale) * 128.0f;
  if (adjusted < 2.0f) {
    destination[output_index] = 0;
    return;
  }
  const __half_raw encoded = __float2half_rn(adjusted);
  destination[output_index] = static_cast<uint8_t>((encoded.x >> 7) & 0xffU);
}

__global__ void initialize_launch_metadata_kernel(
    int32_t* packed_route_indices, int32_t* block_expert_ids,
    int32_t* packed_route_count, float* topk_weights) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (size_t index = 0; index < kRouteSlots; ++index) {
      packed_route_indices[index] = static_cast<int32_t>(index);
      topk_weights[index] = 1.0f;
    }
    block_expert_ids[0] = 0;
    packed_route_count[0] = static_cast<int32_t>(kRouteSlots);
  }
}

void initialize_modules() {
  glmrt_b12x_coordinator_w4a16_q_b_m8_Kernel_Module_Load(&q_b_module);
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Kernel_Module_Load(&o_proj_module);
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Kernel_Module_Load(
      &o_proj_tn64_candidate_module);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    module_init_status = GLMRT_STATUS_INTERNAL_ERROR;
  }
}

void initialize_q_b_m16_candidate_module() {
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Kernel_Module_Load(
      &q_b_m16_candidate_module);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    q_b_m16_candidate_init_status = GLMRT_STATUS_INTERNAL_ERROR;
  }
}

void initialize_o_proj_m16_candidate_module() {
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Kernel_Module_Load(
      &o_proj_m16_candidate_module);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    o_proj_m16_candidate_init_status = GLMRT_STATUS_INTERNAL_ERROR;
  }
}

glmrt_status_t validate_launch_buffers(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t size_k,
    size_t size_n, size_t active_rows) {
  if (buffers == nullptr || active_rows == 0 || active_rows > kMaxQueryRows ||
      size_k > std::numeric_limits<size_t>::max() / size_n ||
      active_rows > std::numeric_limits<size_t>::max() / size_k ||
      active_rows > std::numeric_limits<size_t>::max() / size_n) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      buffer_has_bytes(buffers->input, active_rows * size_k * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->weight, size_n * size_k / 2) &&
      buffer_has_bytes(buffers->output, active_rows * size_n * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->scale, size_n * size_k / 16) &&
      buffer_has_bytes(buffers->global_scale, sizeof(float)) &&
      buffer_has_bytes(buffers->packed_route_indices, kRouteSlots * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->block_expert_ids, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, kRouteSlots * sizeof(float)) &&
      buffer_has_bytes(buffers->c_tmp, kScratchElements * sizeof(float)) &&
      buffer_has_bytes(buffers->locks, kLockElements * sizeof(int32_t));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t validate_m16_candidate_buffers(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t size_k,
    size_t size_n, size_t active_rows) {
  if (buffers == nullptr || active_rows == 0 ||
      active_rows > kM16CandidateRouteSlots) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      buffer_has_bytes(buffers->input, active_rows * size_k * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->weight, size_n * size_k / 2) &&
      buffer_has_bytes(buffers->output, active_rows * size_n * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->scale, size_n * size_k / 16) &&
      buffer_has_bytes(buffers->global_scale, sizeof(float)) &&
      buffer_has_bytes(buffers->packed_route_indices,
                       kM16CandidateRouteSlots * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->block_expert_ids,
                       kM16CandidateRouteBlocks * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights,
                       kM16CandidateRouteSlots * sizeof(float)) &&
      buffer_has_bytes(buffers->c_tmp, kScratchElements * sizeof(float)) &&
      buffer_has_bytes(buffers->locks, kLockElements * sizeof(int32_t));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t reset_w4a16_locks_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, cudaStream_t stream) {
  return status_from_cuda(
      cudaMemsetAsync(buffers->locks.ptr, 0, kLockElements * sizeof(int32_t), stream));
}

glmrt_status_t check_aot_launch(int result, const char* label) {
  if (result == 0) {
    return GLMRT_STATUS_OK;
  }
  glmrt_set_last_error_message(label);
  return GLMRT_STATUS_INTERNAL_ERROR;
}

int launch_q_b(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
               size_t active_rows, cudaStream_t stream) {
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_b_i32_flat_t weight{buffers->weight.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_scales_i32_flat_t scale{buffers->scale.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_global_scale_t global_scale{
      buffers->global_scale.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_packed_route_indices_t routes{
      buffers->packed_route_indices.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_topk_weights_flat_t topk_weights{
      buffers->topk_weights.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_c_tmp_f32_flat_t scratch{
      buffers->c_tmp.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_locks_i32_flat_t locks{
      buffers->locks.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m8_Tensor_trellis_lut_flat_t trellis_lut{
      buffers->scale.ptr};
  return cute_dsl_glmrt_b12x_coordinator_w4a16_q_b_m8_wrapper(
      &q_b_module, buffers->input.ptr, buffers->input.ptr, &weight,
      buffers->output.ptr, &scale, &global_scale, &routes,
      &block_experts, &route_count, &topk_weights, &scratch, &locks, &trellis_lut,
      static_cast<int32_t>(active_rows), GLMRT_B12X_COORDINATOR_Q_B_M8_GRID_X,
      stream);
}

int launch_q_b_m16_candidate(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    cudaStream_t stream) {
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_b_i32_flat_t weight{
      buffers->weight.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_scales_i32_flat_t scale{
      buffers->scale.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_global_scale_t global_scale{
      buffers->global_scale.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_packed_route_indices_t routes{
      buffers->packed_route_indices.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_topk_weights_flat_t topk_weights{
      buffers->topk_weights.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_c_tmp_f32_flat_t scratch{
      buffers->c_tmp.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_locks_i32_flat_t locks{
      buffers->locks.ptr};
  glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_Tensor_trellis_lut_flat_t trellis_lut{
      buffers->scale.ptr};
  return cute_dsl_glmrt_b12x_coordinator_w4a16_q_b_m16_candidate_wrapper(
      &q_b_m16_candidate_module, buffers->input.ptr, buffers->input.ptr, &weight,
      buffers->output.ptr, &scale, &global_scale,
      &routes, &block_experts, &route_count, &topk_weights, &scratch, &locks,
      &trellis_lut,
      static_cast<int32_t>(active_rows),
      GLMRT_B12X_COORDINATOR_Q_B_M16_CANDIDATE_GRID_X, stream);
}

int launch_o_proj_m16_candidate(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    cudaStream_t stream) {
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_b_i32_flat_t weight{
      buffers->weight.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_scales_i32_flat_t scale{
      buffers->scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_global_scale_t global_scale{
      buffers->global_scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_packed_route_indices_t routes{
      buffers->packed_route_indices.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_topk_weights_flat_t topk_weights{
      buffers->topk_weights.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_c_tmp_f32_flat_t scratch{
      buffers->c_tmp.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_locks_i32_flat_t locks{
      buffers->locks.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_Tensor_trellis_lut_flat_t
      trellis_lut{buffers->scale.ptr};
  return cute_dsl_glmrt_b12x_coordinator_w4a16_o_proj_m16_candidate_wrapper(
      &o_proj_m16_candidate_module, buffers->input.ptr, buffers->input.ptr,
      &weight, buffers->output.ptr, &scale,
      &global_scale, &routes, &block_experts, &route_count, &topk_weights,
      &scratch, &locks, &trellis_lut, static_cast<int32_t>(active_rows),
      GLMRT_B12X_COORDINATOR_O_PROJ_M16_CANDIDATE_GRID_X, stream);
}

int launch_o_proj(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
                  cudaStream_t stream) {
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_b_i32_flat_t weight{buffers->weight.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_scales_i32_flat_t scale{
      buffers->scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_global_scale_t global_scale{
      buffers->global_scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_packed_route_indices_t routes{
      buffers->packed_route_indices.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_topk_weights_flat_t topk_weights{
      buffers->topk_weights.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_c_tmp_f32_flat_t scratch{
      buffers->c_tmp.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_locks_i32_flat_t locks{
      buffers->locks.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_Tensor_trellis_lut_flat_t trellis_lut{
      buffers->scale.ptr};
  return cute_dsl_glmrt_b12x_coordinator_w4a16_o_proj_m1_wrapper(
      &o_proj_module, buffers->input.ptr, buffers->input.ptr, &weight,
      buffers->output.ptr, &scale, &global_scale, &routes,
      &block_experts, &route_count, &topk_weights, &scratch, &locks, &trellis_lut, 1,
      GLMRT_B12X_COORDINATOR_O_PROJ_M1_GRID_X, stream);
}

int launch_o_proj_tn64_candidate(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, cudaStream_t stream) {
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_b_i32_flat_t weight{
      buffers->weight.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_scales_i32_flat_t scale{
      buffers->scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_global_scale_t global_scale{
      buffers->global_scale.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_packed_route_indices_t routes{
      buffers->packed_route_indices.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_block_expert_ids_t
      block_experts{buffers->block_expert_ids.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_packed_route_count_t
      route_count{buffers->packed_route_count.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_topk_weights_flat_t
      topk_weights{buffers->topk_weights.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_c_tmp_f32_flat_t scratch{
      buffers->c_tmp.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_locks_i32_flat_t locks{
      buffers->locks.ptr};
  glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_Tensor_trellis_lut_flat_t
      trellis_lut{buffers->scale.ptr};
  return cute_dsl_glmrt_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_wrapper(
      &o_proj_tn64_candidate_module, buffers->input.ptr, buffers->input.ptr,
      &weight, buffers->output.ptr, &scale, &global_scale, &routes,
      &block_experts, &route_count, &topk_weights, &scratch, &locks, &trellis_lut, 1,
      GLMRT_B12X_COORDINATOR_O_PROJ_M1_TN64_CANDIDATE_GRID_X, stream);
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_b12x_coordinator_aot_available(
    int* out_available) {
  if (out_available == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_b12x_coordinator_aot_init(void) {
  std::call_once(module_init_once, initialize_modules);
  return module_init_status;
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
    glmrt_device_buffer_t input_bf16, glmrt_device_buffer_t payload_scratch,
    glmrt_device_buffer_t packed_weight, glmrt_device_buffer_t packed_scale,
    glmrt_device_buffer_t global_scale, size_t size_k, size_t size_n,
    void* cuda_stream) {
  if (size_k == 0 || size_n == 0 || size_k % 16 != 0 || size_n % 64 != 0 ||
      size_n > std::numeric_limits<size_t>::max() / size_k) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t input_bytes = size_n * size_k * sizeof(uint16_t);
  const size_t weight_bytes = size_n * size_k / 2;
  const size_t scale_bytes = size_n * size_k / 16;
  const size_t payload_bytes = weight_bytes + scale_bytes;
  if (!buffer_has_bytes(input_bf16, input_bytes) ||
      !buffer_has_bytes(payload_scratch, payload_bytes) ||
      !buffer_has_bytes(packed_weight, weight_bytes) ||
      !buffer_has_bytes(packed_scale, scale_bytes) ||
      !buffer_has_bytes(global_scale, sizeof(float))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(global_scale.ptr, 0, sizeof(float), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  constexpr size_t threads = 256;
  constexpr size_t max_reduction_blocks = 4'096;
  const size_t values = size_n * size_k;
  const size_t reduction_blocks =
      std::min(max_reduction_blocks, (values + threads - 1) / threads);
  reduce_bf16_max_abs_kernel<<<static_cast<unsigned int>(reduction_blocks), threads, 0,
                               stream>>>(
      static_cast<const uint16_t*>(input_bf16.ptr), values,
      static_cast<float*>(global_scale.ptr));
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  prepare_tensor_scale_kernel<<<1, 1, 0, stream>>>(
      static_cast<float*>(global_scale.ptr));
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  const dim3 quant_grid(static_cast<unsigned int>(size_k / 16),
                        static_cast<unsigned int>(size_n));
  quantize_bf16_nvfp4_normalized_payload_kernel<<<quant_grid, kQuantThreads, 0, stream>>>(
      static_cast<const uint16_t*>(input_bf16.ptr),
      static_cast<uint8_t*>(payload_scratch.ptr),
      static_cast<const float*>(global_scale.ptr), size_n, size_k);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  const size_t words = weight_bytes / sizeof(uint32_t);
  pack_payload_weight_kernel<<<static_cast<unsigned int>((words + threads - 1) / threads),
                               threads, 0, stream>>>(
      static_cast<const uint8_t*>(payload_scratch.ptr),
      static_cast<uint32_t*>(packed_weight.ptr), size_k, size_n);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  pack_payload_scale_kernel<<<
      static_cast<unsigned int>((scale_bytes + threads - 1) / threads), threads, 0,
      stream>>>(static_cast<const uint8_t*>(payload_scratch.ptr),
                static_cast<uint8_t*>(packed_scale.ptr), size_k, size_n);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  finalize_tensor_scale_kernel<<<1, 1, 0, stream>>>(
      static_cast<float*>(global_scale.ptr));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream) {
  if (buffers == nullptr || !buffer_has_bytes(buffers->global_scale, sizeof(float)) ||
      !buffer_has_bytes(buffers->packed_route_indices, kRouteSlots * sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->block_expert_ids, sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->topk_weights, kRouteSlots * sizeof(float)) ||
      !buffer_has_bytes(buffers->locks, kLockElements * sizeof(int32_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(buffers->locks.ptr, 0, kLockElements * sizeof(int32_t),
                                      stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  initialize_launch_metadata_kernel<<<1, 1, 0, stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr));
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_launch_buffers(
      buffers, GLMRT_B12X_COORDINATOR_Q_B_M8_SIZE_K,
      GLMRT_B12X_COORDINATOR_Q_B_M8_SIZE_N, active_rows);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_coordinator_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const glmrt_status_t reset = reset_w4a16_locks_async(buffers, stream);
  if (reset != GLMRT_STATUS_OK) {
    return reset;
  }
  return check_aot_launch(launch_q_b(buffers, active_rows, stream),
                          "B12X coordinator Q-B W4A16 launch failed");
}

extern "C" glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_q_b_m1_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream) {
  return glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async(buffers, 1, cuda_stream);
}

// Benchmark-only dSpark target-verifier candidate; serving does not reference it.
extern "C" glmrt_status_t
glmrt_cuda_b12x_coordinator_w4a16_q_b_m16_candidate_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_m16_candidate_buffers(
          buffers, GLMRT_B12X_COORDINATOR_Q_B_M16_CANDIDATE_SIZE_K,
          GLMRT_B12X_COORDINATOR_Q_B_M16_CANDIDATE_SIZE_N, active_rows);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  std::call_once(q_b_m16_candidate_init_once,
                 initialize_q_b_m16_candidate_module);
  if (q_b_m16_candidate_init_status != GLMRT_STATUS_OK) {
    return q_b_m16_candidate_init_status;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const glmrt_status_t reset = reset_w4a16_locks_async(buffers, stream);
  if (reset != GLMRT_STATUS_OK) {
    return reset;
  }
  return check_aot_launch(
      launch_q_b_m16_candidate(buffers, active_rows, stream),
      "B12X coordinator Q-B M16 candidate launch failed");
}

// Benchmark-only dSpark target-verifier candidate; serving does not reference it.
extern "C" glmrt_status_t
glmrt_cuda_b12x_coordinator_w4a16_o_proj_m16_candidate_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    void* cuda_stream) {
  const glmrt_status_t valid =
      validate_m16_candidate_buffers(
          buffers, GLMRT_B12X_COORDINATOR_O_PROJ_M16_CANDIDATE_SIZE_K,
          GLMRT_B12X_COORDINATOR_O_PROJ_M16_CANDIDATE_SIZE_N, active_rows);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  std::call_once(o_proj_m16_candidate_init_once,
                 initialize_o_proj_m16_candidate_module);
  if (o_proj_m16_candidate_init_status != GLMRT_STATUS_OK) {
    return o_proj_m16_candidate_init_status;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const glmrt_status_t reset = reset_w4a16_locks_async(buffers, stream);
  if (reset != GLMRT_STATUS_OK) {
    return reset;
  }
  return check_aot_launch(
      launch_o_proj_m16_candidate(buffers, active_rows, stream),
      "B12X coordinator O-projection M16 candidate launch failed");
}

extern "C" glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream) {
  const glmrt_status_t valid = validate_launch_buffers(
      buffers, GLMRT_B12X_COORDINATOR_O_PROJ_M1_SIZE_K,
      GLMRT_B12X_COORDINATOR_O_PROJ_M1_SIZE_N, 1);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_coordinator_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const glmrt_status_t reset = reset_w4a16_locks_async(buffers, stream);
  if (reset != GLMRT_STATUS_OK) {
    return reset;
  }
  return check_aot_launch(
      launch_o_proj(buffers, stream),
      "B12X coordinator O-projection W4A16 launch failed");
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream) {
  const glmrt_status_t valid = validate_launch_buffers(
      buffers, GLMRT_B12X_COORDINATOR_O_PROJ_M1_TN64_CANDIDATE_SIZE_K,
      GLMRT_B12X_COORDINATOR_O_PROJ_M1_TN64_CANDIDATE_SIZE_N, 1);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_coordinator_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const glmrt_status_t reset = reset_w4a16_locks_async(buffers, stream);
  if (reset != GLMRT_STATUS_OK) {
    return reset;
  }
  return check_aot_launch(
      launch_o_proj_tn64_candidate(buffers, stream),
      "B12X coordinator O-projection TN64 candidate launch failed");
}
