#include "common.h"

#include "w8a16_packed_o_aot_config.h"
#include "w8a16_packed_o_m16.h"
#include "w8a16_packed_o_m32.h"
#include "w8a16_packed_o_m64.h"
#include "w8a16_packed_o_m128.h"
#include "w8a16_packed_o_m256.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>

namespace {

constexpr size_t kSizeN = 6'144;
constexpr size_t kSizeK = 16'384;
constexpr size_t kScaleGroups = kSizeK / 256;
constexpr size_t kLockElements = 1'024;

glmrt_w8a16_packed_o_m16_Kernel_Module_t module_m16;
glmrt_w8a16_packed_o_m32_Kernel_Module_t module_m32;
glmrt_w8a16_packed_o_m64_Kernel_Module_t module_m64;
glmrt_w8a16_packed_o_m128_Kernel_Module_t module_m128;
glmrt_w8a16_packed_o_m256_Kernel_Module_t module_m256;
std::once_flag module_init_once;
glmrt_status_t module_init_status = GLMRT_STATUS_OK;

bool buffer_has_bytes(glmrt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

size_t round_up_rows(size_t rows, size_t block_m) {
  return ((rows + block_m - 1) / block_m) * block_m;
}

__global__ void initialize_packed_o_metadata_kernel(
    float* global_scale, int32_t* routes, int32_t* block_experts,
    int32_t* route_count, float* topk_weights, size_t route_slots,
    size_t route_blocks) {
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < route_slots; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    routes[index] = static_cast<int32_t>(index);
    topk_weights[index] = 1.0f;
  }
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < route_blocks; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    block_experts[index] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    global_scale[0] = 1.0f;
    route_count[0] = static_cast<int32_t>(route_slots);
  }
}

void initialize_modules() {
  glmrt_w8a16_packed_o_m16_Kernel_Module_Load(&module_m16);
  glmrt_w8a16_packed_o_m32_Kernel_Module_Load(&module_m32);
  glmrt_w8a16_packed_o_m64_Kernel_Module_Load(&module_m64);
  glmrt_w8a16_packed_o_m128_Kernel_Module_Load(&module_m128);
  glmrt_w8a16_packed_o_m256_Kernel_Module_Load(&module_m256);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    module_init_status = status_from_cuda(error);
  }
}

glmrt_status_t validate_buffers(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t rows,
    size_t block_m) {
  if (buffers == nullptr || rows == 0 || rows > 256 || block_m == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t route_slots = round_up_rows(rows, block_m);
  const size_t route_blocks = route_slots / block_m;
  const size_t scratch_elements = std::max(
      kSizeN * route_slots,
      static_cast<size_t>(4) * 256 * block_m * 256);
  if (!buffer_has_bytes(buffers->input, rows * kSizeK * sizeof(uint16_t)) ||
      !buffer_has_bytes(buffers->weight, kSizeN * kSizeK) ||
      !buffer_has_bytes(buffers->output, rows * kSizeN * sizeof(uint16_t)) ||
      !buffer_has_bytes(buffers->scale,
                        kSizeN * kScaleGroups * sizeof(float)) ||
      !buffer_has_bytes(buffers->global_scale, sizeof(float)) ||
      !buffer_has_bytes(buffers->packed_route_indices,
                        route_slots * sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->block_expert_ids,
                        route_blocks * sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->topk_weights,
                        route_slots * sizeof(float)) ||
      !buffer_has_bytes(buffers->c_tmp, scratch_elements * sizeof(float)) ||
      !buffer_has_bytes(buffers->locks, kLockElements * sizeof(int32_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

#define GLMRT_PACKED_O_ARGS(PREFIX)                                           \
  PREFIX##_Tensor_a_bf16_flat_t input{buffers->input.ptr};                    \
  PREFIX##_Tensor_b_i32_flat_t weight{buffers->weight.ptr};                   \
  PREFIX##_Tensor_c_bf16_flat_t output{buffers->output.ptr};                  \
  PREFIX##_Tensor_scales_f32_flat_t scales{buffers->scale.ptr};               \
  PREFIX##_Tensor_global_scale_t global_scale{buffers->global_scale.ptr};     \
  PREFIX##_Tensor_packed_route_indices_t routes{                              \
      buffers->packed_route_indices.ptr};                                     \
  PREFIX##_Tensor_block_expert_ids_t block_experts{                           \
      buffers->block_expert_ids.ptr};                                         \
  PREFIX##_Tensor_packed_route_count_t route_count{                           \
      buffers->packed_route_count.ptr};                                       \
  PREFIX##_Tensor_topk_weights_flat_t topk{buffers->topk_weights.ptr};         \
  PREFIX##_Tensor_c_tmp_f32_flat_t scratch{buffers->c_tmp.ptr};               \
  PREFIX##_Tensor_locks_i32_flat_t locks{buffers->locks.ptr}

int launch_m16(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
               size_t rows, cudaStream_t stream) {
  GLMRT_PACKED_O_ARGS(glmrt_w8a16_packed_o_m16);
  return cute_dsl_glmrt_w8a16_packed_o_m16_wrapper(
      &module_m16, &input, &weight, &output, &scales, &global_scale, &routes,
      &block_experts, &route_count, &topk, &scratch, &locks,
      static_cast<int32_t>(rows), GLMRT_W8A16_PACKED_O_M16_GRID_X, stream);
}

int launch_m32(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
               size_t rows, cudaStream_t stream) {
  GLMRT_PACKED_O_ARGS(glmrt_w8a16_packed_o_m32);
  return cute_dsl_glmrt_w8a16_packed_o_m32_wrapper(
      &module_m32, &input, &weight, &output, &scales, &global_scale, &routes,
      &block_experts, &route_count, &topk, &scratch, &locks,
      static_cast<int32_t>(rows), GLMRT_W8A16_PACKED_O_M32_GRID_X, stream);
}

int launch_m64(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
               size_t rows, cudaStream_t stream) {
  GLMRT_PACKED_O_ARGS(glmrt_w8a16_packed_o_m64);
  return cute_dsl_glmrt_w8a16_packed_o_m64_wrapper(
      &module_m64, &input, &weight, &output, &scales, &global_scale, &routes,
      &block_experts, &route_count, &topk, &scratch, &locks,
      static_cast<int32_t>(rows), GLMRT_W8A16_PACKED_O_M64_GRID_X, stream);
}

int launch_m128(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
                size_t rows, cudaStream_t stream) {
  GLMRT_PACKED_O_ARGS(glmrt_w8a16_packed_o_m128);
  return cute_dsl_glmrt_w8a16_packed_o_m128_wrapper(
      &module_m128, &input, &weight, &output, &scales, &global_scale, &routes,
      &block_experts, &route_count, &topk, &scratch, &locks,
      static_cast<int32_t>(rows), GLMRT_W8A16_PACKED_O_M128_GRID_X, stream);
}

int launch_m256(const glmrt_b12x_coordinator_w4a16_buffers_t* buffers,
                size_t rows, cudaStream_t stream) {
  GLMRT_PACKED_O_ARGS(glmrt_w8a16_packed_o_m256);
  return cute_dsl_glmrt_w8a16_packed_o_m256_wrapper(
      &module_m256, &input, &weight, &output, &scales, &global_scale, &routes,
      &block_experts, &route_count, &topk, &scratch, &locks,
      static_cast<int32_t>(rows), GLMRT_W8A16_PACKED_O_M256_GRID_X, stream);
}

#undef GLMRT_PACKED_O_ARGS

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_w8a16_packed_o_aot_init(void) {
  std::call_once(module_init_once, initialize_modules);
  return module_init_status;
}

extern "C" glmrt_status_t
glmrt_cuda_w8a16_packed_o_initialize_launch_buffers_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t rows,
    size_t block_m, void* cuda_stream) {
  const glmrt_status_t valid = validate_buffers(buffers, rows, block_m);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const size_t route_slots = round_up_rows(rows, block_m);
  const size_t route_blocks = route_slots / block_m;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(
      buffers->locks.ptr, 0, kLockElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::max<size_t>(
      1, (route_slots + static_cast<size_t>(threads) - 1) / threads));
  initialize_packed_o_metadata_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float*>(buffers->global_scale.ptr),
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), route_slots, route_blocks);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_w8a16_packed_o_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t rows,
    void* cuda_stream) {
  size_t block_m = 0;
  int (*launch)(const glmrt_b12x_coordinator_w4a16_buffers_t*, size_t,
                cudaStream_t) = nullptr;
  if (rows <= 16) {
    block_m = GLMRT_W8A16_PACKED_O_M16_BLOCK_M;
    launch = launch_m16;
  } else if (rows <= 32) {
    block_m = GLMRT_W8A16_PACKED_O_M32_BLOCK_M;
    launch = launch_m32;
  } else if (rows <= 64) {
    block_m = GLMRT_W8A16_PACKED_O_M64_BLOCK_M;
    launch = launch_m64;
  } else if (rows <= 128) {
    block_m = GLMRT_W8A16_PACKED_O_M128_BLOCK_M;
    launch = launch_m128;
  } else if (rows <= 256) {
    block_m = GLMRT_W8A16_PACKED_O_M256_BLOCK_M;
    launch = launch_m256;
  } else {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_buffers(buffers, rows, block_m);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_w8a16_packed_o_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int result = launch(buffers, rows, stream);
  if (result != 0) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  return status_from_cuda(cudaGetLastError());
}
