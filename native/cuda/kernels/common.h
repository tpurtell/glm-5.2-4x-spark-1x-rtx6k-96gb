#include "glmrt_native.h"


#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <vector>

extern "C" void glmrt_set_last_error_message(const char* message);

namespace {


constexpr int kBlock = 256;
constexpr int kNvfp4RouteReduceBlock = 256;
constexpr size_t kMaxRouterTopK = 64;
constexpr size_t kMaxSampleTopK = 64;
constexpr float kGlm52RoutedScalingFactor = 2.5f;
constexpr uint32_t kRouterTopKLockBit = 0x80000000u;

__device__ float sigmoid_f32(float value) {
  if (value >= 0.0f) {
    const float exp_neg = expf(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = expf(value);
  return exp_pos / (1.0f + exp_pos);
}

__device__ float nvfp4_e2m1_code_value(uint8_t code) {
  constexpr float kCodebook[16] = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      0.0f,  -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return kCodebook[code & 0x0f];
}

__device__ float f8e4m3_to_f32(uint8_t byte) {
  if (byte == 0 || byte == 0x80) {
    return 0.0f;
  }
  const float sign = (byte & 0x80) == 0 ? 1.0f : -1.0f;
  const int exponent = static_cast<int>((byte >> 3) & 0x0f);
  const float mantissa = static_cast<float>(byte & 0x07);
  const float significand = exponent == 0 ? mantissa / 8.0f : 1.0f + mantissa / 8.0f;
  const int exponent_power = exponent == 0 ? -6 : exponent - 7;
  return sign * ldexpf(significand, exponent_power);
}

__device__ float packed_nvfp4_value(const uint8_t* packed_row, const uint8_t* scale_row,
                                    size_t value_idx, float scale_2) {
  const uint8_t packed = packed_row[value_idx / 2];
  const uint8_t code = value_idx % 2 == 0 ? (packed & 0x0f) : (packed >> 4);
  const float scale = f8e4m3_to_f32(scale_row[value_idx / 16]);
  return nvfp4_e2m1_code_value(code) * scale * scale_2;
}

__device__ float bf16_to_f32(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ uint16_t f32_to_bf16(float value) {
  return static_cast<uint16_t>(__float_as_uint(value) >> 16);
}

__device__ float dot_packed_nvfp4_bf16(const uint16_t* input, const uint8_t* packed_row,
                                       const uint8_t* scale_row, size_t input_dim,
                                       float scale_2) {
  float sum = 0.0f;
  for (size_t value_idx = 0; value_idx < input_dim; ++value_idx) {
    sum = fmaf(bf16_to_f32(input[value_idx]),
               packed_nvfp4_value(packed_row, scale_row, value_idx, scale_2), sum);
  }
  return sum;
}


bool checked_mul(size_t lhs, size_t rhs, size_t* out) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t* out) {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  *out = lhs + rhs;
  return true;
}

glmrt_status_t status_from_cuda(cudaError_t err) {
  if (err == cudaSuccess) {
    return GLMRT_STATUS_OK;
  }
  glmrt_set_last_error_message(cudaGetErrorString(err));
  return GLMRT_STATUS_COPY_FAILED;
}

glmrt_status_t find_kernel_node_by_index(void* cuda_graph, size_t kernel_node_index,
                                         cudaGraphNode_t* out_node) {
  if (cuda_graph == nullptr || out_node == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_node = nullptr;
  cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(cuda_graph);
  size_t node_count = 0;
  cudaError_t err = cudaGraphGetNodes(graph, nullptr, &node_count);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  std::vector<cudaGraphNode_t> nodes(node_count);
  if (node_count > 0) {
    size_t copied_nodes = node_count;
    err = cudaGraphGetNodes(graph, nodes.data(), &copied_nodes);
    if (err != cudaSuccess) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
    nodes.resize(copied_nodes);
  }

  size_t kernel_index = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    err = cudaGraphNodeGetType(node, &type);
    if (err != cudaSuccess) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
    if (type != cudaGraphNodeTypeKernel) {
      continue;
    }
    if (kernel_index == kernel_node_index) {
      *out_node = node;
      return GLMRT_STATUS_OK;
    }
    ++kernel_index;
  }
  return GLMRT_STATUS_INVALID_ARGUMENT;
}

}  // namespace
