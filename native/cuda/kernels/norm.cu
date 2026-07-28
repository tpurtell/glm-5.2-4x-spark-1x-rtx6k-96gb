#include "common.h"

namespace {

__global__ void rmsnorm_f32_kernel(const float* x, const float* weight, float* out, int rows,
                                   int hidden, float eps) {
  __shared__ float scratch[kBlock];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = x[row * hidden + col];
    sum += value * value;
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    out[row * hidden + col] = x[row * hidden + col] * inv * weight[col];
  }
}

__global__ void rmsnorm_bf16_kernel(const uint16_t* x, const uint16_t* weight, uint16_t* out,
                                    int rows, int hidden, float eps) {
  __shared__ float scratch[kBlock];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = bf16_to_f32(x[row * hidden + col]);
    sum += value * value;
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = bf16_to_f32(x[row * hidden + col]);
    out[row * hidden + col] = f32_to_bf16(value * inv * bf16_to_f32(weight[col]));
  }
}

__global__ void layernorm_affine_f32_bf16_kernel(const float* x, const uint16_t* weight,
                                                 const uint16_t* bias, float* out, int rows,
                                                 int hidden, float eps) {
  __shared__ float scratch[kBlock];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    sum += x[row * hidden + col];
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(hidden);

  float variance_sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float centered = x[row * hidden + col] - mean;
    variance_sum += centered * centered;
  }
  scratch[tid] = variance_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float normalized = (x[row * hidden + col] - mean) * inv;
    out[row * hidden + col] =
        normalized * bf16_to_f32(weight[col]) + bf16_to_f32(bias[col]);
  }
}

__global__ void layernorm_affine_bf16_kernel(const uint16_t* x, const uint16_t* weight,
                                             const uint16_t* bias, uint16_t* out, int rows,
                                             int hidden, float eps) {
  __shared__ float scratch[kBlock];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    sum += bf16_to_f32(x[row * hidden + col]);
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(hidden);

  float variance_sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float centered = bf16_to_f32(x[row * hidden + col]) - mean;
    variance_sum += centered * centered;
  }
  scratch[tid] = variance_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = bf16_to_f32(x[row * hidden + col]);
    const float normalized = (value - mean) * inv;
    out[row * hidden + col] =
        f32_to_bf16(normalized * bf16_to_f32(weight[col]) + bf16_to_f32(bias[col]));
  }
}

glmrt_status_t validate_rmsnorm_args(const float* x, const float* weight, const float* out,
                                     int rows, int hidden) {
  if (x == nullptr || weight == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows <= 0 || hidden <= 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_rmsnorm_bf16_args(const uint16_t* x, const uint16_t* weight,
                                          const uint16_t* out, int rows, int hidden) {
  if (x == nullptr || weight == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows <= 0 || hidden <= 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_layernorm_affine_f32_bf16_args(const float* x, const uint16_t* weight,
                                                       const uint16_t* bias, const float* out,
                                                       int rows, int hidden) {
  if (x == nullptr || weight == nullptr || bias == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows <= 0 || hidden <= 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_layernorm_affine_bf16_args(const uint16_t* x, const uint16_t* weight,
                                                   const uint16_t* bias, const uint16_t* out,
                                                   int rows, int hidden) {
  if (x == nullptr || weight == nullptr || bias == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows <= 0 || hidden <= 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_rmsnorm_buffers(glmrt_device_buffer_t x,
                                                   glmrt_device_buffer_t weight,
                                                   glmrt_device_buffer_t out, int rows,
                                                   int hidden) {
  const glmrt_status_t valid = validate_rmsnorm_bf16_args(
      static_cast<const uint16_t*>(x.ptr), static_cast<const uint16_t*>(weight.ptr),
      static_cast<const uint16_t*>(out.ptr), rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t row_values = 0;
  if (!checked_mul(static_cast<size_t>(rows), static_cast<size_t>(hidden), &row_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_bytes = 0;
  if (!checked_mul(row_values, sizeof(uint16_t), &row_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t weight_bytes = 0;
  if (!checked_mul(static_cast<size_t>(hidden), sizeof(uint16_t), &weight_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (x.bytes < row_bytes || out.bytes < row_bytes || weight.bytes < weight_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_graph_layernorm_affine_f32_bf16_buffers(
    glmrt_device_buffer_t x, glmrt_device_buffer_t weight, glmrt_device_buffer_t bias,
    glmrt_device_buffer_t out, int rows, int hidden) {
  const glmrt_status_t valid = validate_layernorm_affine_f32_bf16_args(
      static_cast<const float*>(x.ptr), static_cast<const uint16_t*>(weight.ptr),
      static_cast<const uint16_t*>(bias.ptr), static_cast<const float*>(out.ptr), rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t row_values = 0;
  if (!checked_mul(static_cast<size_t>(rows), static_cast<size_t>(hidden), &row_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_bytes = 0;
  size_t vector_bytes = 0;
  if (!checked_mul(row_values, sizeof(float), &row_bytes) ||
      !checked_mul(static_cast<size_t>(hidden), sizeof(uint16_t), &vector_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (x.bytes < row_bytes || out.bytes < row_bytes || weight.bytes < vector_bytes ||
      bias.bytes < vector_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_graph_layernorm_affine_bf16_buffers(
    glmrt_device_buffer_t x, glmrt_device_buffer_t weight, glmrt_device_buffer_t bias,
    glmrt_device_buffer_t out, int rows, int hidden) {
  const glmrt_status_t valid = validate_layernorm_affine_bf16_args(
      static_cast<const uint16_t*>(x.ptr), static_cast<const uint16_t*>(weight.ptr),
      static_cast<const uint16_t*>(bias.ptr), static_cast<const uint16_t*>(out.ptr), rows,
      hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t row_values = 0;
  if (!checked_mul(static_cast<size_t>(rows), static_cast<size_t>(hidden), &row_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t row_bytes = 0;
  size_t vector_bytes = 0;
  if (!checked_mul(row_values, sizeof(uint16_t), &row_bytes) ||
      !checked_mul(static_cast<size_t>(hidden), sizeof(uint16_t), &vector_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (x.bytes < row_bytes || out.bytes < row_bytes || weight.bytes < vector_bytes ||
      bias.bytes < vector_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_rmsnorm_f32_async(const float* x, const float* weight,
                                                       float* out, int rows, int hidden,
                                                       float eps, void* cuda_stream) {
  const glmrt_status_t valid = validate_rmsnorm_args(x, weight, out, rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rmsnorm_f32_kernel<<<rows, kBlock, 0, stream>>>(x, weight, out, rows, hidden, eps);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_rmsnorm_f32(const float* x, const float* weight, float* out,
                                                 int rows, int hidden, float eps) {
  const glmrt_status_t status = glmrt_cuda_rmsnorm_f32_async(x, weight, out, rows, hidden, eps,
                                                            nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_rmsnorm_bf16_async(const uint16_t* x,
                                                        const uint16_t* weight, uint16_t* out,
                                                        int rows, int hidden, float eps,
                                                        void* cuda_stream) {
  const glmrt_status_t valid = validate_rmsnorm_bf16_args(x, weight, out, rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rmsnorm_bf16_kernel<<<rows, kBlock, 0, stream>>>(x, weight, out, rows, hidden, eps);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_mla_scalar_qa_batched_norm_candidate_async(
    const uint16_t* hidden, const uint16_t* input_norm_weight,
    uint16_t* normalized_hidden, const uint16_t* q_a_weight,
    uint16_t* q_a_projected, const uint16_t* q_a_norm_weight,
    uint16_t* q_a_normalized, size_t rows, size_t hidden_dim,
    size_t q_lora_rank, float eps, void* cuda_stream) {
  if (hidden == nullptr || input_norm_weight == nullptr || normalized_hidden == nullptr ||
      q_a_weight == nullptr || q_a_projected == nullptr || q_a_norm_weight == nullptr ||
      q_a_normalized == nullptr || rows < 2 || rows > 16 || hidden_dim != 6144 ||
      q_lora_rank != 2048 || !isfinite(eps) || eps <= 0.0f) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  glmrt_status_t status = glmrt_cuda_rmsnorm_bf16_async(
      hidden, input_norm_weight, normalized_hidden, static_cast<int>(rows),
      static_cast<int>(hidden_dim), eps, cuda_stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  for (size_t row = 0; row < rows; ++row) {
    status = glmrt_cuda_linear_bf16_cublas_async(
        normalized_hidden + row * hidden_dim, q_a_weight, nullptr,
        q_a_projected + row * q_lora_rank, 1, hidden_dim, q_lora_rank,
        cuda_stream);
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
  }
  return glmrt_cuda_rmsnorm_bf16_async(
      q_a_projected, q_a_norm_weight, q_a_normalized, static_cast<int>(rows),
      static_cast<int>(q_lora_rank), eps, cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_rmsnorm_bf16(const uint16_t* x,
                                                  const uint16_t* weight, uint16_t* out,
                                                  int rows, int hidden, float eps) {
  const glmrt_status_t status = glmrt_cuda_rmsnorm_bf16_async(x, weight, out, rows, hidden, eps,
                                                             nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_graph_update_rmsnorm_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t out, int rows, int hidden, float eps) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_rmsnorm_buffers(x, weight, out, rows, hidden);
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
  if (existing.func != reinterpret_cast<void*>(rmsnorm_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* x_ptr = static_cast<const uint16_t*>(x.ptr);
  const uint16_t* weight_ptr = static_cast<const uint16_t*>(weight.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &x_ptr,
      &weight_ptr,
      &out_ptr,
      &rows,
      &hidden,
      &eps,
  };
  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(rmsnorm_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  params.blockDim = dim3(kBlock, 1, 1);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_layernorm_affine_f32_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t bias, glmrt_device_buffer_t out,
    int rows, int hidden, float eps) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_graph_layernorm_affine_f32_bf16_buffers(x, weight, bias, out, rows, hidden);
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
  if (existing.func != reinterpret_cast<void*>(layernorm_affine_f32_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const float* x_ptr = static_cast<const float*>(x.ptr);
  const uint16_t* weight_ptr = static_cast<const uint16_t*>(weight.ptr);
  const uint16_t* bias_ptr = static_cast<const uint16_t*>(bias.ptr);
  float* out_ptr = static_cast<float*>(out.ptr);
  void* args[] = {
      &x_ptr,
      &weight_ptr,
      &bias_ptr,
      &out_ptr,
      &rows,
      &hidden,
      &eps,
  };
  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(layernorm_affine_f32_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  params.blockDim = dim3(kBlock, 1, 1);
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

extern "C" glmrt_status_t glmrt_cuda_graph_update_layernorm_affine_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t bias, glmrt_device_buffer_t out,
    int rows, int hidden, float eps) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_graph_layernorm_affine_bf16_buffers(x, weight, bias, out, rows, hidden);
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
  if (existing.func != reinterpret_cast<void*>(layernorm_affine_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* x_ptr = static_cast<const uint16_t*>(x.ptr);
  const uint16_t* weight_ptr = static_cast<const uint16_t*>(weight.ptr);
  const uint16_t* bias_ptr = static_cast<const uint16_t*>(bias.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &x_ptr,
      &weight_ptr,
      &bias_ptr,
      &out_ptr,
      &rows,
      &hidden,
      &eps,
  };
  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(layernorm_affine_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  params.blockDim = dim3(kBlock, 1, 1);
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

extern "C" glmrt_status_t glmrt_cuda_layernorm_affine_f32_bf16_async(
    const float* x, const uint16_t* weight, const uint16_t* bias, float* out, int rows, int hidden,
    float eps, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_layernorm_affine_f32_bf16_args(x, weight, bias, out, rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  layernorm_affine_f32_bf16_kernel<<<rows, kBlock, 0, stream>>>(x, weight, bias, out, rows,
                                                               hidden, eps);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_layernorm_affine_f32_bf16(
    const float* x, const uint16_t* weight, const uint16_t* bias, float* out, int rows, int hidden,
    float eps) {
  const glmrt_status_t status =
      glmrt_cuda_layernorm_affine_f32_bf16_async(x, weight, bias, out, rows, hidden, eps, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_layernorm_affine_bf16_async(
    const uint16_t* x, const uint16_t* weight, const uint16_t* bias, uint16_t* out, int rows,
    int hidden, float eps, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_layernorm_affine_bf16_args(x, weight, bias, out, rows, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  layernorm_affine_bf16_kernel<<<rows, kBlock, 0, stream>>>(x, weight, bias, out, rows, hidden,
                                                           eps);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_layernorm_affine_bf16(
    const uint16_t* x, const uint16_t* weight, const uint16_t* bias, uint16_t* out, int rows,
    int hidden, float eps) {
  const glmrt_status_t status =
      glmrt_cuda_layernorm_affine_bf16_async(x, weight, bias, out, rows, hidden, eps, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
