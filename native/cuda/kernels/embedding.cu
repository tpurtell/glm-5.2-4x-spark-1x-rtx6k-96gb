#include "common.h"

namespace {

__global__ void embedding_lookup_f32_kernel(const float* embedding, const uint32_t* token_ids,
                                            float* out, size_t rows, size_t vocab,
                                            size_t hidden) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t col = idx % hidden;
  const size_t token_id = static_cast<size_t>(token_ids[row]);
  if (token_id >= vocab) {
    out[idx] = 0.0f;
    return;
  }
  out[idx] = embedding[token_id * hidden + col];
}

__global__ void embedding_lookup_bf16_kernel(const uint16_t* embedding, const uint32_t* token_ids,
                                             uint16_t* out, size_t rows, size_t vocab,
                                             size_t hidden) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t col = idx % hidden;
  const size_t token_id = static_cast<size_t>(token_ids[row]);
  if (token_id >= vocab) {
    out[idx] = 0;
    return;
  }
  out[idx] = embedding[token_id * hidden + col];
}

glmrt_status_t validate_embedding_lookup_args(const float* embedding, const uint32_t* token_ids,
                                              const float* out, size_t rows, size_t vocab,
                                              size_t hidden) {
  if (embedding == nullptr || token_ids == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || hidden == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(vocab, hidden, &ignored) || !checked_mul(rows, hidden, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_embedding_lookup_bf16_args(const uint16_t* embedding,
                                                   const uint32_t* token_ids,
                                                   const uint16_t* out, size_t rows, size_t vocab,
                                                   size_t hidden) {
  if (embedding == nullptr || token_ids == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || hidden == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(vocab, hidden, &ignored) || !checked_mul(rows, hidden, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_embedding_lookup_buffers(
    glmrt_device_buffer_t embedding, glmrt_device_buffer_t token_ids, glmrt_device_buffer_t out,
    size_t rows, size_t vocab, size_t hidden) {
  const glmrt_status_t valid = validate_embedding_lookup_bf16_args(
      static_cast<const uint16_t*>(embedding.ptr), static_cast<const uint32_t*>(token_ids.ptr),
      static_cast<const uint16_t*>(out.ptr), rows, vocab, hidden);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t embedding_values = 0;
  size_t output_values = 0;
  if (!checked_mul(vocab, hidden, &embedding_values) ||
      !checked_mul(rows, hidden, &output_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t embedding_bytes = 0;
  size_t token_bytes = 0;
  size_t output_bytes = 0;
  if (!checked_mul(embedding_values, sizeof(uint16_t), &embedding_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &token_bytes) ||
      !checked_mul(output_values, sizeof(uint16_t), &output_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (embedding.bytes < embedding_bytes || token_ids.bytes < token_bytes ||
      out.bytes < output_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_embedding_lookup_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t embedding, glmrt_device_buffer_t token_ids, glmrt_device_buffer_t out,
    size_t rows, size_t vocab, size_t hidden) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_embedding_lookup_buffers(embedding, token_ids, out, rows, vocab, hidden);
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
  if (existing.func != reinterpret_cast<void*>(embedding_lookup_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* embedding_ptr = static_cast<const uint16_t*>(embedding.ptr);
  const uint32_t* token_ids_ptr = static_cast<const uint32_t*>(token_ids.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &embedding_ptr,
      &token_ids_ptr,
      &out_ptr,
      &rows,
      &vocab,
      &hidden,
  };
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(embedding_lookup_bf16_kernel);
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

extern "C" glmrt_status_t glmrt_cuda_embedding_lookup_f32_async(
    const float* embedding, const uint32_t* token_ids, float* out, size_t rows, size_t vocab,
    size_t hidden, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_embedding_lookup_args(embedding, token_ids, out, rows, vocab, hidden);
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
  embedding_lookup_f32_kernel<<<blocks, threads, 0, stream>>>(embedding, token_ids, out, rows,
                                                              vocab, hidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_embedding_lookup_f32(const float* embedding,
                                                          const uint32_t* token_ids, float* out,
                                                          size_t rows, size_t vocab,
                                                          size_t hidden) {
  const glmrt_status_t status =
      glmrt_cuda_embedding_lookup_f32_async(embedding, token_ids, out, rows, vocab, hidden,
                                            nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_embedding_lookup_bf16_async(
    const uint16_t* embedding, const uint32_t* token_ids, uint16_t* out, size_t rows, size_t vocab,
    size_t hidden, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_embedding_lookup_bf16_args(embedding, token_ids, out, rows, vocab, hidden);
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
  embedding_lookup_bf16_kernel<<<blocks, threads, 0, stream>>>(embedding, token_ids, out, rows,
                                                               vocab, hidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_embedding_lookup_bf16(const uint16_t* embedding,
                                                           const uint32_t* token_ids,
                                                           uint16_t* out, size_t rows,
                                                           size_t vocab, size_t hidden) {
  const glmrt_status_t status =
      glmrt_cuda_embedding_lookup_bf16_async(embedding, token_ids, out, rows, vocab, hidden,
                                             nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

