#include "common.h"

#include <cub/cub.cuh>

namespace {

__global__ void router_topk_f32_kernel(const float* hidden, const float* router_weight,
                                       const float* correction_bias, uint32_t* topk_indices,
                                       float* topk_scores, float* topk_weights, size_t rows,
                                       size_t hidden_dim, size_t experts, size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float best_scores[kMaxRouterTopK];
  float best_corrected[kMaxRouterTopK];
  uint32_t best_indices[kMaxRouterTopK];
  for (size_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = 0.0f;
    best_corrected[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }

  const float* row_hidden = hidden + row * hidden_dim;
  for (size_t expert = 0; expert < experts; ++expert) {
    const float* weight_row = router_weight + expert * hidden_dim;
    float logit = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      logit += row_hidden[col] * weight_row[col];
    }
    const float raw_score = sigmoid_f32(logit);
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    const float corrected = isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected
                                                                            : -CUDART_INF_F;
    for (size_t rank = 0; rank < top_k; ++rank) {
      if (corrected > best_corrected[rank]) {
        for (size_t shift = top_k - 1; shift > rank; --shift) {
          best_corrected[shift] = best_corrected[shift - 1];
          best_scores[shift] = best_scores[shift - 1];
          best_indices[shift] = best_indices[shift - 1];
        }
        best_corrected[rank] = corrected;
        best_scores[rank] = score;
        best_indices[rank] = static_cast<uint32_t>(expert);
        break;
      }
    }
  }

  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    score_sum += best_scores[rank];
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);

  const size_t out_offset = row * top_k;
  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_indices[out_offset + rank] = best_indices[rank];
    topk_scores[out_offset + rank] = best_scores[rank];
    topk_weights[out_offset + rank] = best_scores[rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

__global__ void router_topk_bf16_init_kernel(uint32_t* topk_indices, float* topk_scores,
                                             float* topk_weights, size_t rows, size_t top_k) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * top_k;
  if (idx >= total) {
    return;
  }
  topk_indices[idx] = 0;
  topk_scores[idx] = 0.0f;
  topk_weights[idx] = -CUDART_INF_F;
}

__global__ void router_topk_bf16_score_kernel(const uint16_t* hidden,
                                              const uint16_t* router_weight,
                                              const float* correction_bias,
                                              uint32_t* topk_indices, float* topk_scores,
                                              float* topk_weights, size_t rows,
                                              size_t hidden_dim, size_t experts,
                                              size_t top_k) {
  __shared__ float scratch[kBlock];
  const size_t expert = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || expert >= experts) {
    return;
  }
  const size_t tid = threadIdx.x;
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  const uint16_t* weight_row = router_weight + expert * hidden_dim;
  float partial = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]), partial);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid != 0) {
    return;
  }

  const float raw_score = sigmoid_f32(scratch[0]);
  const float score = isfinite(raw_score) ? raw_score : 0.0f;
  const float raw_corrected = score + correction_bias[expert];
  const float corrected = isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected
                                                                          : -CUDART_INF_F;
  const size_t out_offset = row * top_k;
  uint32_t* lock_word = topk_indices + out_offset;

  while (true) {
    const uint32_t current = atomicAdd(lock_word, 0);
    if ((current & kRouterTopKLockBit) != 0) {
      continue;
    }
    if (atomicCAS(lock_word, current, current | kRouterTopKLockBit) == current) {
      break;
    }
  }

  for (size_t rank = 0; rank < top_k; ++rank) {
    uint32_t current_index = topk_indices[out_offset + rank];
    if (rank == 0) {
      current_index &= ~kRouterTopKLockBit;
    }
    const float current_corrected = topk_weights[out_offset + rank];
    if (corrected > current_corrected ||
        (corrected == current_corrected && static_cast<uint32_t>(expert) < current_index)) {
      for (size_t shift = top_k - 1; shift > rank; --shift) {
        uint32_t shifted_index = topk_indices[out_offset + shift - 1];
        if (shift == 1) {
          shifted_index &= ~kRouterTopKLockBit;
        }
        topk_indices[out_offset + shift] = shifted_index;
        topk_scores[out_offset + shift] = topk_scores[out_offset + shift - 1];
        topk_weights[out_offset + shift] = topk_weights[out_offset + shift - 1];
      }
      topk_indices[out_offset + rank] =
          static_cast<uint32_t>(expert) | (rank == 0 ? kRouterTopKLockBit : 0);
      topk_scores[out_offset + rank] = score;
      topk_weights[out_offset + rank] = corrected;
      break;
    }
  }
  __threadfence();
  atomicAnd(lock_word, ~kRouterTopKLockBit);
}

__global__ void router_topk_bf16_finalize_kernel(uint32_t* topk_indices, float* topk_scores,
                                                 float* topk_weights, size_t rows,
                                                 size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t out_offset = row * top_k;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    const float score = topk_scores[out_offset + rank];
    score_sum += isfinite(score) ? score : 0.0f;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);

  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_indices[out_offset + rank] &= ~kRouterTopKLockBit;
    const float score = topk_scores[out_offset + rank];
    topk_scores[out_offset + rank] = isfinite(score) ? score : 0.0f;
    topk_weights[out_offset + rank] =
        topk_scores[out_offset + rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

__global__ void router_topk_bf16_cub_fill_indices_offsets_kernel(uint32_t* indices,
                                                                 int* segment_offsets,
                                                                 size_t rows, size_t experts) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * experts;
  if (idx < total) {
    indices[idx] = static_cast<uint32_t>(idx % experts);
  }
  if (idx <= rows) {
    segment_offsets[idx] = static_cast<int>(idx * experts);
  }
}

__global__ void router_topk_bf16_cub_score_kernel(const uint16_t* hidden,
                                                  const uint16_t* router_weight,
                                                  const float* correction_bias,
                                                  float* corrected_scores, size_t rows,
                                                  size_t hidden_dim, size_t experts) {
  __shared__ float scratch[kBlock];
  const size_t expert = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || expert >= experts) {
    return;
  }
  const size_t tid = threadIdx.x;
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  const uint16_t* weight_row = router_weight + expert * hidden_dim;
  float partial = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]), partial);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float raw_score = sigmoid_f32(scratch[0]);
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    corrected_scores[row * experts + expert] =
        isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected : -CUDART_INF_F;
  }
}

__global__ void router_topk_bf16_cub_finalize_kernel(
    const float* sorted_corrected_scores, const uint32_t* sorted_indices,
    const float* correction_bias, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, size_t rows, size_t experts, size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t sorted_offset = row * experts;
  const size_t out_offset = row * top_k;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = sorted_indices[sorted_offset + rank];
    const float raw_score =
        sorted_corrected_scores[sorted_offset + rank] - correction_bias[expert];
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    topk_indices[out_offset + rank] = expert;
    topk_scores[out_offset + rank] = score;
    score_sum += score;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_weights[out_offset + rank] =
        topk_scores[out_offset + rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

glmrt_status_t validate_router_topk_args(const float* hidden, const float* router_weight,
                                         const float* correction_bias,
                                         const uint32_t* topk_indices, const float* topk_scores,
                                         const float* topk_weights, size_t rows,
                                         size_t hidden_dim, size_t experts, size_t top_k) {
  if (hidden == nullptr || router_weight == nullptr || correction_bias == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr || topk_weights == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || experts == 0 || top_k == 0 || top_k > experts ||
      top_k > kMaxRouterTopK) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) ||
      !checked_mul(experts, hidden_dim, &ignored) ||
      !checked_mul(rows, top_k, &ignored) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_router_topk_bf16_args(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    const uint32_t* topk_indices, const float* topk_scores, const float* topk_weights,
    size_t rows, size_t hidden_dim, size_t experts, size_t top_k) {
  if (hidden == nullptr || router_weight == nullptr || correction_bias == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr || topk_weights == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || experts == 0 || top_k == 0 || top_k > experts ||
      top_k > kMaxRouterTopK ||
      experts > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) ||
      !checked_mul(experts, hidden_dim, &ignored) ||
      !checked_mul(rows, top_k, &ignored) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_router_topk_bf16_cub_args(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const glmrt_status_t valid = validate_router_topk_bf16_args(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (corrected_scores == nullptr || sorted_corrected_scores == nullptr ||
      unsorted_indices == nullptr || sorted_indices == nullptr || segment_offsets == nullptr ||
      cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t score_values = 0;
  if (!checked_mul(rows, experts, &score_values) ||
      score_values > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_router_topk_buffers(
    glmrt_device_buffer_t hidden, glmrt_device_buffer_t router_weight,
    glmrt_device_buffer_t correction_bias, glmrt_device_buffer_t topk_indices,
    glmrt_device_buffer_t topk_scores, glmrt_device_buffer_t topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const glmrt_status_t valid = validate_router_topk_bf16_args(
      static_cast<const uint16_t*>(hidden.ptr), static_cast<const uint16_t*>(router_weight.ptr),
      static_cast<const float*>(correction_bias.ptr), static_cast<const uint32_t*>(topk_indices.ptr),
      static_cast<const float*>(topk_scores.ptr), static_cast<const float*>(topk_weights.ptr),
      rows, hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t hidden_values = 0;
  size_t weight_values = 0;
  size_t topk_values = 0;
  if (!checked_mul(rows, hidden_dim, &hidden_values) ||
      !checked_mul(experts, hidden_dim, &weight_values) ||
      !checked_mul(rows, top_k, &topk_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t hidden_bytes = 0;
  size_t weight_bytes = 0;
  size_t bias_bytes = 0;
  size_t index_bytes = 0;
  size_t score_bytes = 0;
  if (!checked_mul(hidden_values, sizeof(uint16_t), &hidden_bytes) ||
      !checked_mul(weight_values, sizeof(uint16_t), &weight_bytes) ||
      !checked_mul(experts, sizeof(float), &bias_bytes) ||
      !checked_mul(topk_values, sizeof(uint32_t), &index_bytes) ||
      !checked_mul(topk_values, sizeof(float), &score_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (hidden.bytes < hidden_bytes || router_weight.bytes < weight_bytes ||
      correction_bias.bytes < bias_bytes || topk_indices.bytes < index_bytes ||
      topk_scores.bytes < score_bytes || topk_weights.bytes < score_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  return GLMRT_STATUS_OK;
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_router_topk_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t hidden, glmrt_device_buffer_t router_weight,
    glmrt_device_buffer_t correction_bias, glmrt_device_buffer_t topk_indices,
    glmrt_device_buffer_t topk_scores, glmrt_device_buffer_t topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_bf16_graph_router_topk_buffers(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  const uint16_t* hidden_ptr = static_cast<const uint16_t*>(hidden.ptr);
  const uint16_t* router_weight_ptr = static_cast<const uint16_t*>(router_weight.ptr);
  const float* correction_bias_ptr = static_cast<const float*>(correction_bias.ptr);
  uint32_t* topk_indices_ptr = static_cast<uint32_t*>(topk_indices.ptr);
  float* topk_scores_ptr = static_cast<float*>(topk_scores.ptr);
  float* topk_weights_ptr = static_cast<float*>(topk_weights.ptr);
  void* init_args[] = {
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &top_k,
  };
  void* score_args[] = {
      &hidden_ptr,
      &router_weight_ptr,
      &correction_bias_ptr,
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &hidden_dim,
      &experts,
      &top_k,
  };
  void* finalize_args[] = {
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &top_k,
  };

  const size_t topk_values = rows * top_k;
  const size_t init_blocks = (topk_values - 1) / kBlock + 1;
  if (init_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  auto update_kernel_node = [&](size_t node_index, void* expected_func,
                                cudaKernelNodeParams* params) -> glmrt_status_t {
    cudaGraphNode_t node = nullptr;
    const glmrt_status_t node_status =
        find_kernel_node_by_index(cuda_graph, node_index, &node);
    if (node_status != GLMRT_STATUS_OK) {
      return node_status;
    }
    cudaKernelNodeParams existing = {};
    cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
    if (err != cudaSuccess) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
    if (existing.func != expected_func) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    err = cudaGraphKernelNodeSetParams(node, params);
    if (err != cudaSuccess) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
    err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec),
                                           node, params);
    if (err != cudaSuccess) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
    return GLMRT_STATUS_OK;
  };

  cudaKernelNodeParams init_params = {};
  init_params.func = reinterpret_cast<void*>(router_topk_bf16_init_kernel);
  init_params.gridDim = dim3(static_cast<unsigned int>(init_blocks), 1, 1);
  init_params.blockDim = dim3(kBlock, 1, 1);
  init_params.sharedMemBytes = 0;
  init_params.kernelParams = init_args;
  init_params.extra = nullptr;

  cudaKernelNodeParams score_params = {};
  score_params.func = reinterpret_cast<void*>(router_topk_bf16_score_kernel);
  score_params.gridDim =
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1);
  score_params.blockDim = dim3(kBlock, 1, 1);
  score_params.sharedMemBytes = 0;
  score_params.kernelParams = score_args;
  score_params.extra = nullptr;

  cudaKernelNodeParams finalize_params = {};
  finalize_params.func = reinterpret_cast<void*>(router_topk_bf16_finalize_kernel);
  finalize_params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  finalize_params.blockDim = dim3(1, 1, 1);
  finalize_params.sharedMemBytes = 0;
  finalize_params.kernelParams = finalize_args;
  finalize_params.extra = nullptr;

  glmrt_status_t status = update_kernel_node(
      kernel_node_index, reinterpret_cast<void*>(router_topk_bf16_init_kernel), &init_params);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = update_kernel_node(kernel_node_index + 1,
                              reinterpret_cast<void*>(router_topk_bf16_score_kernel),
                              &score_params);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = update_kernel_node(kernel_node_index + 2,
                              reinterpret_cast<void*>(router_topk_bf16_finalize_kernel),
                              &finalize_params);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_f32_async(
    const float* hidden, const float* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_router_topk_args(hidden, router_weight, correction_bias, topk_indices, topk_scores,
                                topk_weights, rows, hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  router_topk_f32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_f32(
    const float* hidden, const float* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const glmrt_status_t status =
      glmrt_cuda_router_topk_f32_async(hidden, router_weight, correction_bias, topk_indices,
                                       topk_scores, topk_weights, rows, hidden_dim, experts, top_k,
                                       nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_bf16_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const glmrt_status_t valid = validate_router_topk_bf16_args(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t topk_values = rows * top_k;
  const size_t init_blocks = (topk_values - 1) / kBlock + 1;
  if (init_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  router_topk_bf16_init_kernel<<<static_cast<unsigned int>(init_blocks), kBlock, 0, stream>>>(
      topk_indices, topk_scores, topk_weights, rows, top_k);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_score_kernel<<<
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1), kBlock, 0,
      stream>>>(hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights,
                rows, hidden_dim, experts, top_k);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      topk_indices, topk_scores, topk_weights, rows, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_bf16(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const glmrt_status_t status =
      glmrt_cuda_router_topk_bf16_async(hidden, router_weight, correction_bias, topk_indices,
                                        topk_scores, topk_weights, rows, hidden_dim, experts,
                                        top_k, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_bf16_cub_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const glmrt_status_t valid = validate_router_topk_bf16_cub_args(
      hidden, router_weight, correction_bias, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, segment_offsets, topk_indices, topk_scores, topk_weights,
      cub_temp_storage, cub_temp_storage_bytes, rows, hidden_dim, experts, top_k);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t score_values = rows * experts;
  const size_t fill_values = std::max(score_values, rows + 1);
  const size_t fill_blocks = (fill_values - 1) / kBlock + 1;
  if (fill_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  router_topk_bf16_cub_fill_indices_offsets_kernel<<<
      static_cast<unsigned int>(fill_blocks), kBlock, 0, stream>>>(unsorted_indices,
                                                                   segment_offsets, rows, experts);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_cub_score_kernel<<<
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1), kBlock, 0,
      stream>>>(hidden, router_weight, correction_bias, corrected_scores, rows, hidden_dim,
                experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cub_temp_storage, cub_temp_storage_bytes, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, static_cast<int>(score_values), static_cast<int>(rows),
      segment_offsets, segment_offsets + 1, 0, sizeof(float) * 8, stream);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_cub_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      sorted_corrected_scores, sorted_indices, correction_bias, topk_indices, topk_scores,
      topk_weights, rows, experts, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_router_topk_bf16_cub(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const glmrt_status_t status = glmrt_cuda_router_topk_bf16_cub_async(
      hidden, router_weight, correction_bias, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, segment_offsets, topk_indices, topk_scores, topk_weights,
      cub_temp_storage, cub_temp_storage_bytes, rows, hidden_dim, experts, top_k, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
