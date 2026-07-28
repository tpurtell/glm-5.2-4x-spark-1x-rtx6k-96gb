#include "common.h"

namespace {

__global__ void pack_nibbles_kernel(const uint8_t* codes, uint8_t* packed, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t packed_count = (count + 1) / 2;
  if (idx >= packed_count) {
    return;
  }
  const size_t low_idx = idx * 2;
  const size_t high_idx = low_idx + 1;
  const uint8_t low = codes[low_idx] & 0x0f;
  const uint8_t high = high_idx < count ? static_cast<uint8_t>((codes[high_idx] & 0x0f) << 4) : 0;
  packed[idx] = low | high;
}

__global__ void unpack_nibbles_kernel(const uint8_t* packed, uint8_t* codes, size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const uint8_t byte = packed[idx / 2];
  codes[idx] = (idx % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_pack_nibbles(const uint8_t* codes, uint8_t* packed,
                                                  size_t count) {
  if (codes == nullptr || packed == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  const size_t packed_count = (count + 1) / 2;
  const int threads = 256;
  const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
  pack_nibbles_kernel<<<blocks, threads>>>(codes, packed, count);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_unpack_nibbles(const uint8_t* packed, uint8_t* codes,
                                                    size_t count) {
  if (packed == nullptr || codes == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (count == 0) {
    return GLMRT_STATUS_OK;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  unpack_nibbles_kernel<<<blocks, threads>>>(packed, codes, count);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

