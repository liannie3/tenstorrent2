#include "rag/search.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rag {
namespace {

constexpr unsigned kThreadsPerBlock = 256;
constexpr float kMinimumInitialScore = -2.0f;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void validate_threshold(float threshold) {
  const bool outside_cosine_range = threshold < -1.0f || threshold > 1.0f;
  if (!std::isfinite(threshold) || outside_cosine_range) {
    throw std::invalid_argument("Threshold must be in [-1, 1]");
  }
}

// Each block computes the cosine score for one normalized bank key
__global__ void score_keys(
    const float* keys,
    const float* query,
    float* scores) {
  __shared__ float partial_sums[kThreadsPerBlock];

  const unsigned row = blockIdx.x;
  const unsigned lane = threadIdx.x;
  float local_sum = 0.0f;

  // With 256 threads and 2048 dimensions, each thread handles 8 products
  for (unsigned column = lane; column < kTextDim; column += blockDim.x) {
    const std::size_t key_index =
        static_cast<std::size_t>(row) * kTextDim + column;
    local_sum += keys[key_index] * query[column];
  }

  partial_sums[lane] = local_sum;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial_sums[lane] += partial_sums[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    scores[row] = partial_sums[0];
  }
}

Match select_best(const std::vector<float>& scores, float threshold) {
  Match best;
  best.score = kMinimumInitialScore;

  for (std::uint32_t id = 0; id < scores.size(); ++id) {
    if (scores[id] > best.score) {
      best.id = id;
      best.score = scores[id];
    }
  }

  best.accepted = best.score >= threshold;
  return best;
}

}  // namespace

CudaSearcher::CudaSearcher(const Bank& bank) : count_(bank.size()) {
  const std::size_t key_bytes =
      static_cast<std::size_t>(count_) * kTextDim * sizeof(float);
  const std::size_t query_bytes =
      static_cast<std::size_t>(kTextDim) * sizeof(float);
  const std::size_t score_bytes =
      static_cast<std::size_t>(count_) * sizeof(float);

  try {
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_keys_), key_bytes),
        "cudaMalloc keys");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_query_), query_bytes),
        "cudaMalloc query");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_scores_), score_bytes),
        "cudaMalloc scores");
    check_cuda(
        cudaMemcpy(
            device_keys_,
            bank.keys(),
            key_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy keys");
  } catch (...) {
    cudaFree(device_keys_);
    cudaFree(device_query_);
    cudaFree(device_scores_);
    throw;
  }
}

CudaSearcher::~CudaSearcher() {
  cudaFree(device_keys_);
  cudaFree(device_query_);
  cudaFree(device_scores_);
}

Match CudaSearcher::search(
    const std::vector<float>& query,
    float threshold) {
  validate_threshold(threshold);
  const std::vector<float> normalized_query = normalize_query(query);

  const std::size_t query_bytes = normalized_query.size() * sizeof(float);
  check_cuda(
      cudaMemcpy(
          device_query_,
          normalized_query.data(),
          query_bytes,
          cudaMemcpyHostToDevice),
      "cudaMemcpy query");

  score_keys<<<count_, kThreadsPerBlock>>>(
      device_keys_, device_query_, device_scores_);
  check_cuda(cudaGetLastError(), "score_keys launch");

  std::vector<float> scores(count_);
  const std::size_t score_bytes = scores.size() * sizeof(float);
  check_cuda(
      cudaMemcpy(
          scores.data(),
          device_scores_,
          score_bytes,
          cudaMemcpyDeviceToHost),
      "cudaMemcpy scores");

  return select_best(scores, threshold);
}

}  // namespace rag
