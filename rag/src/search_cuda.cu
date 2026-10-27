#include "rag/search.h"
#include "rag/profile.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rag {
namespace {

using Clock = std::chrono::steady_clock;

// CUDA configuration
constexpr unsigned kThreadsPerBlock = 256;
constexpr float kMinimumInitialScore = -2.0f;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed with cuBLAS status " +
        std::to_string(static_cast<int>(status)));
  }
}

double elapsed_us(Clock::time_point start, Clock::time_point finish) {
  return std::chrono::duration<double, std::micro>(finish - start).count();
}

#ifdef RAG_HAS_CUDA_TIMING
class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

double event_elapsed_us(const CudaEvent& start, const CudaEvent& finish) {
  float milliseconds = 0.0f;
  check_cuda(
      cudaEventElapsedTime(&milliseconds, start.get(), finish.get()),
      "cudaEventElapsedTime");
  return static_cast<double>(milliseconds) * 1000.0;
}
#endif

void validate_threshold(float threshold) {
  const bool outside_cosine_range = threshold < -1.0f || threshold > 1.0f;
  if (!std::isfinite(threshold) || outside_cosine_range) {
    throw std::invalid_argument("Threshold must be in [-1, 1]");
  }
}

// Similarity kernel
__global__ void score_keys(
    const float* keys,
    const float* query,
    float* scores) {
  __shared__ float partial_sums[kThreadsPerBlock];

  const unsigned row = blockIdx.x;
  const unsigned lane = threadIdx.x;
  float local_sum = 0.0f;

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

// Result selection
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

struct DeviceMatch {
  float score;
  std::uint32_t id;
  std::uint32_t accepted;
};

__global__ void convert_float_to_bfloat16(
    const float* input,
    __nv_bfloat16* output,
    std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16(input[index]);
  }
}

__device__ bool is_better(
    float candidate_score,
    std::uint32_t candidate_id,
    float current_score,
    std::uint32_t current_id) {
  return candidate_score > current_score ||
      (candidate_score == current_score && candidate_id < current_id);
}

__global__ void select_best_on_device(
    const float* scores,
    std::uint32_t count,
    std::size_t query_count,
    float threshold,
    DeviceMatch* matches) {
  const std::size_t query = blockIdx.x;
  if (query >= query_count) {
    return;
  }

  __shared__ float shared_scores[kThreadsPerBlock];
  __shared__ std::uint32_t shared_ids[kThreadsPerBlock];
  const unsigned lane = threadIdx.x;
  float best_score = kMinimumInitialScore;
  std::uint32_t best_id = 0;

  for (std::uint32_t id = lane; id < count; id += blockDim.x) {
    const float score = scores[query * count + id];
    if (is_better(score, id, best_score, best_id)) {
      best_score = score;
      best_id = id;
    }
  }

  shared_scores[lane] = best_score;
  shared_ids[lane] = best_id;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride && is_better(
        shared_scores[lane + stride],
        shared_ids[lane + stride],
        shared_scores[lane],
        shared_ids[lane])) {
      shared_scores[lane] = shared_scores[lane + stride];
      shared_ids[lane] = shared_ids[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    matches[query] = DeviceMatch{
        shared_scores[0],
        shared_ids[0],
        shared_scores[0] >= threshold ? 1U : 0U};
  }
}

}  // namespace

// Device buffer lifecycle
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

// CUDA retrieval
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

std::vector<Match> CudaSearcher::search_batch(
    const std::vector<float>& queries,
    std::size_t query_count,
    float threshold) {
  if (query_count == 0 || queries.size() != query_count * kTextDim) {
    throw std::invalid_argument("Query batch has an invalid shape");
  }

  std::vector<Match> matches;
  matches.reserve(query_count);
  for (std::size_t index = 0; index < query_count; ++index) {
    const auto begin = queries.begin() + index * kTextDim;
    matches.push_back(search(
        std::vector<float>(begin, begin + kTextDim), threshold));
  }
  return matches;
}

struct CudaOptimizedSearcher::Impl {
  explicit Impl(const Bank& bank, SearchPrecision requested_precision)
      : count(bank.size()), precision(requested_precision) {
    try {
      check_cublas(cublasCreate(&handle), "cublasCreate");
#ifdef RAG_HAS_CUDA_TIMING
      total_start = std::make_unique<CudaEvent>();
      h2d_start = std::make_unique<CudaEvent>();
      h2d_end = std::make_unique<CudaEvent>();
      conversion_start = std::make_unique<CudaEvent>();
      conversion_end = std::make_unique<CudaEvent>();
      gemm_start = std::make_unique<CudaEvent>();
      gemm_end = std::make_unique<CudaEvent>();
      reduction_start = std::make_unique<CudaEvent>();
      reduction_end = std::make_unique<CudaEvent>();
      d2h_start = std::make_unique<CudaEvent>();
      d2h_end = std::make_unique<CudaEvent>();
      total_end = std::make_unique<CudaEvent>();
#endif
      const std::size_t element_count =
          static_cast<std::size_t>(count) * kTextDim;
      if (precision == SearchPrecision::fp32) {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_keys),
                       element_count * sizeof(float)),
            "cudaMalloc optimized keys");
        check_cuda(
            cudaMemcpy(device_keys, bank.keys(), element_count * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy optimized keys");
      } else {
        float* temporary_keys = nullptr;
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&temporary_keys),
                       element_count * sizeof(float)),
            "cudaMalloc temporary keys");
        try {
          check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_keys),
                         element_count * sizeof(__nv_bfloat16)),
              "cudaMalloc BF16 keys");
          check_cuda(
              cudaMemcpy(temporary_keys, bank.keys(), element_count * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy temporary keys");
          const auto blocks = static_cast<unsigned>(
              (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
          convert_float_to_bfloat16<<<blocks, kThreadsPerBlock>>>(
              temporary_keys,
              static_cast<__nv_bfloat16*>(device_keys),
              element_count);
          check_cuda(cudaGetLastError(), "convert BF16 keys launch");
          check_cuda(cudaDeviceSynchronize(), "convert BF16 keys");
        } catch (...) {
          cudaFree(temporary_keys);
          throw;
        }
        cudaFree(temporary_keys);
      }
      ensure_capacity(1);
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    cudaFree(device_keys);
    cudaFree(device_queries);
    cudaFree(device_float_queries);
    cudaFree(device_scores);
    cudaFree(device_matches);
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
    device_keys = nullptr;
    device_queries = nullptr;
    device_float_queries = nullptr;
    device_scores = nullptr;
    device_matches = nullptr;
    handle = nullptr;
    capacity = 0;
  }

  void ensure_capacity(std::size_t required) {
    if (required <= capacity) {
      return;
    }

    void* new_queries = nullptr;
    float* new_float_queries = nullptr;
    float* new_scores = nullptr;
    DeviceMatch* new_matches = nullptr;
    const std::size_t query_element_size =
        precision == SearchPrecision::fp32 ? sizeof(float) : sizeof(__nv_bfloat16);
    try {
      check_cuda(
          cudaMalloc(&new_queries, required * kTextDim * query_element_size),
          "cudaMalloc optimized queries");
      if (precision == SearchPrecision::bf16) {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&new_float_queries),
                       required * kTextDim * sizeof(float)),
            "cudaMalloc temporary BF16 queries");
      }
      check_cuda(
          cudaMalloc(reinterpret_cast<void**>(&new_scores),
                     required * count * sizeof(float)),
          "cudaMalloc optimized scores");
      check_cuda(
          cudaMalloc(reinterpret_cast<void**>(&new_matches),
                     required * sizeof(DeviceMatch)),
          "cudaMalloc optimized matches");
    } catch (...) {
      cudaFree(new_queries);
      cudaFree(new_float_queries);
      cudaFree(new_scores);
      cudaFree(new_matches);
      throw;
    }

    cudaFree(device_queries);
    cudaFree(device_float_queries);
    cudaFree(device_scores);
    cudaFree(device_matches);
    device_queries = new_queries;
    device_float_queries = new_float_queries;
    device_scores = new_scores;
    device_matches = new_matches;
    capacity = required;
  }

  std::uint32_t count;
  SearchPrecision precision;
  cublasHandle_t handle = nullptr;
  void* device_keys = nullptr;
  void* device_queries = nullptr;
  float* device_float_queries = nullptr;
  float* device_scores = nullptr;
  DeviceMatch* device_matches = nullptr;
  std::size_t capacity = 0;
#ifdef RAG_HAS_CUDA_TIMING
  std::unique_ptr<CudaEvent> total_start;
  std::unique_ptr<CudaEvent> h2d_start;
  std::unique_ptr<CudaEvent> h2d_end;
  std::unique_ptr<CudaEvent> conversion_start;
  std::unique_ptr<CudaEvent> conversion_end;
  std::unique_ptr<CudaEvent> gemm_start;
  std::unique_ptr<CudaEvent> gemm_end;
  std::unique_ptr<CudaEvent> reduction_start;
  std::unique_ptr<CudaEvent> reduction_end;
  std::unique_ptr<CudaEvent> d2h_start;
  std::unique_ptr<CudaEvent> d2h_end;
  std::unique_ptr<CudaEvent> total_end;
#endif
};

CudaOptimizedSearcher::CudaOptimizedSearcher(
    const Bank& bank,
    SearchPrecision precision)
    : impl_(std::make_unique<Impl>(bank, precision)) {}

CudaOptimizedSearcher::~CudaOptimizedSearcher() = default;

Match CudaOptimizedSearcher::search(
    const std::vector<float>& query,
    float threshold) {
  return search_batch(query, 1, threshold).front();
}

std::vector<Match> CudaOptimizedSearcher::search_batch(
    const std::vector<float>& queries,
    std::size_t query_count,
    float threshold) {
  return search_batch_timed(queries, query_count, threshold).matches;
}

TimedSearchResult CudaOptimizedSearcher::search_batch_timed(
    const std::vector<float>& queries,
    std::size_t query_count,
    float threshold) {
  validate_threshold(threshold);
  if (query_count == 0 || queries.size() != query_count * kTextDim) {
    throw std::invalid_argument("Query batch has an invalid shape");
  }

  SearchTiming timing;
  std::vector<float> normalized;
  const auto normalization_start = Clock::now();
  {
    RAG_PROFILE_RANGE("rag/normalize_cpu");
    normalized.reserve(queries.size());
    for (std::size_t index = 0; index < query_count; ++index) {
      const auto begin = queries.begin() + index * kTextDim;
      std::vector<float> query(begin, begin + kTextDim);
      std::vector<float> one_normalized = normalize_query(query);
      normalized.insert(
          normalized.end(), one_normalized.begin(), one_normalized.end());
    }
  }
  timing.cpu_normalization_us =
      elapsed_us(normalization_start, Clock::now());

  const auto preparation_start = Clock::now();
  {
    RAG_PROFILE_RANGE("rag/ensure_capacity");
  impl_->ensure_capacity(query_count);
  }
  timing.buffer_preparation_us =
      elapsed_us(preparation_start, Clock::now());
  const std::size_t element_count = query_count * kTextDim;
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(cudaEventRecord(impl_->total_start->get()), "record total start");
  check_cuda(cudaEventRecord(impl_->h2d_start->get()), "record H2D start");
#endif
  {
    RAG_PROFILE_RANGE("rag/h2d");
  if (impl_->precision == SearchPrecision::fp32) {
    check_cuda(
        cudaMemcpy(impl_->device_queries, normalized.data(),
                   element_count * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy optimized queries");
  } else {
    check_cuda(
        cudaMemcpy(impl_->device_float_queries, normalized.data(),
                   element_count * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy temporary queries");
  }
  }
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(cudaEventRecord(impl_->h2d_end->get()), "record H2D end");
  check_cuda(
      cudaEventRecord(impl_->conversion_start->get()),
      "record conversion start");
#endif
  {
    RAG_PROFILE_RANGE("rag/bf16_conversion");
  if (impl_->precision == SearchPrecision::bf16) {
    const auto blocks = static_cast<unsigned>(
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    convert_float_to_bfloat16<<<blocks, kThreadsPerBlock>>>(
        impl_->device_float_queries,
        static_cast<__nv_bfloat16*>(impl_->device_queries),
        element_count);
    check_cuda(cudaGetLastError(), "convert BF16 queries launch");
  }
  }
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(
      cudaEventRecord(impl_->conversion_end->get()),
      "record conversion end");
  check_cuda(cudaEventRecord(impl_->gemm_start->get()), "record GEMM start");
#endif

  const float alpha = 1.0f;
  const float beta = 0.0f;
  {
    RAG_PROFILE_RANGE("rag/gemm");
  if (impl_->precision == SearchPrecision::fp32) {
    check_cublas(
        cublasSgemm(
            impl_->handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(impl_->count),
            static_cast<int>(query_count),
            static_cast<int>(kTextDim),
            &alpha,
            static_cast<const float*>(impl_->device_keys),
            static_cast<int>(kTextDim),
            static_cast<const float*>(impl_->device_queries),
            static_cast<int>(kTextDim),
            &beta,
            impl_->device_scores,
            static_cast<int>(impl_->count)),
        "cublasSgemm");
  } else {
    check_cublas(
        cublasGemmEx(
            impl_->handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(impl_->count),
            static_cast<int>(query_count),
            static_cast<int>(kTextDim),
            &alpha,
            impl_->device_keys,
            CUDA_R_16BF,
            static_cast<int>(kTextDim),
            impl_->device_queries,
            CUDA_R_16BF,
            static_cast<int>(kTextDim),
            &beta,
            impl_->device_scores,
            CUDA_R_32F,
            static_cast<int>(impl_->count),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx BF16");
  }
  }
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(cudaEventRecord(impl_->gemm_end->get()), "record GEMM end");
  check_cuda(
      cudaEventRecord(impl_->reduction_start->get()),
      "record reduction start");
#endif

  {
    RAG_PROFILE_RANGE("rag/top1_reduction");
  select_best_on_device<<<
      static_cast<unsigned>(query_count), kThreadsPerBlock>>>(
      impl_->device_scores,
      impl_->count,
      query_count,
      threshold,
      impl_->device_matches);
  check_cuda(cudaGetLastError(), "select_best_on_device launch");
  }
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(
      cudaEventRecord(impl_->reduction_end->get()),
      "record reduction end");
  check_cuda(cudaEventRecord(impl_->d2h_start->get()), "record D2H start");
#endif

  std::vector<DeviceMatch> device_results(query_count);
  {
    RAG_PROFILE_RANGE("rag/d2h");
  check_cuda(
      cudaMemcpy(device_results.data(), impl_->device_matches,
                 query_count * sizeof(DeviceMatch), cudaMemcpyDeviceToHost),
      "cudaMemcpy optimized matches");
  }
#ifdef RAG_HAS_CUDA_TIMING
  check_cuda(cudaEventRecord(impl_->d2h_end->get()), "record D2H end");
  check_cuda(cudaEventRecord(impl_->total_end->get()), "record total end");
  check_cuda(cudaEventSynchronize(impl_->total_end->get()), "wait for timing");
  timing.h2d_us = event_elapsed_us(*impl_->h2d_start, *impl_->h2d_end);
  timing.conversion_us =
      event_elapsed_us(*impl_->conversion_start, *impl_->conversion_end);
  timing.gemm_us = event_elapsed_us(*impl_->gemm_start, *impl_->gemm_end);
  timing.reduction_us =
      event_elapsed_us(*impl_->reduction_start, *impl_->reduction_end);
  timing.d2h_us = event_elapsed_us(*impl_->d2h_start, *impl_->d2h_end);
  timing.device_total_us =
      event_elapsed_us(*impl_->total_start, *impl_->total_end);
  timing.h2d_bytes = element_count * sizeof(float);
  timing.d2h_bytes = query_count * sizeof(DeviceMatch);
  timing.device_timing_available = true;
#endif

  std::vector<Match> matches(query_count);
  const auto construction_start = Clock::now();
  {
    RAG_PROFILE_RANGE("rag/construct_matches");
    std::transform(
        device_results.begin(),
        device_results.end(),
        matches.begin(),
        [](const DeviceMatch& result) {
          return Match{result.accepted != 0, result.id, result.score};
        });
  }
  timing.host_result_construction_us =
      elapsed_us(construction_start, Clock::now());
  return TimedSearchResult{std::move(matches), timing};
}

}  // namespace rag
