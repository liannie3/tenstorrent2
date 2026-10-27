#pragma once

#include "rag/bank.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rag {

// Search result
struct Match {
  bool accepted = false;
  std::uint32_t id = 0;
  float score = -1.0f;
};

struct SearchTiming {
  double cpu_normalization_us = 0.0;
  double buffer_preparation_us = 0.0;
  double host_result_construction_us = 0.0;
  double h2d_us = 0.0;
  double conversion_us = 0.0;
  double gemm_us = 0.0;
  double reduction_us = 0.0;
  double d2h_us = 0.0;
  double device_total_us = 0.0;
  std::size_t h2d_bytes = 0;
  std::size_t d2h_bytes = 0;
  bool device_timing_available = false;
};

struct TimedSearchResult {
  std::vector<Match> matches;
  SearchTiming timing;
};

// Shared preprocessing
std::vector<float> normalize_query(const std::vector<float>& query);

// CPU backend
Match search_cpu(
    const Bank& bank,
    const std::vector<float>& query,
    float threshold = 0.7f);

std::vector<Match> search_cpu_batch(
    const Bank& bank,
    const std::vector<float>& queries,
    std::size_t query_count,
    float threshold = 0.7f);

#ifdef RAG_HAS_CUDA
// Simple CUDA reference backend. Scores are reduced on the host.
class CudaSearcher {
 public:
  explicit CudaSearcher(const Bank& bank);
  ~CudaSearcher();

  CudaSearcher(const CudaSearcher&) = delete;
  CudaSearcher& operator=(const CudaSearcher&) = delete;

  Match search(
      const std::vector<float>& query,
      float threshold = 0.7f);

  std::vector<Match> search_batch(
      const std::vector<float>& queries,
      std::size_t query_count,
      float threshold = 0.7f);

 private:
  std::uint32_t count_;
  float* device_keys_ = nullptr;    // [count, 2048]
  float* device_query_ = nullptr;   // [2048]
  float* device_scores_ = nullptr;  // [count]
};

enum class SearchPrecision {
  fp32,
  bf16,
};

// Persistent optimized CUDA backend. It uses cuBLAS for similarity scoring,
// performs top-1 selection and thresholding on the device, and copies only the
// final Match values to the host.
class CudaOptimizedSearcher {
 public:
  explicit CudaOptimizedSearcher(
      const Bank& bank,
      SearchPrecision precision = SearchPrecision::fp32);
  ~CudaOptimizedSearcher();

  CudaOptimizedSearcher(const CudaOptimizedSearcher&) = delete;
  CudaOptimizedSearcher& operator=(const CudaOptimizedSearcher&) = delete;

  Match search(
      const std::vector<float>& query,
      float threshold = 0.7f);

  std::vector<Match> search_batch(
      const std::vector<float>& queries,
      std::size_t query_count,
      float threshold = 0.7f);

  TimedSearchResult search_batch_timed(
      const std::vector<float>& queries,
      std::size_t query_count,
      float threshold = 0.7f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace rag
