#pragma once

#include "rag/bank.h"
#include <cstdint>
#include <vector>

namespace rag {

// Search result
struct Match {
  bool accepted = false;
  std::uint32_t id = 0;
  float score = -1.0f;
};

// Shared preprocessing
std::vector<float> normalize_query(const std::vector<float>& query);

// CPU backend
Match search_cpu(
    const Bank& bank,
    const std::vector<float>& query,
    float threshold = 0.7f);

#ifdef RAG_HAS_CUDA
// CUDA backend
class CudaSearcher {
 public:
  explicit CudaSearcher(const Bank& bank);
  ~CudaSearcher();

  CudaSearcher(const CudaSearcher&) = delete;
  CudaSearcher& operator=(const CudaSearcher&) = delete;

  Match search(
      const std::vector<float>& query,
      float threshold = 0.7f);

 private:
  std::uint32_t count_;
  float* device_keys_ = nullptr;    // [count, 2048]
  float* device_query_ = nullptr;   // [2048]
  float* device_scores_ = nullptr;  // [count]
};
#endif

}  // namespace rag
