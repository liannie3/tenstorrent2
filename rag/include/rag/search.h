#pragma once

#include "rag/bank.h"
#include <cstdint>
#include <vector>

namespace rag {

struct Match {
  bool accepted = false;
  std::uint32_t id = 0;
  float score = -1.0f;
};

// Shared validation and normalization used by both search backends.
std::vector<float> normalize_query(const std::vector<float>& query);

Match search_cpu(
    const Bank& bank,
    const std::vector<float>& query,
    float threshold = 0.7f);

#ifdef RAG_HAS_CUDA
// Owns persistent GPU buffers so bank keys are uploaded only once.
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
