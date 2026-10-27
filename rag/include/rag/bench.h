#pragma once

#include "rag/retrieval.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rag::bench {

using BatchSearch = std::function<TimedSearchResult(
    const std::vector<float>&, std::size_t, float)>;

struct Config {
  std::size_t batch_size = 1;
  std::size_t warmup_iterations = 10;
  std::size_t measured_iterations = 100;
  float threshold = 0.7f;
};

struct Record {
  std::string backend;
  std::size_t batch = 1;
  std::string mode;
  std::int64_t iteration = 0;
  std::int64_t query_index = -1;
  std::size_t query_count = 0;
  double end_to_end_us = 0.0;
  double vector_retrieval_us = 0.0;
  double payload_retrieval_us = 0.0;
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
  bool cuda_timing_enabled = false;
  bool nvtx_enabled = false;
  std::size_t payload_bytes = 0;
  std::size_t peak_host_rss_bytes = 0;
  Match match;
};

struct RunResult {
  std::vector<Record> records;
  double measured_wall_us = 0.0;
  std::size_t measured_queries = 0;
  std::size_t measured_submissions = 0;
};

std::vector<float> load_queries(const std::string& path);

RunResult run(
    const Bank& bank,
    const std::vector<float>& queries,
    const std::string& backend_name,
    const Config& config,
    const BatchSearch& search);

}  // namespace rag::bench
