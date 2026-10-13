#pragma once

#include "rag/retrieval.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rag::bench {

using BatchSearch = std::function<std::vector<Match>(
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
