#include "rag/bench.h"
#include "rag/profile.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace rag::bench {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> slice_queries(
    const std::vector<float>& queries,
    std::size_t query_offset,
    std::size_t query_count) {
  const auto begin = queries.begin() + query_offset * kTextDim;
  return std::vector<float>(begin, begin + query_count * kTextDim);
}

double elapsed_us(Clock::time_point start, Clock::time_point finish) {
  return std::chrono::duration<double, std::micro>(finish - start).count();
}

}  // namespace

std::vector<float> load_queries(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Cannot open query corpus: " + path);
  }

  const auto byte_count = input.tellg();
  const auto bytes_per_query =
      static_cast<std::streamoff>(kTextDim * sizeof(float));
  if (byte_count <= 0 || byte_count % bytes_per_query != 0) {
    throw std::runtime_error(
        "Query corpus must contain complete 2048-float queries");
  }

  input.seekg(0);
  std::vector<float> queries(
      static_cast<std::size_t>(byte_count) / sizeof(float));
  if (!input.read(
          reinterpret_cast<char*>(queries.data()),
          static_cast<std::streamsize>(byte_count))) {
    throw std::runtime_error("Cannot read query corpus");
  }
  return queries;
}

RunResult run(
    const Bank& bank,
    const std::vector<float>& queries,
    const std::string& backend_name,
    const Config& config,
    const BatchSearch& search) {
  if (config.batch_size == 0 || config.measured_iterations == 0) {
    throw std::invalid_argument("Batch size and iterations must be positive");
  }
  if (queries.empty() || queries.size() % kTextDim != 0) {
    throw std::invalid_argument("Query corpus has an invalid shape");
  }

  const std::size_t corpus_size = queries.size() / kTextDim;
  if (corpus_size % config.batch_size != 0) {
    throw std::invalid_argument(
        "Query count must be divisible by batch size; partial batches would "
        "mislabel the workload");
  }

  const auto execute_pass = [&](bool record,
                                std::vector<Record>& records,
                                std::int64_t& submission) {
    for (std::size_t offset = 0; offset < corpus_size;
         offset += config.batch_size) {
      RAG_PROFILE_RANGE("rag/request");
      const std::size_t count = config.batch_size;
      const std::vector<float> batch = slice_queries(queries, offset, count);
      const auto start = Clock::now();
      TimedSearchResult search_result;
      {
        RAG_PROFILE_RANGE("rag/vector_retrieval");
        search_result = search(batch, count, config.threshold);
      }
      const auto vector_finish = Clock::now();
      const std::vector<Match>& matches = search_result.matches;
      if (matches.size() != count) {
        throw std::runtime_error("Backend returned the wrong result count");
      }
      std::vector<RetrievalResponse> responses;
      {
        RAG_PROFILE_RANGE("rag/payload_retrieval");
        responses = retrieve_payloads(bank, matches);
      }
      const auto response_finish = Clock::now();
      const double vector_retrieval_us = elapsed_us(start, vector_finish);
      const bool has_accepted_response = std::any_of(
          matches.begin(), matches.end(),
          [](const Match& match) { return match.accepted; });
      const double payload_retrieval_us = has_accepted_response
          ? elapsed_us(vector_finish, response_finish)
          : 0.0;
      const double end_to_end_us = elapsed_us(start, response_finish);

      if (record) {
        for (std::size_t query = 0; query < responses.size(); ++query) {
          Record measured;
          measured.backend = backend_name;
          measured.batch = count;
          measured.mode = "measured";
          measured.iteration = submission;
          measured.query_index =
              static_cast<std::int64_t>(offset + query);
          measured.query_count = count;
          measured.end_to_end_us = end_to_end_us;
          measured.vector_retrieval_us = vector_retrieval_us;
          measured.payload_retrieval_us = payload_retrieval_us;
          measured.cpu_normalization_us =
              search_result.timing.cpu_normalization_us;
          measured.buffer_preparation_us =
              search_result.timing.buffer_preparation_us;
          measured.host_result_construction_us =
              search_result.timing.host_result_construction_us;
          measured.h2d_us = search_result.timing.h2d_us;
          measured.conversion_us = search_result.timing.conversion_us;
          measured.gemm_us = search_result.timing.gemm_us;
          measured.reduction_us = search_result.timing.reduction_us;
          measured.d2h_us = search_result.timing.d2h_us;
          measured.device_total_us = search_result.timing.device_total_us;
          measured.h2d_bytes = search_result.timing.h2d_bytes;
          measured.d2h_bytes = search_result.timing.d2h_bytes;
          measured.device_timing_available =
              search_result.timing.device_timing_available;
#ifdef RAG_HAS_CUDA_TIMING
          measured.cuda_timing_enabled = true;
#endif
#ifdef RAG_HAS_NVTX
          measured.nvtx_enabled = true;
#endif
          measured.payload_bytes = rag::payload_bytes(responses[query]);
          measured.match = responses[query].match;
          records.push_back(measured);
        }
        ++submission;
      }
    }
  };

  std::vector<Record> ignored;
  std::int64_t ignored_submission = 0;
  {
    RAG_PROFILE_RANGE("rag/warmup_phase");
    for (std::size_t pass = 0; pass < config.warmup_iterations; ++pass) {
      execute_pass(false, ignored, ignored_submission);
    }
  }

  RunResult result;
  result.records.reserve(config.measured_iterations * corpus_size);
  std::int64_t submission = 0;
  const auto measured_start = Clock::now();
  {
    RAG_PROFILE_RANGE("rag/measured_phase");
    for (std::size_t pass = 0; pass < config.measured_iterations; ++pass) {
      execute_pass(true, result.records, submission);
    }
  }
  result.measured_wall_us = elapsed_us(measured_start, Clock::now());
  result.measured_queries = config.measured_iterations * corpus_size;
  result.measured_submissions = static_cast<std::size_t>(submission);
  return result;
}

}  // namespace rag::bench
