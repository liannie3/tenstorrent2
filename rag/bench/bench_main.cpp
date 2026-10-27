#include "rag/bench.h"
#include "rag/profile.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kUsage =
    "Usage: rag_bench BANK QUERIES EXPECTED "
    "[--backend cpu|cuda-reference|cuda-optimized] "
    "[--precision fp32|bf16] [--batch N] [--warmup N] [--iters N] "
    "[--threshold SCORE]";

struct Options {
  std::string bank_path;
  std::string queries_path;
  std::string expected_path;
  std::string backend = "cpu";
  std::string precision = "fp32";
  rag::bench::Config benchmark;
};

std::size_t parse_size(
    const std::string& value,
    const char* name,
    bool allow_zero = false) {
  std::size_t consumed = 0;
  if (!value.empty() && value.front() == '-') {
    throw std::invalid_argument(std::string(name) + " cannot be negative");
  }
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || (!allow_zero && parsed == 0)) {
    throw std::invalid_argument(
        std::string(name) + (allow_zero ? " must be non-negative" : " must be positive"));
  }
  return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  if (argc < 4) {
    throw std::invalid_argument(kUsage);
  }

  Options options;
  options.bank_path = argv[1];
  options.queries_path = argv[2];
  options.expected_path = argv[3];
  for (int index = 4; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--cuda") {
      options.backend = "cuda-reference";
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("Missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--backend") {
      options.backend = value;
    } else if (argument == "--precision") {
      options.precision = value;
    } else if (argument == "--batch") {
      options.benchmark.batch_size = parse_size(value, "batch");
    } else if (argument == "--warmup") {
      options.benchmark.warmup_iterations = parse_size(value, "warmup", true);
    } else if (argument == "--iters") {
      options.benchmark.measured_iterations = parse_size(value, "iters");
    } else if (argument == "--threshold") {
      options.benchmark.threshold = std::stof(value);
    } else {
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }

  if (!std::filesystem::is_regular_file(options.expected_path)) {
    throw std::invalid_argument("EXPECTED must name an existing file");
  }
  if (options.precision != "fp32" && options.precision != "bf16") {
    throw std::invalid_argument("precision must be fp32 or bf16");
  }
  return options;
}

void print_record(const rag::bench::Record& record) {
  std::cout << record.backend << ','
            << record.batch << ','
            << record.mode << ','
            << record.iteration << ','
            << record.query_index << ','
            << record.query_count << ','
            << std::setprecision(12) << record.end_to_end_us << ','
            << record.vector_retrieval_us << ','
            << record.payload_retrieval_us << ',';
  const auto print_timing = [&record](double value) {
    if (record.device_timing_available) {
      std::cout << value;
    }
    std::cout << ',';
  };
  print_timing(record.cpu_normalization_us);
  print_timing(record.buffer_preparation_us);
  print_timing(record.host_result_construction_us);
  print_timing(record.h2d_us);
  print_timing(record.conversion_us);
  print_timing(record.gemm_us);
  print_timing(record.reduction_us);
  print_timing(record.d2h_us);
  print_timing(record.device_total_us);
  if (record.device_timing_available) {
    std::cout << record.h2d_bytes;
  }
  std::cout << ',';
  if (record.device_timing_available) {
    std::cout << record.d2h_bytes;
  }
  std::cout << ','
            << record.device_timing_available << ','
            << record.cuda_timing_enabled << ','
            << record.nvtx_enabled << ','
            << record.payload_bytes << ','
            << record.peak_host_rss_bytes << ','
            << record.match.accepted << ','
            << record.match.id << ','
            << std::setprecision(9) << record.match.score << '\n';
}

rag::bench::Record make_phase_record(
    const std::string& backend,
    const std::string& mode,
    double duration_us) {
  rag::bench::Record record;
  record.backend = backend;
  record.mode = mode;
  record.end_to_end_us = duration_us;
  return record;
}

void apply_timing(
    rag::bench::Record& record,
    const rag::SearchTiming& timing) {
  record.cpu_normalization_us = timing.cpu_normalization_us;
  record.buffer_preparation_us = timing.buffer_preparation_us;
  record.host_result_construction_us = timing.host_result_construction_us;
  record.h2d_us = timing.h2d_us;
  record.conversion_us = timing.conversion_us;
  record.gemm_us = timing.gemm_us;
  record.reduction_us = timing.reduction_us;
  record.d2h_us = timing.d2h_us;
  record.device_total_us = timing.device_total_us;
  record.h2d_bytes = timing.h2d_bytes;
  record.d2h_bytes = timing.d2h_bytes;
  record.device_timing_available = timing.device_timing_available;
#ifdef RAG_HAS_CUDA_TIMING
  record.cuda_timing_enabled = true;
#endif
#ifdef RAG_HAS_NVTX
  record.nvtx_enabled = true;
#endif
}

std::size_t peak_host_rss_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(
          GetCurrentProcess(), &counters, sizeof(counters))) {
    throw std::runtime_error("Cannot read peak host RSS");
  }
  return static_cast<std::size_t>(counters.PeakWorkingSetSize);
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("Cannot read peak host RSS");
  }
#ifdef __APPLE__
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const auto process_start = Clock::now();
  try {
    const Options options = parse_options(argc, argv);

    const auto bank_start = Clock::now();
    const rag::Bank bank(options.bank_path);
    const auto bank_finish = Clock::now();
    const auto query_load_start = Clock::now();
    const std::vector<float> queries =
        rag::bench::load_queries(options.queries_path);
    const auto query_load_finish = Clock::now();

    rag::bench::BatchSearch search;
#ifdef RAG_HAS_CUDA
    std::unique_ptr<rag::CudaSearcher> cuda_reference;
    std::unique_ptr<rag::CudaOptimizedSearcher> cuda_optimized;
#endif

    const auto backend_init_start = Clock::now();
    {
      RAG_PROFILE_RANGE("rag/backend_init");
      if (options.backend == "cpu") {
      search = [&bank](const std::vector<float>& batch,
                       std::size_t count,
                       float threshold) {
        return rag::TimedSearchResult{
            rag::search_cpu_batch(bank, batch, count, threshold), {}};
      };
#ifdef RAG_HAS_CUDA
    } else if (options.backend == "cuda-reference") {
      cuda_reference = std::make_unique<rag::CudaSearcher>(bank);
      search = [&cuda_reference](const std::vector<float>& batch,
                                 std::size_t count,
                                 float threshold) {
        return rag::TimedSearchResult{
            cuda_reference->search_batch(batch, count, threshold), {}};
      };
    } else if (options.backend == "cuda-optimized") {
      const auto precision = options.precision == "bf16"
          ? rag::SearchPrecision::bf16
          : rag::SearchPrecision::fp32;
      cuda_optimized =
          std::make_unique<rag::CudaOptimizedSearcher>(bank, precision);
      search = [&cuda_optimized](const std::vector<float>& batch,
                                 std::size_t count,
                                 float threshold) {
        return cuda_optimized->search_batch_timed(batch, count, threshold);
      };
#endif
      } else {
        throw std::invalid_argument(
            "Backend is unavailable in this build: " + options.backend);
      }
    }
    const auto backend_init_finish = Clock::now();

    const std::vector<float> first_query(
        queries.begin(), queries.begin() + rag::kTextDim);
    const auto first_request_start = Clock::now();
    rag::TimedSearchResult cold_search;
    Clock::time_point first_vector_finish;
    rag::RetrievalResponse cold_response;
    {
      RAG_PROFILE_RANGE("rag/first_request");
      {
        RAG_PROFILE_RANGE("rag/vector_retrieval");
        cold_search = search(first_query, 1, options.benchmark.threshold);
      }
      first_vector_finish = Clock::now();
      if (cold_search.matches.size() != 1) {
        throw std::runtime_error("Backend returned the wrong result count");
      }
      {
        RAG_PROFILE_RANGE("rag/payload_retrieval");
        cold_response = rag::retrieve_payload(bank, cold_search.matches.front());
      }
    }
    const auto first_result = Clock::now();

    std::cout
        << "backend,batch,mode,iteration,query_index,query_count,"
           "end_to_end_us,vector_retrieval_us,payload_retrieval_us,"
           "cpu_normalization_us,buffer_preparation_us,"
           "host_result_construction_us,h2d_us,conversion_us,gemm_us,"
           "reduction_us,d2h_us,device_total_us,h2d_bytes,d2h_bytes,"
           "device_timing_available,cuda_timing_enabled,nvtx_enabled,"
           "payload_bytes,peak_host_rss_bytes,accepted,id,score\n";
    print_record(make_phase_record(
        options.backend,
        "bank_load",
        std::chrono::duration<double, std::micro>(
            bank_finish - bank_start).count()));
    print_record(make_phase_record(
        options.backend,
        "query_load",
        std::chrono::duration<double, std::micro>(
            query_load_finish - query_load_start).count()));
    print_record(make_phase_record(
        options.backend,
        "backend_init",
        std::chrono::duration<double, std::micro>(
            backend_init_finish - backend_init_start).count()));
    rag::bench::Record cold_start = make_phase_record(
        options.backend,
        "cold_start",
        std::chrono::duration<double, std::micro>(
            first_result - process_start).count());
    cold_start.query_count = 1;
    cold_start.payload_bytes = rag::payload_bytes(cold_response);
    cold_start.match = cold_response.match;
    print_record(cold_start);

    rag::bench::Record first_request;
    first_request.backend = options.backend;
    first_request.mode = "first_request";
    first_request.query_count = 1;
    first_request.end_to_end_us =
        std::chrono::duration<double, std::micro>(
            first_result - first_request_start).count();
    first_request.vector_retrieval_us =
        std::chrono::duration<double, std::micro>(
            first_vector_finish - first_request_start).count();
    first_request.payload_retrieval_us = cold_response.match.accepted
        ? std::chrono::duration<double, std::micro>(
              first_result - first_vector_finish).count()
        : 0.0;
    first_request.payload_bytes = rag::payload_bytes(cold_response);
    first_request.match = cold_response.match;
    apply_timing(first_request, cold_search.timing);
    print_record(first_request);

    const rag::bench::RunResult result = rag::bench::run(
        bank, queries, options.backend, options.benchmark, search);
    for (const rag::bench::Record& record : result.records) {
      print_record(record);
    }
    rag::bench::Record measured_phase = make_phase_record(
        options.backend, "measured_phase", result.measured_wall_us);
    measured_phase.batch = options.benchmark.batch_size;
    measured_phase.query_count = result.measured_queries;
    measured_phase.peak_host_rss_bytes = peak_host_rss_bytes();
    measured_phase.iteration =
        static_cast<std::int64_t>(result.measured_submissions);
    print_record(measured_phase);
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << "rag_bench: " << error.what() << '\n' << kUsage << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "rag_bench: " << error.what() << '\n';
    return 1;
  }
}
