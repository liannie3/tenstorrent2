#include "rag/search.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kDefaultThreshold = 0.7f;
constexpr const char* kUsage =
    "Usage: rag_query BANK QUERY_F32 [THRESHOLD] [--cuda]";

struct Options {
  std::string bank_path;
  std::string query_path;
  float threshold = kDefaultThreshold;
  bool use_cuda = false;
};

Options parse_options(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    throw std::invalid_argument(kUsage);
  }

  Options options;
  options.bank_path = argv[1];
  options.query_path = argv[2];

  bool threshold_was_set = false;
  for (int index = 3; index < argc; ++index) {
    const std::string argument = argv[index];

    if (argument == "--cuda") {
      if (options.use_cuda) {
        throw std::invalid_argument("--cuda was provided more than once");
      }
      options.use_cuda = true;
      continue;
    }

    if (threshold_was_set) {
      throw std::invalid_argument("Only one threshold may be provided");
    }
    options.threshold = std::stof(argument);
    threshold_was_set = true;
  }

  return options;
}

std::vector<float> read_query(const std::string& path) {
  constexpr std::size_t kQueryBytes =
      static_cast<std::size_t>(rag::kTextDim) * sizeof(float);

  // Opening at the end lets us validate the file size before allocating or reading.
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != static_cast<std::streamoff>(kQueryBytes)) {
    throw std::runtime_error(
        "Query file must contain exactly 2048 float32 values");
  }

  input.seekg(0);
  std::vector<float> query(rag::kTextDim);
  if (!input.read(reinterpret_cast<char*>(query.data()), kQueryBytes)) {
    throw std::runtime_error("Cannot read query file");
  }

  return query;
}

rag::Match run_search(
    const rag::Bank& bank,
    const std::vector<float>& query,
    const Options& options) {
  if (!options.use_cuda) {
    return rag::search_cpu(bank, query, options.threshold);
  }

#ifdef RAG_HAS_CUDA
  rag::CudaSearcher searcher(bank);
  return searcher.search(query, options.threshold);
#else
  throw std::runtime_error(
      "Build with -DRAG_ENABLE_CUDA=ON to use --cuda");
#endif
}

void print_result(const rag::Bank& bank, const rag::Match& match) {
  std::cout << std::setprecision(9)
            << "accepted=" << match.accepted
            << " id=" << match.id
            << " score=" << match.score << '\n';

  if (!match.accepted) {
    return;
  }

  // Large visual features stay in the payload; the CLI prints only their shape.
  const rag::Payload payload = bank.load(match.id);
  std::cout << "description=" << payload.description << '\n'
            << "visual_shape=" << rag::kVisualTokens
            << 'x' << rag::kVisualDim << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const rag::Bank bank(options.bank_path);
    const std::vector<float> query = read_query(options.query_path);
    const rag::Match match = run_search(bank, query, options);
    print_result(bank, match);
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << "rag_query: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "rag_query: " << error.what() << '\n';
    return 1;
  }
}
