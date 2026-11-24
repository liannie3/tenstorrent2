#include "rag/bank.h"
#include "rag/search.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load_first_query(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> query(rag::kTextDim);
  if (!input.read(
          reinterpret_cast<char*>(query.data()),
          static_cast<std::streamsize>(query.size() * sizeof(float)))) {
    throw std::runtime_error("unable to read first query from " + path);
  }
  return query;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: rag_tt_metal_smoke BANK QUERY [THRESHOLD]\n";
    return 2;
  }
  try {
    const float threshold = argc == 4 ? std::stof(argv[3]) : 0.7F;
    const rag::Bank bank(argv[1]);
    const std::vector<float> query = load_first_query(argv[2]);
    const rag::Match expected = rag::search_cpu(bank, query, threshold);
    rag::TtMetalSearcher searcher(bank, threshold);
    const rag::Match actual = searcher.search(query);

    std::cout << "expected: accepted=" << expected.accepted
              << " id=" << expected.id << " score=" << expected.score << '\n';
    std::cout << "metalium: accepted=" << actual.accepted
              << " id=" << actual.id << " score=" << actual.score << '\n';
    if (actual.id != expected.id || actual.accepted != expected.accepted) {
      std::cerr << "multicore TT-Metal RAG scoring smoke: FAIL\n";
      return 1;
    }
    std::cout << "multicore TT-Metal RAG scoring smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rag_tt_metal_smoke: " << error.what() << '\n';
    return 1;
  }
}
