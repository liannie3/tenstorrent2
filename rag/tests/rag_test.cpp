#include "rag/search.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

// Bank fixture
template <typename T>
void write_value(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_test_bank(const std::filesystem::path& path) {
  constexpr std::uint32_t kEntryCount = 2;
  constexpr std::uint32_t kDescriptionBytes = 5;
  constexpr std::uint64_t kHeaderBytes = 24;
  constexpr std::uint64_t kOffsetTableBytes =
      kEntryCount * sizeof(std::uint64_t);

  std::ofstream output(path, std::ios::binary);
  output.write("RAGBANK1", 8);

  for (const std::uint32_t value : {
           kEntryCount,
           rag::kTextDim,
           rag::kVisualTokens,
           rag::kVisualDim}) {
    write_value(output, value);
  }

  std::vector<float> keys(
      static_cast<std::size_t>(kEntryCount) * rag::kTextDim);
  keys[0] = 1.0f;
  keys[rag::kTextDim + 1] = 1.0f;
  output.write(
      reinterpret_cast<const char*>(keys.data()),
      keys.size() * sizeof(float));

  const std::uint64_t first_payload =
      kHeaderBytes + keys.size() * sizeof(float) + kOffsetTableBytes;
  const std::uint64_t payload_bytes =
      sizeof(std::uint32_t) +
      kDescriptionBytes +
      rag::kVisualFeatureCount * sizeof(float);
  const std::uint64_t second_payload = first_payload + payload_bytes;
  write_value(output, first_payload);
  write_value(output, second_payload);

  const std::vector<float> visual_features(rag::kVisualFeatureCount);
  for (const char* description : {"first", "other"}) {
    write_value(output, kDescriptionBytes);
    output.write(description, kDescriptionBytes);
    output.write(
        reinterpret_cast<const char*>(visual_features.data()),
        visual_features.size() * sizeof(float));
  }
}

}  // namespace

int main() {
  // Fixture setup
  const auto path =
      std::filesystem::temp_directory_path() / "rag_test.bank";
  write_test_bank(path);

  const rag::Bank bank(path.string());
  std::vector<float> query(rag::kTextDim);

  query[1] = 2.0f;
  rag::Match match = rag::search_cpu(bank, query, 0.7f);
  assert(match.accepted);
  assert(match.id == 1);
  assert(std::abs(match.score - 1.0f) < 1e-6f);
  assert(bank.load(match.id).description == "other");

  // Retrieval cases
  match = rag::search_cpu(bank, query, 1.0f);
  assert(match.accepted);

  query[0] = 2.0f;
  match = rag::search_cpu(bank, query, 0.8f);
  assert(!match.accepted);
  assert(match.id == 0);

  query[0] = 0.0f; 
  query[1] = 0.0f;
  // Validation cases
  bool rejected = false;
  try {
    rag::search_cpu(bank, query);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  std::filesystem::remove(path);
}
