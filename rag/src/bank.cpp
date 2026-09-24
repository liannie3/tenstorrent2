#include "rag/bank.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rag {
namespace {

constexpr std::array<char, 8> kMagic = {'R', 'A', 'G', 'B', 'A', 'N', 'K', '1'};
constexpr std::uint32_t kMaxEntries = 1'000'000;
constexpr std::uint32_t kMaxDescriptionBytes = 1024 * 1024;
constexpr double kNormalizedNormTolerance = 0.01;
constexpr std::uint64_t kHeaderBytes = 8 + 4 * sizeof(std::uint32_t);

template <typename T>
T read_value(std::istream& input) {
  T value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) {
    throw std::runtime_error("Truncated bank");
  }
  return value;
}

void validate_dimensions(
    std::uint32_t count,
    std::uint32_t text_dim,
    std::uint32_t visual_tokens,
    std::uint32_t visual_dim) {
  const bool invalid_count = count == 0 || count > kMaxEntries;
  const bool invalid_shape =
      text_dim != kTextDim ||
      visual_tokens != kVisualTokens ||
      visual_dim != kVisualDim;

  if (invalid_count || invalid_shape) {
    throw std::runtime_error("Unsupported bank dimensions or count");
  }
}

void validate_keys(const std::vector<float>& keys, std::uint32_t count) {
  for (std::uint32_t row = 0; row < count; ++row) {
    const std::size_t row_start = static_cast<std::size_t>(row) * kTextDim;
    double squared_norm = 0.0;

    for (std::uint32_t column = 0; column < kTextDim; ++column) {
      const float value = keys[row_start + column];
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite bank key");
      }
      squared_norm += static_cast<double>(value) * value;
    }

    // Unit-length keys make each dot product a cosine similarity.
    if (std::abs(squared_norm - 1.0) > kNormalizedNormTolerance) {
      throw std::runtime_error("Bank key is not normalized");
    }
  }
}

void validate_offsets(
    const std::vector<std::uint64_t>& offsets,
    std::uint64_t payload_start,
    std::uint64_t file_size) {
  for (const std::uint64_t offset : offsets) {
    const bool before_payloads = offset < payload_start;
    const bool missing_length_field = offset + sizeof(std::uint32_t) > file_size;

    if (before_payloads || missing_length_field) {
      throw std::runtime_error("Invalid payload offset");
    }
  }
}

}  // namespace

Bank::Bank(const std::string& path) : path_(path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open bank: " + path);
  }

  std::array<char, kMagic.size()> magic{};
  if (!input.read(magic.data(), magic.size()) || magic != kMagic) {
    throw std::runtime_error("Invalid bank magic");
  }

  count_ = read_value<std::uint32_t>(input);
  const auto text_dim = read_value<std::uint32_t>(input);
  const auto visual_tokens = read_value<std::uint32_t>(input);
  const auto visual_dim = read_value<std::uint32_t>(input);
  validate_dimensions(count_, text_dim, visual_tokens, visual_dim);

  const std::size_t key_count = static_cast<std::size_t>(count_) * kTextDim;
  keys_.resize(key_count);
  if (!input.read(
          reinterpret_cast<char*>(keys_.data()),
          keys_.size() * sizeof(float))) {
    throw std::runtime_error("Truncated bank keys");
  }
  validate_keys(keys_, count_);

  offsets_.resize(count_);
  if (!input.read(
          reinterpret_cast<char*>(offsets_.data()),
          offsets_.size() * sizeof(std::uint64_t))) {
    throw std::runtime_error("Truncated bank offsets");
  }

  input.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::uint64_t>(input.tellg());
  const std::uint64_t payload_start =
      kHeaderBytes +
      keys_.size() * sizeof(float) +
      offsets_.size() * sizeof(std::uint64_t);
  validate_offsets(offsets_, payload_start, file_size);
}

Payload Bank::load(std::uint32_t id) const {
  if (id >= count_) {
    throw std::out_of_range("Bank entry ID");
  }

  std::ifstream input(path_, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot reopen bank");
  }
  input.seekg(static_cast<std::streamoff>(offsets_[id]));

  const auto description_size = read_value<std::uint32_t>(input);
  if (description_size > kMaxDescriptionBytes) {
    throw std::runtime_error("Description too large");
  }

  Payload payload;
  payload.description.resize(description_size);
  if (!input.read(payload.description.data(), description_size)) {
    throw std::runtime_error("Truncated description");
  }

  payload.visual_features.resize(kVisualFeatureCount);
  if (!input.read(
          reinterpret_cast<char*>(payload.visual_features.data()),
          payload.visual_features.size() * sizeof(float))) {
    throw std::runtime_error("Truncated visual features");
  }

  return payload;
}

}  // namespace rag
