#include "rag/search.h"

#include <cmath>
#include <stdexcept>

namespace rag {
namespace {

constexpr float kMinimumInitialScore = -2.0f;

void validate_threshold(float threshold) {
  const bool outside_cosine_range = threshold < -1.0f || threshold > 1.0f;
  if (!std::isfinite(threshold) || outside_cosine_range) {
    throw std::invalid_argument("Threshold must be in [-1, 1]");
  }
}

float dot_product(const float* left, const float* right) {
  double sum = 0.0;
  for (std::uint32_t index = 0; index < kTextDim; ++index) {
    sum += static_cast<double>(left[index]) * right[index];
  }
  return static_cast<float>(sum);
}

}  // namespace

std::vector<float> normalize_query(const std::vector<float>& query) {
  if (query.size() != kTextDim) {
    throw std::invalid_argument("Query must have 2048 floats");
  }

  double squared_norm = 0.0;
  for (const float value : query) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Non-finite query");
    }
    squared_norm += static_cast<double>(value) * value;
  }

  if (squared_norm == 0.0) {
    throw std::invalid_argument("Zero query");
  }

  const float inverse_norm =
      static_cast<float>(1.0 / std::sqrt(squared_norm));

  std::vector<float> normalized = query;
  for (float& value : normalized) {
    value *= inverse_norm;
  }
  return normalized;
}

Match search_cpu(
    const Bank& bank,
    const std::vector<float>& query,
    float threshold) {
  validate_threshold(threshold);
  const std::vector<float> normalized_query = normalize_query(query);

  Match best;
  best.score = kMinimumInitialScore;

  for (std::uint32_t id = 0; id < bank.size(); ++id) {
    const std::size_t row_start = static_cast<std::size_t>(id) * kTextDim;
    const float* key = bank.keys() + row_start;
    const float score = dot_product(normalized_query.data(), key);

    // If scores are equal, make the lower ID win
    if (score > best.score) {
      best.id = id;
      best.score = score;
    }
  }

  best.accepted = best.score >= threshold;
  return best;
}

}
