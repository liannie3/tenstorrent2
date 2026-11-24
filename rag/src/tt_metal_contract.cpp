#include "rag/tt_metal_contract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace rag::tt_metal {
namespace {

void validate_candidate(const Candidate& candidate) {
  if (candidate.valid && !std::isfinite(candidate.score)) {
    throw std::invalid_argument("valid candidate score must be finite");
  }
}

}  // namespace

static_assert(std::is_standard_layout_v<CompactResult>);
static_assert(std::is_trivially_copyable_v<CompactResult>);
static_assert(sizeof(CompactResult) == 12);

std::vector<WorkRange> partition_entries(
    const std::uint32_t valid_entries,
    const std::uint32_t available_workers) {
  if (valid_entries == 0 || available_workers == 0) {
    throw std::invalid_argument("entry and worker counts must be positive");
  }

  const std::uint32_t active_workers =
      std::min(valid_entries, available_workers);
  const std::uint32_t base = valid_entries / active_workers;
  const std::uint32_t extra = valid_entries % active_workers;

  std::vector<WorkRange> ranges;
  ranges.reserve(active_workers);
  std::uint32_t begin = 0;
  for (std::uint32_t worker = 0; worker < active_workers; ++worker) {
    const std::uint32_t size = base + (worker < extra ? 1U : 0U);
    ranges.push_back({begin, begin + size});
    begin += size;
  }
  return ranges;
}

Candidate better_candidate(const Candidate& left, const Candidate& right) {
  validate_candidate(left);
  validate_candidate(right);
  if (!left.valid) {
    return right;
  }
  if (!right.valid) {
    return left;
  }
  if (right.score > left.score ||
      (right.score == left.score && right.id < left.id)) {
    return right;
  }
  return left;
}

Candidate reduce_candidates(
    const std::vector<Candidate>& candidates,
    const float threshold) {
  if (candidates.empty() || !std::isfinite(threshold)) {
    throw std::invalid_argument(
        "candidate list must be non-empty and threshold finite");
  }

  Candidate winner{};
  for (const Candidate& candidate : candidates) {
    winner = better_candidate(winner, candidate);
  }
  if (!winner.valid) {
    throw std::invalid_argument("at least one candidate must be valid");
  }
  winner.valid = winner.score >= threshold;
  return winner;
}

CompactResult compact_result_from_scores(
    const std::vector<float>& scores,
    const std::uint32_t available_workers,
    const float threshold) {
  const auto ranges = partition_entries(
      static_cast<std::uint32_t>(scores.size()), available_workers);
  std::vector<Candidate> local_winners;
  local_winners.reserve(ranges.size());
  for (const WorkRange range : ranges) {
    Candidate local_winner{};
    for (std::uint32_t id = range.begin; id < range.end; ++id) {
      local_winner = better_candidate(
          local_winner, Candidate{scores[id], id, true});
    }
    local_winners.push_back(local_winner);
  }
  return encode_result(reduce_candidates(local_winners, threshold));
}

CompactResult encode_result(const Candidate& candidate) {
  if (!std::isfinite(candidate.score)) {
    throw std::invalid_argument("compact result score must be finite");
  }
  return {candidate.valid ? 1U : 0U, candidate.id, candidate.score};
}

Match decode_result(
    const CompactResult& result,
    const std::uint32_t valid_entries) {
  if (result.accepted > 1U || !std::isfinite(result.score)) {
    throw std::runtime_error("device returned an invalid compact result");
  }
  if (valid_entries == 0 || result.id >= valid_entries) {
    throw std::out_of_range("device result id is outside the valid bank");
  }
  return {result.accepted != 0U, result.id, result.score};
}

}  // namespace rag::tt_metal
