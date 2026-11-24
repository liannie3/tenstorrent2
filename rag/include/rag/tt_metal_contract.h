#pragma once

#include "rag/search.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rag::tt_metal {

// Host-side specification shared by the launcher and its portable tests.  This
// header deliberately contains no Metalium types so non-Tenstorrent builds do
// not acquire a dependency on the device SDK.
struct WorkRange {
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
};

struct Candidate {
  float score = 0.0f;
  std::uint32_t id = 0;
  bool valid = false;
};

// Stable 12-byte device/host result contract.  A fixed-width accepted field is
// used instead of bool because bool's object representation is not a wire ABI.
struct CompactResult {
  std::uint32_t accepted = 0;
  std::uint32_t id = 0;
  float score = 0.0f;
};

std::vector<WorkRange> partition_entries(
    std::uint32_t valid_entries,
    std::uint32_t available_workers);

Candidate better_candidate(const Candidate& left, const Candidate& right);

Candidate reduce_candidates(
    const std::vector<Candidate>& candidates,
    float threshold);

// Portable oracle for the device reduction pipeline. Scores are partitioned
// exactly as the multicore launcher partitions bank entries, each shard emits
// one local winner, and the global winner is encoded with the wire contract.
// The TT-Metal kernels must produce byte-for-byte equivalent results.
CompactResult compact_result_from_scores(
    const std::vector<float>& scores,
    std::uint32_t available_workers,
    float threshold);

CompactResult encode_result(const Candidate& candidate);
Match decode_result(const CompactResult& result, std::uint32_t valid_entries);

}  // namespace rag::tt_metal
