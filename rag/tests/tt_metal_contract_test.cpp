#include "rag/tt_metal_contract.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable) {
  try {
    callable();
  } catch (const Exception&) {
    return;
  }
  assert(false && "expected exception was not thrown");
}

void test_partitions() {
  for (const auto [entries, workers] :
       std::vector<std::pair<std::uint32_t, std::uint32_t>>{
           {1, 64}, {31, 8}, {32, 8}, {33, 8}, {3400, 63}}) {
    const auto ranges = rag::tt_metal::partition_entries(entries, workers);
    assert(ranges.size() == std::min(entries, workers));
    assert(ranges.front().begin == 0);
    assert(ranges.back().end == entries);
    std::uint32_t smallest = entries;
    std::uint32_t largest = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      assert(ranges[i].begin < ranges[i].end);
      if (i != 0) {
        assert(ranges[i - 1].end == ranges[i].begin);
      }
      const auto size = ranges[i].end - ranges[i].begin;
      smallest = std::min(smallest, size);
      largest = std::max(largest, size);
    }
    assert(largest - smallest <= 1);
  }
  expect_throws<std::invalid_argument>(
      [] { rag::tt_metal::partition_entries(0, 1); });
  expect_throws<std::invalid_argument>(
      [] { rag::tt_metal::partition_entries(1, 0); });
}

void test_reduction() {
  using rag::tt_metal::Candidate;
  assert(rag::tt_metal::better_candidate(
             Candidate{0.5F, 9, true}, Candidate{0.5F, 2, true})
             .id == 2);
  assert(rag::tt_metal::better_candidate(
             Candidate{0.5F, 2, true}, Candidate{100.0F, 99, false})
             .id == 2);
  assert(rag::tt_metal::reduce_candidates({{0.7F, 4, true}}, 0.7F).valid);
  assert(!rag::tt_metal::reduce_candidates({{0.699F, 4, true}}, 0.7F)
              .valid);
  assert(rag::tt_metal::reduce_candidates(
             {{0.8F, 7, true}, {0.8F, 3, true}}, 0.7F)
             .id == 3);
  expect_throws<std::invalid_argument>([] {
    rag::tt_metal::better_candidate(
        {std::numeric_limits<float>::quiet_NaN(), 0, true}, {});
  });
  expect_throws<std::invalid_argument>(
      [] { rag::tt_metal::reduce_candidates({{0.1F, 0, false}}, 0.0F); });
  expect_throws<std::invalid_argument>([] {
    rag::tt_metal::reduce_candidates(
        {{0.1F, 0, true}}, std::numeric_limits<float>::infinity());
  });
}

void test_wire_contract() {
  const auto encoded =
      rag::tt_metal::encode_result({0.875F, 17, true});
  assert(encoded.accepted == 1 && encoded.id == 17);
  const auto decoded = rag::tt_metal::decode_result(encoded, 18);
  assert(decoded.accepted && decoded.id == 17);
  assert(std::abs(decoded.score - 0.875F) < 1e-7F);
  expect_throws<std::runtime_error>([] {
    rag::tt_metal::decode_result({2, 0, 1.0F}, 1);
  });
  expect_throws<std::out_of_range>([] {
    rag::tt_metal::decode_result({0, 10, 0.1F}, 10);
  });
  expect_throws<std::runtime_error>([] {
    rag::tt_metal::decode_result(
        {1, 0, std::numeric_limits<float>::quiet_NaN()}, 1);
  });
  expect_throws<std::out_of_range>([] {
    rag::tt_metal::decode_result({0, 0, 0.1F}, 0);
  });
}

void test_sharded_compact_result() {
  using rag::tt_metal::compact_result_from_scores;
  const std::vector<float> scores{0.1F, 0.8F, 0.2F, 0.8F, 0.7F};
  for (const std::uint32_t workers : {1U, 2U, 3U, 5U, 64U}) {
    const auto accepted = compact_result_from_scores(scores, workers, 0.8F);
    assert(accepted.accepted == 1U);
    assert(accepted.id == 1U);
    assert(std::abs(accepted.score - 0.8F) < 1e-7F);

    const auto rejected = compact_result_from_scores(scores, workers, 0.81F);
    assert(rejected.accepted == 0U);
    assert(rejected.id == 1U);
  }
  expect_throws<std::invalid_argument>(
      [] { compact_result_from_scores({}, 1, 0.7F); });
  expect_throws<std::invalid_argument>(
      [] { compact_result_from_scores({0.5F}, 0, 0.7F); });
  expect_throws<std::invalid_argument>([] {
    compact_result_from_scores(
        {std::numeric_limits<float>::quiet_NaN()}, 1, 0.7F);
  });
}

}  // namespace

int main() {
  test_partitions();
  test_reduction();
  test_wire_contract();
  test_sharded_compact_result();
}
