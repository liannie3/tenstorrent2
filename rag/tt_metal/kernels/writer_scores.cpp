#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kFaceElements = 16 * 16;
constexpr uint32_t kCandidateWords = 8;

uint16_t canonical_bf16(const uint16_t bits) {
    return (bits & 0x7FFFU) == 0U ? 0U : bits;
}

bool is_nan_bf16(const uint16_t bits) {
    return (bits & 0x7F80U) == 0x7F80U && (bits & 0x007FU) != 0U;
}

uint16_t ordered_bf16(const uint16_t bits) {
    const uint16_t canonical = canonical_bf16(bits);
    return (canonical & 0x8000U) != 0U
        ? static_cast<uint16_t>(~canonical)
        : static_cast<uint16_t>(canonical ^ 0x8000U);
}

uint32_t first_row_offset(const uint32_t column) {
    return column < 16U ? column : kFaceElements + column - 16U;
}

}  // namespace

void kernel_main() {
    const uint32_t candidates_addr = get_arg_val<uint32_t>(0);
    const uint32_t worker = get_arg_val<uint32_t>(1);
    const uint32_t tile_begin = get_arg_val<uint32_t>(2);
    const uint32_t local_nt = get_arg_val<uint32_t>(3);
    const uint32_t valid_entries = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_scores = 16;
    constexpr uint32_t cb_candidate = 2;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto candidates = TensorAccessor(args, candidates_addr);

    uint16_t best_score = 0U;
    uint16_t best_ordered = 0U;
    uint32_t best_id = 0U;
    uint32_t valid = 0U;
    uint32_t error = 0U;

    for (uint32_t tile = 0; tile < local_nt; ++tile) {
        cb_wait_front(cb_scores, 1);
        const auto score_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            get_read_ptr(cb_scores));
        const uint32_t global_tile = tile_begin + tile;
        for (uint32_t column = 0; column < kTileWidth; ++column) {
            const uint32_t id = global_tile * kTileWidth + column;
            if (id >= valid_entries) {
                break;
            }
            const uint16_t score = canonical_bf16(
                score_ptr[first_row_offset(column)]);
            if (is_nan_bf16(score)) {
                error |= 1U;
                continue;
            }
            const uint16_t ordered = ordered_bf16(score);
            // IDs are visited in ascending order, so retaining the current
            // candidate on equality implements the lower-ID tie rule.
            if (valid == 0U || ordered > best_ordered) {
                best_score = score;
                best_ordered = ordered;
                best_id = id;
                valid = 1U;
            }
        }
        cb_pop_front(cb_scores, 1);
    }

    if (valid == 0U) {
        error |= 2U;
    }

    cb_reserve_back(cb_candidate, 1);
    auto candidate_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(cb_candidate));
    candidate_ptr[0] = best_score;
    candidate_ptr[1] = best_id;
    candidate_ptr[2] = valid;
    candidate_ptr[3] = error;
    for (uint32_t index = 4; index < kCandidateWords; ++index) {
        candidate_ptr[index] = 0U;
    }
    cb_push_back(cb_candidate, 1);
    cb_wait_front(cb_candidate, 1);
    noc_async_write_page(worker, candidates, get_read_ptr(cb_candidate));
    noc_async_write_barrier();
    cb_pop_front(cb_candidate, 1);
}
