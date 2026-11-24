#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kPageWords = 8;

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

uint32_t canonical_fp32(const uint32_t bits) {
    return (bits & 0x7FFFFFFFU) == 0U ? 0U : bits;
}

uint32_t ordered_fp32(const uint32_t bits) {
    const uint32_t canonical = canonical_fp32(bits);
    return (canonical & 0x80000000U) != 0U
        ? ~canonical
        : canonical ^ 0x80000000U;
}

}  // namespace

void kernel_main() {
    const uint32_t candidates_addr = get_arg_val<uint32_t>(0);
    const uint32_t result_addr = get_arg_val<uint32_t>(1);
    const uint32_t candidate_count = get_arg_val<uint32_t>(2);
    const uint32_t threshold_fp32 = get_arg_val<uint32_t>(3);
    const uint32_t valid_entries = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_candidate = 0;
    constexpr uint32_t cb_result = 1;

    constexpr auto candidate_args = TensorAccessorArgs<0>();
    const auto candidates = TensorAccessor(candidate_args, candidates_addr);
    constexpr auto result_args =
        TensorAccessorArgs<candidate_args.next_compile_time_args_offset()>();
    const auto result = TensorAccessor(result_args, result_addr);

    uint16_t best_score = 0U;
    uint16_t best_ordered = 0U;
    uint32_t best_id = 0U;
    uint32_t valid = 0U;
    uint32_t error = 0U;

    for (uint32_t worker = 0; worker < candidate_count; ++worker) {
        cb_reserve_back(cb_candidate, 1);
        noc_async_read_page(
            worker, candidates, get_write_ptr(cb_candidate));
        noc_async_read_barrier();
        cb_push_back(cb_candidate, 1);
        cb_wait_front(cb_candidate, 1);
        const auto page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(cb_candidate));
        const uint16_t score = canonical_bf16(
            static_cast<uint16_t>(page[0]));
        const uint32_t id = page[1];
        const uint32_t candidate_valid = page[2];
        const uint32_t candidate_error = page[3];
        if (candidate_error != 0U || candidate_valid != 1U ||
            id >= valid_entries || is_nan_bf16(score)) {
            error |= 1U;
        } else {
            const uint16_t ordered = ordered_bf16(score);
            if (valid == 0U || ordered > best_ordered ||
                (ordered == best_ordered && id < best_id)) {
                best_score = score;
                best_ordered = ordered;
                best_id = id;
                valid = 1U;
            }
        }
        for (uint32_t index = 4; index < kPageWords; ++index) {
            if (page[index] != 0U) {
                error |= 2U;
            }
        }
        cb_pop_front(cb_candidate, 1);
    }

    if (valid == 0U) {
        error |= 4U;
    }
    const uint32_t score_fp32 = static_cast<uint32_t>(best_score) << 16U;
    const uint32_t accepted =
        error == 0U && ordered_fp32(score_fp32) >= ordered_fp32(threshold_fp32)
        ? 1U
        : 0U;

    cb_reserve_back(cb_result, 1);
    auto output = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(cb_result));
    output[0] = accepted;
    output[1] = best_id;
    output[2] = score_fp32;
    output[3] = error;
    for (uint32_t index = 4; index < kPageWords; ++index) {
        output[index] = 0U;
    }
    cb_push_back(cb_result, 1);
    cb_wait_front(cb_result, 1);
    noc_async_write_page(0, result, get_read_ptr(cb_result));
    noc_async_write_barrier();
    cb_pop_front(cb_result, 1);
}
