#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t query_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_addr = get_arg_val<uint32_t>(1);
    const uint32_t mt = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_query = 0;
    constexpr uint32_t cb_bank = 1;
    constexpr auto query_args = TensorAccessorArgs<0>();
    const auto query = TensorAccessor(query_args, query_addr);
    constexpr auto bank_args =
        TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    const auto bank = TensorAccessor(bank_args, bank_addr);

    for (uint32_t m = 0; m < mt; ++m) {
        for (uint32_t n = 0; n < nt; ++n) {
            for (uint32_t k = 0; k < kt; ++k) {
                cb_reserve_back(cb_query, 1);
                noc_async_read_page(m * kt + k, query, get_write_ptr(cb_query));
                noc_async_read_barrier();
                cb_push_back(cb_query, 1);

                cb_reserve_back(cb_bank, 1);
                noc_async_read_page(k * nt + n, bank, get_write_ptr(cb_bank));
                noc_async_read_barrier();
                cb_push_back(cb_bank, 1);
            }
        }
    }
}
