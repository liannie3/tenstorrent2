#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t mt = get_arg_val<uint32_t>(1);
    const uint32_t global_nt = get_arg_val<uint32_t>(2);
    const uint32_t tile_begin = get_arg_val<uint32_t>(3);
    const uint32_t local_nt = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_scores = 16;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto scores = TensorAccessor(args, scores_addr);

    for (uint32_t m = 0; m < mt; ++m) {
        for (uint32_t n = 0; n < local_nt; ++n) {
            cb_wait_front(cb_scores, 1);
            noc_async_write_page(
                m * global_nt + tile_begin + n,
                scores,
                get_read_ptr(cb_scores));
            noc_async_write_barrier();
            cb_pop_front(cb_scores, 1);
        }
    }
}
