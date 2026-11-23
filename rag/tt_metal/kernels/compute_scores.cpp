#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const std::uint32_t mt = get_compile_time_arg_val(0);
    const std::uint32_t kt = get_compile_time_arg_val(1);
    const std::uint32_t nt = get_compile_time_arg_val(2);
    constexpr tt::CBIndex query = tt::CBIndex::c_0;
    constexpr tt::CBIndex bank = tt::CBIndex::c_1;
    constexpr tt::CBIndex scores = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(query, bank, scores);
    matmul_init(query, bank);
    for (std::uint32_t m = 0; m < mt; ++m) {
        for (std::uint32_t n = 0; n < nt; ++n) {
            tile_regs_acquire();
            for (std::uint32_t k = 0; k < kt; ++k) {
                cb_wait_front(query, 1);
                cb_wait_front(bank, 1);
                matmul_tiles(query, bank, 0, 0, 0);
                cb_pop_front(query, 1);
                cb_pop_front(bank, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(scores, 1);
            pack_tile(0, scores);
            cb_push_back(scores, 1);
            tile_regs_release();
        }
    }
}
