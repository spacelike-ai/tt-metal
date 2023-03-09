#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

void pack_main()
{
uint32_t scaler = get_compile_time_arg_val(0);
llk_pack_init();
llk_pack_reduce_hw_configure_disaggregated<false,PoolType::SUM,ReduceDim::REDUCE_SCALAR>(16);
llk_setup_outputs();
llk_pack_dest_init<SyncFull, DstTileFaceLayout::RowMajor, false>();
uint32_t Ht = get_compile_time_arg_val(1);
uint32_t Wt = get_compile_time_arg_val(2);
uint32_t NC = get_compile_time_arg_val(3);
for (uint32_t nc = 0U; nc < NC; nc++) {
  constexpr int onetile = 1;
  int reduce_dst_idx = 0;
// tiles are expected to be coming in in NCHW order (W-contiguous)
// reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
// in this case we just sequentially add to accumulator all the W-tiles in a row
  llk_packer_wait_for_math_done();
  llk_wait_for_free_tiles<false,false,false>(16,1);
  llk_pack<false, SyncFull, false >(reduce_dst_idx,16);
  llk_push_tiles<false,false>(16,1);
  llk_pack_dest_section_done<SyncFull>();
}
}
}
