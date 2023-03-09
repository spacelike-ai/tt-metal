#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_reduce.h"
namespace NAMESPACE
{

void math_main()
{
uint32_t int_scaler = get_compile_time_arg_val(0);
uint32_t Ht = get_compile_time_arg_val(1);
uint32_t Wt = get_compile_time_arg_val(2);
uint32_t NC = get_compile_time_arg_val(3);
llk_math_reduce_init<PoolType::SUM,ReduceDim::REDUCE_COL,MATH_FIDELITY>();
llk_math_pack_sync_init<SyncFull>();
for (uint32_t nc = 0U; nc < NC; nc++) {
  constexpr int onetile = 1;
  int reduce_dst_idx = 0;
  for (uint32_t wt = 0U; wt < Wt; ++wt) {
    llk_math_wait_for_dest_available<SyncFull>();
    for (uint32_t ht = 0U; ht < Ht; ++ht) {
      llk_math_reduce<PoolType::SUM,ReduceDim::REDUCE_COL,MATH_FIDELITY>(reduce_dst_idx);
    }
    llk_math_dest_section_done<SyncFull>();
  }
}
}
}
