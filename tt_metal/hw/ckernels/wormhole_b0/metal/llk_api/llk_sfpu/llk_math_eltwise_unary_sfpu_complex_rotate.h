// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_complex_rotate.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_complex_rotate_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::complex_rotate, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_complex_rotate(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_complex_rotate<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
