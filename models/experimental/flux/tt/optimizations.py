# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any

import ttnn


class AttentionPartType(Enum):
    SPATIAL = 1
    PROMPT = 2


class AttentionPartOptimization:
    def __init__(self, part: AttentionPartType, /) -> None:
        self._part = part

    def prepare_qkv_projection(self, x: ttnn.Tensor) -> ttnn.Tensor:
        cores = x.device().compute_with_storage_grid_size()

        # if self._part == AttentionPartType.SPATIAL:
        #     memory_config = ttnn.create_sharded_memory_config(
        #         x.shape,
        #         core_grid=ttnn.CoreGrid(x=cores.x, y=cores.y),
        #         strategy=ttnn.ShardStrategy.BLOCK,
        #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     )

        #     return ttnn.to_memory_config(x, memory_config)

        return x

    def qkv_projection_settings(self, device: ttnn.Device) -> dict[str, Any]:
        cores = device.compute_with_storage_grid_size()

        if self._part == AttentionPartType.SPATIAL:
            return {
                # "core_grid": ttnn.CoreGrid(x=cores.x, y=cores.y),
                # "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            }

        return {
            # "core_grid": ttnn.CoreGrid(x=8, y=6),
            # "memory_config": ttnn.L1_MEMORY_CONFIG,
        }

    def prepare_split(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x = ttnn.reallocate(x)
        # x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return x

    def postprocess_split(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        # q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        # k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        # v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        return q, k, v

    def postprocess_out_projection(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # return ttnn.to_memory_config(
        #     result,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     dtype=ttnn.bfloat16,
        # )
        return x


class AttentionOptimization:
    def spatial_part(self) -> AttentionPartOptimization:
        return AttentionPartOptimization(AttentionPartType.SPATIAL)

    def prompt_part(self) -> AttentionPartOptimization:
        return AttentionPartOptimization(AttentionPartType.PROMPT)

    def sdpa_settings(self, *, device: ttnn.Device) -> dict[str, Any]:
        return {
            "program_config": ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=256,
                k_chunk_size=512,
                exp_approx_mode=True,
            ),
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            ),
        }
