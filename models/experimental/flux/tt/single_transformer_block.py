# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .attention import TtAttention, TtAttentionParameters
from .linear import TtLinear, TtLinearParameters
from .normalization import TtLayerNorm, TtLayerNormParameters
from .substate import substate
from .transformer_block import chunk_time

if TYPE_CHECKING:
    import torch


@dataclass
class TtFluxSingleTransformerBlockParameters:
    attn: TtAttentionParameters
    norm: TtLayerNormParameters
    time_embed: TtLinearParameters
    proj_mlp: TtLinearParameters
    proj_out: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        linear_on_host: bool = False,
    ) -> TtFluxSingleTransformerBlockParameters:
        return cls(
            attn=TtAttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            norm=TtLayerNormParameters.from_torch(substate(state, "norm"), dtype=dtype, device=device),
            time_embed=TtLinearParameters.from_torch(
                substate(state, "norm.linear"), dtype=dtype, device=device, unsqueeze_bias=True
            ),
            proj_mlp=TtLinearParameters.from_torch(
                substate(state, "proj_mlp"), dtype=dtype, device=device, on_host=linear_on_host
            ),
            proj_out=TtLinearParameters.from_torch(
                substate(state, "proj_out"), dtype=dtype, device=device, on_host=linear_on_host
            ),
        )


class TtFluxSingleTransformerBlock:
    def __init__(
        self,
        parameters: TtFluxSingleTransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        self._attn = TtAttention(parameters.attn, num_heads=num_heads)
        self._norm = TtLayerNorm(parameters.norm, eps=1e-6)
        self._time_embed = TtLinear(parameters.time_embed)

        self._proj_mlp = TtLinear(parameters.proj_mlp)
        self._proj_out = TtLinear(parameters.proj_out)

    def __call__(
        self,
        *,
        combined: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        t = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        time = self._time_embed(t)

        shift_msa, scale_msa, gate_msa = chunk_time(time, 3)
        norm_combined = self._norm(combined) * (1 + scale_msa) + shift_msa

        mlp_combined = self._proj_mlp(norm_combined)
        ttnn.gelu(mlp_combined, output_tensor=mlp_combined, fast_and_approximate_mode=False)
        attn, _ = self._attn(spatial=norm_combined, image_rotary_emb=image_rotary_emb)
        # TODO: PCC of attn seems a bit low

        ttnn.deallocate(norm_combined)

        additional = ttnn.concat([attn, mlp_combined], dim=2)
        additional = gate_msa * self._proj_out(additional)

        combined += additional

        return combined
        # return ttnn.clamp(combined, -65504, 65504)  # TODO: clamp gives worse PCC
