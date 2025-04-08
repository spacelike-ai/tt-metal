# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .attention import Attention, AttentionParameters
from .linear import Linear, LinearParameters
from .normalization import LayerNorm, LayerNormParameters
from .substate import substate
from .transformer_block import chunk_time

if TYPE_CHECKING:
    import torch


@dataclass
class FluxSingleTransformerBlockParameters:
    attn: AttentionParameters
    norm: LayerNormParameters
    time_embed: LinearParameters
    proj_mlp: LinearParameters
    proj_out: LinearParameters
    gather: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        linear_on_host: bool = False,
    ) -> FluxSingleTransformerBlockParameters:
        with ttnn.distribute(ttnn.ShardTensorToMesh(device, dim=-1)):
            proj_mlp = LinearParameters.from_torch(
                substate(state, "proj_mlp"), dtype=dtype, device=device, on_host=linear_on_host
            )
            proj_out = LinearParameters.from_torch(
                substate(state, "proj_out"), dtype=dtype, device=device, on_host=linear_on_host
            )

        return cls(
            attn=AttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            norm=LayerNormParameters.from_torch(substate(state, "norm"), dtype=dtype, device=device),
            time_embed=LinearParameters.from_torch(
                substate(state, "norm.linear"), dtype=dtype, device=device, unsqueeze_bias=True
            ),
            proj_mlp=proj_mlp,
            proj_out=proj_out,
            gather=device.get_num_devices() > 1,
        )


class FluxSingleTransformerBlock:
    def __init__(
        self,
        parameters: FluxSingleTransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        self._attn = Attention(parameters.attn, num_heads=num_heads)
        self._norm = LayerNorm(parameters.norm, eps=1e-6)
        self._time_embed = Linear(parameters.time_embed)

        self._proj_mlp = Linear(parameters.proj_mlp)
        self._proj_out = Linear(parameters.proj_out)

        self._gather = parameters.gather

    def forward(
        self,
        *,
        combined: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        t = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        time = self._time_embed.forward(t)

        shift_msa, scale_msa, gate_msa = chunk_time(time, 3)
        norm_combined = self._norm.forward(combined) * (1 + scale_msa) + shift_msa

        mlp_combined = self._proj_mlp.forward(norm_combined)
        ttnn.gelu(mlp_combined, output_tensor=mlp_combined, fast_and_approximate_mode=False)
        if self._gather:
            mlp_combined = ttnn.all_gather(mlp_combined, dim=-1)
        attn, _ = self._attn.forward(spatial=norm_combined, image_rotary_emb=image_rotary_emb)
        # TODO: PCC of attn seems a bit low

        del norm_combined

        additional = ttnn.concat([attn, mlp_combined], dim=2)
        proj_out = self._proj_out.forward(additional)
        if self._gather:
            proj_out = ttnn.all_gather(proj_out, dim=-1)
        additional = gate_msa * proj_out

        combined += additional

        return combined
        # return ttnn.clamp(combined, -65504, 65504)  # TODO: clamp gives worse PCC
