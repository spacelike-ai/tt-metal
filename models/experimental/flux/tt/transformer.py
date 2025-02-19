# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtLayerNorm, TtLayerNormParameters
from .substate import indexed_substates, substate
from .timestep_embedding import TtCombinedTimestepTextProjEmbeddings, TtCombinedTimestepTextProjEmbeddingsParameters
from .transformer_block import (
    TtFluxSingleTransformerBlock,
    TtFluxSingleTransformerBlockParameters,
    TtTransformerBlock,
    TtTransformerBlockParameters,
    chunk_time,
)

if TYPE_CHECKING:
    import torch


@dataclass
class TtFluxTransformer2DModelParameters:
    x_embedder: TtLinearParameters
    time_text_embed: TtCombinedTimestepTextProjEmbeddingsParameters
    context_embedder: TtLinearParameters
    transformer_blocks: list[TtTransformerBlockParameters]
    single_transformer_blocks: list[TtFluxSingleTransformerBlockParameters]
    time_embed_out: TtLinearParameters
    norm_out: TtLayerNormParameters
    proj_out: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtFluxTransformer2DModelParameters:
        return cls(
            x_embedder=TtLinearParameters.from_torch(substate(state, "x_embedder"), dtype=dtype, device=device),
            time_text_embed=TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"), dtype=dtype, device=device
            ),
            context_embedder=TtLinearParameters.from_torch(
                substate(state, "context_embedder"), dtype=dtype, device=device
            ),
            transformer_blocks=[
                TtTransformerBlockParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "transformer_blocks")
            ],
            single_transformer_blocks=[
                TtFluxSingleTransformerBlockParameters.from_torch(s, dtype=dtype, device=device, linear_on_host=i > 10)
                for i, s in enumerate(indexed_substates(state, "single_transformer_blocks"))
            ],
            time_embed_out=TtLinearParameters.from_torch(
                substate(state, "norm_out.linear"), dtype=dtype, device=device, unsqueeze_bias=True
            ),
            norm_out=TtLayerNormParameters.from_torch(substate(state, "norm_out.norm"), dtype=dtype, device=device),
            proj_out=TtLinearParameters.from_torch(substate(state, "proj_out"), dtype=dtype, device=device),
        )


class TtFluxTransformer2DModel:
    def __init__(self, parameters: TtFluxTransformer2DModelParameters, *, num_attention_heads: int) -> None:
        super().__init__()

        self._x_embedder = TtLinear(parameters.x_embedder)
        self._time_text_embed = TtCombinedTimestepTextProjEmbeddings(parameters.time_text_embed)
        self._context_embedder = TtLinear(parameters.context_embedder)
        self._transformer_blocks = [
            TtTransformerBlock(block, num_heads=num_attention_heads) for block in parameters.transformer_blocks
        ]
        self._single_transformer_blocks = [
            TtFluxSingleTransformerBlock(block, num_heads=num_attention_heads)
            for block in parameters.single_transformer_blocks
        ]
        self._time_embed_out = TtLinear(parameters.time_embed_out)
        self._norm_out = TtLayerNorm(parameters.norm_out, eps=1e-6)
        self._proj_out = TtLinear(parameters.proj_out)

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        height, width = list(spatial.shape)[-2:]

        spatial = self._x_embedder(spatial)
        time_embed = self._time_text_embed(timestep=timestep, pooled_projection=pooled_projection)
        prompt = self._context_embedder(prompt)

        time_embed = time_embed.reshape([time_embed.shape[0], 1, time_embed.shape[1]])

        for i, block in enumerate(self._transformer_blocks, start=1):
            print(f"iteration {i}...")
            spatial, prompt = block(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                image_rotary_emb=image_rotary_emb,
            )

            if i % 6 == 0:
                ttnn.DumpDeviceProfiler(spatial.device())

        combined = ttnn.concat([prompt, spatial], dim=1)

        for i, block in enumerate(self._single_transformer_blocks, start=1):
            print(f"single iteration {i}...")
            combined = block(
                combined=combined,
                time_embed=time_embed,
                image_rotary_emb=image_rotary_emb,
            )

            if i % 6 == 0:
                ttnn.DumpDeviceProfiler(combined.device())

        spatial = combined[:, prompt.shape[1] :]

        spatial_time = self._time_embed_out(ttnn.silu(time_embed))
        [scale, shift] = chunk_time(spatial_time, 2)
        spatial = self._norm_out(spatial) * (1 + scale) + shift

        return self._proj_out(spatial)
