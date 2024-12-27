from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinear, TtLinearParameters

from .normalization import TtLayerNorm, TtLayerNormParameters
from .patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from .substate import has_substate, substate
from .timestep_embedding import TtCombinedTimestepTextProjEmbeddings, TtCombinedTimestepTextProjEmbeddingsParameters
from .transformer_block import TtTransformerBlock, TtTransformerBlockParameters


@dataclass
class TtSD3Transformer2DModelParameters:
    pos_embed: TtPatchEmbedParameters
    time_text_embed: TtCombinedTimestepTextProjEmbeddingsParameters
    context_embedder: TtLinearParameters
    transformer_blocks: list[TtTransformerBlockParameters]
    norm_out: TtLayerNormParameters
    proj_out: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtSD3Transformer2DModelParameters:
        transformer_blocks = []
        for i in itertools.count():
            key = f"transformer_blocks.{i}"
            if not has_substate(state, key):
                break

            transformer_blocks.append(
                TtTransformerBlockParameters.from_torch(substate(state, key), dtype=dtype, device=device)
            )

            if i == 2:
                break  # TODO: remove

        return cls(
            pos_embed=TtPatchEmbedParameters.from_torch(substate(state, "pos_embed"), dtype=dtype, device=device),
            time_text_embed=TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"), dtype=dtype, device=device
            ),
            context_embedder=TtLinearParameters.from_torch(
                substate(state, "context_embedder"), dtype=dtype, device=device
            ),
            transformer_blocks=transformer_blocks,
            norm_out=TtLayerNormParameters.from_torch(substate(state, "norm_out"), dtype=dtype, device=device),
            proj_out=TtLinearParameters.from_torch(substate(state, "proj_out"), dtype=dtype, device=device),
        )


class TtSD3Transformer2DModel:
    def __init__(
        self,
        parameters: TtSD3Transformer2DModelParameters,
        *,
        # in_channels: int = 16,
        num_attention_heads: int,
    ) -> None:
        super().__init__()

        self._pos_embed = TtPatchEmbed(parameters.pos_embed)
        self._time_text_embed = TtCombinedTimestepTextProjEmbeddings(parameters.time_text_embed)
        self._context_embedder = TtLinear(parameters.context_embedder)
        self._transformer_blocks = [
            TtTransformerBlock(block, num_heads=num_attention_heads) for block in parameters.transformer_blocks
        ]
        self._norm_out = TtLayerNorm(parameters.norm_out, eps=1e-6)
        self._proj_out = TtLinear(parameters.proj_out)

        # self._out_channels = out_channels
        self._patch_size = parameters.pos_embed.patch_size

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt_embed: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        torch_timestep: torch.Tensor,
    ) -> ttnn.Tensor:
        height, width = list(spatial.shape)[-2:]

        spatial = self._pos_embed(spatial)
        time_embed = self._time_text_embed(torch_timestep=torch_timestep, pooled_projection=pooled_projection)
        prompt_embed = self._context_embedder(prompt_embed)

        time_embed = ttnn.from_torch(
            ttnn.to_torch(time_embed).unsqueeze(1),
            device=time_embed.device(),
            layout=ttnn.TILE_LAYOUT,
        )
        # time_embed = ttnn.untilize(time_embed)
        # time_embed = time_embed.reshape([time_embed.shape[0], 1, time_embed.shape[1]])
        # time_embed = ttnn.tilize(time_embed)

        for block in self._transformer_blocks[0:1]:  # TODO: remove range
            spatial, prompt_embed = block(
                spatial=spatial,
                prompt=prompt_embed,
                time_embed=time_embed,
            )
            return prompt_embed

        time_embed = self._norm_out.linear(torch.nn.functional.silu(time_embed))
        scale, shift = torch.chunk(time_embed, 2, dim=1)
        spatial = self._norm_out.norm(spatial) * (1 + scale)[:, None, :] + shift[:, None, :]

        spatial = self._proj_out(spatial)

        patch_count_y = height // self._patch_size[0]
        patch_count_x = width // self._patch_size[1]

        spatial = spatial.reshape(
            shape=(
                spatial.shape[0],
                patch_count_y,
                patch_count_x,
                self._patch_size[0],
                self._patch_size[1],
                self._out_channels,
            )
        )

        spatial = torch.einsum("nhwpqc->nchpwq", spatial)

        return spatial.reshape(
            shape=(
                spatial.shape[0],
                self._out_channels,
                patch_count_y * self._patch_size,
                patch_count_x * self._patch_size,
            )
        )
