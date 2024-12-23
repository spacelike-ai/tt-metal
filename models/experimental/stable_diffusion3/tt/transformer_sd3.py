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
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 18,
        # attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        # attention_dim: int = 4096,
        # caption_projection_dim: int = 1152,
        # pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        # pos_embed_max_size: int = 96,
        # dual_attention_layers: tuple[int, ...] = (),
        # qk_norm: str = "rms_norm",
        # device: ttnn.Device,
        # parameters: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()

        eps = 123123123213

        self._pos_embed = TtPatchEmbed(parameters.pos_embed)
        self._time_text_embed = TtCombinedTimestepTextProjEmbeddings(parameters.time_text_embed)
        self._context_embedder = TtLinear(parameters.context_embedder)
        self._transformer_blocks = [TtTransformerBlock(block) for block in parameters.transformer_blocks]
        self._norm_out = TtLayerNorm(parameters.norm_out, eps=eps)
        self._proj_out = TtLinear(parameters.proj_out)

        # inner_dim = num_attention_heads * attention_head_dim

        # self._device = device
        # self._out_channels = out_channels
        # self._patch_size = patch_size
        # self._in_channels = in_channels

        # self.pos_embed = TtPatchEmbed(
        #     patch_size=patch_size,
        #     in_channels=in_channels,
        #     embed_dim=inner_dim,
        #     pos_embed_max_size=pos_embed_max_size,
        # )
        # self.time_text_embed = TtCombinedTimestepTextProjEmbeddings(
        #     embedding_dim=inner_dim,
        #     pooled_projection_dim=pooled_projection_dim,
        # )
        # self.context_embedder = torch.nn.Linear(attention_dim, caption_projection_dim)

        # self.transformer_blocks = torch.nn.ModuleList(
        #     [
        #         TtTransformerBlock(
        #             dim=inner_dim,
        #             num_heads=num_attention_heads,
        #             head_dim=attention_head_dim,
        #             context_pre_only=i == num_layers - 1,
        #             qk_norm=qk_norm,
        #             use_dual_attention=i in dual_attention_layers,
        #         )
        #         for i in range(num_layers)
        #     ]
        # )

        # self.norm_out = TtAdaLayerNormContinuous(inner_dim, inner_dim)
        # self.proj_out = torch.nn.Linear(inner_dim, patch_size * patch_size * self._out_channels)

    def __call__(
        self,
        spatial: ttnn.Tensor,
        prompt_embed: ttnn.Tensor,
        pooled_projections: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        height, width = list(spatial.shape)[-2:]

        # spatial = self.pos_embed(spatial)
        # time_embed = self.time_text_embed(timestep, pooled_projections)
        # prompt_embed = self.context_embedder(prompt_embed)

        # for block in self.transformer_blocks:
        #     prompt_embed, spatial = block(
        #         spatial=spatial,
        #         prompt=prompt_embed,
        #         time_embed=time_embed,
        #     )

        # time_embed = self.norm_out.linear(torch.nn.functional.silu(time_embed))
        # scale, shift = torch.chunk(time_embed, 2, dim=1)
        # spatial = self.norm_out.norm(spatial) * (1 + scale)[:, None, :] + shift[:, None, :]

        # spatial = self.proj_out(spatial)

        # patch_count_y = height // self._patch_size
        # patch_count_x = width // self._patch_size

        # spatial = spatial.reshape(
        #     shape=(
        #         spatial.shape[0],
        #         patch_count_y,
        #         patch_count_x,
        #         self._patch_size,
        #         self._patch_size,
        #         self._out_channels,
        #     )
        # )

        # spatial = torch.einsum("nhwpqc->nchpwq", spatial)

        # return spatial.reshape(
        #     shape=(
        #         spatial.shape[0],
        #         self._out_channels,
        #         patch_count_y * self._patch_size,
        #         patch_count_x * self._patch_size,
        #     )
        # )
