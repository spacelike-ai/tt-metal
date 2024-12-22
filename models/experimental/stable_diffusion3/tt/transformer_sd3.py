from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinearParameters

from .normalization import TtLayerNormParameters
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
        *,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[int, ...] = (),
        qk_norm: str = "rms_norm",
        device: ttnn.Device,
        parameters: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self._device = device
        self._out_channels = out_channels
        self._patch_size = patch_size
        self._in_channels = in_channels

        self.pos_embed = TtPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = TtCombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )
        self.context_embedder = torch.nn.Linear(attention_dim, caption_projection_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=i in dual_attention_layers,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = TtAdaLayerNormContinuous(inner_dim, inner_dim)
        self.proj_out = torch.nn.Linear(inner_dim, patch_size * patch_size * self._out_channels)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        pooled_projections: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        height, width = list(hidden_states.shape_without_padding())[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_count_y = height // self._patch_size
        patch_count_x = width // self._patch_size

        torch_hidden_states = hidden_states.to_torch(dtype=torch.float32)

        torch_hidden_states = torch_hidden_states.reshape(
            shape=(
                torch_hidden_states.shape[0],
                patch_count_y,
                patch_count_x,
                self._patch_size,
                self._patch_size,
                self._out_channels,
            )
        )

        torch_hidden_states = torch.einsum("nhwpqc->nchpwq", torch_hidden_states)
        torch_hidden_states = torch_hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self._out_channels,
                patch_count_y * self._patch_size,
                patch_count_x * self._patch_size,
            )
        )

        return ttnn.from_torch(
            torch_hidden_states,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=hidden_states.device,
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size
