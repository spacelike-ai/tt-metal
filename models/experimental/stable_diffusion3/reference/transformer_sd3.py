from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

from .joint_transformer_block import JointTransformerBlock
from .normalization import AdaLayerNormContinuous
from .patch_embedding import PatchEmbed
from .timestep_embedding import CombinedTimestepTextProjEmbeddings


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/controlnet_sd3.py
class SD3Transformer2DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        *,
        sample_size: int = 128,  # noqa: ARG002
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[int, ...] = (),
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self._out_channels = out_channels
        self._patch_size = patch_size
        self._in_channels = in_channels

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )
        self.context_embedder = torch.nn.Linear(joint_attention_dim, caption_projection_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=i in dual_attention_layers,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim)
        self.proj_out = torch.nn.Linear(inner_dim, patch_size * patch_size * self._out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]

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

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                patch_count_y,
                patch_count_x,
                self._patch_size,
                self._patch_size,
                self._out_channels,
            )
        )

        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)

        return hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self._out_channels,
                patch_count_y * self._patch_size,
                patch_count_x * self._patch_size,
            )
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size
