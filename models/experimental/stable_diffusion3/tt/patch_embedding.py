from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn

from .conv2d import TtConv2d, TtConv2dParameters
from .substate import substate


@dataclass
class TtPatchEmbedParameters:
    proj: TtConv2dParameters
    pos_embed: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtPatchEmbedParameters:
        return cls(
            proj=TtConv2dParameters.from_torch(substate(state, "proj"), dtype=dtype, device=device),
            pos_embed=ttnn.from_torch(state["pos_embed"], dtype=dtype, device=device),
        )


class TtPatchEmbed:
    def __init__(self, parameters: TtPatchEmbedParameters) -> None:
        super().__init__()

        weight_shape = list(parameters.proj.weight.shape)
        self._pos_embed_max_size = math.isqrt(parameters.pos_embed.shape[1])

        self._proj = TtConv2d(parameters.proj, stride=weight_shape[-2:])
        self._pos_embed = parameters.pos_embed

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self._proj(latent)

        batch_size, c, height, width = list(latent.shape)

        assert latent.layout == ttnn.ROW_MAJOR_LAYOUT
        latent = latent.reshape([batch_size, c, height * width])

        # latent = ttnn.transpose(latent, 1, 2)
        latent = ttnn.from_torch(ttnn.to_torch(latent).transpose(1, 2), device=latent.device())

        pos_embed = self._cropped_pos_embed(height, width)

        return ttnn.tilize(latent) + ttnn.tilize(pos_embed)

    def _cropped_pos_embed(self, height: int, width: int) -> ttnn.Tensor:
        top = (self._pos_embed_max_size - height) // 2
        left = (self._pos_embed_max_size - width) // 2

        spatial_pos_embed = self._pos_embed.reshape([1, self._pos_embed_max_size, self._pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
