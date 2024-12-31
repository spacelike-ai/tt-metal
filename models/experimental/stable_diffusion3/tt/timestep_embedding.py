from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinear, TtLinearParameters

from .substate import substate


@dataclass
class TtEmbeddingParameters:
    linear_1: TtLinearParameters
    linear_2: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtEmbeddingParameters:
        return cls(
            linear_1=TtLinearParameters.from_torch(substate(state, "linear_1"), dtype=dtype, device=device),
            linear_2=TtLinearParameters.from_torch(substate(state, "linear_2"), dtype=dtype, device=device),
        )


@dataclass
class TtCombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: TtEmbeddingParameters
    text_embedder: TtEmbeddingParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtCombinedTimestepTextProjEmbeddingsParameters:
        return cls(
            timestep_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "text_embedder"), dtype=dtype, device=device
            ),
        )


class TtCombinedTimestepTextProjEmbeddings:
    def __init__(self, parameters: TtCombinedTimestepTextProjEmbeddingsParameters) -> None:
        super().__init__()

        self._timestep_embedder = _TimestepEmbedding(parameters.timestep_embedder)
        self._text_embedder = _TimestepEmbedding(parameters.text_embedder)

    def __call__(self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        timesteps_proj = _time_proj(num_channels=256, timesteps=timestep)

        time_embed = self._timestep_embedder(timesteps_proj)
        text_embed = self._text_embedder(pooled_projection)

        return time_embed + text_embed


class _TimestepEmbedding:
    def __init__(self, parameters: TtEmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = TtLinear(parameters.linear_1)
        self._linear_2 = TtLinear(parameters.linear_2)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1(x)
        x = ttnn.silu(x)
        return self._linear_2(x)


def _time_proj(*, num_channels: int, timesteps: ttnn.Tensor) -> ttnn.Tensor:
    assert num_channels % 2 == 0
    half_dim = num_channels // 2

    max_period = 10000

    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / half_dim

    emb = torch.exp(exponent).unsqueeze(0)
    emb = ttnn.to_torch(timesteps) * emb
    result = torch.concat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return ttnn.from_torch(result, device=timesteps.device(), dtype=timesteps.dtype, layout=timesteps.layout)

    # TODO
    # torch_emb = torch.exp(exponent).unsqueeze(0)
    # emb = ttnn.from_torch(torch_emb, device=timesteps.device(), dtype=timesteps.dtype, layout=timesteps.layout)
    # emb = timesteps * emb
    # return ttnn.concat([ttnn.cos(emb), ttnn.sin(emb)], dim=-1)
