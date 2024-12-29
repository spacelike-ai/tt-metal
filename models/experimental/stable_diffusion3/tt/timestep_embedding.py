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

    def __call__(self, *, torch_timestep: torch.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        torch_timesteps_proj = _time_proj(num_channels=256, timesteps=torch_timestep)
        timesteps_proj = ttnn.from_torch(
            torch_timesteps_proj, device=pooled_projection.device(), dtype=pooled_projection.dtype
        )

        return self._timestep_embedder(timesteps_proj) + self._text_embedder(pooled_projection)


class _TimestepEmbedding:
    def __init__(self, parameters: TtEmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = TtLinear(parameters.linear_1)
        self._linear_2 = TtLinear(parameters.linear_2)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1(x)
        # x = ttnn.silu(x) # imprecise
        x = ttnn.from_torch(
            torch.nn.functional.silu(ttnn.to_torch(x)),
            device=x.device(),
            layout=x.layout,
        )
        return self._linear_2(x)


# tensors involved here are so small, there is no real optimization potential by converting them to ttnn
def _time_proj(*, num_channels: int, timesteps: torch.Tensor) -> torch.Tensor:
    assert num_channels % 2 == 0
    half_dim = num_channels // 2

    max_period = 10000

    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
