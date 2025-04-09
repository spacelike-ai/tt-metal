# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from .linear import Linear, LinearParameters
from .substate import substate


@dataclass
class EmbeddingParameters:
    linear_1: LinearParameters
    linear_2: LinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> EmbeddingParameters:
        return cls(
            linear_1=LinearParameters.from_torch(substate(state, "linear_1"), dtype=dtype, device=device),
            linear_2=LinearParameters.from_torch(substate(state, "linear_2"), dtype=dtype, device=device),
        )


@dataclass
class CombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: EmbeddingParameters
    text_embedder: EmbeddingParameters
    device: ttnn.MeshDevice

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> CombinedTimestepTextProjEmbeddingsParameters:
        return cls(
            timestep_embedder=EmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=EmbeddingParameters.from_torch(substate(state, "text_embedder"), dtype=dtype, device=device),
            device=device,
        )


class CombinedTimestepTextProjEmbeddings:
    def __init__(self, parameters: CombinedTimestepTextProjEmbeddingsParameters) -> None:
        super().__init__()

        device = parameters.device

        self._timestep_embedder = _TimestepEmbedding(parameters.timestep_embedder)
        self._text_embedder = _TimestepEmbedding(parameters.text_embedder)

        self._time_proj_factor = self._create_time_proj_factor(num_channels=256, device=device)

    def forward(self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32

        batch_size = timestep.shape[0]

        # ttnn.repeat currently does not work with mesh devices
        assert batch_size == 1
        time_proj_factor = self._time_proj_factor
        # time_proj_factor = ttnn.repeat(self._time_proj_factor, ttnn.Shape([batch_size, 1]))

        time_proj_factor = ttnn.to_layout(time_proj_factor, ttnn.TILE_LAYOUT)

        emb = timestep * time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)

        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

        time_embed = self._timestep_embedder.forward(timesteps_proj)
        text_embed = self._text_embedder.forward(pooled_projection)

        return time_embed + text_embed

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent).unsqueeze(0)

        return ttnn.from_torch(factor, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device))


class _TimestepEmbedding:
    def __init__(self, parameters: EmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = Linear(parameters.linear_1)
        self._linear_2 = Linear(parameters.linear_2)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1.forward(x)
        x = ttnn.silu(x)
        return self._linear_2.forward(x)
