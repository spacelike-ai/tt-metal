# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .utils import from_torch_fast

if TYPE_CHECKING:
    import torch


@dataclass
class RmsNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device | ttnn.MeshDevice | None = None,
    ) -> RmsNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"].unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
        )


class RmsNorm:
    def __init__(self, parameters: RmsNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self._weight, epsilon=self._eps)


@dataclass
class LayerNormParameters:
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device | ttnn.MeshDevice | None = None,
    ) -> LayerNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"], layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
            if "weight" in state
            else None,
            bias=from_torch_fast(state["bias"], layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
            if "bias" in state
            else None,
        )


class LayerNorm:
    def __init__(self, parameters: LayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias

    def forward(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.ProgramConfig | None = None,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self._weight,
            bias=self._bias,
            epsilon=self._eps,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
