# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    import torch


@dataclass
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None = None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        unsqueeze_bias: bool = False,
    ) -> TtLinearParameters:
        # There appears to be an issue when using `ttnn.linear` if the input tensor has
        # shape (n, 1, m) and the bias has rank two. `unsqueeze_bias` makes bias a rank three
        # tensor to circumvent that.
        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
            if unsqueeze_bias:
                bias = bias.unsqueeze(0)
        else:
            bias = None

        return cls(
            weight=ttnn.from_torch(
                state["weight"].transpose(0, 1), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device
            ),
            bias=ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
            if bias is not None
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class TtLinear:
    def __init__(
        self,
        parameters: TtLinearParameters,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
    ) -> None:
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias

        self._memory_config = memory_config
        self._program_config = program_config
        self._core_grid = core_grid
        self._output_tile = output_tile

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_channels, "input tensor does not have the expected shape"

        return ttnn.linear(
            x,
            self._weight,
            bias=self._bias,
            memory_config=self._memory_config,
            program_config=self._program_config,
            core_grid=self._core_grid,
            output_tile=self._output_tile,
        )

    @property
    def device(self) -> ttnn.Device:
        return self._weight.device()
