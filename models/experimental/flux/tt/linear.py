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
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    on_host: bool
    device: ttnn.Device | ttnn.MeshDevice | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device | ttnn.MeshDevice | None,
        on_host: bool = False,
        unsqueeze_bias: bool = False,
    ) -> TtLinearParameters:
        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
            if unsqueeze_bias:
                # TODO: Remove this workaround for issue https://github.com/tenstorrent/tt-metal/issues/16599
                bias = bias.unsqueeze(0)
        else:
            bias = None

        on_host = on_host or device is None

        return cls(
            weight=from_torch_fast(
                state["weight"].transpose(0, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                to_host=on_host,
            ),
            bias=from_torch_fast(bias, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device, to_host=on_host)
            if bias is not None
            else None,
            on_host=on_host,
            device=device,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class TtLinear:
    def __init__(self, parameters: TtLinearParameters) -> None:
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._paramters_on_host = parameters.on_host
        self._device = parameters.device

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        dtype: ttnn.DataType | None = None,
    ) -> ttnn.Tensor:
        msg = f"last value in input shape {list(x.shape)} should be equal to {self._in_channels}"
        assert x.shape[-1] == self._in_channels, msg

        if self._paramters_on_host:
            weight = self._weight.to(self._device)
            bias = self._bias.to(self._device) if self._bias is not None else None
        else:
            weight = self._weight
            bias = self._bias

        return ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
        )
