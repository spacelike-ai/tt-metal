# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .substate import substate

if TYPE_CHECKING:
    import torch


@dataclass
class FeedForwardParameters:
    in_proj: LinearParameters
    out_proj: LinearParameters
    device_count: int

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        linear_on_host: bool = False,
    ) -> FeedForwardParameters:
        return cls(
            in_proj=LinearParameters.from_torch(
                substate(state, "net.0.proj"),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=1,
            ),
            out_proj=LinearParameters.from_torch(
                substate(state, "net.2"),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=0,
            ),
            device_count=device.get_num_devices(),
        )


class FeedForward:
    def __init__(self, parameters: FeedForwardParameters) -> None:
        super().__init__()

        self._device_count = parameters.device_count
        self.in_proj = Linear(parameters.in_proj)
        self.out_proj = Linear(parameters.out_proj)

    def forward(self, x: ttnn.Tensor, *, gather: bool = True) -> ttnn.Tensor:
        x = self.in_proj.forward(x)
        # Turning on fast_and_approximate_mode leads to big changes in the generated image.
        # The image quality might still be okay.
        x = ttnn.gelu(x, fast_and_approximate_mode=False)
        x = self.out_proj.forward(x)

        if self._device_count > 1:
            x = utils.reduce_scatter(x, dim=-1, math_op=ttnn.ReduceType.Sum)
            if gather:
                x = ttnn.all_gather(x, dim=-1)

        return x
