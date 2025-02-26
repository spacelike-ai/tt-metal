# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .linear import TtLinear, TtLinearParameters
from .substate import substate

if TYPE_CHECKING:
    import torch


@dataclass
class TtFeedForwardParameters:
    in_proj: TtLinearParameters
    out_proj: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device | ttnn.MeshDevice | None = None,
        linear_on_host: bool = False,
    ) -> TtFeedForwardParameters:
        return cls(
            in_proj=TtLinearParameters.from_torch(
                substate(state, "net.0.proj"), dtype=dtype, device=device, on_host=linear_on_host
            ),
            out_proj=TtLinearParameters.from_torch(
                substate(state, "net.2"), dtype=dtype, device=device, on_host=linear_on_host
            ),
        )


class TtFeedForward:
    def __init__(self, parameters: TtFeedForwardParameters) -> None:
        super().__init__()

        self.in_proj = TtLinear(parameters.in_proj)
        self.out_proj = TtLinear(parameters.out_proj)

    def forward(self, x: ttnn.Tensor, *, gather: bool = False) -> ttnn.Tensor:
        x2 = self.in_proj.forward(x)
        # Turning on fast_and_approximate_mode leads to big changes in the generated image.
        # The image quality might still be okay.
        x3 = ttnn.gelu(x2, fast_and_approximate_mode=False)
        ttnn.deallocate(x2)

        if gather:
            x3 = ttnn.all_gather(x3, dim=-1)

        result = self.out_proj.forward(x3)
        ttnn.deallocate(x3)

        if gather:
            result = ttnn.all_gather(result, dim=-1)

        return result
