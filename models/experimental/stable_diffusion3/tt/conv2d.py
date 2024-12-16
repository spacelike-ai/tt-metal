from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtConv2dParameters:
        torch_weight = state["weight"]
        torch_bias = state["bias"]

        return cls(
            weight=ttnn.from_torch(
                torch_weight.transpose(0, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            ),
            bias=(
                None
                if torch_bias is None
                else ttnn.from_torch(
                    torch_bias.unsqueeze(0),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    device=device,
                )
            ),
        )
