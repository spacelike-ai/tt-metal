from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn


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
    ) -> TtLinearParameters:
        return cls(
            weight=ttnn.from_torch(
                state["weight"].transpose(0, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            ),
            bias=ttnn.from_torch(
                state["bias"],
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if "bias" in state
            else None,
        )


class TtLinear:
    def __init__(self, parameters: TtLinearParameters) -> None:
        self._in_features = parameters.weight.shape[0]

        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_features, "input tensor does not have the expected shape"

        try:
            return ttnn.linear(x, self._weight, bias=self._bias)
        except Exception:
            result = ttnn.to_torch(x) @ ttnn.to_torch(self._weight)
            if self._bias is not None:
                result += ttnn.to_torch(self._bias)
            return ttnn.from_torch(result, device=x.device(), layout=ttnn.TILE_LAYOUT, dtype=x.dtype)
