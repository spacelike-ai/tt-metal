from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn


@dataclass
class TtRmsNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtRmsNormParameters:
        return cls(
            weight=ttnn.from_torch(
                state["weight"],
                dtype=dtype,
                device=device,
            )
        )


@dataclass
class TtLayerNormParameters:
    weight: ttnn.Tensor | None
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtRmsNormParameters:
        torch_weight = state["weight"]
        torch_bias = state["bias"]

        return cls(
            weight=ttnn.from_torch(
                torch_weight,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if torch_weight is not None
            else None,
            bias=ttnn.from_torch(
                torch_bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if torch_bias is not None
            else None,
        )


class TtRmsNorm:
    def __init__(self, parameters: TtRmsNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = ttnn.to_torch(parameters.weight)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        torch_x = ttnn.to_torch(x)

        variance = torch_x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        torch_x = torch_x * torch.rsqrt(variance + self._eps) * self._weight

        return ttnn.from_torch(torch_x, layout=x.layout, dtype=x.dtype, device=x.device())


class TtLayerNorm:
    def __init__(self, parameters: TtLayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self._weight, bias=self._bias, epsilon=self._eps)
