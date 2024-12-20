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
        return cls(
            weight=ttnn.from_torch(
                state["weight"],
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if "weight" in state
            else None,
            bias=ttnn.from_torch(
                state["bias"],
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if "bias" in state
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

        assert self._weight is None
        assert self._bias is None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # ttnn.layer_norm does currently not work correctly with padded tensors
        # assert list(x.shape) == list(x.shape.with_tile_padding())
        # return ttnn.layer_norm(x, weight=self._weight, bias=self._bias, epsilon=self._eps)

        torch_x = ttnn.to_torch(x)
        torch_weight = ttnn.to_torch(self._weight) if self._weight is not None else None
        torch_bias = ttnn.to_torch(self._bias) if self._bias is not None else None

        torch_result = torch.nn.functional.layer_norm(
            torch_x, [torch_x.shape[-1]], weight=torch_weight, bias=torch_bias
        )
        return ttnn.from_torch(torch_result, device=x.device(), layout=ttnn.TILE_LAYOUT)
