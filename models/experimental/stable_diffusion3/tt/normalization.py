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
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
        )


@dataclass
class TtLayerNormParameters:
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None

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
        self._weight = parameters.weight

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        variance = ttnn.mean(ttnn.pow(x, 2), -1, keepdim=True)
        x *= ttnn.rsqrt(variance + self._eps)
        return x * self._weight

        # x32 = ttnn.clone(x, dtype=ttnn.float32)
        # variance = ttnn.mean(ttnn.pow(x32, 2), -1, keepdim=True)
        # x *= ttnn.rsqrt(variance + self._eps)
        # return ttnn.clone(x, dtype=x.dtype) * self._weight

        # torch_x = ttnn.to_torch(x)
        # dtype = torch_x.dtype
        # variance = torch_x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # torch_x = (torch_x * torch.rsqrt(variance + self._eps)).to(dtype) * ttnn.to_torch(self._weight)
        # return ttnn.from_torch(torch_x, layout=x.layout, dtype=x.dtype, device=x.device())


class TtLayerNorm:
    def __init__(self, parameters: TtLayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # ttnn.layer_norm currently requires padded tensors to only contain zeros in the padded area
        # if list(x.shape) != list(x.shape.with_tile_padding()):
        #     logger.warning("retilizing tensor for layer norm")
        #     x = utils.untilize(x)
        #     x = utils.tilize(x)

        return ttnn.layer_norm(x, weight=self._weight, bias=self._bias, epsilon=self._eps)

        # torch_x = ttnn.to_torch(x)
        # torch_weight = ttnn.to_torch(self._weight).squeeze(0) if self._weight is not None else None
        # torch_bias = ttnn.to_torch(self._bias).squeeze(0) if self._bias is not None else None

        # torch_result = torch.nn.functional.layer_norm(
        #     torch_x,
        #     [torch_x.shape[-1]],
        #     weight=torch_weight,
        #     bias=torch_bias,
        #     eps=self._eps,
        # )
        # return ttnn.from_torch(torch_result, device=x.device(), layout=ttnn.TILE_LAYOUT)
