from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

import ttnn

logger = logging.getLogger(__name__)


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
        self._weight = ttnn.to_torch(parameters.weight)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        torch_x = ttnn.to_torch(x)

        # TODO: convert to ttnn
        dtype = torch_x.dtype
        variance = torch_x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        torch_x = (torch_x * torch.rsqrt(variance + self._eps)).to(dtype) * self._weight

        return ttnn.from_torch(torch_x, layout=x.layout, dtype=x.dtype, device=x.device())


class TtLayerNorm:
    def __init__(self, parameters: TtLayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # ttnn.layer_norm currently requires padded tensors to only contain zeros in the padded area
        # if list(x.shape) != list(x.shape.with_tile_padding()):
        #     x = ttnn.untilize(x)
        #     x = _tilize_with_zero_padding(x)

        # return ttnn.layer_norm(x, weight=self._weight, bias=self._bias, epsilon=self._eps)

        torch_x = ttnn.to_torch(x)
        torch_weight = ttnn.to_torch(self._weight).squeeze(0) if self._weight is not None else None
        torch_bias = ttnn.to_torch(self._bias).squeeze(0) if self._bias is not None else None

        torch_result = torch.nn.functional.layer_norm(
            torch_x,
            [torch_x.shape[-1]],
            weight=torch_weight,
            bias=torch_bias,
            eps=self._eps,
        )
        return ttnn.from_torch(torch_result, device=x.device(), layout=ttnn.TILE_LAYOUT)


# def _increase_to_nearest_multiple(x, factor):
#     """Return smallest multiple of `factor` bigger or equal to `x`."""
#     return (x + factor - 1) // factor * factor


# def _tilize_with_zero_padding(x: ttnn.Tensor) -> ttnn.Tensor:
#     if x.dtype != ttnn.bfloat16:
#         logger.warning("tilize_with_val_padding expects bfloat16 input")

#     padded_shape = [_increase_to_nearest_multiple(s, 32) for s in x.shape]

#     return ttnn.tilize_with_val_padding(x, output_tensor_shape=padded_shape, pad_value=0.0)
