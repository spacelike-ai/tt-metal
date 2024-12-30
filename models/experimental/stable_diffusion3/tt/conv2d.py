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
        return cls(
            weight=ttnn.from_torch(
                state["weight"],
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            ),
            bias=(
                ttnn.from_torch(
                    state["bias"],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    device=device,
                )
            )
            if "bias" in state
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[1]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def kernel_size(self) -> tuple[int, int]:
        return list(self.weight.shape)[-2:]


class TtConv2d:
    def __init__(
        self,
        parameters: TtConv2dParameters,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> None:
        self._stride = stride
        self._padding = padding

        self._in_channels = parameters.in_channels
        self._out_channels = parameters.out_channels
        self._kernel_size = parameters.kernel_size

        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        torch_result = torch.nn.functional.conv2d(
            ttnn.to_torch(x).permute([0, 3, 1, 2]),
            ttnn.to_torch(self._weight),
            bias=ttnn.to_torch(self._bias).squeeze(0) if self._bias is not None else None,
            stride=self._stride,
        ).permute([0, 2, 3, 1])
        return ttnn.from_torch(torch_result, device=x.device(), layout=x.layout, dtype=x.dtype)

        # input_shape = x.shape

        # conv_config = ttnn.Conv2dConfig(
        #     # dtype=x.dtype,
        #     # weights_dtype=self._weight.dtype,
        #     # activation="",
        #     # shard_layout=self.conv1_shard_layout,
        #     # input_channels_alignment=32,
        #     # transpose_shards=False,
        #     # reshard_if_not_optimal=False,
        # )

        # [result, _out_height, _out_width, self._weights, self._bias] = ttnn.conv2d(
        #     input_tensor=x,
        #     weight_tensor=self._weight,
        #     bias_tensor=self._bias,
        #     in_channels=self._in_channels,
        #     out_channels=self._out_channels,
        #     device=x.device(),
        #     kernel_size=self._kernel_size,
        #     stride=self._stride,
        #     padding=self._padding,
        #     batch_size=input_shape[0],
        #     input_height=input_shape[1],
        #     input_width=input_shape[2],
        #     conv_config=conv_config,
        # )

        # return result
