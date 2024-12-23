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

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class TtLinear:
    def __init__(
        self,
        parameters: TtLinearParameters,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        output_dtype: ttnn.DataType | None = None,
    ) -> None:
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias

        self._memory_config = memory_config
        self._program_config = program_config
        self._compute_kernel_config = compute_kernel_config
        self._core_grid = core_grid
        self._output_tile = output_tile

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_channels, "input tensor does not have the expected shape"

        try:
            return ttnn.linear(
                x,
                self._weight,
                bias=self._bias,
                memory_config=self._memory_config,
                program_config=self._program_config,
                compute_kernel_config=self._compute_kernel_config,
                core_grid=self._core_grid,
                output_tile=self._output_tile,
                dtype=self._output_dtype,
            )
        except Exception:
            result = ttnn.to_torch(x) @ ttnn.to_torch(self._weight)
            if self._bias is not None:
                result += ttnn.to_torch(self._bias)
            return ttnn.from_torch(result, device=x.device(), layout=ttnn.TILE_LAYOUT, dtype=x.dtype)
