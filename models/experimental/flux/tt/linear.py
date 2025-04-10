# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .utils import from_torch_fast

if TYPE_CHECKING:
    import torch


@dataclass
class LinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    on_host: bool
    device: ttnn.MeshDevice
    reduce_scatter: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        on_host: bool = False,
        unsqueeze_bias: bool = False,
        mesh_sharding_dim: int | None = None,
    ) -> LinearParameters:
        weight = state["weight"]
        assert len(weight.shape) == 2, "weight should be a rank two tensor"

        if "bias" in state:
            bias = state["bias"]
            assert len(bias.shape) == 1, "bias should be a rank one tensor"

            bias = bias.unsqueeze(0)
            if unsqueeze_bias:
                # TODO: Remove this workaround for issue https://github.com/tenstorrent/tt-metal/issues/16599
                bias = bias.unsqueeze(0)
        else:
            bias = None

        on_host = on_host or device is None

        if mesh_sharding_dim is None:
            weight_mm = bias_mm = ttnn.ReplicateTensorToMesh(device)
            output_sharding = False
        elif mesh_sharding_dim in [1, -1]:
            weight_mm = bias_mm = ttnn.ShardTensorToMesh(device, -1)
            output_sharding = False
        elif mesh_sharding_dim in [0, -2]:
            weight_mm = ttnn.ShardTensorToMesh(device, -2)
            bias_mm = _ShardBias(device)
            output_sharding = True
        else:
            msg = "mesh_sharding_dim must be in the range from -2 to 1, or None"
            raise ValueError(msg)

        return cls(
            weight=from_torch_fast(
                weight.transpose(0, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                to_host=on_host,
                mesh_mapper=weight_mm,
            ),
            bias=from_torch_fast(
                bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                to_host=on_host,
                mesh_mapper=bias_mm,
            )
            if bias is not None
            else None,
            on_host=on_host,
            device=device,
            reduce_scatter=output_sharding and device.get_num_devices() > 1,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class Linear:
    def __init__(self, parameters: LinearParameters) -> None:
        self._reduce_scatter = parameters.reduce_scatter
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._paramters_on_host = parameters.on_host
        self._device = parameters.device

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        dtype: ttnn.DataType | None = None,
    ) -> ttnn.Tensor:
        msg = f"last value in input shape {list(x.shape)} should be equal to {self._in_channels}"
        assert x.shape[-1] == self._in_channels, msg

        if self._paramters_on_host:
            weight = self._weight.to(self._device)
            bias = self._bias.to(self._device) if self._bias is not None else None
        else:
            weight = self._weight
            bias = self._bias

        x = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
        )

        if self._reduce_scatter:
            x = utils.reduce_scatter(x, dim=-1, math_op=ttnn.ReduceType.Sum)

        return x


class _ShardBias(ttnn.TensorToMesh):
    """
    This mesh mapper is intended for sharding the bias of a linear operation on the first dimension.
    A single device receive the bias as is, while the other ones receive zero tensors of the same
    shape so that the bias is not added multiple times after gathering.

    The otherwise problematic behavior of adding the bias mutiple times is currently not observed
    with a bias of type bfloat8_b or bfloat4_b, since ttnn.from_torch pads to such tensors to the
    tile size before sharding, which has the same effect if the number devices is not too big.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__(mesh_device)

    def map(self, tensor: torch.Tensor) -> dict[int, ttnn.Tensor]:
        import torch

        return [tensor] + [torch.zeros_like(tensor)] * (self.mesh_device.get_num_devices() - 1)

    def config(self) -> dict[str, str]:
        return {
            "strategy": "shard",
            "shard_dim": "0",
        }
