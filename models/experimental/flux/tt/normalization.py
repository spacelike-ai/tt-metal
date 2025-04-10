# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .utils import from_torch_fast


@dataclass
class RmsNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device | ttnn.MeshDevice | None = None,
    ) -> RmsNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"].unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
        )


class RmsNorm:
    def __init__(self, parameters: RmsNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self._weight, epsilon=self._eps)


@dataclass
class LayerNormParameters:
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None
    mesh_sharded: bool = False

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        mesh_sharded: bool = False,
        weight_shape: list[int] | None = None,
    ) -> LayerNormParameters:
        mesh_sharded = mesh_sharded and device.get_num_devices() > 1

        weight = state.get("weight")
        bias = state.get("bias")

        if mesh_sharded:
            # ttnn.layer_norm_post_all_gather currently requires weight and bias
            if weight is None:
                assert weight_shape is not None, "weight_shape is required when weight is missing"
                weight = torch.ones(weight_shape)
            if bias is None:
                bias = torch.zeros_like(weight)

            weight = weight.reshape([-1, 32 * device.get_num_devices()])
            bias = bias.reshape([-1, 32 * device.get_num_devices()])

        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=-1) if mesh_sharded else None
        layout = ttnn.ROW_MAJOR_LAYOUT if mesh_sharded else ttnn.TILE_LAYOUT
        if mesh_sharded and dtype != ttnn.float32:
            dtype = ttnn.bfloat16

        return cls(
            weight=from_torch_fast(
                weight,
                layout=layout,
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            if weight is not None
            else None,
            bias=from_torch_fast(
                bias,
                layout=layout,
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            if bias is not None
            else None,
            mesh_sharded=mesh_sharded,
        )


class LayerNorm:
    def __init__(self, parameters: LayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._mesh_sharded = parameters.mesh_sharded
        self._weight = parameters.weight
        self._bias = parameters.bias

    def forward(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.ProgramConfig | None = None,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> ttnn.Tensor:
        if not self._mesh_sharded:
            return ttnn.layer_norm(
                x,
                weight=self._weight,
                bias=self._bias,
                epsilon=self._eps,
                memory_config=memory_config,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )

        rank = len(x.shape)
        if rank < 4:
            shape = [1] * (4 - rank) + list(x.shape)
            x = ttnn.reshape(x, shape)

        stats = ttnn.layer_norm_pre_all_gather(
            x,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        stats = ttnn.all_gather(stats, dim=-1, memory_config=memory_config)

        x = ttnn.layer_norm_post_all_gather(
            x,
            stats,
            weight=self._weight,
            bias=self._bias,
            epsilon=self._eps,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        if rank < 4:
            shape = list(x.shape)[4 - rank :]
            x = ttnn.reshape(x, shape)

        return x
