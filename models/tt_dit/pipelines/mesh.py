# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..parallel.config import DiTParallelConfig


@contextmanager
def reshape_device(device: ttnn.MeshDevice, shape: ttnn.MeshShape | Sequence[int]) -> Iterator[None]:
    """Temporarily rearrange a mesh device into ``shape``, restoring on exit."""
    if not isinstance(shape, ttnn.MeshShape):
        shape = ttnn.MeshShape(*shape)

    # Create a new ttnn.MeshShape instance as the original will be invalidated by the reshape.
    original_shape = ttnn.MeshShape(device.shape)

    if original_shape.mesh_size() != shape.mesh_size():
        msg = f"original shape {original_shape} and target shape {shape} have different device counts"
        raise ValueError(msg)

    if original_shape == shape:
        yield
        return

    device.reshape(shape)
    try:
        yield
    finally:
        device.reshape(original_shape)


def create_submeshes(
    device: ttnn.MeshDevice, parallel_config: DiTParallelConfig
) -> tuple[ttnn.MeshDevice] | tuple[ttnn.MeshDevice, ttnn.MeshDevice]:
    """Slice the mesh into cfg-parallel submeshes sized for tensor and sequence parallelism."""
    tp = parallel_config.tensor_parallel
    sp = parallel_config.sequence_parallel
    cp = parallel_config.cfg_parallel

    if cp.factor not in (1, 2):
        msg = "cfg parallel factor must be 1 or 2"
        raise ValueError(msg)

    submesh_shape = [1] * device.shape.dims()
    submesh_shape[sp.mesh_axis] *= sp.factor
    submesh_shape[tp.mesh_axis] *= tp.factor

    devices = device.create_submeshes(ttnn.MeshShape(*submesh_shape))
    if len(devices) < cp.factor:
        msg = f"not enough submeshes created: expected {cp.factor}, got {len(devices)}"
        raise ValueError(msg)

    return (devices[0],) if cp.factor == 1 else (devices[0], devices[1])
