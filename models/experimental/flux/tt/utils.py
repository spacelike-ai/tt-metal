# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
) -> ttnn.Tensor:
    conversion_device = device
    device = None if to_host else device

    cd_is_mesh_device = hasattr(conversion_device, "create_submesh")  # "is ttnn.MeshDevice" always returns False
    if cd_is_mesh_device and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(conversion_device)

    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host.
    if conversion_device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype, mesh_mapper=mesh_mapper)

    try:
        tensor = ttnn.from_torch(t, device=conversion_device, mesh_mapper=mesh_mapper)
    except RuntimeError as e:
        # https://github.com/tenstorrent/tt-metal/issues/16861
        if "TODO: add support for multi-paged buffer with page size > 64KB" in str(e):
            return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype, mesh_mapper=mesh_mapper)
        raise

    if tensor.shape[-2] == 32 and t.shape[-2] == 1:
        # Work around the fact that the shape is erroneously set to the padded shape under certain conditions.
        assert isinstance(conversion_device, ttnn.MeshDevice)
        assert dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
        tensor = tensor.reshape(ttnn.Shape(t.shape))

    tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)

    if to_host:
        tensor = tensor.cpu()

    return tensor


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    pcc: float | None = None,
    mse: float | None = None,
    mesh_composer: ttnn.MeshToTensor | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = ttnn.to_torch(a, mesh_composer=mesh_composer)
    if isinstance(b, ttnn.Tensor):
        b = ttnn.to_torch(b, mesh_composer=mesh_composer)

    assert a.shape == b.shape, f"{a.shape} != {b.shape}"

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    _, pcc_calculated = comp_pcc(a, b)
    mse_calculated = torch.nn.functional.mse_loss(a, b).item()

    logger.info(f"PCC = {pcc_calculated * 100:.4f} %, MSE = {mse_calculated:.6f}")
    if pcc is not None:
        assert pcc_calculated >= pcc, f"PCC = {pcc_calculated * 100:.4f} % >= {pcc * 100:.4f} %"
    if mse is not None:
        assert mse_calculated <= mse, f"MSE = {mse_calculated:.6f} <= {mse:.6f}"


def reduce_scatter(
    x: ttnn.Tensor,
    dim: int,
    math_op: ttnn.ReduceType,
    *,
    num_links: int = 1,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    if memory_config is None:
        memory_config = x.memory_config()

    # ttnn.reduce_scatter currently supports rank 4 tensors only
    rank = len(x.shape)
    if rank < 4:
        shape = [1] * (4 - rank) + list(x.shape)
        x = ttnn.reshape(x, shape)

    x = ttnn.reduce_scatter(
        x,
        dim=dim,
        math_op=math_op,
        num_links=num_links,
        memory_config=memory_config,
    )

    if rank < 4:
        shape = list(x.shape)[4 - rank :]
        x = ttnn.reshape(x, shape)

    return x
