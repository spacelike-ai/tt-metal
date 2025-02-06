# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    import torch


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
) -> ttnn.Tensor:
    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype)

    tensor = ttnn.from_torch(t, device=device)

    new = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)
    ttnn.deallocate(tensor)
    tensor = new

    if to_host:
        new = tensor.cpu()
        ttnn.deallocate(tensor)
        tensor = new

    return tensor
