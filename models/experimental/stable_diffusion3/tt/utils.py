# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def increase_to_nearest_multiple(x: int, factor: int) -> int:
    """Return smallest multiple of `factor` bigger or equal to `x`."""
    return (x + factor - 1) // factor * factor


def allocate_tensor_on_device_like(t: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device)
