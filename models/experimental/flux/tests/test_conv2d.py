# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import ttnn

from ..tt.conv2d import TtConv2d, TtConv2dParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    ("batch_size", "in_channels", "out_channels", "kernel_size", "stride", "height", "width"),
    [
        (10, 32, 32, (2, 3), (2, 2), 64, 64),
        (10, 20, 32, (3, 3), (2, 3), 128, 256),
        # these are needed in the VAE for an image resolution of 1024x1024:
        # (1, 128, 128, (3, 3), (1, 1), 1024, 1024),
        # (1, 128, 3, (3, 3), (1, 1), 1024, 1024),
        # (1, 16, 512, (3, 3), (1, 1), 128, 128),
        # (1, 256, 128, (3, 3), (1, 1), 1024, 1024),
        # (1, 256, 256, (3, 3), (1, 1), 1024, 1024),
        # (1, 256, 256, (3, 3), (1, 1), 512, 512),
        # (1, 512, 512, (3, 3), (1, 1), 128, 128),
        # (1, 512, 512, (3, 3), (1, 1), 256, 256),
        # (1, 512, 256, (3, 3), (1, 1), 512, 512),
        # (1, 512, 512, (3, 3), (1, 1), 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_conv2d(
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    height: int,
    width: int,
) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    torch.manual_seed(0)

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        parameters = TtConv2dParameters.from_torch(torch_model.state_dict(), dtype=ttnn.bfloat16)
        tt_model = TtConv2d(parameters, stride=stride)

    torch_input_tensor = torch.ones((batch_size, in_channels, height, width))

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=0)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[: tt_output.shape[0]].permute([0, 3, 1, 2])

    assert_quality(torch_output, tt_output_torch, pcc=0.999951)
