# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import ttnn

from ..reference.feed_forward import FeedForward
from ..tt.feed_forward import TtFeedForward, TtFeedForwardParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    ("batch_size", "input_dim", "output_dim"),
    [
        (32, 128, 256),
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_feed_forward(
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    torch.manual_seed(0)

    torch_model = FeedForward(dim=input_dim, dim_out=output_dim)
    torch_model.eval()

    with ttnn.distribute(ttnn.ShardTensorToMesh(device, dim=-1)) if is_mesh_device else nullcontext():
        parameters = TtFeedForwardParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat16)
    tt_model = TtFeedForward(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor, gather=is_mesh_device)

    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=-1)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[..., : tt_output.shape[-1]]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_500)
