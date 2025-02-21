# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import ttnn

from ..reference.normalization import RmsNorm
from ..tt.normalization import TtLayerNorm, TtLayerNormParameters, TtRmsNorm, TtRmsNormParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_layer_norm(
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    input_shape: list[int],
) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    torch.manual_seed(0)

    torch_model = torch.nn.LayerNorm(input_shape[-1:], eps=1.0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        parameters = TtLayerNormParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat16)
    tt_model = TtLayerNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=0)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[: input_shape[0]]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_950)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_rms_norm(
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    input_shape: list[int],
) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    torch.manual_seed(0)

    torch_model = RmsNorm(dim=input_shape[-1], eps=1.0)
    torch.nn.init.normal_(torch_model.weight)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        parameters = TtRmsNormParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)

    tt_model = TtRmsNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=0)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[: input_shape[0]]

    assert_quality(torch_output, tt_output_torch, pcc=0.999933)
