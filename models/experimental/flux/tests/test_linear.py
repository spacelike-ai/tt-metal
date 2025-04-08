# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn

from ..tt.linear import Linear, LinearParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    ("batch_size", "input_dim", "output_dim"),
    [
        (32, 1536, 2048),
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2)], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_linear(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    torch.manual_seed(0)

    torch_model = torch.nn.Linear(input_dim, output_dim)
    torch_model.eval()

    with ttnn.distribute(ttnn.ShardTensorToMesh(mesh_device, dim=-1)):
        parameters = LinearParameters.from_torch(torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b)
    tt_model = Linear(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model.forward(tt_input_tensor)

    with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=-1)):
        assert_quality(torch_output, tt_output, pcc=0.99976)
