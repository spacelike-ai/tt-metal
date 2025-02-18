# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
@pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_feed_forward(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    torch_model = FeedForward(dim=input_dim, dim_out=output_dim)
    torch_model.eval()

    with ttnn.distribute(ttnn.ShardTensorToMesh(mesh_device, dim=-1)):
        parameters = TtFeedForwardParameters.from_torch(
            torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat16
        )
    tt_model = TtFeedForward(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=-1)):
        assert_quality(torch_output, tt_output, pcc=0.999_500)
