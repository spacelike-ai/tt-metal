# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..tt.timestep_embedding import CombinedTimestepTextProjEmbeddings, CombinedTimestepTextProjEmbeddingsParameters
from ..tt.utils import assert_quality

if TYPE_CHECKING:
    from ..reference import FluxTransformer2DModel as FluxTransformer2DModelReference
    from ..reference.timestep_embedding import (
        CombinedTimestepTextProjEmbeddings as CombinedTimestepTextProjEmbeddingsReference,
    )


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        # 100,
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2)], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_timestep_embedding(
    *, mesh_device: ttnn.MeshDevice, batch_size: int, parent_torch_model: FluxTransformer2DModelReference
) -> None:
    torch.manual_seed(0)

    torch_model: CombinedTimestepTextProjEmbeddingsReference = parent_torch_model.time_text_embed.to(torch.float32)

    # torch_model = CombinedTimestepTextProjEmbeddingsReference(embedding_dim=3072, pooled_projection_dim=768)
    # torch_model.eval()

    parameters = CombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = CombinedTimestepTextProjEmbeddings(parameters)

    timestep = torch.randint(1000, (batch_size,))
    pooled_projection = torch.randn((batch_size, 768))

    rm = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        mesh_mapper=rm,
    )
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=rm,
    )

    with torch.no_grad():
        torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model.forward(timestep=tt_timestep, pooled_projection=tt_pooled_projection)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:batch_size]

    assert_quality(torch_output, tt_output_torch, pcc=0.99983)
