# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import FluxTransformer2DModel
from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings
from ..tt.timestep_embedding import (
    TtCombinedTimestepTextProjEmbeddings,
    TtCombinedTimestepTextProjEmbeddingsParameters,
)
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    "batch_size",
    [
        100,
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_timestep_embedding(
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    batch_size: int,
) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    torch.manual_seed(0)

    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model: CombinedTimestepTextProjEmbeddings = parent_torch_model.time_text_embed.to(torch.float32)
    torch_model.eval()
    del parent_torch_model

    # torch_model = CombinedTimestepTextProjEmbeddings(embedding_dim=3072, pooled_projection_dim=768)
    # torch_model.eval()

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        parameters = TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
            torch_model.state_dict(), device=device, dtype=ttnn.bfloat16
        )
    tt_model = TtCombinedTimestepTextProjEmbeddings(parameters)

    timestep = torch.randint(1000, (batch_size,))
    pooled_projection = torch.randn((batch_size, 768))

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_timestep = ttnn.from_torch(timestep.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        tt_pooled_projection = ttnn.from_torch(
            pooled_projection, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    with torch.no_grad():
        torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model(timestep=tt_timestep, pooled_projection=tt_pooled_projection)
    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=0)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[:batch_size]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900)
