# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import FluxTransformer2DModel
from ..tt.timestep_embedding import (
    TtCombinedTimestepTextProjEmbeddings,
    TtCombinedTimestepTextProjEmbeddingsParameters,
)
from ..tt.utils import assert_quality

if TYPE_CHECKING:
    from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings


@pytest.mark.parametrize(
    "batch_size",
    [
        100,
    ],
)
@pytest.mark.usefixtures("use_program_cache")
def test_timestep_embedding(
    *,
    device: ttnn.Device,
    batch_size: int,
) -> None:
    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model: CombinedTimestepTextProjEmbeddings = parent_torch_model.time_text_embed.to(torch.float32)
    torch_model.eval()
    del parent_torch_model

    parameters = TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b
    )
    tt_model = TtCombinedTimestepTextProjEmbeddings(parameters)

    torch.manual_seed(0)
    timestep = torch.randint(1000, (batch_size,))
    pooled_projection = torch.randn((batch_size, 768))

    tt_timestep = ttnn.from_torch(timestep.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model(timestep=tt_timestep, pooled_projection=tt_pooled_projection)

    assert_quality(torch_output, tt_output, pcc=0.999_900)
