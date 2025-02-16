# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch

import ttnn

from ..reference import FluxTransformer2DModel
from ..tt.pos_embedding import TtFluxPosEmbed
from ..tt.utils import assert_quality

if TYPE_CHECKING:
    from ..reference.pos_embedding import FluxPosEmbed


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_pos_embedding(
    *,
    device: ttnn.Device,
) -> None:
    dtype = torch.bfloat16

    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=dtype
    )
    torch_model: FluxPosEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    tt_model = TtFluxPosEmbed(theta=torch_model.theta, axes_dim=torch_model.axes_dim)

    torch_input_tensor = torch.randn([1536, 3], dtype=dtype)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_cos, torch_sin = torch_model(torch_input_tensor)

    tt_cos, tt_sin = tt_model(tt_input_tensor)

    assert_quality(torch_cos, tt_cos, pcc=0.999_990)
    assert_quality(torch_sin, tt_sin, pcc=0.999_990)
