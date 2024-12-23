import logging

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.transformer_sd3 import SD3Transformer2DModel
from ..tt.transformer_sd3 import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "batch_size, prompt_sequence_length",
    [
        (2, 333),
    ],
)
def test_transformer_sd3(
    *,
    device: ttnn.Device,
    batch_size: int,
    prompt_sequence_length: int,
):
    torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model.eval()

    parameters = TtSD3Transformer2DModelParameters.from_torch(
        torch_model.state_dict(),
        device=device,
        dtype=ttnn.float32,
    )
    tt_model = TtSD3Transformer2DModel(parameters)

    spatial = torch.randn(batch_size, 16, 64, 64)
    prompt_embed = torch.randn(batch_size, prompt_sequence_length, 4096)
    pooled_projection = torch.randn(batch_size, 2048)
    timestep = torch.randn(batch_size)

    tt_spatial = ttnn.from_torch(
        spatial,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_prompt_embed = ttnn.from_torch(
        prompt_embed,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_timestep = ttnn.from_torch(
        timestep[:, None],
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_pooled_projection = ttnn.from_torch(pooled_projection, device=device)

    with torch.no_grad():
        spatial, prompt_embed = torch_model(
            spatial=spatial, prompt_embed=prompt_embed, pooled_projections=pooled_projection, timestep=timestep
        )

    tt_spatial, tt_prompt_embed = tt_model(
        spatial=tt_spatial,
        prompt_embed=tt_prompt_embed,
        pooled_projections=tt_pooled_projection,
        timestep=tt_timestep,
    )
    tt_spatial_torch = ttnn.to_torch(tt_spatial)
    tt_prompt_embed_torch = ttnn.to_torch(tt_prompt_embed)

    assert_with_pcc(spatial, tt_spatial_torch, pcc=0.999)
    assert_with_pcc(prompt_embed, tt_prompt_embed_torch, pcc=0.999)
