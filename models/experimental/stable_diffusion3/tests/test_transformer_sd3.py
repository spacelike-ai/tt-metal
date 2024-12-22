import logging

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.transformer_sd3 import SD3Transformer2DModel
from ..tt.transformer_sd3 import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "block_index, batch_size, spatial_sequence_length, prompt_sequence_length",
    [
        (0, 2, 1024, 333),
    ],
)
def test_transformer_sd3(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
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
    tt_model = TtSD3Transformer2DModel(
        parameters,
        num_heads=torch_model.num_heads,
        head_dim=torch_model.head_dim,
        context_pre_only=torch_model.context_pre_only,
    )

    embedding_dim = 1536

    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt_embed = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time_embed = torch.randn((batch_size, embedding_dim))

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

    tt_time_embed = ttnn.from_torch(
        time_embed[:, None],
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        spatial, prompt_embed = torch_model(spatial=spatial, prompt=prompt_embed, time_embed=time_embed)

    tt_spatial, tt_prompt_embed = tt_model(spatial=tt_spatial, prompt=tt_prompt_embed, time_embed=tt_time_embed)
    tt_spatial_torch = ttnn.to_torch(tt_spatial)
    tt_prompt_embed_torch = ttnn.to_torch(tt_prompt_embed)

    assert_with_pcc(spatial, tt_spatial_torch, pcc=0.999)
    assert_with_pcc(prompt_embed, tt_prompt_embed_torch, pcc=0.999)
