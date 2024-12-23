import logging

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.attention import Attention
from ..tt.attention import TtAttention, TtAttentionParameters

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "block_index, batch_size, spatial_sequence_length, prompt_sequence_length",
    [
        (0, 4, 1024, 333),
    ],
)
def test_attention(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
):
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads, head_dim=torch_model.head_dim)

    spatial = torch.randn((batch_size, spatial_sequence_length, 1536))
    prompt_embed = torch.randn((batch_size, prompt_sequence_length, 1536))

    tt_spatial = ttnn.from_torch(
        spatial,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_prompt_embed = ttnn.from_torch(
        prompt_embed,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        spatial, prompt_embed = torch_model(spatial=spatial, prompt=prompt_embed)

    tt_spatial, tt_prompt_embed = tt_model(spatial=tt_spatial, prompt=tt_prompt_embed)
    tt_spatial_torch = ttnn.to_torch(tt_spatial)
    tt_prompt_embed_torch = ttnn.to_torch(tt_prompt_embed)

    assert_with_pcc(spatial, tt_spatial_torch, pcc=0.999)
    assert_with_pcc(prompt_embed, tt_prompt_embed_torch, pcc=0.999)
