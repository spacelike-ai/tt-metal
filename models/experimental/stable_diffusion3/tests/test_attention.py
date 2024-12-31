import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.attention import Attention
from ..tt.attention import TtAttention, TtAttentionParameters


@pytest.mark.parametrize(
    "block_index, batch_size, spatial_sequence_length, prompt_sequence_length",
    [
        (0, 4, 1024, 333),
        (23, 4, 1024, 333),
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
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn_dtype)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, 1536), dtype=torch_dtype)
    prompt_embed = torch.randn((batch_size, prompt_sequence_length, 1536), dtype=torch_dtype)

    tt_spatial = ttnn.from_torch(spatial, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)
    tt_prompt_embed = ttnn.from_torch(prompt_embed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)

    with torch.no_grad():
        spatial, prompt_embed = torch_model(spatial=spatial, prompt=prompt_embed)

    tt_spatial, tt_prompt_embed = tt_model(spatial=tt_spatial, prompt=tt_prompt_embed)
    tt_spatial_torch = ttnn.to_torch(tt_spatial)
    tt_prompt_embed_torch = ttnn.to_torch(tt_prompt_embed)

    mse = torch.nn.functional.mse_loss(
        spatial.to(dtype=torch.float32),
        tt_spatial_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"spatial mse: {mse:.6f}")
    assert_with_pcc(spatial, tt_spatial_torch, pcc=0.995)

    mse = torch.nn.functional.mse_loss(
        prompt_embed.to(dtype=torch.float32),
        tt_prompt_embed_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"prompt mse: {mse:.6f}")
    assert_with_pcc(prompt_embed, tt_prompt_embed_torch, pcc=0.995)
