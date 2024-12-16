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
    "block_index, batch_size, input_dim",
    [
        (0, 32, 128),
    ],
)
def test_attention(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    input_dim: int,
):
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(torch_state=torch_model.state_dict(), device=device)
    tt_model = TtAttention(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim, 64))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99999)
