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

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads, head_dim=torch_model.head_dim)

    torch_hidden_states = torch.randn((batch_size, 1024, 1536))
    torch_encoder_hidden_states = torch.randn((batch_size, 333, 1536))

    tt_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_hidden_states, torch_encoder_hidden_states = torch_model(torch_hidden_states, torch_encoder_hidden_states)

    tt_hidden_states, tt_encoder_hidden_states = tt_model(tt_hidden_states, tt_encoder_hidden_states)
    tt_hidden_states_torch = ttnn.to_torch(tt_hidden_states)
    tt_encoder_hidden_states_torch = ttnn.to_torch(tt_encoder_hidden_states)

    assert_with_pcc(torch_hidden_states, tt_hidden_states_torch, pcc=0.999)
    assert_with_pcc(torch_encoder_hidden_states, tt_encoder_hidden_states_torch, pcc=0.999)
