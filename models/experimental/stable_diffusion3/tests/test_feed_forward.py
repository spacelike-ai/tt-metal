import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.feed_forward import FeedForward
from ..tt.feed_forward import TtFeedForward, TtFeedForwardParameters


@pytest.mark.parametrize(
    "block_index, batch_size, input_dim",
    [
        (0, 32, 128),
    ],
)
def test_feed_forward(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    input_dim: int,
):
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model: FeedForward = parent_torch_model.transformer_blocks[block_index].ff
    torch_model.eval()

    parameters = TtFeedForwardParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtFeedForward(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim, 1536))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_999_99)
