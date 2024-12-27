from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..tt.patch_embedding import TtConv2d, TtConv2dParameters


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, kernel_size, stride, height, width",
    [
        (2, 3, 4, (3, 5), (1, 2), 10, 10),
        # (10, 20, 15, (3, 5), (1, 2), 128, 256),
    ],
)
def test_conv2d(
    *,
    device: ttnn.Device,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    height: int,
    width: int,
):
    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dtype=torch.bfloat16,
    )
    torch_model.eval()

    parameters = TtConv2dParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtConv2d(parameters, stride=stride)

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch.bfloat16)

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
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_999_99)
