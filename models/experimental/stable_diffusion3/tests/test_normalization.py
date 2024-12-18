import logging

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.normalization import RmsNorm
from ..tt.normalization import TtLayerNorm, TtLayerNormParameters, TtRmsNorm, TtRmsNormParameters

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "batch_size, input_dim",
    [
        (32, 128),
    ],
)
def test_layer_norm(
    *,
    device: ttnn.Device,
    batch_size: int,
    input_dim: int,
):
    torch_model = torch.nn.LayerNorm(input_dim, eps=1.0)

    parameters = TtLayerNormParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtLayerNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.9999)


@pytest.mark.parametrize(
    "batch_size, input_dim",
    [
        (32, 128),
    ],
)
def test_rms_norm(
    *,
    device: ttnn.Device,
    batch_size: int,
    input_dim: int,
):
    torch_model = RmsNorm(dim=input_dim, eps=1.0)
    torch.nn.init.normal_(torch_model.weight)

    parameters = TtRmsNormParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtRmsNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn((batch_size, input_dim))

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
