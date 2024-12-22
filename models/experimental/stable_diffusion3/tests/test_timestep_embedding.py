import logging

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings
from ..tt.timestep_embedding import (
    TtCombinedTimestepTextProjEmbeddings,
    TtCombinedTimestepTextProjEmbeddingsParameters,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "batch_size",
    [
        2,
    ],
)
def test_timestep_embedding(
    *,
    device: ttnn.Device,
    batch_size: int,
):
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model: CombinedTimestepTextProjEmbeddings = parent_torch_model.time_text_embed
    torch_model.eval()

    parameters = TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtCombinedTimestepTextProjEmbeddings(parameters)

    torch_input_tensor = torch.randn((batch_size, 16, 64, 64))

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device)

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99999)
