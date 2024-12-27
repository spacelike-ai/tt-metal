import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings
from ..tt.timestep_embedding import (
    TtCombinedTimestepTextProjEmbeddings,
    TtCombinedTimestepTextProjEmbeddingsParameters,
)


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

    timestep = torch.randn((batch_size,))
    pooled_projection = torch.randn((batch_size, 2048))

    tt_pooled_projection = ttnn.from_torch(pooled_projection, device=device)

    torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model(torch_timestep=timestep, pooled_projection=tt_pooled_projection)
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_999_99)
