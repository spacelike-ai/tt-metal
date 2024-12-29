import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.patch_embedding import PatchEmbed
from ..tt.patch_embedding import TtPatchEmbed, TtPatchEmbedParameters


@pytest.mark.parametrize(
    "batch_size",
    [
        2,
    ],
)
def test_patch_embedding(
    *,
    device: ttnn.Device,
    batch_size: int,
):
    dtype = torch.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=dtype
    )
    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    parameters = TtPatchEmbedParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtPatchEmbed(parameters)

    torch_input_tensor = torch.randn((batch_size, 16, 64, 64), dtype=dtype)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device)

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_995)
