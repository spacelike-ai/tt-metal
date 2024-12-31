import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.transformer import SD3Transformer2DModel
from ..tt.transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters


@pytest.mark.parametrize(
    "batch_size, prompt_sequence_length",
    [
        (2, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_transformer(
    *,
    device: ttnn.Device,
    batch_size: int,
    prompt_sequence_length: int,
):
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()

    parameters = TtSD3Transformer2DModelParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn_dtype)
    tt_model = TtSD3Transformer2DModel(parameters, num_attention_heads=torch_model.config.num_attention_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, 16, 64, 64), dtype=torch_dtype)
    prompt_embed = torch.randn((batch_size, prompt_sequence_length, 4096), dtype=torch_dtype)
    pooled_projection = torch.randn((batch_size, 2048), dtype=torch_dtype)
    timestep = torch.randn(batch_size, dtype=torch_dtype)

    tt_spatial = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_prompt_embed = ttnn.from_torch(
        prompt_embed,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )

    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial, prompt_embed=prompt_embed, pooled_projections=pooled_projection, timestep=timestep
        )

    tt_output = tt_model(
        spatial=tt_spatial,
        prompt_embed=tt_prompt_embed,
        pooled_projection=tt_pooled_projection,
        torch_timestep=timestep,
    )
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.990)
