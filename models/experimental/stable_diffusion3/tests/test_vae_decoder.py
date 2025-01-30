# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.vae_decoder import VaeDecoder
from ..tt.utils import allocate_tensor_on_device_like
from ..tt.vae_decoder import TtVaeDecoder, TtVaeDecoderParameters


@pytest.mark.parametrize(
    "image_size",
    [
        128,
        # 1024,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 716800}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
def test_vae_decoder(*, device: ttnn.Device, image_size: int) -> None:
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="vae", torch_dtype=torch_dtype
    )
    torch_model = VaeDecoder()
    torch_model.load_state_dict(parent_torch_model.decoder.state_dict())
    torch_model.eval()

    parameters = TtVaeDecoderParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn_dtype)
    tt_model = TtVaeDecoder(parameters)

    torch.manual_seed(0)
    latent = torch.randn([1, 16, image_size // 8, image_size // 8], dtype=torch_dtype)

    tt_latent_host = ttnn.from_torch(latent.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)

    with torch.no_grad():
        image = torch_model(latent)

    tt_latent = allocate_tensor_on_device_like(tt_latent_host, device=device)

    # cache
    # tt_image = tt_model.forward(tt_latent)

    # # trace
    # tid = ttnn.begin_trace_capture(device)
    # tt_image = tt_model(tt_latent)
    # ttnn.end_trace_capture(device, tid)

    # # execute
    # ttnn.copy_host_to_device_tensor(tt_latent_host, tt_latent)
    # ttnn.execute_trace(device, tid)

    ttnn.copy_host_to_device_tensor(tt_latent_host, tt_latent)
    tt_image = tt_model(tt_latent)

    tt_image_torch = ttnn.to_torch(tt_image).permute(0, 3, 1, 2)

    assert image.shape == tt_image_torch.shape
    mse = torch.nn.functional.mse_loss(
        image.to(dtype=torch.float32),
        tt_image_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"latent mse: {mse:.6f}")
    assert_with_pcc(image, tt_image_torch, pcc=0.990)
