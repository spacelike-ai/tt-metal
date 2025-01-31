# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
        # 256,
        # 512,
        # 1024,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 716800}], indirect=True)
@pytest.mark.parametrize(
    ("use_program_cache", "use_tracing"),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_vae_decoder(*, device: ttnn.Device, use_program_cache: bool, use_tracing: bool, image_size: int) -> None:
    if use_program_cache:
        ttnn.enable_program_cache(device)

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

    if use_tracing:
        # cache
        logger.info("caching...")
        tt_model(tt_latent)

        # trace
        logger.info("tracing...")
        tid = ttnn.begin_trace_capture(device)
        tt_image = tt_model(tt_latent)
        ttnn.end_trace_capture(device, tid)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_latent_host, tt_latent)
        ttnn.execute_trace(device, tid)
        logger.info("done...")
    else:
        logger.info("compiling...")
        tt_model(tt_latent)

        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_latent_host, tt_latent)
        tt_image = tt_model(tt_latent)
        logger.info("done...")

    tt_image_torch = ttnn.to_torch(tt_image).permute(0, 3, 1, 2)

    assert image.shape == tt_image_torch.shape
    mse = torch.nn.functional.mse_loss(
        image.to(dtype=torch.float32),
        tt_image_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"latent mse: {mse:.6f}")
    assert_with_pcc(image, tt_image_torch, pcc=0.990)
