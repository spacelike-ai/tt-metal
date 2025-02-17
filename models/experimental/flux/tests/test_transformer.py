# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ..reference.transformer import FluxTransformer2DModel
from ..tt.transformer import TtFluxTransformer2DModel, TtFluxTransformer2DModelParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality


@pytest.mark.parametrize(
    ("batch_size", "spatial_sequence_lenght", "prompt_sequence_length"),
    [
        (1, 1024, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15157248}], indirect=True)
@pytest.mark.parametrize(
    ("use_program_cache", "use_tracing"),
    [
        (False, False),
        # (True, False),
        # (True, True),
    ],
)
def test_transformer(  # noqa: PLR0915
    *,
    device: ttnn.Device,
    use_program_cache: bool,
    use_tracing: bool,
    batch_size: int,
    prompt_sequence_length: int,
    spatial_sequence_lenght: int,
) -> None:
    if use_program_cache:
        ttnn.enable_program_cache(device)

    logger.info("loading model...")
    torch_model = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="transformer")
    torch_model.eval()

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_sequence_lenght, 64])
    prompt = torch.randn([batch_size, prompt_sequence_length, 4096])
    pooled_projection = torch.randn([batch_size, 768])
    timestep = torch.randint(1000, [batch_size])
    imagerot1 = torch.randn([spatial_sequence_lenght + prompt_sequence_length, 128])
    imagerot2 = torch.randn([spatial_sequence_lenght + prompt_sequence_length, 128])

    logger.info("running PyTorch model...")
    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial,
            prompt_embed=prompt,
            pooled_projections=pooled_projection,
            timestep=timestep,
            image_rotary_emb=(imagerot1, imagerot2),
        )

    del torch_model

    logger.info("loading model...")
    torch_model_bfloat16 = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model_bfloat16.eval()

    logger.info("creating TT-NN model...")
    parameters = TtFluxTransformer2DModelParameters.from_torch(
        torch_model_bfloat16.state_dict(), device=device, dtype=ttnn.bfloat8_b
    )
    tt_model = TtFluxTransformer2DModel(parameters, num_attention_heads=torch_model_bfloat16.config.num_attention_heads)

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_pooled_projection_host = ttnn.from_torch(pooled_projection, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_timestep_host = ttnn.from_torch(timestep.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    tt_pooled_projection = allocate_tensor_on_device_like(tt_pooled_projection_host, device=device)
    tt_timestep = allocate_tensor_on_device_like(tt_timestep_host, device=device)
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=device)

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled_projection=tt_pooled_projection,
        timestep=tt_timestep,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    if use_tracing:
        # cache
        logger.info("caching...")
        tt_model(**model_args)

        # trace
        logger.info("tracing...")
        tid = ttnn.begin_trace_capture(device)
        tt_output = tt_model(**model_args)
        ttnn.end_trace_capture(device, tid)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_pooled_projection_host, tt_pooled_projection)
        ttnn.copy_host_to_device_tensor(tt_timestep_host, tt_timestep)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(device, tid)
    else:
        # compile
        logger.info("compiling...")
        tt_model(**model_args)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_pooled_projection_host, tt_pooled_projection)
        ttnn.copy_host_to_device_tensor(tt_timestep_host, tt_timestep)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_output = tt_model(**model_args)

    assert_quality(torch_output, tt_output, pcc=0.999_500, mse=13)
