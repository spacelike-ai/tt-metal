# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from ..reference.transformer import FluxTransformer as FluxTransformerReference
from ..tt.transformer import FluxTransformer, FluxTransformerParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality


@pytest.mark.parametrize(
    ("spatial_sequence_length", "prompt_sequence_length", "block_count", "pcc", "mse"),
    [
        # (1024, 512, None, 0.99944, 13.8),
        (4096, 512, None, 0.984, 19),
        # (4096, 512, 1, 0.992, 320),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 18006016}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize(
    ("mesh_device", "use_tracing"),
    [
        # Tracing is not supported on single devices, since not all weights fit on the device.
        ((1, 1), False),
        # Tracing on multiple devices currently causes hangs.
        ((1, 2), False),
        ((2, 2), False),
    ],
    indirect=["mesh_device"],
)
def test_transformer(  # noqa: PLR0915
    *,
    mesh_device: ttnn.MeshDevice,
    use_tracing: bool,
    prompt_sequence_length: int,
    spatial_sequence_length: int,
    block_count: int | None,
    pcc: float,
    mse: float,
) -> None:
    mesh_height, _ = mesh_device.shape
    batch_size = mesh_height

    torch.manual_seed(0)

    logger.info("loading model...")
    torch_model = FluxTransformerReference.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="transformer")
    torch_model.eval()
    torch_model.keep_blocks_only(block_count, block_count)

    spatial = torch.randn([batch_size, spatial_sequence_length, 64])
    prompt = torch.randn([batch_size, prompt_sequence_length, 4096])
    pooled_projection = torch.randn([batch_size, 768])
    timestep = torch.tensor([500], dtype=torch.float32)
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])

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
    torch_model_bfloat16 = FluxTransformerReference.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model_bfloat16.eval()
    torch_model_bfloat16.keep_blocks_only(block_count, block_count)

    logger.info("creating TT-NN model...")
    parameters = FluxTransformerParameters.from_torch(
        torch_model_bfloat16.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = FluxTransformer(parameters, num_attention_heads=torch_model_bfloat16.config.num_attention_heads)

    batch_sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None))
    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_spatial_host = ttnn.from_torch(
        spatial,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -2)),
    )
    tt_prompt_host = ttnn.from_torch(
        prompt,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )
    tt_pooled_projection_host = ttnn.from_torch(
        pooled_projection, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=batch_sharded
    )
    tt_timestep_host = ttnn.from_torch(
        timestep.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=mesh_device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=mesh_device)
    tt_pooled_projection = allocate_tensor_on_device_like(tt_pooled_projection_host, device=mesh_device)
    tt_timestep = allocate_tensor_on_device_like(tt_timestep_host, device=mesh_device)
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=mesh_device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=mesh_device)

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
        tt_model.forward(**model_args)

        # trace
        logger.info("tracing...")
        tid = ttnn.begin_trace_capture(mesh_device)
        tt_output = tt_model.forward(**model_args)
        ttnn.end_trace_capture(mesh_device, tid)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_pooled_projection_host, tt_pooled_projection)
        ttnn.copy_host_to_device_tensor(tt_timestep_host, tt_timestep)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(mesh_device, tid)
    else:
        # compile
        logger.info("compiling...")
        tt_model.forward(**model_args)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_pooled_projection_host, tt_pooled_projection)
        ttnn.copy_host_to_device_tensor(tt_timestep_host, tt_timestep)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_output = tt_model.forward(**model_args)

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -2))
    assert_quality(torch_output, tt_output, pcc=pcc, mse=mse, mesh_composer=composer)
