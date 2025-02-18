# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import FluxTransformer2DModel
from ..tt.transformer_block import (
    TtFluxSingleTransformerBlock,
    TtFluxSingleTransformerBlockParameters,
    TtTransformerBlock,
    TtTransformerBlockParameters,
)
from ..tt.utils import allocate_tensor_on_device_like, assert_quality

if TYPE_CHECKING:
    from ..reference.transformer_block import SingleTransformerBlock, TransformerBlock


@pytest.mark.parametrize(
    ("block_index", "batch_size", "sequence_length"),
    [
        (0, 1, 1024 + 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.parametrize(
    ("use_program_cache", "use_tracing"),
    [
        # (False, False),
        # (True, False),
        (True, True),
    ],
)
def test_single_transformer_block(
    *,
    device: ttnn.Device,
    use_program_cache: bool,
    use_tracing: bool,
    block_index: int,
    batch_size: int,
    sequence_length: int,
) -> None:
    if use_program_cache:
        ttnn.enable_program_cache(device)

    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer"
    )
    torch_model: SingleTransformerBlock = parent_torch_model.single_transformer_blocks[block_index]
    torch_model.eval()

    parameters = TtFluxSingleTransformerBlockParameters.from_torch(
        torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b
    )
    tt_model = TtFluxSingleTransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 3072

    torch.manual_seed(0)
    combined = torch.randn((batch_size, sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))
    imagerot1 = torch.randn([sequence_length, 128], dtype=torch.float32)
    imagerot2 = torch.randn([sequence_length, 128], dtype=torch.float32)

    tt_combined_host = ttnn.from_torch(combined, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_time_host = ttnn.from_torch(time.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)

    with torch.no_grad():
        combined_output = torch_model(combined=combined, time_embed=time, image_rotary_emb=(imagerot1, imagerot2))

    tt_combined = allocate_tensor_on_device_like(tt_combined_host, device=device)
    tt_time = allocate_tensor_on_device_like(tt_time_host, device=device)
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=device)

    model_args = dict(  # noqa: C408
        combined=tt_combined,
        time_embed=tt_time,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    if use_tracing:
        # cache
        tt_model(**model_args)

        # trace
        tid = ttnn.begin_trace_capture(device)
        tt_combined_output = tt_model(**model_args)
        ttnn.end_trace_capture(device, tid)

        # execute
        ttnn.copy_host_to_device_tensor(tt_combined_host, tt_combined)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(device, tid)
    else:
        # compile
        tt_model(**model_args)

        # execute
        ttnn.copy_host_to_device_tensor(tt_combined_host, tt_combined)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_combined_output = tt_model(**model_args)

    assert_quality(combined_output, tt_combined_output, pcc=0.999, mse=830.0)


@pytest.mark.parametrize(
    ("block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 1, 1024, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.parametrize(
    ("use_program_cache", "use_tracing"),
    [
        # (False, False),
        # (True, False),
        (True, True),
    ],
)
def test_transformer_block(
    *,
    device: ttnn.Device,
    use_program_cache: bool,
    use_tracing: bool,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
) -> None:
    if use_program_cache:
        ttnn.enable_program_cache(device)

    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index].to(torch.float32)
    torch_model.eval()
    del parent_torch_model

    parameters = TtTransformerBlockParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtTransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 3072

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128], dtype=torch.float32)
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128], dtype=torch.float32)

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_time_host = ttnn.from_torch(time.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(
            spatial=spatial, prompt=prompt, time_embed=time, image_rotary_emb=(imagerot1, imagerot2)
        )

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    tt_time = allocate_tensor_on_device_like(tt_time_host, device=device)
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=device)

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
        time_embed=tt_time,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    if use_tracing:
        # cache
        tt_model(**model_args)

        # trace
        tid = ttnn.begin_trace_capture(device)
        tt_spatial_output, tt_prompt_output = tt_model(**model_args)
        ttnn.end_trace_capture(device, tid)

        # execute
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(device, tid)
    else:
        # compile
        tt_model(**model_args)

        # execute
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_spatial_output, tt_prompt_output = tt_model(**model_args)

    assert (prompt_output is None) == (tt_prompt_output is None)

    if prompt_output is not None and tt_prompt_output is not None:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.999_500, mse=1500.0)

    assert_quality(spatial_output, tt_spatial_output, pcc=0.999, mse=15.0)
