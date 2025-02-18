# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import FluxTransformer2DModel
from ..tt.attention import TtAttention, TtAttentionParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality

if TYPE_CHECKING:
    from ..reference.attention import Attention


@pytest.mark.parametrize(
    ("block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 1, 1024, 512),
        (0, 1, 1024 + 512, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 517120}], indirect=True)
@pytest.mark.parametrize(
    ("use_program_cache", "use_tracing"),
    [
        # (False, False),
        # (True, False),
        (True, True),
    ],
)
def test_attention(
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

    separate_prompt = prompt_sequence_length != 0

    parent_torch_model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn.to(torch.float32)
    torch_model.eval()
    del parent_torch_model

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, 3072))
    prompt = torch.randn((batch_size, prompt_sequence_length, 3072)) if separate_prompt else None
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16) if separate_prompt else None
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(
            spatial=spatial, prompt=prompt, image_rotary_emb=(imagerot1, imagerot2)
        )

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device) if separate_prompt else None
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=device)

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
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
        if separate_prompt:
            ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(device, tid)
    else:
        # compile
        tt_model(**model_args)

        # execute
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        if separate_prompt:
            ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_spatial_output, tt_prompt_output = tt_model(**model_args)

    assert_quality(spatial_output, tt_spatial_output, pcc=0.994, mse=0.0005)

    if separate_prompt:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.995, mse=0.05)
