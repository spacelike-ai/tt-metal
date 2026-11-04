# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux3.text_encoder import OUTPUT_LAYERS, Flux3TextEncoder, reference_prompt_embeds
from ....utils.check import assert_quality
from .device_params import line_params_flux3

CHECKPOINT = "black-forest-labs/flux-3-image-klein"

PROMPTS = [
    "A red fox sitting in a snowy forest at dawn",
    "A cozy coffee shop interior with warm lighting, wooden tables, a chalkboard menu that reads "
    "'TODAY: OAT LATTE $4.50', plants hanging from the ceiling, and rain streaking down the large front "
    "windows while people read and work on laptops. " * 2,
]


@pytest.mark.parametrize(
    "mesh_device, tp_axis, num_links, device_params",
    [
        [(2, 2), 1, 2, line_params_flux3],
        [(2, 4), 1, 1, line_params_flux3],
    ],
    ids=["2x2tp1", "2x4tp1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "long"])
def test_text_encoder(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    prompt: str,
    model_location_generator,  # noqa: ANN001
) -> None:
    name = model_location_generator(CHECKPOINT)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis)
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    encoder = Flux3TextEncoder(
        checkpoint_name=name,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        max_length=640,
    )
    encoder.load_weights()

    processor = transformers.AutoProcessor.from_pretrained(name, subfolder="text_encoder")
    torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    torch_model.eval()
    expected = reference_prompt_embeds(prompt, processor=processor, model=torch_model).float()

    actual = encoder.encode(prompt).float()

    assert actual.shape == expected.shape
    hidden_size = expected.shape[-1] // len(OUTPUT_LAYERS)
    for i, layer in enumerate(OUTPUT_LAYERS):
        span = slice(i * hidden_size, (i + 1) * hidden_size)
        logger.info(f"layer {layer}:")
        assert_quality(expected[..., span], actual[..., span], pcc=0.99)
