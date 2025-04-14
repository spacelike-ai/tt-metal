# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import pytest
import torch
import ttnn
from loguru import logger
from transformers.models.t5.modeling_t5 import T5EncoderModel

from ..reference.t5_encoder import T5Config
from ..reference.t5_encoder import T5Encoder as T5EncoderReference
from ..tt.t5_encoder import T5Encoder, T5EncoderParameters
from ..tt.utils import assert_quality, from_torch_fast


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
        (1, 2),
        (2, 2),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("use_program_cache")
def test_t5_encoder(*, mesh_device: ttnn.MeshDevice) -> None:
    mesh_height, _ = mesh_device.shape
    batch_size = mesh_height
    input_count = 2

    hf_model = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
    )

    with torch.device("meta"):
        torch_model = T5EncoderReference(
            T5Config(
                vocab_size=hf_model.config.vocab_size,
                d_model=hf_model.config.d_model,
                d_ff=hf_model.config.d_ff,
                d_kv=hf_model.config.d_kv,
                num_layers=hf_model.config.num_layers,
                num_heads=hf_model.config.num_heads,
                relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
                relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
                layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
            )
        )
    torch_model.load_state_dict(hf_model.state_dict(), assign=True)
    torch_model.eval()

    start_time = time.time()
    parameters = T5EncoderParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    tt_model = T5Encoder(
        parameters,
        num_heads=hf_model.config.num_heads,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
        layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
    )
    logger.info(f"model creation time: {time.time() - start_time}")

    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [input_count, 256])

    start_time = time.time()
    with torch.no_grad():
        output = torch_model(tokens)
    logger.info(f"CPU runtime: {time.time() - start_time}")

    tt_tokens_host = [
        from_torch_fast(
            token_batch,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None)),
        )
        for token_batch in tokens.chunk(input_count // batch_size)
    ]

    logger.info("compiling...")
    tt_model.forward(tt_tokens_host[0].to(mesh_device))

    logger.info("executing...")
    ttnn.synchronize_device(mesh_device)
    start_time = time.time()

    tt_output = [tt_model.forward(tt_tokens.to(mesh_device)) for tt_tokens in tt_tokens_host]

    ttnn.synchronize_device(mesh_device)
    logger.info(f"TT-NN runtime: {time.time() - start_time}")
    logger.info("done...")

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))
    tt_output_torch = torch.cat([ttnn.to_torch(x, mesh_composer=composer) for x in tt_output])

    assert_quality(output, tt_output_torch, pcc=0.947, mse=0.0031)
