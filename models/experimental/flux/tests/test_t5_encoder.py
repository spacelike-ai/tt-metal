# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from contextlib import nullcontext

import pytest
import torch
import ttnn
from loguru import logger
from transformers.models.t5.modeling_t5 import T5EncoderModel

from ..reference.t5_encoder import T5Config, T5Encoder
from ..tt.t5_encoder import TtT5Encoder, TtT5EncoderParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("program_cache_enabled", [True], indirect=True)
@pytest.mark.parametrize("device_type", [ttnn.Device, ttnn.MeshDevice], indirect=True)
def test_t5_encoder(*, device: ttnn.Device | ttnn.MeshDevicel) -> None:
    is_mesh_device = isinstance(device, ttnn.MeshDevice)

    batch_size = 1

    hf_model = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2")

    with torch.device("meta"):
        torch_model = T5Encoder(
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
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        parameters = TtT5EncoderParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat16)
    tt_model = TtT5Encoder(
        parameters,
        num_heads=hf_model.config.num_heads,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
        layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
    )
    logger.info(f"model creation time: {time.time() - start_time}")

    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [batch_size, 256])

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(device)) if is_mesh_device else nullcontext():
        tt_tokens_host = ttnn.from_torch(tokens, layout=ttnn.TILE_LAYOUT)

    start_time = time.time()
    with torch.no_grad():
        output = torch_model(tokens)
    logger.info(f"CPU runtime: {time.time() - start_time}")

    tt_tokens = tt_tokens_host.to(device)

    logger.info("compiling...")
    tt_model(tt_tokens)

    logger.info("executing...")
    start_time = time.time()
    tt_output = tt_model(tt_tokens)
    logger.info(f"TT-NN runtime: {time.time() - start_time}")
    logger.info("done...")

    with ttnn.distribute(ttnn.ConcatMeshToTensor(device, dim=0)) if is_mesh_device else nullcontext():
        tt_output_torch = ttnn.to_torch(tt_output)[:batch_size]

    assert output.shape == tt_output_torch.shape
    assert_quality(output, tt_output, pcc=0.945)
