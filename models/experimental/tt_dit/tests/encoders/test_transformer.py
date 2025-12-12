# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
import transformers.models.mistral.modeling_mistral
import ttnn
from loguru import logger

from ...encoders.rope import RopeConfig
from ...encoders.transformer import MISTRAL3_CONVERSION, Transformer
from ...parallel.config import EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "batch_size", "skip_layers"),
    [
        pytest.param((1, 1), 2, 32, id="1x1"),
        # pytest.param((1, 2), 10, 0, id="1x2"),
        # pytest.param((1, 8), 10, 0, id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        # pytest.param(False, id="unmasked"),
    ],
)
def test_transformer(*, mesh_device: ttnn.MeshDevice, batch_size: int, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    sequence_length = 512
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    config = torch_model.model.language_model.config

    mid = len(torch_model.model.language_model.layers) // 2
    del torch_model.model.language_model.layers[mid - skip_layers // 2 : mid - (-skip_layers // 2)]

    model = Transformer(
        vocab_size=config.vocab_size,
        head_size=config.head_dim,
        embed_size=config.hidden_size,
        ff_size=config.intermediate_size,
        num_layers=config.num_hidden_layers - skip_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        norm_eps=config.rms_norm_eps,
        attn_qkv_bias=False,
        attn_out_bias=False,
        rope_config=RopeConfig(theta=config.rope_theta),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    state_dict = MISTRAL3_CONVERSION.convert(torch_model.state_dict())
    model.load_torch_state_dict(state_dict)

    tokens = torch.randint(0, config.vocab_size, [batch_size, sequence_length])
    m = torch.randint(0, sequence_length + 1, [batch_size])
    attention_mask = torch.arange(sequence_length) < m.unsqueeze(1) if masked else None
    # cos, sin = model.create_rope_tensors(batch_size, sequence_length, attention_mask)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    # tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    # tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    logger.info("running ttnn model...")
    tt_prompt_embeds = model.forward(
        tt_tokens,
        mask=tt_attention_mask,
        # pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
        skip_final_linear=True,
    )
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.forward(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = out.hidden_states[-1]

    if masked:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.952, relative_rmse=0.31)
    else:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.991, relative_rmse=0.14)
