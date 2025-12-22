# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
import transformers.generation.utils
import transformers.models.mistral.modeling_mistral
import ttnn
from loguru import logger

from ...encoders.rope import RopeConfig
from ...encoders.transformer import MISTRAL3_CONVERSION, Transformer
from ...parallel.config import EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "skip_layers"),
    [
        # pytest.param((1, 1), 32, id="1x1"),
        # pytest.param((1, 2), 22, id="1x2"),
        pytest.param((1, 4), 0, id="1x4"),
        # pytest.param((1, 8), 0, id="1x8"), CRASHES HOST
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
def test_generate(*, mesh_device: ttnn.MeshDevice, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    tp_axis = 1
    max_length = 20

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    tokenizer = transformers.LlamaTokenizerFast.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="tokenizer")

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    config = torch_model.model.language_model.config

    num_layers = len(torch_model.model.language_model.layers)
    del torch_model.model.language_model.layers[num_layers - skip_layers :]

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

    state_dict = torch_model.state_dict()
    state_dict = MISTRAL3_CONVERSION.convert(state_dict)
    if not cache.initialize_from_cache(
        tt_model=model,
        torch_state_dict=state_dict,
        model_name="flux2",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        model.load_torch_state_dict(state_dict)

    out = tokenizer.__call__(
        ["Once upon a time", "Hello"],
        padding="longest",
        # padding side does not matter for our implementation but the
        # transformers library complains if right padding is used
        padding_side="left",
        return_tensors="pt",
        return_attention_mask=True,
    )
    tokens = out["input_ids"].to(torch_model.device)
    mask = out["attention_mask"].to(torch_model.device) if masked else None

    generation_config = torch_model.generation_config
    assert isinstance(generation_config, transformers.GenerationConfig)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    print("running ttnn model...")
    out = model.generate(
        tt_tokens,
        mask=tt_mask,
        eos_tokens=generation_config.eos_token_id,
        max_length=generation_config.max_length,
        top_k=generation_config.top_k if generation_config.do_sample else 1,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        return_logits=True,
    )

    generation_config.max_length = max_length
    generation_config.repetition_penalty = None  # repetition penalty is not implemented
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = True

    print("running torch model...")
    out_ref = torch_model.generate(tokens, attention_mask=mask)
    assert isinstance(out_ref, transformers.generation.utils.GenerateOutput)

    tokens_out = tensor.to_torch(out.tokens)
    tokens_out_ref = out_ref.sequences
    logits = tensor.to_torch(ttnn.stack(out.logits, dim=1), mesh_axes=[..., tp_axis])
    logits_ref = torch.stack(out_ref.logits, dim=1)

    # diffusers somtimes generates longer sequences than max_length, in particular when `max_length
    # = 20` for whatever reason.
    tokens_out_ref = tokens_out_ref[:, :max_length]
    logits_ref = logits_ref[:, : max_length - tokens.size(1)]

    # for i in range(tokens_out.size(0)):
    #     print(tokenizer.decode(tokens_out_ref[i]))
    #     print(tokenizer.decode(tokens_out[i]))

    if mask is not None:
        # Masked positions on the start of the sequence contain random values from computing softmax over all -inf
        # so we remove them before comparison.
        _, s, d = logits_ref.shape
        padded_mask = torch.nn.functional.pad(mask.bool(), [0, s - mask.size(1)], value=True)
        logits_ref = logits_ref.masked_select(padded_mask.unsqueeze(-1)).view([-1, d])
        logits = logits.masked_select(padded_mask.unsqueeze(-1)).view([-1, d])

    assert_quality(logits_ref, logits, ccc=0.99999, relative_rmse=0.001)
    assert tokens_out.eq(tokens_out_ref).all()


@pytest.mark.parametrize(
    ("mesh_device", "batch_size", "skip_layers"),
    [
        pytest.param((1, 2), 2, 22, id="1x2"),
        pytest.param((1, 1), 2, 32, id="1x1"),
        pytest.param((1, 4), 2, 0, id="1x4"),
        # pytest.param((1, 8), 2, 0, id="1x8"), CRASHES HOST
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
        pytest.param(False, id="unmasked"),
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

    num_layers = len(torch_model.model.language_model.layers)
    del torch_model.model.language_model.layers[num_layers - skip_layers :]

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

    state_dict = torch_model.state_dict()
    state_dict = MISTRAL3_CONVERSION.convert(state_dict)
    if not cache.initialize_from_cache(
        tt_model=model,
        torch_state_dict=state_dict,
        model_name="flux2",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        model.load_torch_state_dict(state_dict)

    tokens = torch.randint(0, config.vocab_size, [batch_size, sequence_length])
    lengths = torch.randint(sequence_length // 4, 3 * sequence_length // 4, [batch_size])
    mask = torch.arange(sequence_length).flip([0]) < lengths.unsqueeze(1) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    logger.info("running ttnn model...")
    tt_prompt_embeds = model.forward(
        tt_tokens,
        mask=tt_mask,
        skip_final_linear=True,
    )
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.forward(tokens, attention_mask=mask, output_hidden_states=True)
        prompt_embeds = out.hidden_states[-1]

    if mask is not None:
        # Masked positions on the start of the sequence contain random values from computing softmax over all -inf
        # so we remove them before comparison.
        _, _, d = prompt_embeds.shape
        prompt_embeds = prompt_embeds.masked_select(mask.unsqueeze(-1)).view([-1, d])
        tt_prompt_embeds_torch = tt_prompt_embeds_torch.masked_select(mask.unsqueeze(-1)).view([-1, d])

    if masked:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.952, relative_rmse=0.31)
    else:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.991, relative_rmse=0.14)
