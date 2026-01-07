# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from loguru import logger

from ..layers.module import Module

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from ..parallel.config import EncoderParallelConfig
    from ..parallel.manager import CCLManager


class TokenizerEncoderPair:
    def __init__(
        self,
        *,
        embedding_dim: int | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        tokenizer: PreTrainedTokenizerBase,
        text_encoder: Module | torch.nn.Module | None,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._tokenizer = tokenizer
        self._encoder = text_encoder

        self._embedding_dim = embedding_dim

    def encode(
        self,
        prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        sequence_length: int,
        zero_masking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            embedding_dim=self._embedding_dim,
            zero_masking=zero_masking,
            sequence_length=sequence_length,
            mesh_device=self._device,
        )


def _get_prompt_embeds(
    *,
    text_encoder: Module | torch.nn.Module | None,
    prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    num_images_per_prompt: int,
    mesh_device: ttnn.MeshDevice | None,
    embedding_dim: int | None,
    zero_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = list(prompts)

    if text_encoder is None:
        assert embedding_dim is not None
        return torch.zeros([len(prompts) * num_images_per_prompt, sequence_length, embedding_dim])

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    )

    tokens = tokenizer_out.input_ids
    attention_mask = tokenizer_out.attention_mask

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("input text was truncated")

    if isinstance(text_encoder, Module):
        assert mesh_device is not None

        tt_tokens = ttnn.from_torch(
            tokens,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            device=mesh_device,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        tt_attention_mask = (
            ttnn.from_torch(
                attention_mask[:, None, None, :],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
            if attention_mask is not None
            else None
        )
        tt_hidden_states = text_encoder(prompt=tt_tokens, attention_mask=tt_attention_mask, device=mesh_device)
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, attention_mask=attention_mask)
        prompt_embeds = output.last_hidden_state.to("cpu")

    if zero_masking:
        prompt_embeds = prompt_embeds * (tokens != tokenizer.pad_token_id).unsqueeze(-1)

    if embedding_dim is not None:
        assert prompt_embeds.shape[-1] == embedding_dim

    return prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
