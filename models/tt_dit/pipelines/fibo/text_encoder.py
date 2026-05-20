# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn
from models.tt_dit.encoders.smollm3.encoder_pair import SmolLm3TokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback

if TYPE_CHECKING:
    from collections.abc import Sequence


# Beginning-of-text token id used by SmolLM3 when the user prompt is empty.
_BOT_TOKEN_ID = 128000


class TextEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
    ) -> None:
        self._encoder = SmolLm3TokenizerEncoderPair(
            checkpoint_name,
            tokenizer_subfolder="tokenizer",
            encoder_subfolder="text_encoder",
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch,
        )

    def encoder_loaded(self) -> bool:
        return self._encoder.encoder_loaded()

    def reload_encoder_weights(self) -> None:
        self._encoder.reload_encoder_weights()

    def deallocate_encoder_weights(self) -> None:
        self._encoder.deallocate_encoder_weights()

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        max_sequence_length: int,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode prompts (and optionally negatives) into FIBO's text representation.

        Returns ``(embeds, layers, mask)`` where ``embeds`` is the concatenation of the last two
        SmolLM3 hidden states (DiT context input), ``layers`` is the full per-layer hidden state
        tuple (DimFusion input), and ``mask`` is the attention mask. When CFG is enabled the
        batch dimension is laid out as ``[negative, positive]``.
        """
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        all_prompts = [*negative_prompts, *prompts] if cfg_enabled else list(prompts)

        on_event(SectionStart("smollm3_encoding"))
        hidden_states, mask = self._encoder.encode(
            all_prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=max_sequence_length,
            empty_token_id=_BOT_TOKEN_ID,
            output_hidden_states=True,
            enable_tracing=traced,
        )
        on_event(SectionEnd("smollm3_encoding"))

        # FIBO uses concat(last_layer, second_to_last_layer) along the channel axis as the
        # transformer's encoder_hidden_states input.
        embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)

        return embeds, list(hidden_states), mask
