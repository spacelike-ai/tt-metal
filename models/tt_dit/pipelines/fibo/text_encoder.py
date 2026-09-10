# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers

import ttnn
from models.tt_dit.encoders.smollm3 import SmolLm3Checkpoint
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


_BOT_TOKEN_ID = 128000


class TextEncoder:
    """FIBO's SmolLM3 text encoder wrapper with PyTorch fallback.

    Each prompt is encoded on its own at the smallest of ``sequence_lengths`` it fits and padded
    to the largest, which is the maximum prompt length. Every length is one trace.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        sequence_lengths: Sequence[int],
        use_torch: bool,
    ) -> None:
        self._device = device
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")

        sp = parallel_config.sequence_parallel
        self._sp_axis = sp.mesh_axis if sp is not None and sp.factor != 1 else None
        self._sp_factor = device.shape[self._sp_axis] if self._sp_axis is not None else 1

        for length in sequence_lengths:
            if length % (128 * self._sp_factor) != 0:
                msg = f"sequence length {length} must be a multiple of {128 * self._sp_factor}"
                raise ValueError(msg)

        self._sequence_lengths = sorted(sequence_lengths)
        self._max_sequence_length = self._sequence_lengths[-1]
        self._empty_prompt_output: tuple[list[torch.Tensor], torch.Tensor] | None = None

        if use_torch:
            self._torch_encoder = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint_name,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )
            self._torch_encoder.eval()
            self._encoder = None
        else:
            self._torch_encoder = None
            self._encoder = SmolLm3Checkpoint(checkpoint_name).build(
                device=device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )

        self._tracers: dict[int, Tracer] = {}

    def warmup(self) -> None:
        """Runs every sequence length once untraced."""
        if self._encoder is None:
            return

        for length in self._sequence_lengths:
            tokens = torch.full((1, length), _BOT_TOKEN_ID, dtype=torch.int64)
            mask = torch.ones((1, length), dtype=torch.int64)
            self._encode_tokens(tokens, mask, traced=False)

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        traced: bool,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        all_prompts = [*negative_prompts, *prompts] if cfg_enabled else list(prompts)

        on_event(SectionStart("smollm3_encoding"))
        if self._torch_encoder is not None:
            tokens, mask = self._tokenize(all_prompts, sequence_length=self._max_sequence_length)
            outputs = self._torch_encoder.forward(
                input_ids=tokens,
                attention_mask=mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states)
        else:
            # One batch-2 encode was measured to be slower than two batch-1 encodes.
            prompt_hidden_states = []
            prompt_masks = []
            for prompt in all_prompts:
                hs, m = self._encode_prompt(prompt, traced=traced)
                prompt_hidden_states.append(hs)
                prompt_masks.append(m)
            hidden_states = [torch.cat(layer) for layer in zip(*prompt_hidden_states, strict=True)]
            mask = torch.cat(prompt_masks)
        on_event(SectionEnd("smollm3_encoding"))

        mask_inv = ~mask.unsqueeze(-1).bool()
        for h in hidden_states:
            h.masked_fill_(mask_inv, 0)

        hidden_states = [h.repeat_interleave(num_images_per_prompt, dim=0) for h in hidden_states]
        mask = mask.repeat_interleave(num_images_per_prompt, dim=0)

        # FIBO uses concat(last_layer, second_to_last_layer) along the channel axis as the
        # transformer's encoder_hidden_states input.
        embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)

        return embeds, hidden_states, mask

    def _encode_prompt(self, prompt: str, *, traced: bool) -> tuple[list[torch.Tensor], torch.Tensor]:
        # The negative prompt is usually empty, so its output is kept.
        if prompt == "" and self._empty_prompt_output is not None:
            return self._empty_prompt_output

        tokens, mask = self._tokenize([prompt], sequence_length=self._max_sequence_length)
        count = int(mask.sum())
        length = min(length for length in self._sequence_lengths if length >= count)
        hidden_states = self._encode_tokens(tokens[:, :length], mask[:, :length], traced=traced)
        hidden_states = [_pad_sequence(h, self._max_sequence_length) for h in hidden_states]

        if prompt == "":
            self._empty_prompt_output = hidden_states, mask
        return hidden_states, mask

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, *, traced: bool) -> list[torch.Tensor]:
        assert self._encoder is not None

        length = tokens.shape[1]
        if length not in self._tracers:
            self._tracers[length] = Tracer(self._encoder.forward, device=self._device, prep_run=False)

        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32, mesh_axes=[None, self._sp_axis])
        tt_mask = tensor.from_torch(mask, device=self._device)
        tt_hidden_states = self._tracers[length](
            tt_tokens,
            mask=tt_mask,
            skip_final_linear=True,
            output_hidden_states=True,
            traced=traced,
        )

        # The tracer reuses its output tensors on every call, so read back before the next one.
        return [tensor.to_torch(h, mesh_axes=[None, self._sp_axis, None]) for h in tt_hidden_states]

    def _tokenize(self, prompts: Sequence[str], *, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(
            list(prompts),
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        empty_rows = torch.tensor([p == "" for p in prompts], dtype=torch.bool)
        input_ids[empty_rows, 0] = _BOT_TOKEN_ID
        attention_mask[empty_rows, 0] = 1
        attention_mask[empty_rows, 1:] = 0

        return input_ids, attention_mask


def _pad_sequence(x: torch.Tensor, length: int) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, 0, 0, length - x.shape[1]))
