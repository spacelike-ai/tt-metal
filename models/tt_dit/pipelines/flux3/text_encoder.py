# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.3 text encoder: stacked hidden states of Qwen3-VL's language model.

As diffusers ``Flux3TextEncoderStep``: the prompt goes through the chat template and is right-padded
to a multiple of 80 tokens. The transformer attends to the padding, so the padding tokens' states
matter; their positions are 1, as in the training environment.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlEncoder
from models.tt_dit.encoders.transformer import WEIGHT_CACHE_DTYPE
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache, tensor

# The text window is right-padded to a multiple of this many tokens.
PAD_MULTIPLE = 80
# The hidden states whose concatenation is the text embedding: the inputs of these layers.
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class Flux3TextEncoder:
    """Encodes one prompt at a time into embeddings of shape [1, length, 8 * hidden size].

    The length is the prompt's token count rounded up to a multiple of ``PAD_MULTIPLE`` and capped at
    ``max_length``. Only the layers up to the last one of ``OUTPUT_LAYERS`` are built.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        max_length: int,
        use_torch: bool = False,
    ) -> None:
        if max_length % PAD_MULTIPLE != 0:
            msg = f"max_length {max_length} must be a multiple of {PAD_MULTIPLE}"
            raise ValueError(msg)

        self._checkpoint_name = checkpoint_name
        self._device = device
        self._parallel_config = parallel_config
        self._max_length = max_length
        self._processor = transformers.AutoProcessor.from_pretrained(checkpoint_name, subfolder="text_encoder")

        self._torch_encoder: transformers.Qwen3VLForConditionalGeneration | None = None
        self._encoder: Qwen3VlEncoder | None = None

        if use_torch:
            self._torch_encoder = _load_torch_encoder(checkpoint_name)
        else:
            hf_config = transformers.AutoConfig.from_pretrained(checkpoint_name, subfolder="text_encoder")
            config = dataclasses.replace(
                Qwen3VlEncoder.config_from_hf(hf_config),
                num_layers=OUTPUT_LAYERS[-1],
                # The last kept state is the input of the first left-out layer, without a norm.
                final_norm=False,
                final_linear=False,
            )
            self._encoder = Qwen3VlEncoder(
                config, device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
            )

    @property
    def max_length(self) -> int:
        return self._max_length

    def load_weights(self) -> None:
        if self._encoder is None:
            return
        cache.load_model(
            self._encoder,
            get_torch_state_dict=self._state_dict,
            model_name=Path(self._checkpoint_name).name,
            subfolder=f"text_encoder_L{OUTPUT_LAYERS[-1]}",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            mesh_device=self._device,
            dtype=WEIGHT_CACHE_DTYPE,
        )

    def deallocate_weights(self) -> None:
        if self._encoder is not None:
            self._encoder.deallocate_weights()

    def is_loaded(self) -> bool:
        return self._encoder is None or self._encoder.is_loaded()

    def prompt_length(self, prompt: str) -> int:
        """The length of the embeddings of ``prompt``: its token count padded to the window multiple."""
        real_length = len(self._tokenize(prompt, padded_length=None)[0][0])
        return min(math.ceil(real_length / PAD_MULTIPLE) * PAD_MULTIPLE, self._max_length)

    def lengths(self) -> list[int]:
        """Every length ``encode`` returns."""
        return list(range(PAD_MULTIPLE, self._max_length + 1, PAD_MULTIPLE))

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """The embeddings of ``prompt``, of shape [1, ``prompt_length(prompt)``, 8 * hidden size].

        Not traced: the multimodal RoPE of ``Qwen3VlEncoder`` writes its frequency table to the
        device on every call, which a trace capture does not allow.
        """
        length = self.prompt_length(prompt)
        tokens, mask = self._tokenize(prompt, padded_length=length)
        # The training environment, transformers 4.57, gave the padding position 1.
        positions = torch.arange(length)[None].masked_fill(mask == 0, 1)

        if self._torch_encoder is not None:
            output = self._torch_encoder.forward(
                input_ids=tokens,
                attention_mask=mask,
                position_ids=positions,
                output_hidden_states=True,
                use_cache=False,
            )
            return torch.cat([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=-1).to(torch.bfloat16)

        assert self._encoder is not None
        device = self._device
        # One multimodal rope axis per section; text tokens have the same position on each.
        mrope_axes = len(self._encoder.config.rope_config.mrope_section or [None])
        tt_hidden_states = self._encoder.forward(
            tensor.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            mask=tensor.from_torch(mask, device=device),
            positions=tensor.from_torch(
                positions.float().expand(mrope_axes, -1, -1).contiguous(), device=device, dtype=ttnn.float32
            ),
            skip_final_linear=True,
            output_hidden_states=OUTPUT_LAYERS,
        )
        assert isinstance(tt_hidden_states, list)
        tt_embeds = ttnn.concat(tt_hidden_states, dim=-1)
        # The states are replicated; read one device's copy.
        return ttnn.to_torch(ttnn.get_device_tensors(tt_embeds)[0]).to(torch.bfloat16)

    def _tokenize(self, prompt: str, *, padded_length: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self._processor.tokenizer
        if padded_length is None:
            out = tokenizer(text, return_tensors="pt", truncation=True, max_length=self._max_length)
            if len(tokenizer(text).input_ids) > self._max_length:
                logger.warning(f"the prompt was truncated to {self._max_length} tokens")
        else:
            out = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                padding_side="right",
                truncation=True,
                max_length=padded_length,
            )
        return out.input_ids, out.attention_mask

    def _state_dict(self) -> dict[str, torch.Tensor]:
        state = Qwen3VlEncoder.convert_state(_load_torch_encoder(self._checkpoint_name).state_dict())
        num_layers = OUTPUT_LAYERS[-1]

        def keep(key: str) -> bool:
            if key.startswith(("final_norm.", "final_linear.")):
                return False
            if key.startswith("layers."):
                return int(key.split(".")[1]) < num_layers
            return True

        return {k: v for k, v in state.items() if keep(k)}


def _load_torch_encoder(checkpoint_name: str) -> transformers.Qwen3VLForConditionalGeneration:
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def reference_prompt_embeds(
    prompt: str,
    *,
    processor: transformers.ProcessorMixin,
    model: transformers.Qwen3VLForConditionalGeneration,
    max_sequence_length: int = 8192,
    layers: Sequence[int] = OUTPUT_LAYERS,
) -> torch.Tensor:
    """diffusers ``Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds``, for tests."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    real_length = processor.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_sequence_length)[
        "input_ids"
    ].shape[1]
    padded_length = min(math.ceil(real_length / PAD_MULTIPLE) * PAD_MULTIPLE, max_sequence_length)
    inputs = processor.tokenizer(
        text, return_tensors="pt", padding="max_length", padding_side="right", truncation=True, max_length=padded_length
    )
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    position_ids = torch.arange(padded_length)[None].masked_fill(attention_mask == 0, 1)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    out = torch.stack([output.hidden_states[k] for k in layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
