# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module


@dataclass
class RopeConfig:
    theta: float
    llama3_scale_factor: float | None = None
    mrope_section: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RopeConfig:
        return cls(
            theta=data["theta"],
            llama3_scale_factor=data.get("llama3_scale_factor"),
            mrope_section=data.get("mrope_section"),
        )


class RotaryEmbedding(Module):
    def __init__(self, *, head_size: int, config: RopeConfig) -> None:
        super().__init__()

        self._head_size = head_size  # ty:ignore
        self._config = config  # ty:ignore

    def forward(self, positions: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 2

        device = positions.device
        size = self._head_size
        theta = self._config.theta

        # https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/models/llama/modeling_llama.py#L73
        inv_freq = theta ** (-torch.arange(0, size, 2, dtype=torch.int64, device=device) / size)

        if self._config.llama3_scale_factor is not None:
            inv_freq = _apply_llama3_scaling(inv_freq, self._config.llama3_scale_factor)

        if self._config.mrope_section is not None:
            warnings.warn("mrope_section is not implemented yet", stacklevel=2)
            # this only seems to affect decode mode
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1577
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L513
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L577-L583

        freqs = positions[:, :, None].float() @ inv_freq[None, :].float()  # outer product
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype), sin.to(dtype)


# https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py#L45
def _apply_llama3_scaling(freqs: torch.Tensor, factor: float) -> torch.Tensor:
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / factor + smooth * new_freqs,
        new_freqs,
    )
