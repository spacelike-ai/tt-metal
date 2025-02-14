# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import ttnn


class TtFluxPosEmbed:
    def __init__(self, *, axes_dim: list[int]) -> None:
        self.axes_dim = axes_dim

    def __call__(self, ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []

        torch_ids = ttnn.to_torch(ids)

        for i in range(n_axes):
            cos, sin = self._get_1d_rotary_pos_embed(self.axes_dim[i], torch_ids[:, i])
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1)
        freqs_sin = torch.cat(sin_out, dim=-1)

        return (
            ttnn.from_torch(freqs_cos, device=ids.device(), layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(freqs_sin, device=ids.device(), layout=ttnn.TILE_LAYOUT),
        )

    @staticmethod
    def _get_1d_rotary_pos_embed(
        dim: int,
        pos: torch.Tensor,
        theta: float = 10000.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dim % 2 == 0

        range_ = torch.arange(0, dim, step=2, dtype=torch.float32, device=pos.device)
        freqs = 1.0 / (theta ** (range_[: dim // 2] / dim))
        freqs = torch.outer(pos, freqs)

        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()

        return freqs_cos, freqs_sin
