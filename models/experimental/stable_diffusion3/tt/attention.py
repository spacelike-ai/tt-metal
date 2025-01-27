# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import has_substate, substate


@dataclass
class TtAttentionPartParameters:
    qkv_proj: TtLinearParameters
    norm_q: TtRmsNormParameters
    norm_k: TtRmsNormParameters
    out_proj: TtLinearParameters | None


@dataclass
class TtAttentionParameters:
    spatial: TtAttentionPartParameters
    prompt: TtAttentionPartParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtAttentionParameters:
        spatial_qkv_proj = _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v"))
        prompt_qkv_proj = _merge_qkv_proj(
            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
        )

        return cls(
            spatial=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch(spatial_qkv_proj, dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_out.0"), dtype=dtype, device=device),
            ),
            prompt=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch(prompt_qkv_proj, dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_added_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_added_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_add_out"), dtype=dtype, device=device)
                if has_substate(state, "to_add_out")
                else None,
            )
            if has_substate(state, "add_q_proj")
            else None,
        )

    def head_dim(self, *, num_heads: int) -> int:
        return self.spatial.qkv_proj.out_channels // num_heads // 3


class TtAttentionPart:
    def __init__(self, parameters: TtAttentionPartParameters) -> None:
        super().__init__()

        eps = 1e-6

        self._qkv_proj = TtLinear(parameters.qkv_proj)
        self._out_proj = TtLinear(parameters.out_proj) if parameters.out_proj is not None else None
        self._norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self._norm_k = TtRmsNorm(parameters.norm_k, eps=eps)

    def qkv(self, x: ttnn.Tensor, *, num_heads: int, head_dim: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        qkv = self._qkv_proj(x)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=num_heads, transpose_key=False)

        q = self._norm_q(q)
        k = self._norm_k(k)

        return q, k, v

    def out_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._out_proj is None:
            return x
        return self._out_proj(x)


class TtAttention:
    def __init__(self, parameters: TtAttentionParameters, *, num_heads: int) -> None:
        super().__init__()

        self._num_heads = num_heads
        self._head_dim = parameters.head_dim(num_heads=num_heads)

        self._spatial_attn = TtAttentionPart(parameters.spatial)
        self._prompt_attn = TtAttentionPart(parameters.prompt) if parameters.prompt is not None else None

    def __call__(
        self, *, spatial: ttnn.Tensor, prompt: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H * E1)
        prompt: N ⊗ S2 ⊗ (H * E2)
        """
        spatial_sequence_length = spatial.shape[1]

        q, k, v = self._spatial_attn.qkv(spatial, num_heads=self._num_heads, head_dim=self._head_dim)

        if prompt is not None:
            assert self._prompt_attn is not None

            q2, k2, v2 = self._prompt_attn.qkv(prompt, num_heads=self._num_heads, head_dim=self._head_dim)

            q = ttnn.concat([q, q2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            k = ttnn.concat([k, k2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            v = ttnn.concat([v, v2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Ev

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            q_chunk_size=256,
            k_chunk_size=512,
            exp_approx_mode=True,
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        # operands must be in DRAM
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        concatenated_attn = ttnn.transformer.concatenate_heads(attn)
        ttnn.deallocate(attn)

        if prompt is not None:
            spatial = concatenated_attn[:, :spatial_sequence_length]
            prompt = concatenated_attn[:, spatial_sequence_length:]

            prompt = self._prompt_attn.out_proj(prompt)
        else:
            spatial = concatenated_attn

        spatial = self._spatial_attn.out_proj(spatial)

        return spatial, prompt


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor],
    k_state: dict[str, torch.Tensor],
    v_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]) if "weight" in q_state else None,
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]) if "bias" in q_state else None,
    }
