from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import has_substate, substate


@dataclass
class TtAttentionPartParameters:
    q_proj: TtLinearParameters
    k_proj: TtLinearParameters
    v_proj: TtLinearParameters
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
        return cls(
            spatial=TtAttentionPartParameters(
                q_proj=TtLinearParameters.from_torch(substate(state, "to_q"), dtype=dtype, device=device),
                k_proj=TtLinearParameters.from_torch(substate(state, "to_k"), dtype=dtype, device=device),
                v_proj=TtLinearParameters.from_torch(substate(state, "to_v"), dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_out.0"), dtype=dtype, device=device),
            ),
            prompt=TtAttentionPartParameters(
                q_proj=TtLinearParameters.from_torch(substate(state, "add_q_proj"), dtype=dtype, device=device),
                k_proj=TtLinearParameters.from_torch(substate(state, "add_k_proj"), dtype=dtype, device=device),
                v_proj=TtLinearParameters.from_torch(substate(state, "add_v_proj"), dtype=dtype, device=device),
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
        return self.spatial.q_proj.out_channels // num_heads


class TtAttentionPart:
    def __init__(self, parameters: TtAttentionPartParameters) -> None:
        super().__init__()

        eps = 1e-6

        self.q_proj = TtLinear(parameters.q_proj)
        self.k_proj = TtLinear(parameters.k_proj)
        self.v_proj = TtLinear(parameters.v_proj)
        self.out_proj = TtLinear(parameters.out_proj) if parameters.out_proj is not None else None
        self.norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self.norm_k = TtRmsNorm(parameters.norm_k, eps=eps)


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
        batch_size = spatial.shape[0]
        spatial_sequence_length = spatial.shape[1]

        q = self._spatial_attn.q_proj(spatial)  # N ⊗ S1 ⊗ (H * Eq)
        k = self._spatial_attn.k_proj(spatial)  # N ⊗ S1 ⊗ (H * Eq)
        v = self._spatial_attn.v_proj(spatial)  # N ⊗ S1 ⊗ (H * Ev)

        q = ttnn.transpose(ttnn.reshape(q, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2)  # N ⊗ H ⊗ S1 ⊗ Eq
        k = ttnn.transpose(ttnn.reshape(k, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2)  # N ⊗ H ⊗ S1 ⊗ Eq
        v = ttnn.transpose(ttnn.reshape(v, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2)  # N ⊗ H ⊗ S1 ⊗ Ev

        q = self._spatial_attn.norm_q(q)
        k = self._spatial_attn.norm_k(k)

        if prompt is not None:
            assert self._prompt_attn is not None

            q2 = self._prompt_attn.q_proj(prompt)
            k2 = self._prompt_attn.k_proj(prompt)
            v2 = self._prompt_attn.v_proj(prompt)

            q2 = ttnn.transpose(
                ttnn.reshape(q2, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2
            )  # N ⊗ H ⊗ S2 ⊗ Eq
            k2 = ttnn.transpose(
                ttnn.reshape(k2, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2
            )  # N ⊗ H ⊗ S2 ⊗ Eq
            v2 = ttnn.transpose(
                ttnn.reshape(v2, (batch_size, -1, self._num_heads, self._head_dim)), 1, 2
            )  # N ⊗ H ⊗ S2 ⊗ Ev

            q2 = self._prompt_attn.norm_q(q2)
            k2 = self._prompt_attn.norm_k(k2)

            q = ttnn.concat([q, q2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            k = ttnn.concat([k, k2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            v = ttnn.concat([v, v2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Ev

        k = ttnn.transpose(k, 2, 3)

        attention_scores = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attention_probs = ttnn.transformer.attention_softmax(
            attention_scores, attention_mask=None, head_size=self._head_dim
        )
        ttnn.deallocate(attention_scores)

        attn = ttnn.matmul(attention_probs, v)
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(v)

        concatenated_attn = ttnn.transformer.concatenate_heads(attn)
        ttnn.deallocate(attn)

        if prompt is not None:
            spatial = concatenated_attn[:, :spatial_sequence_length]
            prompt = concatenated_attn[:, spatial_sequence_length:]

            if self._prompt_attn.out_proj is not None:
                prompt = self._prompt_attn.out_proj(prompt)
        else:
            spatial = concatenated_attn

        if self._spatial_attn.out_proj is not None:
            spatial = self._spatial_attn.out_proj(spatial)

        return spatial, prompt
