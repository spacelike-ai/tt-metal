from __future__ import annotations

import math
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
    part_a: TtAttentionPartParameters
    part_b: TtAttentionPartParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtAttentionParameters:
        return cls(
            part_a=TtAttentionPartParameters(
                q_proj=TtLinearParameters.from_torch(substate(state, "to_q"), dtype=dtype, device=device),
                k_proj=TtLinearParameters.from_torch(substate(state, "to_k"), dtype=dtype, device=device),
                v_proj=TtLinearParameters.from_torch(substate(state, "to_v"), dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_out.0"), dtype=dtype, device=device),
            ),
            part_b=TtAttentionPartParameters(
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
        return self.part_a.q_proj.out_channels // num_heads


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

        self._spatial_attn = TtAttentionPart(parameters.part_a)
        self._prompt_attn = TtAttentionPart(parameters.part_b) if parameters.part_b is not None else None

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

        q = ttnn.to_torch(q)
        k = ttnn.to_torch(k)
        v = ttnn.to_torch(v)

        # TODO: port to ttnn
        q = q.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S1 ⊗ Eq
        k = k.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S1 ⊗ Eq
        v = v.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S1 ⊗ Ev

        q = ttnn.from_torch(q, device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        k = ttnn.from_torch(k, device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        v = ttnn.from_torch(v, device=spatial.device(), layout=ttnn.TILE_LAYOUT)

        q = self._spatial_attn.norm_q(q)
        k = self._spatial_attn.norm_k(k)

        if prompt is not None:
            assert self._prompt_attn is not None

            q2 = self._prompt_attn.q_proj(prompt)
            k2 = self._prompt_attn.k_proj(prompt)
            v2 = self._prompt_attn.v_proj(prompt)

            q2 = ttnn.to_torch(q2)
            k2 = ttnn.to_torch(k2)
            v2 = ttnn.to_torch(v2)

            # TODO: port to ttnn
            q2 = q2.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S2 ⊗ Eq
            k2 = k2.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S2 ⊗ Eq
            v2 = v2.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H ⊗ S2 ⊗ Ev

            q2 = ttnn.from_torch(q2, device=spatial.device(), layout=ttnn.TILE_LAYOUT)
            k2 = ttnn.from_torch(k2, device=spatial.device(), layout=ttnn.TILE_LAYOUT)
            v2 = ttnn.from_torch(v2, device=spatial.device(), layout=ttnn.TILE_LAYOUT)

            q2 = self._prompt_attn.norm_q(q2)
            k2 = self._prompt_attn.norm_k(k2)

            # TODO: `concat` does not work correctly with tilized tensors and `tilize` does not yield the correct result
            k = ttnn.from_torch(
                torch.cat([ttnn.to_torch(k), ttnn.to_torch(k2)], dim=2),
                device=spatial.device(),
                layout=ttnn.TILE_LAYOUT,
            )
            q = ttnn.from_torch(
                torch.cat([ttnn.to_torch(q), ttnn.to_torch(q2)], dim=2),
                device=spatial.device(),
                layout=ttnn.TILE_LAYOUT,
            )
            v = ttnn.from_torch(
                torch.cat([ttnn.to_torch(v), ttnn.to_torch(v2)], dim=2),
                device=spatial.device(),
                layout=ttnn.TILE_LAYOUT,
            )
            # q = ttnn.concat([q, q2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            # k = ttnn.concat([k, k2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Eq
            # v = ttnn.concat([v, v2], dim=2)  # N ⊗ H ⊗ (S1 + S2) ⊗ Ev

        k = ttnn.transpose(k, 2, 3)

        attention_scores = ttnn.from_torch(
            ttnn.to_torch(q) @ ttnn.to_torch(k), device=q.device(), layout=ttnn.TILE_LAYOUT
        )
        # attention_scores = ttnn.matmul(
        #     q,
        #     k,
        #     compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        #         math_fidelity=ttnn.MathFidelity.HiFi4,
        #     ),
        # )  # not very precise
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attention_probs = ttnn.from_torch(
            torch.softmax(ttnn.to_torch(attention_scores) / math.sqrt(q.shape[-1]), dim=-1),
            device=attention_scores.device(),
            layout=ttnn.TILE_LAYOUT,
        )
        # attention_probs = ttnn.transformer.attention_softmax(
        #     attention_scores, attention_mask=None, head_size=self._head_dim
        # ) # imprecise
        ttnn.deallocate(attention_scores)

        attn = ttnn.from_torch(
            ttnn.to_torch(attention_probs) @ ttnn.to_torch(v), device=v.device(), layout=ttnn.TILE_LAYOUT
        )
        # attn = ttnn.matmul(attention_probs, v) # imprecise
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(v)

        concatenated_attn = ttnn.transformer.concatenate_heads(attn)
        ttnn.deallocate(attn)

        if prompt is not None:
            torch_concatenated_attn = ttnn.to_torch(concatenated_attn)
            torch_spatial, torch_prompt = (
                torch_concatenated_attn[:, :spatial_sequence_length],
                torch_concatenated_attn[:, spatial_sequence_length:],
            )
            spatial = ttnn.from_torch(torch_spatial, device=spatial.device(), layout=ttnn.TILE_LAYOUT)
            prompt = ttnn.from_torch(torch_prompt, device=prompt.device(), layout=ttnn.TILE_LAYOUT)

            if self._prompt_attn.out_proj is not None:
                prompt = self._prompt_attn.out_proj(prompt)
        else:
            spatial = concatenated_attn

        if self._spatial_attn.out_proj is not None:
            spatial = self._spatial_attn.out_proj(spatial)

        return spatial, prompt


# def _concat(tensors: list[ttnn.Tensor], dim: int) -> ttnn.Tensor:
#     shape = list(tensors[0].shape)
#     for t in tensors[1:]:
#         shape[dim] += t.shape[dim]

#     result = ttnn.concat(tensors, dim=dim)
#     return ttnn.reshape(result, ttnn.Shape(shape, result.shape.with_tile_padding()))
