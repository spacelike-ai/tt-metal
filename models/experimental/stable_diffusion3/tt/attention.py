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
    out_proj: TtLinearParameters


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
                out_proj=TtLinearParameters.from_torch(substate(state, "to_add_out"), dtype=dtype, device=device),
            )
            if has_substate(state, "add_q_proj")
            else None,
        )


class TtAttentionPart:
    def __init__(self, parameters: TtAttentionPartParameters) -> None:
        super().__init__()

        eps = 1e-6

        self.q_proj = TtLinear(parameters.q_proj)
        self.k_proj = TtLinear(parameters.k_proj)
        self.v_proj = TtLinear(parameters.v_proj)
        self.out_proj = TtLinear(parameters.out_proj)
        self.norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self.norm_k = TtRmsNorm(parameters.norm_k, eps=eps)


class TtAttention:
    def __init__(self, parameters: TtAttentionParameters, *, num_heads: int, head_dim: int) -> None:
        super().__init__()

        self._num_heads = num_heads
        self._head_dim = head_dim

        self._part_a = TtAttentionPart(parameters.part_a)
        self._part_b = TtAttentionPart(parameters.part_b) if parameters.part_b is not None else None

    def __call__(
        self, spatial: ttnn.Tensor, prompt_embed: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H1 * E1)
        prompt_embed: N ⊗ S2 ⊗ (H2 * E2)
        """
        batch_size = spatial.shape[0]

        q = self._part_a.q_proj(spatial)  # N ⊗ S1 ⊗ (H1 * Eq1)
        k = self._part_a.k_proj(spatial)  # N ⊗ S1 ⊗ (H1 * Eq1)
        v = self._part_a.v_proj(spatial)  # N ⊗ S1 ⊗ (H1 * Ev1)

        q = ttnn.to_torch(q)
        k = ttnn.to_torch(k)
        v = ttnn.to_torch(v)

        q = q.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H1 ⊗ S1 ⊗ Eq1
        k = k.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H1 ⊗ Eq1 ⊗ S1
        v = v.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)  # N ⊗ H1 ⊗ S1 ⊗ Ev1

        q = ttnn.from_torch(q, device=spatial.device())
        k = ttnn.from_torch(k, device=spatial.device())
        v = ttnn.from_torch(v, device=spatial.device())

        q = self._part_a.norm_q(q)
        k = self._part_a.norm_k(k)

        # if self._part_b is not None:
        #     prompt_embed_query_proj = self.q_proj_2(prompt_embed)
        #     prompt_embed_key_proj = self.k_proj_2(prompt_embed)
        #     prompt_embed_value_proj = self.v_proj_2(prompt_embed)

        #     prompt_embed_query_proj = prompt_embed_query_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)
        #     prompt_embed_key_proj = prompt_embed_key_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)
        #     prompt_embed_value_proj = prompt_embed_value_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)

        #     prompt_embed_query_proj = self.norm_q_2(prompt_embed_query_proj)
        #     prompt_embed_key_proj = self.norm_added_k(prompt_embed_key_proj)

        #     q = torch.cat([q, prompt_embed_query_proj], dim=2)
        #     k = torch.cat([k, prompt_embed_key_proj], dim=2)
        #     v = torch.cat([v, prompt_embed_value_proj], dim=2)

        k = ttnn.transpose(k, 2, 3)
        k = ttnn.tilize(k)
        q = ttnn.tilize(q)
        v = ttnn.tilize(v)

        attention_scores = ttnn.matmul(q, k)
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

        spatial = self._part_a.out_proj(concatenated_attn)
        ttnn.deallocate(concatenated_attn)

        # if prompt_embed is not None:
        #     spatial, prompt_embed = (
        #         spatial[:, : residual.shape[1]],
        #         spatial[:, residual.shape[1] :],
        #     )
        #     if not self.context_pre_only:
        #         prompt_embed = self.to_add_out(prompt_embed)

        return spatial, prompt_embed
