from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import substate, substate_exists


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
            if substate_exists(state, "add_q_proj")
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
    def __init__(self, parameters: TtAttentionParameters) -> None:
        super().__init__()

        self.part_a = TtAttentionPart(parameters.part_a)
        self.part_b = TtAttentionPart(parameters.part_b) if parameters.part_b is not None else None

    def forward(
        self, hidden_states: ttnn.Tensor, encoder_hidden_states: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        batch_size = hidden_states.shape[0]

        num_heads = self.num_heads
        head_dim = self.head_dim

        residual = hidden_states

        q = self.part_a.k_proj(hidden_states)
        k = self.part_a.k_proj(hidden_states)
        v = self.part_a.v_proj(hidden_states)

        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        q = self.part_a.norm_q(q)
        k = self.part_a.norm_k(k)

        if self.part_b is not None:
            encoder_hidden_states_query_proj = self.q_proj_2(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.k_proj_2(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.v_proj_2(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)

            encoder_hidden_states_query_proj = self.norm_q_2(encoder_hidden_states_query_proj)
            encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

            q = torch.cat([q, encoder_hidden_states_query_proj], dim=2)
            k = torch.cat([k, encoder_hidden_states_key_proj], dim=2)
            v = torch.cat([v, encoder_hidden_states_value_proj], dim=2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        hidden_states = hidden_states.to(q.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not self.context_pre_only:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.out_proj_1[0](hidden_states)

        return hidden_states, encoder_hidden_states
