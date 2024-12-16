from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import substate, substate_exists


@dataclass
class TtAttentionParameters:
    to_q: TtLinearParameters
    to_k: TtLinearParameters
    to_v: TtLinearParameters
    to_out: TtLinearParameters
    norm_q: TtRmsNormParameters
    norm_k: TtRmsNormParameters
    add_q_proj: TtLinearParameters | None
    add_k_proj: TtLinearParameters | None
    add_v_proj: TtLinearParameters | None
    to_add_out: TtLinearParameters | None
    norm_added_q: TtRmsNormParameters | None
    norm_added_k: TtRmsNormParameters | None

    @classmethod
    def prepare(
        cls,
        torch_state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtAttentionParameters:
        return cls(
            to_q=TtLinearParameters.prepare(substate(torch_state, "to_q"), dtype=dtype, device=device),
            to_k=TtLinearParameters.prepare(substate(torch_state, "to_k"), dtype=dtype, device=device),
            to_v=TtLinearParameters.prepare(substate(torch_state, "to_v"), dtype=dtype, device=device),
            to_out=TtLinearParameters.prepare(substate(torch_state, "to_out.0"), dtype=dtype, device=device),
            norm_q=TtRmsNormParameters.prepare(substate(torch_state, "norm_q"), dtype=dtype, device=device),
            norm_k=TtRmsNormParameters.prepare(substate(torch_state, "norm_k"), dtype=dtype, device=device),
            add_q_proj=TtLinearParameters.prepare(substate(torch_state, "add_q_proj"), dtype=dtype, device=device)
            if substate_exists(torch_state, "add_q_proj")
            else None,
            add_k_proj=TtLinearParameters.prepare(substate(torch_state, "add_k_proj"), dtype=dtype, device=device)
            if substate_exists(torch_state, "add_k_proj")
            else None,
            add_v_proj=TtLinearParameters.prepare(substate(torch_state, "add_v_proj"), dtype=dtype, device=device)
            if substate_exists(torch_state, "add_v_proj")
            else None,
            to_add_out=TtLinearParameters.prepare(substate(torch_state, "to_add_out"), dtype=dtype, device=device)
            if substate_exists(torch_state, "to_add_out")
            else None,
            norm_added_q=TtRmsNormParameters.prepare(substate(torch_state, "norm_added_q"), dtype=dtype, device=device)
            if substate_exists(torch_state, "norm_added_q")
            else None,
            norm_added_k=TtRmsNormParameters.prepare(substate(torch_state, "norm_added_k"), dtype=dtype, device=device)
            if substate_exists(torch_state, "norm_added_k")
            else None,
        )


# TODO: split between one class with and one without encoder_hidden_states
class TtAttention(torch.nn.Module):
    def __init__(self, parameters: TtAttentionParameters) -> None:
        super().__init__()

        eps = 1e-6

        self.to_q = TtLinear(parameters.to_q)
        self.to_k = TtLinear(parameters.to_k)
        self.to_v = TtLinear(parameters.to_v)
        self.to_out = TtLinear(parameters.to_out)
        self.norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self.norm_k = TtRmsNorm(parameters.norm_k, eps=eps)
        self.add_q_proj = TtLinear(parameters.add_q_proj) if parameters.add_q_proj is not None else None
        self.add_k_proj = TtLinear(parameters.add_k_proj) if parameters.add_k_proj is not None else None
        self.add_v_proj = TtLinear(parameters.add_v_proj) if parameters.add_v_proj is not None else None
        self.norm_added_q = TtRmsNorm(parameters.norm_added_q, eps=eps) if parameters.norm_added_q is not None else None
        self.norm_added_k = TtRmsNorm(parameters.norm_added_k, eps=eps) if parameters.norm_added_k is not None else None
        self.to_add_out = TtLinear(parameters.to_add_out) if parameters.to_add_out is not None else None

        # self.head_dim = dim_head
        # self.num_heads = heads

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        batch_size = hidden_states.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        residual = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, num_heads, head_dim
            ).transpose(1, 2)

            if self.norm_added_q is not None:
                encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
            if self.norm_added_k is not None:
                encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not self.context_pre_only:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out[0](hidden_states)

        return hidden_states, encoder_hidden_states
