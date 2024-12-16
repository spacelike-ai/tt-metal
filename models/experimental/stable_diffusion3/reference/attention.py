from __future__ import annotations

import torch

from .normalization import RmsNorm


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Attention(torch.nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        dim_head: int,
        heads: int,
        out_dim: int,
        qk_norm: str,
        added_kv_proj_dim: int = 0,
        context_pre_only: bool = True,
    ) -> None:
        super().__init__()

        if qk_norm != "rms_norm":
            msg = "invalid qk_norm"
            raise ValueError(msg)

        eps = 1e-6

        self.context_pre_only = context_pre_only
        self.head_dim = dim_head
        self.num_heads = heads

        self.norm_q = RmsNorm(dim=dim_head, eps=eps)
        self.norm_k = RmsNorm(dim=dim_head, eps=eps)

        self.to_q = torch.nn.Linear(query_dim, out_dim)
        self.to_k = torch.nn.Linear(query_dim, out_dim)
        self.to_v = torch.nn.Linear(query_dim, out_dim)

        if added_kv_proj_dim > 0:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(out_dim, out_dim))

        if not self.context_pre_only:
            self.to_add_out = torch.nn.Linear(out_dim, out_dim)

        if added_kv_proj_dim > 0:
            self.norm_added_q = RmsNorm(dim=dim_head, eps=eps)
            self.norm_added_k = RmsNorm(dim=dim_head, eps=eps)
        else:
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        # if encoder_hidden_states is not None:
        #     encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
        #     encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
        #     encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

        #     encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)
        #     encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)
        #     encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        #         batch_size, -1, num_heads, head_dim
        #     ).transpose(1, 2)

        #     if self.norm_added_q is not None:
        #         encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
        #     if self.norm_added_k is not None:
        #         encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

        #     query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        #     key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        #     value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # if encoder_hidden_states is not None:
        #     hidden_states, encoder_hidden_states = (
        #         hidden_states[:, : residual.shape[1]],
        #         hidden_states[:, residual.shape[1] :],
        #     )
        #     if not self.context_pre_only:
        #         encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out[0](hidden_states)

        return hidden_states, encoder_hidden_states
