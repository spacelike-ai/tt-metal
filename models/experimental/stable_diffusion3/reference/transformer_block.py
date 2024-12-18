from __future__ import annotations

import torch

from .attention import Attention
from .feed_forward import FeedForward
from .normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    SD35AdaLayerNormZeroX,
)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention.py
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool,
        qk_norm: str,
        use_dual_attention: bool,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only

        self.attn = Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, approximate="tanh")

        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(dim, dim)
            self.norm2_context = None
            self.ff_context = None
        else:
            self.norm1_context = AdaLayerNormZero(dim)
            self.norm2_context = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, approximate="tanh")

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
            self.attn2 = Attention(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                qk_norm=qk_norm,
            )
        else:
            self.norm1 = AdaLayerNormZero(dim)
            self.attn2 = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.attn2 is None:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_hidden_states2 = None
            gate_msa2 = None
        else:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                norm_hidden_states2,
                gate_msa2,
            ) = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            c_gate_msa = None
            c_shift_mlp = None
            c_scale_mlp = None
            c_gate_mlp = None
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.attn2 is not None:
            assert gate_msa2 is not None
            attn_output2, _ = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if self.context_pre_only:
            return None, hidden_states

        assert self.norm2_context is not None
        assert self.ff_context is not None
        assert c_gate_msa is not None
        assert c_scale_mlp is not None
        assert c_shift_mlp is not None
        assert c_gate_mlp is not None

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states
