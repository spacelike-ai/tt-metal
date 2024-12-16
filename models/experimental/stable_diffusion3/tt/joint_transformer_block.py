from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .attention import TtAttention, TtAttentionParameters
from .feed_forward import TtFeedForward, TtFeedForwardParameters
from .linear import TtLinearParameters
from .normalization import TtLayerNorm, TtLayerNormParameters
from .substate import has_substate, substate


@dataclass
class TtJointTransformerBlockParameters:
    spatial_t_embed: TtLinearParameters
    prompt_t_embed: TtLinearParameters
    attn_1: TtAttentionParameters
    attn_2: TtAttentionParameters | None
    spatial_norm_1: TtLayerNormParameters
    spatial_norm_2: TtLayerNormParameters
    prompt_norm_1: TtLayerNormParameters
    pormpt_norm_2: TtLayerNormParameters | None
    spatial_ff: TtFeedForwardParameters
    prompt_ff: TtFeedForwardParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtFeedForwardParameters:
        return cls(
            attn_1=TtAttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            attn_2=TtAttentionParameters.from_torch(substate(state, "attn2"), dtype=dtype, device=device)
            if has_substate(state, "attn2")
            else None,
            spatial_norm_1=TtLayerNormParameters.from_torch(substate(state, "norm1.norm"), dtype=dtype, device=device),
            spatial_norm_2=TtLayerNormParameters.from_torch(substate(state, "norm2"), dtype=dtype, device=device),
            prompt_norm_1=TtLayerNormParameters.from_torch(
                substate(state, "norm1_context.norm"), dtype=dtype, device=device
            ),
            pormpt_norm_2=TtLayerNormParameters.from_torch(substate(state, "norm2_context"), dtype=dtype, device=device)
            if has_substate(state, "norm2_context")
            else None,
            spatial_t_embed=TtLinearParameters.from_torch(substate(state, "norm1.linear"), dtype=dtype, device=device),
            prompt_t_embed=TtLayerNormParameters.from_torch(
                substate(state, "norm1_context.linear"), dtype=dtype, device=device
            ),
            spatial_ff=TtFeedForwardParameters.from_torch(substate(state, "ff"), dtype=dtype, device=device),
            prompt_ff=TtFeedForwardParameters.from_torch(substate(state, "ff_context"), dtype=dtype, device=device)
            if has_substate(state, "ff_context")
            else None,
        )


class TtJointTransformerBlock:
    def __init__(
        self,
        parameters: TtJointTransformerBlockParameters,
        *,
        num_attention_heads: int,
        attention_head_dim: int,
    ) -> None:
        self._attn_1 = TtAttention(parameters.attn_1, head_dim=attention_head_dim, num_heads=num_attention_heads)
        self._attn_2 = (
            TtAttention(parameters.attn_2, head_dim=attention_head_dim, num_heads=num_attention_heads)
            if parameters.attn2 is not None
            else None
        )

        self._spatial_norm_1 = TtLayerNorm(parameters.spatial_norm_1)
        self._spatial_norm_2 = TtLayerNorm(parameters.spatial_norm_2, eps=1e-6)
        self._prompt_norm_1 = TtLayerNorm(parameters.prompt_norm_1)
        self._pormpt_norm_2 = (
            TtLayerNorm(parameters.pormpt_norm_2, eps=1e-6) if parameters.pormpt_norm_2 is not None else None
        )

        self._spatial_ff = TtFeedForward(parameters.spatial_ff, approximate="tanh")
        self._prompt_ff = (
            TtFeedForward(parameters.prompt_ff, approximate="tanh") if parameters.prompt_ff is not None else None
        )

        self._spatial_t_embed = TtLayerNorm(parameters.spatial_t_embed)
        self._prompt_t_embed = TtLayerNorm(parameters.prompt_t_embed)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        temb: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        if self._attn_2 is None:
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

        if self._pormpt_norm_2 is not None:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            c_gate_msa = None
            c_shift_mlp = None
            c_scale_mlp = None
            c_gate_mlp = None

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.attn2 is not None:
            assert gate_msa2 is not None
            attn_output2, _ = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        spatial_ff_output = self.spatial_ff(norm_hidden_states)
        spatial_ff_output = gate_mlp.unsqueeze(1) * spatial_ff_output

        hidden_states = hidden_states + spatial_ff_output

        if self.context_pre_only:
            return None, hidden_states

        assert self.norm2_context is not None
        assert self.prompt_ff is not None
        assert c_gate_msa is not None
        assert c_scale_mlp is not None
        assert c_shift_mlp is not None
        assert c_gate_mlp is not None

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.prompt_ff(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states
