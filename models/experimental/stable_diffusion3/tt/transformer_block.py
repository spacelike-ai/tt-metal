from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .attention import TtAttention, TtAttentionParameters
from .feed_forward import TtFeedForward, TtFeedForwardParameters
from .linear import TtLinear, TtLinearParameters
from .normalization import TtLayerNorm, TtLayerNormParameters
from .substate import has_substate, substate


@dataclass
class TtTransformerBlockParameters:
    dual_attn: TtAttentionParameters
    spatial_attn: TtAttentionParameters | None
    prompt_time_embed: TtLinearParameters
    spatial_time_embed: TtLinearParameters
    prompt_norm_1: TtLayerNormParameters
    spatial_norm_1: TtLayerNormParameters
    spatial_norm_2: TtLayerNormParameters
    prompt_ff: TtFeedForwardParameters | None
    spatial_ff: TtFeedForwardParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtFeedForwardParameters:
        return cls(
            dual_attn=TtAttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            spatial_attn=TtAttentionParameters.from_torch(substate(state, "attn2"), dtype=dtype, device=device)
            if has_substate(state, "attn2")
            else None,
            spatial_norm_1=TtLayerNormParameters.from_torch(substate(state, "norm1.norm"), dtype=dtype, device=device),
            spatial_norm_2=TtLayerNormParameters.from_torch(substate(state, "norm2"), dtype=dtype, device=device),
            prompt_norm_1=TtLayerNormParameters.from_torch(
                substate(state, "norm1_context.norm"), dtype=dtype, device=device
            ),
            spatial_time_embed=TtLinearParameters.from_torch(
                substate(state, "norm1.linear"), dtype=dtype, device=device
            ),
            prompt_time_embed=TtLinearParameters.from_torch(
                substate(state, "norm1_context.linear"), dtype=dtype, device=device
            ),
            spatial_ff=TtFeedForwardParameters.from_torch(substate(state, "ff"), dtype=dtype, device=device),
            prompt_ff=TtFeedForwardParameters.from_torch(substate(state, "ff_context"), dtype=dtype, device=device)
            if has_substate(state, "ff_context")
            else None,
        )


class TtTransformerBlock:
    def __init__(
        self,
        parameters: TtTransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        eps = 1e-6

        self._dual_attn = TtAttention(parameters.dual_attn, num_heads=num_heads)
        self._spatial_attn = (
            TtAttention(parameters.spatial_attn, num_heads=num_heads) if parameters.spatial_attn is not None else None
        )

        self._spatial_norm_1 = TtLayerNorm(parameters.spatial_norm_1, eps=eps)
        self._spatial_norm_2 = TtLayerNorm(parameters.spatial_norm_2, eps=eps)
        self._prompt_norm_1 = TtLayerNorm(parameters.prompt_norm_1, eps=eps)
        self._prompt_norm_2 = TtLayerNorm(TtLayerNormParameters(), eps=eps)

        self._spatial_ff = TtFeedForward(parameters.spatial_ff, approximate="tanh")
        self._prompt_ff = (
            TtFeedForward(parameters.prompt_ff, approximate="tanh") if parameters.prompt_ff is not None else None
        )

        self._spatial_time_embed = TtLinear(parameters.spatial_time_embed)
        self._prompt_time_embed = TtLinear(parameters.prompt_time_embed)

        self._context_pre_only = self._prompt_ff is None

    def _spatial_attn_block(
        self,
        inp: ttnn.Tensor,
        *,
        gate: ttnn.Tensor,
        scale: ttnn.Tensor,
        shift: ttnn.Tensor,
    ) -> ttnn.Tensor:
        assert self._spatial_attn is not None

        scaled = inp * (1 + scale) + shift
        attn, _ = self._spatial_attn(spatial=scaled)

        result = gate * attn

        ttnn.deallocate(scaled)
        ttnn.deallocate(attn)
        return result

    def _dual_attn_block(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        spatial_gate: ttnn.Tensor,
        prompt_gate: ttnn.Tensor | None,
        prompt_scale: ttnn.Tensor,
        prompt_shift: ttnn.Tensor,
        spatial_scale: ttnn.Tensor,
        spatial_shift: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        # spatial = ttnn.from_torch(torch.load("spatial.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        # prompt = ttnn.from_torch(torch.load("prompt.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        # spatial_gate = ttnn.from_torch(torch.load("spatial_gate.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        # prompt_gate = ttnn.from_torch(torch.load("prompt_gate.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        # prompt_scale = ttnn.from_torch(torch.load("prompt_scale.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)
        # prompt_shift = ttnn.from_torch(torch.load("prompt_shift.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT)

        # spatial_scale = ttnn.from_torch(
        #     torch.load("spatial_scale.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT
        # )
        # spatial_shift = ttnn.from_torch(
        #     torch.load("spatial_shift.pt"), device=spatial.device(), layout=ttnn.TILE_LAYOUT
        # )

        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift

        # spatial_scaled = ttnn.from_torch(
        #     torch.load("spatial_scaled.pt"), device=prompt.device(), layout=ttnn.TILE_LAYOUT
        # )
        # prompt_scaled = ttnn.from_torch(torch.load("prompt_scaled.pt"), device=prompt.device(), layout=ttnn.TILE_LAYOUT)
        spatial_attn, prompt_attn = self._dual_attn(spatial=spatial_scaled, prompt=prompt_scaled)

        # spatial_attn_scaled = ttnn.from_torch(
        #     ttnn.to_torch(spatial_gate) * ttnn.to_torch(spatial_attn),
        #     device=spatial.device(),
        #     layout=ttnn.TILE_LAYOUT,
        # )
        # prompt_attn_scaled = (
        #     ttnn.from_torch(
        #         ttnn.to_torch(prompt_gate) * ttnn.to_torch(prompt_attn),
        #         device=prompt.device(),
        #         layout=ttnn.TILE_LAYOUT,
        #     )
        #     if prompt_gate is not None
        #     else None
        # )
        spatial_attn_scaled = spatial_gate * spatial_attn
        prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None

        ttnn.deallocate(spatial_attn)
        ttnn.deallocate(prompt_attn)
        return spatial_attn_scaled, prompt_attn_scaled

    def _spatial_ff_block(
        self,
        inp: ttnn.Tensor,
        *,
        gate: ttnn.Tensor,
        scale: ttnn.Tensor,
        shift: ttnn.Tensor,
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        result = gate * self._spatial_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def _prompt_ff_block(
        self,
        inp: ttnn.Tensor,
        *,
        gate: ttnn.Tensor,
        scale: ttnn.Tensor,
        shift: ttnn.Tensor,
    ) -> ttnn.Tensor:
        assert self._prompt_ff is not None

        scaled = inp * (1 + scale) + shift
        result = gate * self._prompt_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def __call__(
        self, *, spatial: ttnn.Tensor, prompt: ttnn.Tensor, time_embed: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        t = ttnn.silu(time_embed)
        spatial_time = self._spatial_time_embed(t)
        prompt_time = self._prompt_time_embed(t)
        ttnn.deallocate(t)

        if self._spatial_attn is not None:
            [
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
                spatial_shift_attn,
                spatial_scale_attn,
                spatial_gate_attn,
            ] = chunk_time(spatial_time, 9)
        else:
            [
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
            ] = chunk_time(spatial_time, 6)

            spatial_gate_attn = None
            spatial_shift_attn = None
            spatial_scale_attn = None

        if self._context_pre_only:
            [
                prompt_scale_attn,
                prompt_shift_attn,
            ] = chunk_time(prompt_time, 2)

            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            [
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ] = chunk_time(prompt_time, 6)

        spatial_normed = self._spatial_norm_1(spatial)
        prompt_normed = self._prompt_norm_1(prompt)

        spatial_attn, prompt_attn = self._dual_attn_block(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_gate=spatial_gate_dual_attn,
            prompt_gate=prompt_gate_attn,
            prompt_scale=prompt_scale_attn,
            prompt_shift=prompt_shift_attn,
            spatial_scale=spatial_scale_dual_attn,
            spatial_shift=spatial_shift_dual_attn,
        )
        ttnn.deallocate(prompt_normed)
        ttnn.deallocate(spatial_gate_dual_attn)
        if prompt_gate_attn is not None:
            ttnn.deallocate(prompt_gate_attn)
        ttnn.deallocate(prompt_scale_attn)
        ttnn.deallocate(prompt_shift_attn)
        ttnn.deallocate(spatial_scale_dual_attn)
        ttnn.deallocate(spatial_shift_dual_attn)

        spatial += spatial_attn
        ttnn.deallocate(spatial_attn)

        if self._spatial_attn is not None:
            assert spatial_gate_attn is not None
            assert spatial_scale_attn is not None
            assert spatial_shift_attn is not None

            spatial += self._spatial_attn_block(
                spatial_normed,
                gate=spatial_gate_attn,
                scale=spatial_scale_attn,
                shift=spatial_shift_attn,
            )
            ttnn.deallocate(spatial_normed)
            ttnn.deallocate(spatial_gate_attn)
            ttnn.deallocate(spatial_scale_attn)
            ttnn.deallocate(spatial_shift_attn)

        spatial_normed = self._spatial_norm_2(spatial)
        spatial += self._spatial_ff_block(
            spatial_normed,
            gate=spatial_gate_ff,
            scale=spatial_scale_ff,
            shift=spatial_shift_ff,
        )
        ttnn.deallocate(spatial_normed)
        ttnn.deallocate(spatial_gate_ff)
        ttnn.deallocate(spatial_scale_ff)
        ttnn.deallocate(spatial_shift_ff)

        if self._context_pre_only:
            return spatial, None

        assert prompt_scale_ff is not None
        assert prompt_shift_ff is not None
        assert prompt_gate_ff is not None

        prompt += prompt_attn
        ttnn.deallocate(prompt_attn)

        prompt_normed = self._prompt_norm_2(prompt)
        prompt += self._prompt_ff_block(
            prompt_normed,
            gate=prompt_gate_ff,
            scale=prompt_scale_ff,
            shift=prompt_shift_ff,
        )
        ttnn.deallocate(prompt_normed)
        ttnn.deallocate(prompt_gate_ff)
        ttnn.deallocate(prompt_scale_ff)
        ttnn.deallocate(prompt_shift_ff)

        return spatial, prompt


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    # TODO: the ttnn implementation does not give the correct result
    torch_chunks = ttnn.to_torch(t).chunk(count, dim=-1)
    return [ttnn.from_torch(x, device=t.device(), layout=ttnn.TILE_LAYOUT) for x in torch_chunks]

    # s = t.shape
    # batch_size = s[0]

    # t = ttnn.untilize(t)
    # t = ttnn.reshape(t, [s[0], s[1], count, s[2] // count])
    # t = ttnn.permute(t, [2, 0, 1, 3])
    # t = ttnn.reshape(t, [count * s[0], s[1], s[2] // count])
    # t = ttnn.tilize(t)

    # return [t[i * batch_size : (i + 1) * batch_size] for i in range(count)]
