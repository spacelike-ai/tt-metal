# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .normalization import RmsNorm, RmsNormParameters
from .optimizations import AttentionOptimization, AttentionPartOptimization
from .substate import has_substate, substate


@dataclass
class AttentionPartParameters:
    qkv_proj: LinearParameters
    norm_q: RmsNormParameters
    norm_k: RmsNormParameters
    mesh_width: int
    out_proj: LinearParameters | None


@dataclass
class AttentionParameters:
    spatial: AttentionPartParameters
    prompt: AttentionPartParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> AttentionParameters:
        _, mesh_width = device.shape

        return cls(
            spatial=AttentionPartParameters(
                qkv_proj=LinearParameters.from_torch(
                    _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v")),
                    dtype=dtype,
                    device=device,
                    mesh_sharding_dim=0,
                    chunks=3,
                ),
                norm_q=RmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=RmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=(
                    LinearParameters.from_torch(
                        substate(state, "to_out.0"),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                    )
                    if has_substate(state, "to_out.0")
                    else None
                ),
                mesh_width=mesh_width,
            ),
            prompt=AttentionPartParameters(
                qkv_proj=(
                    LinearParameters.from_torch(
                        _merge_qkv_proj(
                            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
                        ),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                        chunks=3,
                    )
                    if has_substate(state, "add_q_proj")
                    else None
                ),
                norm_q=RmsNormParameters.from_torch(substate(state, "norm_added_q"), dtype=dtype, device=device),
                norm_k=RmsNormParameters.from_torch(substate(state, "norm_added_k"), dtype=dtype, device=device),
                out_proj=(
                    LinearParameters.from_torch(
                        substate(state, "to_add_out"),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                    )
                    if has_substate(state, "add_q_proj")
                    else None
                ),
                mesh_width=mesh_width,
            )
            if has_substate(state, "add_q_proj")
            else None,
        )


class AttentionPart:
    def __init__(self, parameters: AttentionPartParameters, optimization: AttentionPartOptimization) -> None:
        super().__init__()

        eps = 1e-6
        self._opt = optimization

        self._qkv_proj = Linear(parameters.qkv_proj)
        self._out_proj = Linear(parameters.out_proj) if parameters.out_proj is not None else None
        self._norm_q = RmsNorm(parameters.norm_q, eps=eps)
        self._norm_k = RmsNorm(parameters.norm_k, eps=eps)

        self._mesh_width = parameters.mesh_width

    def qkv(self, x: ttnn.Tensor, *, num_heads: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        utils.signpost("qkv preparation")

        x = self._opt.prepare_qkv_projection(x)

        x = self._qkv_proj.forward(x, **self._opt.qkv_projection_settings(x.device()))

        x = self._opt.prepare_split(x)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            x,
            num_heads=num_heads // self._mesh_width,
            transpose_key=False,
        )
        del x

        q = self._norm_q.forward(q)
        k = self._norm_k.forward(k)

        return self._opt.postprocess_split(q, k, v)

    def out_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._out_proj is None:
            return x

        x = self._out_proj.forward(x)
        return self._opt.postprocess_out_projection(x)


class Attention:
    def __init__(self, parameters: AttentionParameters, *, num_heads: int) -> None:
        super().__init__()

        self._num_heads = num_heads
        self._opt = AttentionOptimization()

        self._spatial_attn = AttentionPart(parameters.spatial, optimization=self._opt.spatial_part())
        self._prompt_attn = (
            AttentionPart(parameters.prompt, optimization=self._opt.prompt_part())
            if parameters.prompt is not None
            else None
        )

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H * E1)
        prompt: N ⊗ S2 ⊗ (H * E2)
        """
        device = spatial.device()

        q, k, v = self._spatial_attn.qkv(spatial, num_heads=self._num_heads)

        if prompt is None:
            if image_rotary_emb is not None:
                utils.signpost("rotary embedding path I")
                q = _apply_rotary_emb(q, image_rotary_emb)
                k = _apply_rotary_emb(k, image_rotary_emb)

            utils.signpost("dot product attention path I")
            # operands must be in DRAM
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, **self._opt.sdpa_settings(device=device)
            )
            del q, k, v

            attn = ttnn.transformer.concatenate_heads(attn)

            spatial = self._spatial_attn.out_proj(attn)
            return spatial, None

        assert self._prompt_attn is not None

        q2, k2, v2 = self._prompt_attn.qkv(prompt, num_heads=self._num_heads)

        q = ttnn.concat([q2, q], dim=2)
        k = ttnn.concat([k2, k], dim=2)
        v = ttnn.concat([v2, v], dim=2)
        del q2, k2, v2

        if image_rotary_emb is not None:
            utils.signpost("rotary embedding path II")
            q = _apply_rotary_emb(q, image_rotary_emb)
            k = _apply_rotary_emb(k, image_rotary_emb)

        utils.signpost("dot product attention path II")
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, **self._opt.sdpa_settings(device=device)
        )
        del q, k, v

        attn = ttnn.transformer.concatenate_heads(attn)

        if prompt is not None:
            prompt = attn[:, : prompt.shape[1]]
            spatial = attn[:, prompt.shape[1] :]
        else:
            spatial = attn

        # if image_rotary_emb is not None:
        #     emb = (image_rotary_emb[0][-q.shape[2] :], image_rotary_emb[1][-q.shape[2] :])
        #     q = _apply_rotary_emb(q, emb)
        #     k = _apply_rotary_emb(k, emb)
        #     emb = (image_rotary_emb[0][: q2.shape[2]], image_rotary_emb[1][: q2.shape[2]])
        #     q2 = _apply_rotary_emb(q2, emb)
        #     k2 = _apply_rotary_emb(k2, emb)
        #
        # prompt, spatial = ttnn.transformer.joint_scaled_dot_product_attention(
        #     q2, k2, v2, q, k, v, joint_strategy="rear", **self._opt.sdpa_settings(device=device)
        # )
        #
        # spatial = ttnn.transformer.concatenate_heads(spatial)
        # prompt = ttnn.transformer.concatenate_heads(prompt)

        spatial = self._spatial_attn.out_proj(spatial)
        prompt = self._prompt_attn.out_proj(prompt)

        return spatial, prompt


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor | None],
    k_state: dict[str, torch.Tensor | None],
    v_state: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]),
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]),
    }


def _apply_rotary_emb(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])

    return x * cos + ttnn.alt_complex_rotate90(x) * sin
