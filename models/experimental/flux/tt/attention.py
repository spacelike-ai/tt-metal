# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import has_substate, substate


@dataclass
class TtAttentionPartParameters:
    qkv_proj: TtLinearParameters
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
        spatial_qkv_proj = _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v"))
        prompt_qkv_proj = _merge_qkv_proj(
            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
        )

        return cls(
            spatial=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch(spatial_qkv_proj, dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_out.0"), dtype=dtype, device=device)
                if has_substate(state, "to_out.0")
                else None,
            ),
            prompt=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch(prompt_qkv_proj, dtype=dtype, device=device),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_added_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_added_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(substate(state, "to_add_out"), dtype=dtype, device=device)
                if has_substate(state, "to_add_out")
                else None,
            )
            if has_substate(state, "add_q_proj")
            else None,
        )


class TtAttentionPart:
    def __init__(self, parameters: TtAttentionPartParameters) -> None:
        super().__init__()

        eps = 1e-6

        self._qkv_proj = TtLinear(parameters.qkv_proj)
        self._out_proj = TtLinear(parameters.out_proj) if parameters.out_proj is not None else None
        self._norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self._norm_k = TtRmsNorm(parameters.norm_k, eps=eps)

    def qkv(self, x: ttnn.Tensor, *, num_heads: int, deallocate: bool) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        _batch_size, sequence_length, _embedding_dim = x.shape

        # # Input sharding
        # if sequence_length > 1024:
        #     # sharding leads to worse PCC, so disable it until further investigation
        #     mm_a_x = 8
        #     mm_a_y = 8
        #     mm_a_x_memory_config = ttnn.L1_MEMORY_CONFIG
        # elif sequence_length >= 512:
        #     mm_a_y = 8
        #     mm_a_x = 8
        #     mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        #     mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        #     x = to_memory_config(
        #         x,
        #         memory_config=ttnn.create_sharded_memory_config(
        #             x.shape,
        #             core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        #             strategy=mm_a_x_strategy,
        #             orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #         ),
        #         deallocate=deallocate,
        #     )
        #     deallocate = True
        # else:
        #     mm_a_x = 8
        #     mm_a_y = 6
        #     mm_a_x_memory_config = ttnn.L1_MEMORY_CONFIG

        qkv = self._qkv_proj(
            x,
            # memory_config=mm_a_x_memory_config,
            # core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            # dtype=ttnn.bfloat8_b,
            deallocate=deallocate,
        )

        # qkv = ttnn.reallocate(qkv)
        # qkv = to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG, deallocate=True)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=num_heads, transpose_key=False)
        ttnn.deallocate(qkv)

        q = self._norm_q(q, deallocate=True)
        k = self._norm_k(k, deallocate=True)

        # q = to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)
        # k = to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)
        # v = to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)

        return q, k, v

    def out_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._out_proj is None:
            return x

        return self._out_proj(x)

        # return to_memory_config(
        #     result,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     dtype=ttnn.bfloat16,
        #     deallocate=True,
        # )


class TtAttention:
    def __init__(self, parameters: TtAttentionParameters, *, num_heads: int) -> None:
        super().__init__()

        self._num_heads = num_heads

        self._spatial_attn = TtAttentionPart(parameters.spatial)
        self._prompt_attn = TtAttentionPart(parameters.prompt) if parameters.prompt is not None else None

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        deallocate: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H * E1)
        prompt: N ⊗ S2 ⊗ (H * E2)
        """
        device = spatial.device()

        q, k, v = self._spatial_attn.qkv(spatial, num_heads=self._num_heads, deallocate=deallocate)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=256,
            k_chunk_size=512,
            exp_approx_mode=True,
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        if prompt is None:
            if image_rotary_emb is not None:
                q = _apply_rotary_emb(q, image_rotary_emb)
                k = _apply_rotary_emb(k, image_rotary_emb)

            # operands must be in DRAM
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            concatenated_attn = ttnn.transformer.concatenate_heads(attn)
            ttnn.deallocate(attn)

            spatial = self._spatial_attn.out_proj(concatenated_attn)
            return spatial, None

        assert self._prompt_attn is not None

        q2, k2, v2 = self._prompt_attn.qkv(prompt, num_heads=self._num_heads, deallocate=deallocate)

        q = ttnn.concat([q2, q], dim=2)
        k = ttnn.concat([k2, k], dim=2)
        v = ttnn.concat([v2, v], dim=2)

        if image_rotary_emb is not None:
            q = _apply_rotary_emb(q, image_rotary_emb)
            k = _apply_rotary_emb(k, image_rotary_emb)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
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
        #     q2,
        #     k2,
        #     v2,
        #     q,
        #     k,
        #     v,
        #     joint_strategy="rear",
        #     program_config=program_config,
        #     compute_kernel_config=compute_kernel_config,
        # )
        #
        # spatial = ttnn.transformer.concatenate_heads(spatial)
        # prompt = ttnn.transformer.concatenate_heads(prompt)

        spatial = self._spatial_attn.out_proj(spatial)
        prompt = self._prompt_attn.out_proj(prompt)

        return spatial, prompt


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor],
    k_state: dict[str, torch.Tensor],
    v_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]) if "weight" in q_state else None,
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]) if "bias" in q_state else None,
    }


def _rotate(x: ttnn.Tensor) -> ttnn.Tensor:
    [*batch_dims, _] = x.shape

    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = x.reshape([*batch_dims, -1, 2])

    x = ttnn.permute(x, [4, 0, 1, 2, 3])

    x_real = ttnn.to_layout(x[0:1], ttnn.TILE_LAYOUT)
    x_imag = ttnn.to_layout(x[1:2], ttnn.TILE_LAYOUT)

    x_real, x_imag = (ttnn.neg(x_imag), x_real)

    x = ttnn.concat([x_real, x_imag], dim=0)
    x = ttnn.permute(x, [1, 2, 3, 4, 0])

    return x.reshape([*batch_dims, -1])


def _apply_rotary_emb(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])

    return x * cos + _rotate(x) * sin
