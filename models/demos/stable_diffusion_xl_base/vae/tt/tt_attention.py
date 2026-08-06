# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtAttention(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim: int,
        heads: int = 8,
        out_dim: int = None,
        kv_heads=None,
        dim_head: int = 64,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        self.norm_core_grid = ttnn.CoreGrid(y=8, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-6

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.sdpa_program_config = model_config.get_sdpa_config(module_path=module_path, is_self_attention=True)

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        norm_weights = state_dict[f"{module_path}.group_norm.weight"]
        norm_bias = state_dict[f"{module_path}.group_norm.bias"]
        (
            self.groupnorm_config,
            self.groupnorm_memory_config,
            self.input_mask,
            self.input_negative_mask,
            self.gamma_t,
            self.beta_t,
        ) = model_config.get_groupnorm_params(f"{module_path}.norm", norm_weights, norm_bias, self.norm_groups, device)
        assert (
            self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        q_bias = state_dict[f"{module_path}.to_q.bias"]
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        k_bias = state_dict[f"{module_path}.to_k.bias"]
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)
        v_bias = state_dict[f"{module_path}.to_v.bias"]

        fused_qkv_weights = torch.cat(
            [
                torch.transpose(q_weights, -2, -1),
                torch.transpose(k_weights, -2, -1),
                torch.transpose(v_weights, -2, -1),
            ],
            dim=-1,
        )
        self.tt_qkv_weights = ttnn.from_torch(fused_qkv_weights, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        fused_qkv_bias = torch.concat(
            [q_bias, k_bias, v_bias],
            dim=-1,
        ).unsqueeze(0)
        self.tt_qkv_bias = ttnn.from_torch(fused_qkv_bias, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT)

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(device, out_weights, out_bias, ttnn.bfloat16)

    def forward(self, input_tensor, input_shape, encoder_hidden_states=None):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            negative_mask=self.input_negative_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            **self.groupnorm_config,
        )

        assert encoder_hidden_states is None, "VAE does self attention only"
        encoder_hidden_states = hidden_states

        centre_v = os.environ.get("VAE_CENTRE_V")
        zero_in = ttnn.multiply(hidden_states, 0.0) if centre_v == "bias" else None

        qkv_fused = ttnn.linear(
            hidden_states,
            self.tt_qkv_weights,
            bias=self.tt_qkv_bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(hidden_states)

        (
            q_heads,
            k_heads,
            v_heads,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            qkv_fused, num_heads=self.heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv_fused)

        v_bar = None
        if centre_v:
            # Attention weights sum to 1, so sum_j p_j (v_j - v_bar) = out - v_bar: removing any
            # fixed vector from V and adding it back is exact. The VAE's V is a groupnorm'd feature
            # map -- one dominant shared direction plus a small residual -- and that common mode is
            # what the bf16 pack-accumulate onto cb_out_im drifts on, coherently, across k chunks.
            # "bias" takes only the token-independent part (the qkv bias, recovered by projecting a
            # zeroed input so it lands in the same head layout); "mean" also takes Wv @ x_bar.
            if centre_v == "bias":
                qkv_bias_only = ttnn.linear(
                    zero_in,
                    self.tt_qkv_weights,
                    bias=self.tt_qkv_bias,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=self.compute_kernel_config,
                )
                v_bias = ttnn.experimental.nlp_create_qkv_heads(
                    qkv_bias_only, num_heads=self.heads, transpose_k_heads=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )[2]
                v_bar = ttnn.mean(v_bias, dim=2, keepdim=True)
            else:
                v_bar = ttnn.mean(v_heads, dim=2, keepdim=True)
            v_heads = ttnn.subtract(v_heads, v_bar)

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            attn_mask=None,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        if v_bar is not None:
            hidden_states = ttnn.add(hidden_states, v_bar)
        hidden_states = ttnn.experimental.nlp_concat_heads(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = ttnn.add(hidden_states, input_tensor)

        return hidden_states
