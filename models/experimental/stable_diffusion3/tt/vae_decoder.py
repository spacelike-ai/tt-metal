# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .conv2d import TtConv2d, TtConv2dParameters
from .linear import TtLinear, TtLinearParameters
from .substate import has_substate, indexed_substates, substate


@dataclass
class TtVaeDecoderParameters:
    conv_in: TtConv2dParameters
    mid_block: TtUNetMidBlock2DParameters
    up_blocks: list[TtUpDecoderBlock2DParameters]
    conv_norm_out: TtGroupNormParameters
    conv_out: TtConv2dParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtVaeDecoderParameters:
        return cls(
            conv_in=TtConv2dParameters.from_torch(substate(state, "conv_in"), dtype=dtype),
            conv_out=TtConv2dParameters.from_torch(substate(state, "conv_out"), dtype=dtype),
            conv_norm_out=TtGroupNormParameters.from_torch(
                substate(state, "conv_norm_out"), dtype=dtype, device=device
            ),
            mid_block=TtUNetMidBlock2DParameters.from_torch(substate(state, "mid_block"), dtype=dtype, device=device),
            up_blocks=[
                TtUpDecoderBlock2DParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "up_blocks")
            ],
        )


class TtVaeDecoder:
    def __init__(self, parameters: TtVaeDecoderParameters, *, norm_num_groups: int = 32) -> None:
        super().__init__()

        attention_head_dim = parameters.up_blocks[0].resnets[0].conv1.in_channels

        self._conv_in = TtConv2d(parameters.conv_in, padding=(1, 1))
        self._mid_block = TtUNetMidBlock2D(
            parameters.mid_block, attention_head_dim=attention_head_dim, resnet_groups=norm_num_groups
        )
        self._up_blocks = [TtUpDecoderBlock2D(p, resnet_groups=norm_num_groups) for p in parameters.up_blocks]
        self._conv_norm_out = TtGroupNorm(parameters.conv_norm_out, num_groups=norm_num_groups, eps=1e-6)
        self._conv_out = TtConv2d(parameters.conv_out, padding=(1, 1))

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._conv_in(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # TODO: remove
        x = self._mid_block(x)

        for up_block in self._up_blocks:
            x = up_block(x)

        x = self._conv_norm_out(x, inplace=False)  # TODO: change to inplace=True

        x = ttnn.silu(x)
        return self._conv_out(x)


@dataclass
class TtUpDecoderBlock2DParameters:
    resnets: list[TtResnetBlock2DParameters]
    upsampler: TtConv2dParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtUpDecoderBlock2DParameters:
        return cls(
            resnets=[
                TtResnetBlock2DParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "resnets")
            ],
            upsampler=TtConv2dParameters.from_torch(substate(state, "upsamplers.0.conv"), dtype=dtype)
            if has_substate(state, "upsamplers.0.conv")
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.resnets[0].in_channels


class TtUpDecoderBlock2D:
    def __init__(self, parameters: TtUpDecoderBlock2DParameters, *, resnet_groups: int) -> None:
        super().__init__()

        self._resnets = [TtResnetBlock2D(p, num_groups=resnet_groups) for p in parameters.resnets]
        self._upsampler_conv = (
            TtConv2d(parameters.upsampler, padding=(1, 1)) if parameters.upsampler is not None else None
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for resnet in self._resnets:
            x = resnet(x)

        if self._upsampler_conv is not None:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.upsample(x, 2)
            x = self._upsampler_conv(x)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # TODO: remove

        return x


@dataclass
class TtUNetMidBlock2DParameters:
    attention: TtAttentionParameters
    resnet1: TtResnetBlock2DParameters
    resnet2: TtResnetBlock2DParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtUNetMidBlock2DParameters:
        return cls(
            resnet1=TtResnetBlock2DParameters.from_torch(substate(state, "resnets.0"), dtype=dtype, device=device),
            resnet2=TtResnetBlock2DParameters.from_torch(substate(state, "resnets.1"), dtype=dtype, device=device),
            attention=TtAttentionParameters.from_torch(substate(state, "attentions.0"), dtype=dtype, device=device),
        )


class TtUNetMidBlock2D:
    def __init__(
        self,
        parameters: TtUNetMidBlock2DParameters,
        *,
        resnet_groups: int,
        attention_head_dim: int,
    ) -> None:
        super().__init__()

        self._attention = TtAttention(
            parameters.attention,
            dim_head=attention_head_dim,
            norm_num_groups=resnet_groups,
        )

        self._resnet1 = TtResnetBlock2D(parameters.resnet1, num_groups=resnet_groups)
        self._resnet2 = TtResnetBlock2D(parameters.resnet2, num_groups=resnet_groups)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._resnet1(x)
        x = self._attention(x)
        return self._resnet2(x)


@dataclass
class TtResnetBlock2DParameters:
    norm1: TtGroupNormParameters
    norm2: TtGroupNormParameters
    conv1: TtConv2dParameters
    conv2: TtConv2dParameters
    conv_shortcut: TtConv2dParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtResnetBlock2DParameters:
        return cls(
            norm1=TtGroupNormParameters.from_torch(substate(state, "norm1"), dtype=dtype, device=device),
            norm2=TtGroupNormParameters.from_torch(substate(state, "norm2"), dtype=dtype, device=device),
            conv1=TtConv2dParameters.from_torch(substate(state, "conv1"), dtype=dtype),
            conv2=TtConv2dParameters.from_torch(substate(state, "conv2"), dtype=dtype),
            conv_shortcut=TtConv2dParameters.from_torch(substate(state, "conv_shortcut"), dtype=dtype)
            if has_substate(state, "conv_shortcut")
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.conv1.in_channels


class TtResnetBlock2D:
    def __init__(
        self,
        parameters: TtResnetBlock2DParameters,
        *,
        num_groups: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.norm1 = TtGroupNorm(parameters.norm1, num_groups=num_groups, eps=eps)
        self.norm2 = TtGroupNorm(parameters.norm2, num_groups=num_groups, eps=eps)
        self.conv1 = TtConv2d(parameters.conv1, padding=(1, 1))
        self.conv2 = TtConv2d(parameters.conv2, padding=(1, 1))
        self.conv_shortcut = TtConv2d(parameters.conv_shortcut) if parameters.conv_shortcut is not None else None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x

        x = self.norm1(x, inplace=False)  # TODO: change to inplace=True

        x = ttnn.silu(x)
        x = self.conv1(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # TODO: remove

        x = self.norm2(x, inplace=False)  # TODO: change to inplace=True

        x = ttnn.silu(x)
        x = self.conv2(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # TODO: remove

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
            residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT)  # TODO: remove

        return residual + x


@dataclass
class TtAttentionParameters:
    group_norm: TtGroupNormParameters
    to_q: TtLinearParameters
    to_k: TtLinearParameters
    to_v: TtLinearParameters
    to_out: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtAttentionParameters:
        return cls(
            group_norm=TtGroupNormParameters.from_torch(substate(state, "group_norm"), dtype=dtype, device=device),
            to_q=TtLinearParameters.from_torch(substate(state, "to_q"), dtype=dtype, device=device),
            to_k=TtLinearParameters.from_torch(substate(state, "to_k"), dtype=dtype, device=device),
            to_v=TtLinearParameters.from_torch(substate(state, "to_v"), dtype=dtype, device=device),
            to_out=TtLinearParameters.from_torch(substate(state, "to_out.0"), dtype=dtype, device=device),
        )


class TtAttention:
    def __init__(self, parameters: TtAttentionParameters, *, norm_num_groups: int, dim_head: int) -> None:
        super().__init__()

        self._num_heads = parameters.to_q.out_channels // dim_head

        self._group_norm = TtGroupNorm(parameters.group_norm, num_groups=norm_num_groups, eps=1e-6)
        self.to_q = TtLinear(parameters.to_q)
        self.to_k = TtLinear(parameters.to_k)
        self.to_v = TtLinear(parameters.to_v)
        self.to_out = TtLinear(parameters.to_out)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x

        # x = ttnn.permute(x, [0, 3, 1, 2])  # TODO: remove
        # x = x.reshape([batch_size, features, height * width])

        x = self._group_norm(x, inplace=False)  # TODO: inplace=True
        x = ttnn.permute(x, [0, 3, 1, 2])  # TODO: remove
        batch_size, features, height, width = list(x.shape)
        x = x.reshape([batch_size, features, height * width])

        x = ttnn.transpose(x, 1, 2)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = ttnn.concat([q, k, v], dim=-1)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self._num_heads, transpose_key=False
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=True,
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        # operands must be in DRAM
        x = ttnn.transformer.scaled_dot_product_attention(
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

        x = ttnn.transpose(x, 1, 2)
        x = x.reshape([batch_size, -1, features])  # TODO: is this the correct shape?

        x = self.to_out(x)

        x = ttnn.transpose(x, -1, -2).reshape([batch_size, features, height, width])

        x = ttnn.permute(x, [0, 2, 3, 1])  # TODO: remove

        return x + residual


@dataclass
class TtGroupNormParameters:
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtGroupNormParameters:
        return cls(
            weight=ttnn.from_torch(state["weight"], layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
            if "weight" in state
            else None,
            bias=ttnn.from_torch(state["bias"], layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
            if "bias" in state
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[1]  # TODO: correct?


class TtGroupNorm:
    def __init__(self, parameters: TtGroupNormParameters, *, num_groups: int, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._num_groups = num_groups

        # if input_memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        #     num_cores_across_channel = self.group_norm_core_grid.y
        # elif input_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        #     num_cores_across_channel = 1
        # else:
        #     num_cores_across_channel = int(self.group_norm_core_grid.x * self.group_norm_core_grid.y)
        num_cores_across_channel = 8

        device = parameters.weight.device()

        torch_weight = ttnn.create_group_norm_weight_bias_rm(
            ttnn.to_torch(parameters.weight), parameters.in_channels, num_cores_across_channel
        )
        torch_bias = ttnn.create_group_norm_weight_bias_rm(
            ttnn.to_torch(parameters.bias), parameters.in_channels, num_cores_across_channel
        )
        torch_norm_input_mask = ttnn.create_group_norm_input_mask(
            parameters.in_channels, num_groups, num_cores_across_channel
        )
        self._weight = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._bias = ttnn.from_torch(
            torch_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._norm_input_mask = ttnn.from_torch(
            torch_norm_input_mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._torch_weight = ttnn.to_torch(parameters.weight).squeeze(0)
        self._torch_bias = ttnn.to_torch(parameters.bias).squeeze(0)

    def __call__(self, x: ttnn.Tensor, *, inplace: bool = False) -> ttnn.Tensor:
        if x.shape[1] > 0:
            assert not inplace

            torch_x = ttnn.to_torch(x).permute([0, 3, 1, 2])
            torch_result = torch.nn.functional.group_norm(
                torch_x, self._num_groups, self._torch_weight, self._torch_bias, eps=self._eps
            )
            torch_result = torch_result.permute([0, 2, 3, 1])

            return ttnn.from_torch(torch_result, device=x.device(), layout=x.layout, memory_config=x.memory_config())

        [batch_size, height, width, in_channels] = list(x.shape)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        (
            memory_config,
            core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=x.device(),
            num_channels=in_channels,
            num_groups=self._num_groups,
            input_nhw=batch_size * height * width,
            is_height_sharded=False,
        )

        x = ttnn.reshape(x, [batch_size, 1, width * height, in_channels])
        x = ttnn.to_memory_config(x, memory_config)
        x = ttnn.reallocate(x)

        x = ttnn.group_norm(
            x,
            weight=self._weight,
            bias=self._bias,
            input_mask=self._norm_input_mask,
            num_groups=self._num_groups,
            epsilon=self._eps,
            core_grid=core_grid,
            memory_config=memory_config,
            inplace=inplace,
        )

        x = x.reshape([batch_size, height, width, in_channels])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # TODO: remove?
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x
