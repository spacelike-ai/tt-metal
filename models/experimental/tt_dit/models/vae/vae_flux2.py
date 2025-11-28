# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import GroupNorm
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.substate import pop_substate, rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


@dataclass
class Flux2VaeContext:
    tp_axis: int | None
    device: ttnn.MeshDevice
    ccl_manager: CCLManager | None


class Flux2VaeImageConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        tensor_parallel: bool = True,
        ctx: Flux2VaeContext,
    ) -> None:
        super().__init__()

        # Shard bigger dimension to minimize communication. If both are equal, shard rows to
        # minimize memory requirements.
        out_is_greater = out_channels > in_channels

        self.inner = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            mesh_device=ctx.device,
            in_mesh_axis=ctx.tp_axis if tensor_parallel and not out_is_greater else None,
            out_mesh_axis=ctx.tp_axis if tensor_parallel and out_is_greater else None,
            ccl_manager=ctx.ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "", "inner")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.inner.forward(x, use_persistent_buffer=False)


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L97
class Flux2VaeUpsample(Module):
    def __init__(self, *, num_channels: int, ctx: Flux2VaeContext) -> None:
        super().__init__()
        self.conv = Flux2VaeImageConv(num_channels, num_channels, kernel_size=3, padding=1, ctx=ctx)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = tensor.upsample(x, scale_factor=2)
        return self.conv.forward(x)


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L54
class Flux2VaeResnetBlock(Module):
    def __init__(self, *, in_channels: int, out_channels: int, ctx: Flux2VaeContext) -> None:
        super().__init__()

        self.norm1 = GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, mesh_axis=ctx.tp_axis, mesh_device=ctx.device
        )
        self.conv1 = Flux2VaeImageConv(in_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)
        self.norm2 = GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, mesh_axis=ctx.tp_axis, mesh_device=ctx.device
        )
        self.conv2 = Flux2VaeImageConv(out_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)

        self.conv_shortcut = (
            RowParallelLinear(
                in_channels,
                out_channels,
                mesh_axis=ctx.tp_axis,
                mesh_device=ctx.device,
                ccl_manager=ctx.ccl_manager,
            )
            if in_channels != out_channels
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = state["conv_shortcut.weight"].squeeze(2, 3)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = x

        h = self.norm1.forward(h)
        h = ttnn.silu(h)
        h = self.conv1.forward(h)

        h = self.norm2.forward(h)
        h = ttnn.silu(h)
        h = self.conv2.forward(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut.forward(x)

        return x + h


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L24
class Flux2VaeAttnBlock(Module):
    def __init__(self, *, num_channels: int, ctx: Flux2VaeContext) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        linear_args = dict(mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager)

        self.group_norm = GroupNorm(
            num_groups=32, num_channels=num_channels, eps=1e-6, mesh_axis=ctx.tp_axis, mesh_device=ctx.device
        )
        self.to_q = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_k = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_v = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_out = ColParallelLinear(num_channels, num_channels, **linear_args)

        grid_size = ctx.device.compute_with_storage_grid_size()
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        identity = x

        x = self.group_norm.forward(x)

        if self._tp_axis is not None:
            assert self._ccl_manager is not None
            q = self.to_q.forward(x)
            q = self._ccl_manager.all_gather(q, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            k = self.to_k.forward(x)
            k = self._ccl_manager.all_gather(k, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            v = self.to_v.forward(x)
            v = self._ccl_manager.all_gather(v, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        else:
            q = self.to_q.forward(x)
            k = self.to_k.forward(x)
            v = self.to_v.forward(x)
        del x

        n, h, w, c = q.shape

        # convert to 1d sequence and insert head dimension; there is only one head
        q = q.reshape([n, 1, h * w, c])
        k = k.reshape([n, 1, h * w, c])
        v = v.reshape([n, 1, h * w, c])

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = self.to_out.forward(x)

        # convert back to 2d
        x = x.reshape([n, h, w, -1])

        return x + identity


class Flux2VaeMidBlock(Module):
    def __init__(
        self,
        *,
        num_channels: int,
        ctx: Flux2VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            Flux2VaeResnetBlock(in_channels=num_channels, out_channels=num_channels, ctx=ctx) for _ in range(2)
        )

        self.attentions = ModuleList([Flux2VaeAttnBlock(num_channels=num_channels, ctx=ctx)])

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = self.resnets[0].forward(z)
        z = self.attentions[0].forward(z)
        z = self.resnets[1].forward(z)

        return z


class Flux2VaeUpBlock(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        upsample: bool,
        ctx: Flux2VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            Flux2VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                ctx=ctx,
            )
            for i in range(num_res_blocks + 1)
        )

        self.upsampler = Flux2VaeUpsample(num_channels=out_channels, ctx=ctx) if upsample else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsampler")

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.resnets:
            z = block.forward(z)

        if self.upsampler is not None:
            z = self.upsampler.forward(z)

        return z


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L184
class Flux2VaeDecoder(Module):
    def __init__(
        self,
        *,
        out_channels: int,
        block_out_channels: Sequence[int],
        num_res_blocks: int,
        z_channels: int,
        ctx: Flux2VaeContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]

        self.post_quant_conv = Linear(z_channels, z_channels, mesh_device=ctx.device)
        self.conv_in = Flux2VaeImageConv(z_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)

        self.mid_block = Flux2VaeMidBlock(num_channels=channel_counts[0], ctx=ctx)

        self.up_blocks = ModuleList(
            Flux2VaeUpBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                upsample=i != len(channel_counts) - 2,
                num_res_blocks=num_res_blocks,
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )

        self.conv_norm_out = GroupNorm(
            num_groups=32, num_channels=channel_counts[-1], eps=1e-6, mesh_axis=ctx.tp_axis, mesh_device=ctx.device
        )
        self.conv_out = Flux2VaeImageConv(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = self.post_quant_conv.forward(z)
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)
        z = ttnn.silu(z)

        if self._tp_axis is not None:
            assert self._ccl_manager is not None
            z = self._ccl_manager.all_gather(z, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        z = self.conv_out.forward(z)

        return z


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L271
class Flux2Vae(Module):
    def __init__(
        self,
        *,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
        z_channels: int = 32,
        parallel_config: VAEParallelConfig | None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = Flux2VaeContext(
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            device=device,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        self.decoder = Flux2VaeDecoder(
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            ctx=ctx,
        )

        self.patch_size = 2

        bn_size = self.patch_size**2 * z_channels
        self.bn_running_mean = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_running_var = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_eps = 1e-4

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # remove encoder state
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

        rename_substate(state, "post_quant_conv", "decoder.post_quant_conv")

        if "bn.running_mean" in state:
            state["bn_running_mean"] = state.pop("bn.running_mean")
        if "bn.running_var" in state:
            state["bn_running_var"] = state.pop("bn.running_var")
        state.pop("bn.num_batches_tracked", None)

    def _inv_normalize(self, z: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.sqrt(self.bn_running_var.data + self.bn_eps)
        m = self.bn_running_mean.data
        return z * s + m

    def decode(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        n, h, w, _ = z.shape
        p = self.patch_size

        z = self._inv_normalize(z)

        # N H W (C P P) -> N (H P) (W P) C
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = z.reshape([n, h, w, -1, p, p])
        z = ttnn.permute(z, [0, 1, 4, 2, 5, 3])
        z = z.reshape([n, h * p, w * p, -1])
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)

        return self.decoder.forward(z)

    def forward(self) -> None:
        msg = "call decode() instead of forward()"
        raise RuntimeError(msg)
