# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.3 transformer (diffusers ``Flux3Transformer2DModel``).

Three token modes share one joint sequence: the noisy image tokens, the optional reference image
tokens and the text. Each mode first runs through its own per-mode blocks, in which it attends only
to itself, and then all of them run through the joint blocks, which share one set of weights but
modulate each mode by its own timestep embedding. Only the image tokens reach the output layers.

Every block is a parallel attention + SwiGLU block, so all of them are ``Flux3Block``: the per-mode
blocks run it on one sequence, the joint blocks on the prompt and the spatial sequence (image, then
reference) as the two streams of the attention.

Layout: like FLUX.2, activations are sharded on the sequence over the SP axis and on the hidden
dimension over the TP axis. The prompt is sharded over SP too and padded to whole tiles per device.
The attention masks the padding through its logical length, so the prompt can have any length; the
image and reference sequences must be whole tiles per device.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import diffusers.models.transformers.transformer_flux2
import huggingface_hub
import safetensors.torch
import torch

import ttnn

from ...blocks.attention_opt import Attention
from ...layers.embeddings import TimestepEmbedding
from ...layers.linear import ColParallelLinear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils import cache, tensor
from ...utils.substate import rename_substate
from ...utils.tensor import bf16_tensor, prepare_weight_for_concatenated_input

if TYPE_CHECKING:
    from ...parallel.config import DiTGParallelConfigNoCFG, DiTParallelConfig
    from ...parallel.manager import CCLManager

# The model was trained with the image token positions scaled by this factor.
IMAGE_ROPE_SCALE = 0.5

# Every time-embedding projection consumes one of two activated embeddings: the image tokens' (at
# the flow timestep) or the one at timestep 0 that the text and the reference tokens share. Each
# group is merged into one chunked matmul. The tables list the checkpoint prefix of each source, how
# many chunks it contributes and which of them are `scale` (those get the consumers' `+1` folded into
# the bias). Row order is the chunk order of the merged weight, which is also the order `forward`
# unpacks.
_IMAGE_MOD_SOURCES = (
    ("early_stream_modulations.image.linear", 3, (1,)),  # (shift, scale, gate)
    ("single_stream_modulations.image.linear", 3, (1,)),
    ("norm_out.linear", 2, (0,)),  # (scale, shift)
)
_ZERO_MOD_SOURCES = (
    ("early_stream_modulations.txt.linear", 3, (1,)),
    ("early_stream_modulations.image_cond.linear", 3, (1,)),
    ("single_stream_modulations.txt.linear", 3, (1,)),
    ("single_stream_modulations.image_cond.linear", 3, (1,)),
)


def _chunk_count(sources: Sequence[tuple[str, int, tuple[int, ...]]]) -> int:
    return sum(count for _, count, _ in sources)


class Flux3Block(Module):
    """A parallel attention + SwiGLU feedforward block, sharing one modulated input.

    The attention's output projection and the feedforward's second linear are merged into
    ``proj_out``, whose input is the attention output and the feedforward hidden state side by side.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        eps: float,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig | DiTGParallelConfigNoCFG,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = int(dim * mlp_ratio)
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=eps,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            use_spatial_weights_for_prompt=True,
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=True,
        )

        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )

        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        to_out = state.pop("attn.to_out.0.weight", None)
        linear_out = state.pop("ff.linear_out.weight", None)
        if to_out is not None and linear_out is not None:
            state["proj_out.weight"] = prepare_weight_for_concatenated_input(
                torch.cat([to_out, linear_out], dim=1),
                [self._dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

        linear_in = state.pop("ff.linear_in.weight", None)
        if linear_in is not None:
            # The fused SwiGLU takes the halves of the packed weight in the other order.
            state["proj_mlp.weight"] = torch.roll(linear_in, shifts=linear_in.shape[0] // 2, dims=0)

    def forward(
        self,
        x: ttnn.Tensor,
        modulation: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        sequence_length: int,
        prompt: ttnn.Tensor | None = None,
        prompt_modulation: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_sequence_length: int = 0,
        compute_prompt_output: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Run the block on ``x``, and on ``prompt`` as the attention's second stream if given.

        Args:
            x: Tensor of shape [1, padded sequence length / sp_factor, dim / tp_factor].
            modulation: ``(shift, 1 + scale, gate)``, each of shape [1, 1, dim / tp_factor], or
                [1, padded sequence length / sp_factor, dim / tp_factor] for one row per token.
            rope: cos and sin of shape [1, 1, padded sequence length / sp_factor, head_dim].
            sequence_length: The length of ``x`` without padding, over all devices. The padding
                must be at the end of the sequence.
            prompt, prompt_modulation, prompt_rope, prompt_sequence_length: The same for the
                prompt.
            compute_prompt_output: Whether to return the prompt's output. Without, the prompt only
                serves as context to ``x``.
        """
        x_normed = self._norm(x, modulation)
        if prompt is None:
            x_attn, _ = self.attn.forward(sequence_1=x_normed, sequence_1_length=sequence_length, sequence_1_rope=rope)
            return self._project(x_normed, x_attn, residual=x, gate=modulation[2]), None

        assert prompt_modulation is not None
        assert prompt_rope is not None
        prompt_normed = self._norm(prompt, prompt_modulation)

        # The prompt goes in the attention's first stream, the one passed around the ring, and x in
        # the joint stream. The prompt length need not be a whole number of tiles, and ring joint SDPA
        # garbles some outputs when the sharded joint stream ends in a partial tile (seen at 80 of 128
        # tokens on four devices); it masks a partial tile of the ring stream correctly. Both orders
        # take the same time.
        prompt_attn, x_attn = self.attn.forward(
            sequence_1=prompt_normed,
            sequence_1_length=prompt_sequence_length,
            sequence_1_rope=prompt_rope,
            sequence_2=x_normed,
            sequence_2_length=sequence_length,
            sequence_2_rope=rope,
        )
        assert x_attn is not None

        x = self._project(x_normed, x_attn, residual=x, gate=modulation[2])
        if not compute_prompt_output:
            return x, None
        prompt = self._project(prompt_normed, prompt_attn, residual=prompt, gate=prompt_modulation[2])
        return x, prompt

    def _norm(self, x: ttnn.Tensor, modulation: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        shift, scale, _ = modulation
        return ttnn.squeeze(self.norm(ttnn.unsqueeze(x, 0), dynamic_weight=scale, dynamic_bias=shift), 0)

    def _project(
        self, x_normed: ttnn.Tensor, x_attn: ttnn.Tensor, *, residual: ttnn.Tensor, gate: ttnn.Tensor
    ) -> ttnn.Tensor:
        x_mlp = self.proj_mlp(
            x_normed,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_normed.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )

        # The fused path broadcasts one gate row over the sequence.
        if self._ccl_manager.topology == ttnn.Topology.Ring and gate.shape[-2] == 1:
            return self.proj_out.forward_fused_addcmul([x_attn, x_mlp], residual, gate, scalar=1.0)
        return ttnn.addcmul(residual, self.proj_out([x_attn, x_mlp]), gate)


class Flux3Modulation(NamedTuple):
    """The per-step modulations, each ``(shift, 1 + scale, gate)`` of shape [1, 1, dim / tp]."""

    image_early: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    image_single: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    reference_early: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    reference_single: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    prompt_early: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    prompt_single: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    # (1 + scale, shift) of the output layer norm.
    norm_out: tuple[ttnn.Tensor, ttnn.Tensor]


class Flux3Transformer(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        cond_in_channels: int,
        num_layers: int,
        num_single_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        timestep_channels: int,
        mlp_ratio: float,
        eps: float,
        flow_map: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig | DiTGParallelConfigNoCFG,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        if num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            msg = (
                f"the {num_attention_heads} attention heads must divide evenly over the tensor parallel factor "
                f"{parallel_config.tensor_parallel.factor}"
            )
            raise ValueError(msg)
        # Without sequence parallelism, the attention can mask neither the padding of the prompt
        # in the per-mode blocks nor take a sharded prompt in the joint blocks.
        if parallel_config.sequence_parallel.factor == 1:
            msg = "the FLUX.3 transformer needs sequence parallelism"
            raise ValueError(msg)

        self.timestep_embedder = TimestepEmbedding(timestep_channels, inner_dim, bias=False, mesh_device=device)
        self.target_timestep_embedder = (
            TimestepEmbedding(timestep_channels, inner_dim, bias=False, mesh_device=device) if flow_map else None
        )

        self.image_time_mod = ColParallelLinear(
            inner_dim,
            _chunk_count(_IMAGE_MOD_SOURCES) * inner_dim,
            bias=True,
            mesh_device=device,
            mesh_axis=tp_axis,
            chunks=_chunk_count(_IMAGE_MOD_SOURCES),
        )
        self.zero_time_mod = ColParallelLinear(
            inner_dim,
            _chunk_count(_ZERO_MOD_SOURCES) * inner_dim,
            bias=True,
            mesh_device=device,
            mesh_axis=tp_axis,
            chunks=_chunk_count(_ZERO_MOD_SOURCES),
        )

        self.x_embedder = ColParallelLinear(in_channels, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis)
        self.cond_embedder = ColParallelLinear(
            cond_in_channels, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis
        )
        self.context_embedder = ColParallelLinear(
            joint_attention_dim, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis
        )

        def blocks(count: int) -> ModuleList:
            return ModuleList(
                Flux3Block(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    device=device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                )
                for _ in range(count)
            )

        self.image_blocks = blocks(num_layers)
        self.reference_blocks = blocks(num_layers)
        self.prompt_blocks = blocks(num_layers)
        self.joint_blocks = blocks(num_single_layers)

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.proj_out = RowParallelLinear(
            inner_dim,
            in_channels,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        # The sinusoidal timestep projection (diffusers `Timesteps` with `flip_sin_to_cos=True`,
        # `downscale_freq_shift=0`) and its value at timestep 0.
        half = timestep_channels // 2
        factor = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self._time_proj_factor = ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=device))
        zero_proj = torch.cat([torch.ones(half), torch.zeros(half)])
        self._zero_time_proj = ttnn.unsqueeze_to_4D(bf16_tensor(zero_proj.unsqueeze(0), device=device))

        self.device = device
        self.flow_map = flow_map
        self._inner_dim = inner_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._sp_factor = parallel_config.sequence_parallel.factor

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name, sources in (("image_time_mod", _IMAGE_MOD_SOURCES), ("zero_time_mod", _ZERO_MOD_SOURCES)):
            _merge_time_mod(state, name, sources, inner_dim=self._inner_dim, tp_factor=self._tp_factor)

        rename_substate(state, "x_embedders.image", "x_embedder")
        rename_substate(state, "x_embedders.image_cond", "cond_embedder")
        rename_substate(state, "mode_blocks.image", "image_blocks")
        rename_substate(state, "mode_blocks.image_cond", "reference_blocks")
        rename_substate(state, "txt_mode_blocks", "prompt_blocks")
        rename_substate(state, "single_transformer_blocks", "joint_blocks")

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        target_timestep: ttnn.Tensor | None,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        reference: ttnn.Tensor | None = None,
        reference_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        joint_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        reference_sequence_length: int = 0,
    ) -> ttnn.Tensor:
        """Predict the velocity of the image tokens.

        Args:
            spatial: The image tokens, of shape [1, spatial_sequence_length / sp_factor, in_channels].
            prompt: ``context_embedder`` applied to the prompt, of shape
                [1, padded prompt length / sp_factor, inner_dim / tp_factor]. The caller applies it
                because the prompt is the same in every denoising step.
            timestep: The flow timestep times 1000, of shape [1, 1], float32.
            target_timestep: For a flow map, the timestep the step jumps to, likewise.
            spatial_rope, prompt_rope: cos and sin of shape [1, 1, length / sp_factor, head_dim],
                sharded like the tokens.
            spatial_sequence_length: The number of image tokens, a multiple of the tile size times
                sp_factor.
            prompt_sequence_length: The prompt length without padding.
            reference: The reference image tokens, like ``spatial``.
            reference_rope: Their RoPE, like ``spatial_rope``.
            joint_rope: With a reference, the RoPE of each device's image tokens followed by its
                reference tokens.
            reference_sequence_length: The number of reference tokens, a multiple of the tile size
                times sp_factor.

        Returns:
            The velocity, of shape [1, spatial_sequence_length / sp_factor, in_channels].
        """
        if (target_timestep is not None) != self.flow_map:
            msg = "target_timestep is required by, and only by, a flow-map transformer"
            raise ValueError(msg)
        has_reference = reference is not None
        if has_reference and (reference_rope is None or joint_rope is None or reference_sequence_length == 0):
            msg = "a reference needs reference_rope, joint_rope and reference_sequence_length"
            raise ValueError(msg)

        mod = self.modulation(timestep, target_timestep)

        spatial = self.x_embedder(spatial)
        local_image_length = spatial.shape[1]

        for block in self.prompt_blocks:
            prompt, _ = block.forward(
                prompt, mod.prompt_early, rope=prompt_rope, sequence_length=prompt_sequence_length
            )

        for block in self.image_blocks:
            spatial, _ = block.forward(
                spatial, mod.image_early, rope=spatial_rope, sequence_length=spatial_sequence_length
            )

        if reference is not None:
            assert reference_rope is not None
            assert joint_rope is not None
            ref = self.cond_embedder(reference)
            for block in self.reference_blocks:
                ref, _ = block.forward(
                    ref, mod.reference_early, rope=reference_rope, sequence_length=reference_sequence_length
                )
            spatial = ttnn.concat([spatial, ref], dim=1)
            local_reference_length = ref.shape[1]
            # The image rows and the reference rows are modulated differently.
            spatial_mod = tuple(
                ttnn.concat(
                    [
                        ttnn.repeat(i, [1, local_image_length, 1]),
                        ttnn.repeat(r, [1, local_reference_length, 1]),
                    ],
                    dim=1,
                )
                for i, r in zip(mod.image_single, mod.reference_single, strict=True)
            )
        else:
            joint_rope = spatial_rope
            spatial_mod = mod.image_single

        joint_length = spatial_sequence_length + reference_sequence_length
        for i, block in enumerate(self.joint_blocks):
            spatial, next_prompt = block.forward(
                spatial,
                spatial_mod,
                rope=joint_rope,
                sequence_length=joint_length,
                prompt=prompt,
                prompt_modulation=mod.prompt_single,
                prompt_rope=prompt_rope,
                prompt_sequence_length=prompt_sequence_length,
                # The last block's prompt output is unused.
                compute_prompt_output=i < len(self.joint_blocks) - 1,
            )
            if next_prompt is not None:
                prompt = next_prompt

        if has_reference:
            spatial = spatial[:, :local_image_length, :]

        scale, shift = mod.norm_out
        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0), dynamic_weight=scale, dynamic_bias=shift), 0)
        return self.proj_out(spatial, gather_output=True)

    def modulation(self, timestep: ttnn.Tensor, target_timestep: ttnn.Tensor | None) -> Flux3Modulation:
        """The modulations of every block for the flow timestep and, for a flow map, the target one."""
        image_embed = self.timestep_embedder(self._time_proj(timestep))
        zero_embed = self.timestep_embedder(self._zero_time_proj)
        if self.target_timestep_embedder is not None:
            assert target_timestep is not None
            # The target timestep's embedding is added to every mode's.
            target_embed = self.target_timestep_embedder(self._time_proj(target_timestep))
            image_embed = image_embed + target_embed
            zero_embed = zero_embed + target_embed

        image_embed = _to_batch_row(ttnn.silu(image_embed))
        zero_embed = _to_batch_row(ttnn.silu(zero_embed))

        # The `+1` of each scale is folded into the bias already.
        image_mod = self.image_time_mod(image_embed, dtype=ttnn.bfloat16)
        zero_mod = self.zero_time_mod(zero_embed, dtype=ttnn.bfloat16)

        return Flux3Modulation(
            image_early=tuple(image_mod[0:3]),
            image_single=tuple(image_mod[3:6]),
            norm_out=(image_mod[6], image_mod[7]),
            prompt_early=tuple(zero_mod[0:3]),
            reference_early=tuple(zero_mod[3:6]),
            prompt_single=tuple(zero_mod[6:9]),
            reference_single=tuple(zero_mod[9:12]),
        )

    def _time_proj(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32, "timesteps require float32 precision"
        emb = timestep * self._time_proj_factor
        return ttnn.concat([ttnn.cos(emb), ttnn.sin(emb)], dim=-1)


def _to_batch_row(x: ttnn.Tensor) -> ttnn.Tensor:
    """[..., batch, dim] to [batch, 1, dim]."""
    return x.reshape([x.shape[-2], 1, x.shape[-1]])


def _merge_time_mod(
    state: dict[str, torch.Tensor],
    name: str,
    sources: Sequence[tuple[str, int, tuple[int, ...]]],
    *,
    inner_dim: int,
    tp_factor: int,
) -> None:
    """Merge the projections of ``sources`` into the chunked linear ``name``.

    Leaves the weight absent, to be reported as missing, rather than build a partial one if any
    source is missing.
    """
    weights = [state.pop(f"{prefix}.weight", None) for prefix, _, _ in sources]
    assert not any(
        state.pop(f"{prefix}.bias", None) is not None for prefix, _, _ in sources
    ), "the merged time-embedding projection assumes bias-free sources"
    present = [w for w in weights if w is not None]
    if len(present) != len(weights):
        return

    state[f"{name}.weight"] = torch.cat(present, dim=0)
    # The bias carries the consumers' `+1` on each scale chunk.
    state[f"{name}.bias"] = torch.cat(
        [
            torch.full([inner_dim], 1.0 if i in scale_indices else 0.0)
            for _, count, scale_indices in sources
            for i in range(count)
        ]
    )
    prepare_chunked_linear_output(state, prefix=name, device_count=tp_factor, chunks=_chunk_count(sources))


class Flux3Checkpoint:
    """A FLUX.3 checkpoint: reads the transformer's configuration and builds loaded transformers.

    Reads only ``config.json`` in ``__init__``; the weights are loaded on a cache miss only. The
    weights are read from the checkpoint's safetensors directly, so that diffusers need not know
    FLUX.3.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._directory = _checkpoint_directory(name, subfolder="transformer", weights=False)
        with (self._directory / "config.json").open() as f:
            self.config: dict = json.load(f)

        self._pos_embed = diffusers.models.transformers.transformer_flux2.Flux2PosEmbed(
            theta=self.config["rope_theta"], axes_dim=list(self.config["axes_dims_rope"])
        )

    @property
    def flow_map(self) -> bool:
        return bool(self.config.get("flow_map", False))

    @property
    def in_channels(self) -> int:
        return self.config["in_channels"]

    @property
    def joint_attention_dim(self) -> int:
        return self.config["joint_attention_dim"]

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig | DiTGParallelConfigNoCFG,
        is_fsdp: bool = False,
        num_layers: int | None = None,
        num_single_layers: int | None = None,
        get_torch_state_dict: Callable[[], dict[str, torch.Tensor]] | None = None,
    ) -> Flux3Transformer:
        """Construct a ``Flux3Transformer`` for this checkpoint and load its weights.

        ``num_layers`` and ``num_single_layers`` build only the first blocks, for tests, which can
        also pass the torch weights in ``get_torch_state_dict``.
        """
        device = ccl_manager.mesh_device
        c = self.config
        num_layers = c["num_layers"] if num_layers is None else num_layers
        num_single_layers = c["num_single_layers"] if num_single_layers is None else num_single_layers

        model = Flux3Transformer(
            in_channels=c["in_channels"],
            cond_in_channels=c["cond_in_channels"],
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=c["attention_head_dim"],
            num_attention_heads=c["num_attention_heads"],
            joint_attention_dim=c["joint_attention_dim"],
            timestep_channels=c["timestep_channels"],
            mlp_ratio=c["mlp_ratio"],
            eps=c["eps"],
            flow_map=self.flow_map,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )

        full = num_layers == c["num_layers"] and num_single_layers == c["num_single_layers"]

        def load_state_dict() -> dict[str, torch.Tensor]:
            return self.state_dict(num_layers=num_layers, num_single_layers=num_single_layers)

        cache.load_model(
            tt_model=model,
            get_torch_state_dict=get_torch_state_dict or load_state_dict,
            model_name=self._name,
            subfolder="transformer" if full else f"transformer_L{num_layers}_S{num_single_layers}",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
            is_fsdp=is_fsdp,
        )
        return model

    def state_dict(self, *, num_layers: int | None = None, num_single_layers: int | None = None) -> dict:
        """The torch weights, without the blocks past ``num_layers`` and ``num_single_layers``."""
        directory = _checkpoint_directory(self._name, subfolder="transformer", weights=True)
        state = {}
        for path in sorted(directory.glob("*.safetensors")):
            state.update(safetensors.torch.load_file(path))

        def keep(key: str) -> bool:
            for prefix, limit in (
                ("mode_blocks.image.", num_layers),
                ("mode_blocks.image_cond.", num_layers),
                ("txt_mode_blocks.", num_layers),
                ("single_transformer_blocks.", num_single_layers),
            ):
                if limit is not None and key.startswith(prefix):
                    return int(key.removeprefix(prefix).split(".")[0]) < limit
            return True

        return {k: v for k, v in state.items() if keep(k)}

    def rope(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin of shape [len(ids), head_dim] for position ids of shape [len(ids), 4]."""
        cos, sin = self._pos_embed.forward(ids)
        return cos, sin

    @staticmethod
    def image_ids(*, height: int, width: int, index: int = 0) -> torch.Tensor:
        """The position ids of an image of ``height`` x ``width`` tokens.

        The output is image 0; reference image ``k`` (from 0) is image ``-(k + 1)`` on the last axis.
        """
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.tensor([index])
        ids = torch.cartesian_prod(t, h, w, l).to(torch.float32)
        ids[:, 1:3] *= IMAGE_ROPE_SCALE
        return ids

    @staticmethod
    def text_ids(length: int) -> torch.Tensor:
        """The position ids of ``length`` text tokens, which advance on the last axis."""
        zero = torch.arange(1)
        return torch.cartesian_prod(zero, zero, zero, torch.arange(length)).to(torch.float32)

    def rope_tensor(
        self,
        ids: torch.Tensor,
        *,
        device: ttnn.MeshDevice,
        sp_axis: int,
        padded_length: int | None = None,
        on_host: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The RoPE of ``ids`` pre-shaped as [1, 1, length, head_dim] and sharded on the sequence.

        The rows past the ids up to ``padded_length`` are zero.
        """
        cos, sin = self.rope(ids)
        padding = (padded_length or len(ids)) - len(ids)

        def upload(x: torch.Tensor) -> ttnn.Tensor:
            return tensor.from_torch(
                torch.nn.functional.pad(x, (0, 0, 0, padding))[None, None],
                device=device,
                mesh_axes=[None, None, sp_axis, None],
                on_host=on_host,
            )

        return upload(cos), upload(sin)


def interleave_shards(a: torch.Tensor, b: torch.Tensor, num_shards: int) -> torch.Tensor:
    """Concatenate ``a`` and ``b`` on the first dim so that each of ``num_shards`` equal shards of the
    result is the matching shard of ``a`` followed by that of ``b``."""
    shards = zip(a.chunk(num_shards), b.chunk(num_shards), strict=True)
    return torch.cat([t for pair in shards for t in pair])


def padded_prompt_length(length: int, *, sp_factor: int) -> int:
    """The prompt length padded to whole tiles on each of ``sp_factor`` devices."""
    multiple = ttnn.TILE_SIZE * sp_factor
    return -(-length // multiple) * multiple


def _checkpoint_directory(name: str, *, subfolder: str, weights: bool) -> Path:
    path = Path(name)
    if path.is_dir():
        return path / subfolder
    patterns = [f"{subfolder}/*"] if weights else [f"{subfolder}/config.json"]
    return Path(huggingface_hub.snapshot_download(name, allow_patterns=patterns)) / subfolder
