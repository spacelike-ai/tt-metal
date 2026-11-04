# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.3 [klein] pipeline: text-to-image and editing with the step-distilled checkpoint.

As diffusers ``Flux3DistilledModularPipeline``: Qwen3-VL text embeddings, a flow-map transformer
sampled without guidance on a constant-shift schedule, and the FLUX.2 VAE.

Unlike diffusers, the output and each reference image have one of the configured resolutions: a
reference is center-cropped to the output's aspect ratio and resized to the output size.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import tqdm
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.models.transformers.transformer_flux3 import (
    Flux3Checkpoint,
    Flux3Transformer,
    interleave_shards,
    padded_prompt_length,
)
from models.tt_dit.models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from models.tt_dit.parallel.config import (
    DiTGParallelConfigNoCFG,
    EncoderParallelConfig,
    Flux2VaeParallelConfig,
    ParallelFactor,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, event_section, null_callback
from models.tt_dit.pipelines.flux3.text_encoder import Flux3TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.utils import cache
from models.tt_dit.utils.tensor import fast_device_to_host, float_to_uint8, from_torch, to_torch
from models.tt_dit.utils.tracing import StateTensor, Tracer

_T = TypeVar("_T")

_DEFAULT_CHECKPOINT = "black-forest-labs/flux-3-image-klein"
_VAE_SCALE_FACTOR = 8
_PATCH_SIZE = 2
_LATENT_CHANNELS = 32
# The constant timestep shift of the FLUX.3 sampling recipe.
_SHIFT = 6.93

# The image sizes (width, height) supported by default.
_DEFAULT_RESOLUTIONS = ((1024, 1024), (2048, 2048))

# The parallel layout per mesh shape: the DiT's SP and TP axes, the text encoder's TP axis, the VAE's
# channel (TP) and row (H) axes, and the CCL links and topology. The transformer needs SP.
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp_axis": 1,
        "tp_axis": 0,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 0,
        "vae_h_axis": 1,
        "num_links": 1,
        "topology": ttnn.Topology.Linear,
    },
}
_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 1,
        "vae_h_axis": 0,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
    (2, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 1,
        "vae_h_axis": 0,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
    (4, 8): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 0,
        "vae_h_axis": 1,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
}


@dataclass(frozen=True, kw_only=True)
class Flux3PipelineConfig:
    checkpoint_name: str
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTGParallelConfigNoCFG
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: Flux2VaeParallelConfig

    # The supported output sizes, (width, height).
    resolutions: tuple[tuple[int, int], ...]
    # The longest prompt in tokens, a multiple of 80; longer prompts are truncated.
    max_prompt_length: int
    # The most reference images of an edit; 0 disables editing and the VAE encoder.
    max_references: int
    is_fsdp: bool
    use_torch_text_encoder: bool

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: Sequence[int],
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        encoder_tp_axis: int | None = None,
        vae_tp_axis: int | None = None,
        vae_h_axis: int | None = None,
        vae_w_axis: int | None = None,
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        resolutions: Sequence[tuple[int, int]] = _DEFAULT_RESOLUTIONS,
        max_prompt_length: int = 640,
        max_references: int = 1,
        is_fsdp: bool = False,
        use_torch_text_encoder: bool = False,
    ) -> Flux3PipelineConfig:
        """The preset for ``mesh_shape``; the layout arguments left as ``None`` come from it.

        The VAE axes come from the preset only when none of them is given.
        """
        mesh_shape = tuple(mesh_shape)
        preset = (_PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH).get(mesh_shape)

        def from_preset(name: str, value: _T | None) -> _T:
            if value is not None:
                return value
            if preset is None:
                msg = f"{name} is required: there is no FLUX.3 preset for mesh shape {mesh_shape}"
                raise ValueError(msg)
            return preset[name]

        sp_axis = from_preset("sp_axis", sp_axis)
        tp_axis = from_preset("tp_axis", tp_axis)
        encoder_tp_axis = from_preset("encoder_tp_axis", encoder_tp_axis)
        if vae_tp_axis is None and vae_h_axis is None and vae_w_axis is None:
            vae_tp_axis = from_preset("vae_tp_axis", vae_tp_axis)
            vae_h_axis = from_preset("vae_h_axis", vae_h_axis)

        def factor(axis: int) -> ParallelFactor:
            return ParallelFactor(factor=int(mesh_shape[axis]), mesh_axis=axis)

        return cls(
            checkpoint_name=checkpoint_name,
            topology=from_preset("topology", topology),
            num_links=from_preset("num_links", num_links),
            dit_parallel_config=DiTGParallelConfigNoCFG(
                tensor_parallel=factor(tp_axis), sequence_parallel=factor(sp_axis)
            ),
            encoder_parallel_config=EncoderParallelConfig(tensor_parallel=factor(encoder_tp_axis)),
            vae_parallel_config=Flux2VaeParallelConfig.from_axes(
                ttnn.MeshShape(list(mesh_shape)), tp_axis=vae_tp_axis, h_axis=vae_h_axis, w_axis=vae_w_axis
            ),
            resolutions=tuple(resolutions),
            max_prompt_length=max_prompt_length,
            max_references=max_references,
            is_fsdp=is_fsdp,
            use_torch_text_encoder=use_torch_text_encoder,
        )


class _DenoiseState:
    """The device inputs of the denoising step for one output size, reference count and prompt
    length, and its trace. A trace bakes its inputs' addresses, so they are updated in place."""

    def __init__(self, *, step_tracer: Tracer) -> None:
        self.step_tracer = step_tracer
        self.latents = StateTensor()
        self.reference = StateTensor()
        self.prompt = StateTensor()
        self.timestep = StateTensor()
        self.target_timestep = StateTensor()
        self.sigma_difference = StateTensor()
        self.spatial_rope = (StateTensor(), StateTensor())
        self.prompt_rope = (StateTensor(), StateTensor())
        self.reference_rope = (StateTensor(), StateTensor())
        self.joint_rope = (StateTensor(), StateTensor())


class Flux3Pipeline(PipelineAPIMixin):
    @classmethod
    def create_pipeline(
        cls, *, mesh_device: ttnn.MeshDevice, checkpoint_name: str = _DEFAULT_CHECKPOINT, **kwargs: object
    ) -> Flux3Pipeline:
        """Build a pipeline with the preset of ``mesh_device``'s shape; ``kwargs`` go to
        ``Flux3PipelineConfig.default``."""
        config = Flux3PipelineConfig.default(
            mesh_shape=tuple(mesh_device.shape),
            checkpoint_name=checkpoint_name,
            **kwargs,  # type: ignore[arg-type]
        )
        return cls(mesh_device=mesh_device, config=config)

    def __init__(self, *, mesh_device: ttnn.MeshDevice, config: Flux3PipelineConfig, trace_warmup: bool = True) -> None:
        """Build the pipeline and run it once untraced at every size, reference count and prompt
        length, and with ``trace_warmup`` once traced too, which captures every trace."""
        self._device = mesh_device
        self._config = config
        self._parallel_config = config.dit_parallel_config
        self._vae_parallel = config.vae_parallel_config
        self._resolutions = config.resolutions
        sp_factor = config.dit_parallel_config.sequence_parallel.factor

        for width, height in self._resolutions:
            tokens = (width // 16) * (height // 16)
            if width % 16 != 0 or height % 16 != 0 or tokens % (ttnn.TILE_SIZE * sp_factor) != 0:
                msg = (
                    f"resolution {width}x{height} must be a multiple of 16 px with a token count divisible by "
                    f"{ttnn.TILE_SIZE * sp_factor}"
                )
                raise ValueError(msg)

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Encoder parallel config: {config.encoder_parallel_config}")
        logger.info(f"VAE parallel: {config.vae_parallel_config}")

        self._ccl_manager = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)
        self._encoder_ccl_manager = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)
        self._vae_ccl_manager = CCLManager(mesh_device, num_links=config.num_links, topology=ttnn.Topology.Linear)

        self._checkpoint = Flux3Checkpoint(config.checkpoint_name)
        if not self._checkpoint.flow_map:
            msg = "only the step-distilled FLUX.3 checkpoints are supported; the base model needs CFG"
            raise NotImplementedError(msg)

        # The denoising states by (width, height, reference count, prompt length), and the VAE
        # traces by image size.
        self._denoise_states: dict[tuple[int, int, int, int], _DenoiseState] = {}
        self._vae_decoder_tracers: dict[tuple[int, int], Tracer] = {}
        self._vae_encoder_tracers: dict[tuple[int, int], Tracer] = {}

        logger.info("creating the text encoder...")
        self._text_encoder = Flux3TextEncoder(
            checkpoint_name=config.checkpoint_name,
            device=mesh_device,
            ccl_manager=self._encoder_ccl_manager,
            parallel_config=config.encoder_parallel_config,
            max_length=config.max_prompt_length,
            use_torch=config.use_torch_text_encoder,
        )
        self._text_encoder.load_weights()

        logger.info("creating the transformer...")
        self.transformer: Flux3Transformer = self._checkpoint.build(
            ccl_manager=self._ccl_manager, parallel_config=self._parallel_config, is_fsdp=config.is_fsdp
        )

        logger.info("creating the VAE...")
        torch_vae = AutoencoderKLFlux2.from_pretrained(config.checkpoint_name, subfolder="vae")
        assert isinstance(torch_vae, AutoencoderKLFlux2)
        vae_config = torch_vae.config
        model_name = os.path.basename(config.checkpoint_name)
        self._vae_decoder = Flux2VaeDecoder(
            out_channels=vae_config["out_channels"],
            block_out_channels=list(vae_config["block_out_channels"]),
            layers_per_block=vae_config["layers_per_block"],
            z_channels=vae_config["latent_channels"],
            device=mesh_device,
            parallel_config=self._vae_parallel,
            ccl_manager=self._vae_ccl_manager,
        )
        cache.load_model(
            tt_model=self._vae_decoder,
            get_torch_state_dict=torch_vae.state_dict,
            model_name=model_name,
            subfolder="vae",
            parallel_config=self._vae_parallel,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
        )
        if config.max_references > 0:
            self._vae_encoder: Flux2VaeEncoder | None = Flux2VaeEncoder(
                in_channels=vae_config["in_channels"],
                block_out_channels=list(vae_config["block_out_channels"]),
                layers_per_block=vae_config["layers_per_block"],
                z_channels=vae_config["latent_channels"],
                device=mesh_device,
                parallel_config=self._vae_parallel,
                ccl_manager=self._vae_ccl_manager,
            )
            cache.load_model(
                tt_model=self._vae_encoder,
                get_torch_state_dict=torch_vae.state_dict,
                model_name=model_name,
                subfolder="vae_encoder",
                parallel_config=self._vae_parallel,
                mesh_shape=tuple(mesh_device.shape),
                mesh_device=mesh_device,
            )
        else:
            self._vae_encoder = None
        self._latents_bn_mean = torch_vae.bn.running_mean.float()
        self._latents_bn_std = torch.sqrt(torch_vae.bn.running_var.float() + vae_config["batch_norm_eps"])
        ttnn.synchronize_device(mesh_device)

        # Every size runs untraced before the first capture. The buffers that the first run at a
        # size allocates and keeps, such as conv weights and CCL buffers, would otherwise be
        # allocated after an earlier trace was captured, and that trace's replay would overwrite them.
        self._warm_up(traced=False)
        if trace_warmup:
            self._warm_up(traced=True)

    @property
    def resolutions(self) -> tuple[tuple[int, int], ...]:
        """The supported image sizes as (width, height)."""
        return self._resolutions

    def release_traces(self) -> None:
        """Release every trace captured by this pipeline."""
        for state in self._denoise_states.values():
            state.step_tracer.release_trace()
        for tracer in itertools.chain(self._vae_decoder_tracers.values(), self._vae_encoder_tracers.values()):
            tracer.release_trace()

    def _warm_up(self, *, traced: bool) -> None:
        """Run every resolution with every reference count and prompt length."""
        prompts = {length: self._prompt_of_length(length) for length in self._text_encoder.lengths()}
        for width, height in self._resolutions:
            for reference_count in range(self._config.max_references + 1):
                images = [Image.new("RGB", (width, height), (128, 128, 128))] * reference_count
                for length, prompt in prompts.items():
                    logger.info(
                        f"warming up{' traced' if traced else ''} at {width}x{height} with {reference_count} "
                        f"references and {length} prompt tokens..."
                    )
                    self(
                        prompts=[prompt],
                        num_inference_steps=2,
                        seed=0,
                        width=width,
                        height=height,
                        images=images or None,
                        traced=traced,
                    )

    def _prompt_of_length(self, length: int) -> str:
        """A prompt that the text encoder pads to ``length`` tokens."""
        words = 0
        while self._text_encoder.prompt_length("a " * words) < length:
            words += 1
        assert self._text_encoder.prompt_length("a " * words) == length
        return "a " * words

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int = 16,
        seed: int = 0,
        width: int | None = None,
        height: int | None = None,
        images: Sequence[Image.Image] | None = None,
        traced: bool = True,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        """Generate an image, or edit the reference ``images``.

        ``width`` and ``height`` default to the first resolution, and must be one of them. The
        distilled model samples without guidance, so ``negative_prompts`` must be empty. Each size,
        reference count and prompt length gets its own trace.
        """
        if len(prompts) != 1:
            msg = "exactly one prompt is supported"
            raise NotImplementedError(msg)
        if negative_prompts and any(negative_prompts):
            msg = "the step-distilled model samples without guidance and takes no negative prompt"
            raise ValueError(msg)
        images = list(images or [])
        if len(images) > self._config.max_references:
            msg = f"got {len(images)} reference images, at most {self._config.max_references} are supported"
            raise ValueError(msg)
        assert self._vae_encoder is not None or not images

        if (width is None) != (height is None):
            msg = "pass both width and height, or neither"
            raise ValueError(msg)
        if width is None or height is None:
            width, height = self._resolutions[0]
        if (width, height) not in self._resolutions:
            msg = f"{width}x{height} is not one of the resolutions {self._resolutions}"
            raise ValueError(msg)

        on_event = on_event if on_event is not None else null_callback
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        sp_factor = self._parallel_config.sequence_parallel.factor
        token_height = height // (_VAE_SCALE_FACTOR * _PATCH_SIZE)
        token_width = width // (_VAE_SCALE_FACTOR * _PATCH_SIZE)
        seq_axes = [None, sp_axis, None]
        rope_axes = [None, None, sp_axis, None]

        with event_section(on_event, "total"):
            logger.info("encoding the prompt...")
            with event_section(on_event, "encoder"):
                prompt_embeds = self._text_encoder.encode(prompts[0])
            prompt_length = prompt_embeds.shape[1]
            padded_length = padded_prompt_length(prompt_length, sp_factor=sp_factor)

            references = []
            if images:
                logger.info("encoding the reference images...")
                with event_section(on_event, "image_encoder"):
                    references = [
                        self._encode_reference(image, width=width, height=height, traced=traced) for image in images
                    ]

            sigmas = _sigmas(num_inference_steps)

            # The noise of diffusers' FLUX.3 pipelines for the same seed: drawn in float32 on the
            # CPU, cast to bfloat16 and packed to one token per row.
            generator = torch.Generator("cpu").manual_seed(seed)
            shape = [1, _LATENT_CHANNELS * _PATCH_SIZE**2, token_height, token_width]
            latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
            latents = latents.flatten(2).transpose(1, 2).float()

            state = self._denoise_state(width, height, len(references), prompt_length)
            checkpoint = self._checkpoint
            device = self._device

            def update_rope(
                target: tuple[StateTensor, StateTensor], ids: torch.Tensor, length: int | None = None
            ) -> None:
                cos, sin = checkpoint.rope(ids)
                padding = (length or len(ids)) - len(ids)
                for buffer, x in zip(target, (cos, sin), strict=True):
                    x = torch.nn.functional.pad(x, (0, 0, 0, padding))[None, None]
                    buffer.update(x, traced, mesh_axes=rope_axes, device=device)

            image_ids = checkpoint.image_ids(height=token_height, width=token_width)
            state.latents.update(latents, traced, mesh_axes=seq_axes, device=device)
            update_rope(state.spatial_rope, image_ids)
            update_rope(state.prompt_rope, checkpoint.text_ids(prompt_length), padded_length)

            reference_length = 0
            if references:
                reference_ids = torch.cat(
                    [
                        checkpoint.image_ids(height=token_height, width=token_width, index=-(1 + k))
                        for k in range(len(references))
                    ]
                )
                reference_tokens = torch.cat(references, dim=1)
                reference_length = reference_tokens.shape[1]
                state.reference.update(reference_tokens, traced, mesh_axes=seq_axes, device=device)
                update_rope(state.reference_rope, reference_ids)
                update_rope(state.joint_rope, interleave_shards(image_ids, reference_ids, sp_factor))

            # The prompt is the same in every step, so its input projection runs once here.
            tt_prompt_embeds = from_torch(
                torch.nn.functional.pad(prompt_embeds.float(), (0, 0, 0, padded_length - prompt_length)),
                device=device,
                mesh_axes=seq_axes,
            )
            state.prompt.update(self.transformer.context_embedder(tt_prompt_embeds), traced)
            ttnn.deallocate(tt_prompt_embeds)

            logger.info("denoising...")
            with event_section(on_event, "denoising"):
                for i in tqdm.tqdm(range(num_inference_steps)):
                    with event_section(on_event, f"denoising_step_{i}"):
                        for buffer, value in (
                            (state.timestep, _timestep(sigmas[i])),
                            (state.target_timestep, _timestep(sigmas[i + 1])),
                            (state.sigma_difference, sigmas[i + 1] - sigmas[i]),
                        ):
                            buffer.update(
                                torch.full([1, 1], fill_value=value), traced, dtype=ttnn.float32, device=device
                            )

                        step_inputs = {
                            "spatial": state.latents.data,
                            "prompt": state.prompt.data,
                            "timestep": state.timestep.data,
                            "target_timestep": state.target_timestep.data,
                            "sigma_difference": state.sigma_difference.data,
                            "spatial_rope": tuple(t.data for t in state.spatial_rope),
                            "prompt_rope": tuple(t.data for t in state.prompt_rope),
                            "spatial_sequence_length": token_height * token_width,
                            "prompt_sequence_length": prompt_length,
                            "reference": state.reference.data if references else None,
                            "reference_rope": tuple(t.data for t in state.reference_rope) if references else None,
                            "joint_rope": tuple(t.data for t in state.joint_rope) if references else None,
                            "reference_sequence_length": reference_length,
                        }
                        if traced:
                            state.step_tracer(**step_inputs)
                        else:
                            self._step(**step_inputs)

            logger.info("decoding the image...")
            with event_section(on_event, "vae"):
                return [self._decode(state.latents.data, width=width, height=height, traced=traced)]

    def _denoise_state(self, width: int, height: int, reference_count: int, prompt_length: int) -> _DenoiseState:
        key = (width, height, reference_count, prompt_length)
        state = self._denoise_states.get(key)
        if state is None:
            # The step updates the latents in place, so the run before the capture works on copies.
            state = _DenoiseState(step_tracer=Tracer(self._step, device=self._device, clone_prep_inputs=True))
            self._denoise_states[key] = state
        return state

    def _step(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        target_timestep: ttnn.Tensor,
        sigma_difference: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        reference: ttnn.Tensor | None,
        reference_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        joint_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        reference_sequence_length: int,
    ) -> None:
        """One Euler step, updating ``spatial`` in place."""
        velocity = self.transformer.forward(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            target_timestep=target_timestep,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            reference=reference,
            reference_rope=reference_rope,
            joint_rope=joint_rope,
            reference_sequence_length=reference_sequence_length,
        )
        ttnn.multiply_(velocity, sigma_difference)
        ttnn.add_(spatial, velocity)

    def _encode_reference(self, image: Image.Image, *, width: int, height: int, traced: bool) -> torch.Tensor:
        """The normalized latent tokens of ``image``, center-cropped to the aspect ratio of the
        output and resized to its size."""
        assert self._vae_encoder is not None

        image = image.convert("RGB")
        crop_width = min(image.width, round(image.height * width / height))
        crop_height = min(image.height, round(image.width * height / width))
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        pixels = torch.from_numpy(np.asarray(image)).float()[None] / 127.5 - 1.0
        h_axis = self._vae_parallel.h_parallel.mesh_axis if self._vae_parallel.h_parallel is not None else None
        w_axis = self._vae_parallel.w_parallel.mesh_axis if self._vae_parallel.w_parallel is not None else None
        tt_pixels = from_torch(
            pixels, device=self._device, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, h_axis, w_axis, None]
        )

        if traced:
            tracer = self._vae_encoder_tracers.get((width, height))
            if tracer is None:
                tracer = Tracer(self._vae_encoder.forward, device=self._device, clone_prep_inputs=False)
                self._vae_encoder_tracers[width, height] = tracer
            tt_latents = tracer(tt_pixels)
        else:
            tt_latents = self._vae_encoder.forward(tt_pixels)

        latents = to_torch(tt_latents).float()
        return (_patchify(latents) - self._latents_bn_mean) / self._latents_bn_std

    def _decode(self, latents: ttnn.Tensor, *, width: int, height: int, traced: bool) -> Image.Image:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        h_parallel = self._vae_parallel.h_parallel
        w_parallel = self._vae_parallel.w_parallel

        # The latents are sharded on the tokens over SP. The VAE takes them sharded on rows over its
        # H axis, and on columns over its W axis once unpatchified.
        if h_parallel is None or h_parallel.mesh_axis != sp_axis:
            latents = self._ccl_manager.all_gather(latents, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
            if h_parallel is not None:
                latents = ttnn.mesh_partition(
                    latents, dim=1, cluster_axis=h_parallel.mesh_axis, memory_config=latents.memory_config()
                )

        latents = self._vae_decoder.preprocess_and_unpatchify(
            latents, height=height // _VAE_SCALE_FACTOR, width=width // _VAE_SCALE_FACTOR
        )
        if w_parallel is not None:
            latents = ttnn.mesh_partition(
                latents, dim=2, cluster_axis=w_parallel.mesh_axis, memory_config=latents.memory_config()
            )

        if traced:
            tracer = self._vae_decoder_tracers.get((width, height))
            if tracer is None:
                tracer = Tracer(self._vae_decoder.forward, device=self._device, clone_prep_inputs=False)
                self._vae_decoder_tracers[width, height] = tracer
            decoded = tracer(latents)
        else:
            decoded = self._vae_decoder.forward(latents)

        pixels = fast_device_to_host(
            decoded, self._device, [None, None], ccl_manager=self._vae_ccl_manager, pre_transfer_fn=float_to_uint8
        )
        return Image.fromarray(pixels[0].numpy())


def _sigmas(num_inference_steps: int) -> list[float]:
    """The noise levels of the steps followed by 0: diffusers' constant-shift schedule."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=_SHIFT)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
    sigmas = scheduler.sigmas
    assert isinstance(sigmas, torch.Tensor)
    return sigmas.tolist()


def _timestep(sigma: float) -> float:
    """The timestep the reference embeds for ``sigma``: rounded to bfloat16, then times 1000 in
    bfloat16."""
    return float(torch.tensor(sigma, dtype=torch.bfloat16) * 1000)


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    # N, H, W, C -> N, (H / P) * (W / P), C * P * P
    batch_size, height, width, channels = latents.shape
    p = _PATCH_SIZE
    latents = latents.reshape([batch_size, height // p, p, width // p, p, channels])
    return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)
