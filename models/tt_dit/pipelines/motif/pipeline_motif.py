# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from loguru import logger

import ttnn

from ...models.transformers.transformer_motif import MotifCheckpoint
from ...models.vae.vae_sd35 import VAEDecoderAdapter
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...solvers import EulerSolver
from ...utils import tensor
from ...utils.tracing import Tracer
from ..cfg import CFGCombiner
from ..events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from ..mesh import create_submeshes, reshape_device
from ..pipeline_api import PipelineAPIMixin
from .text_encoder import TextEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from PIL import Image

_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16

_PRESETS: dict[tuple[int, int], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
    },
}


@dataclass(frozen=True, kw_only=True)
class MotifPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    enable_t5_text_encoder: bool
    use_torch_t5_text_encoder: bool
    use_torch_clip_text_encoder: bool
    use_torch_vae: bool

    height: int
    width: int
    cfg_enabled: bool

    checkpoint_name: str


class MotifPipeline(PipelineAPIMixin):
    def __init__(self, *, device: ttnn.MeshDevice, config: MotifPipelineConfig) -> None:
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor == 2
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._encoder_tp = config.encoder_parallel_config.tensor_parallel
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._submesh_devices = create_submeshes(device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._submesh_devices[0].shape}")

        self._prediction_tracers = [Tracer(self._prediction, device=d, prep_run=False) for d in self._submesh_devices]

        self._ccl_managers = [
            CCLManager(d, num_links=config.num_links, topology=config.topology) for d in self._submesh_devices
        ]

        self._combiner = CFGCombiner(self._submesh_devices)
        self._solvers = (EulerSolver(), EulerSolver()) if self._cfg_parallel else (EulerSolver(),)
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

        logger.info("creating transformer...")
        checkpoint = MotifCheckpoint(config.checkpoint_name)
        self._transformers = [
            checkpoint.build(
                latents_height=config.height // _VAE_SCALE_FACTOR,
                latents_width=config.width // _VAE_SCALE_FACTOR,
                parallel_config=config.dit_parallel_config,
                ccl_manager=m,
            )
            for m in self._ccl_managers
        ]

        with self._reshape_encoder_device():
            logger.info("creating encoder...")
            self._text_encoder = TextEncoder(
                parallel_config=config.encoder_parallel_config,
                enable_t5=config.enable_t5_text_encoder,
                use_torch_clip_encoder=config.use_torch_clip_text_encoder,
                use_torch_t5_encoder=config.use_torch_t5_text_encoder,
                ccl_manager=self._ccl_managers[0],
            )

            logger.info("creating VAE decoder...")
            self._vae = VAEDecoderAdapter(
                checkpoint_name="stabilityai/stable-diffusion-3.5-large",
                parallel_config=config.vae_parallel_config,
                skip_shift=True,  # Motif omits the VAE shift.
                use_torch=config.use_torch_vae,
                ccl_manager=self._ccl_managers[0],
            )

        for d in self._submesh_devices:
            ttnn.synchronize_device(d)

        logger.info("pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, traced=False, cfg_scale=2 if config.cfg_enabled else 1)

    def _reshape_encoder_device(self) -> AbstractContextManager[None]:
        device = self._submesh_devices[0]
        tp = self._encoder_tp

        shape = list(device.shape)
        shape[tp.mesh_axis] = tp.factor
        shape[1 - tp.mesh_axis] = device.shape.mesh_size() // tp.factor

        return reshape_device(self._submesh_devices[0], ttnn.MeshShape(*shape))

    @classmethod
    def create_pipeline(
        cls,
        mesh_device: ttnn.MeshDevice,
        dit_cfg: tuple[int, int] | None = None,
        dit_sp: tuple[int, int] | None = None,
        dit_tp: tuple[int, int] | None = None,
        encoder_tp: tuple[int, int] | None = None,
        vae_tp: tuple[int, int] | None = None,
        enable_t5_text_encoder: bool = True,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        use_torch_vae: bool = False,
        num_links: int | None = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = "Motif-Technologies/Motif-Image-6B-Preview",
        model_checkpoint_path: str | None = None,
    ) -> MotifPipeline:
        """Factory that picks parallelism defaults based on the mesh shape."""
        if model_checkpoint_path is not None:
            checkpoint_name = model_checkpoint_path
            logger.warning("DEPRECATED: model_checkpoint_path is deprecated. Use checkpoint_name instead.")

        preset = _PRESETS.get(tuple(mesh_device.shape), {})

        config = MotifPipelineConfig(
            topology=topology,
            num_links=num_links or preset["num_links"],
            dit_parallel_config=DiTParallelConfig.from_tuples(
                cfg=dit_cfg or preset["cfg"],
                sp=dit_sp or preset["sp"],
                tp=dit_tp or preset["tp"],
            ),
            encoder_parallel_config=EncoderParallelConfig.from_tuple(encoder_tp or preset["encoder_tp"]),
            vae_parallel_config=VAEParallelConfig.from_tuple(vae_tp or preset["vae_tp"]),
            enable_t5_text_encoder=enable_t5_text_encoder,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            use_torch_vae=use_torch_vae,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Encoder parallel config: {config.encoder_parallel_config}")
        logger.info(f"VAE parallel config: {config.vae_parallel_config}")
        logger.info(f"T5 enabled: {enable_t5_text_encoder}")

        return cls(device=mesh_device, config=config)

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        prompts_2: Sequence[str] | None = None,
        prompts_3: Sequence[str] | None = None,
        negative_prompts: Sequence[str | None] | None = None,
        negative_prompts_2: Sequence[str | None] | None = None,
        negative_prompts_3: Sequence[str | None] | None = None,
        num_inference_steps: int = 40,
        seed: int | None = None,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 5.0,
        linear_quadratic_emulating_steps: int = 100,
        negative_strategy_switch_time: float = 0.85,
        traced: bool = False,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts or [None] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        logger.info("encoding prompts...")
        on_event(SectionStart("total"))

        on_event(SectionStart("encoder"))
        with self._reshape_encoder_device():
            (
                torch_early_context,
                torch_early_pooled,
                torch_late_context,
                torch_late_pooled,
            ) = self._text_encoder.encode_cfg(
                (prompts, prompts_2 or prompts, prompts_3 or prompts),
                (negative_prompts, negative_prompts_2 or negative_prompts, negative_prompts_3 or negative_prompts),
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=self._cfg_enabled,
                traced=encoder_traced,
                on_event=on_event,
            )
        on_event(SectionEnd("encoder"))

        logger.info("preparing timesteps...")
        sigmas, alphas = _schedule(
            step_count=num_inference_steps,
            linear_quadratic_emulating_steps=linear_quadratic_emulating_steps,
        )
        for solver in self._solvers:
            solver.set_schedule(sigmas=sigmas, alphas=alphas)

        logger.info("preparing inputs...")
        latents = self._random_latents(batch_size=prompt_count * num_images_per_prompt, seed=seed)
        early_context = self._distribute_cfg(torch_early_context, on_host=traced)
        early_pooled = self._distribute_cfg(torch_early_pooled, on_host=traced)
        late_context = self._distribute_cfg(torch_late_context, on_host=traced)
        late_pooled = self._distribute_cfg(torch_late_pooled, on_host=traced)

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for i, t in enumerate(tqdm.tqdm(sigmas[:-1])):
            on_event(SectionStart(f"denoising_step_{i}"))

            early = t >= negative_strategy_switch_time

            velocity_pred = []

            for device_idx, device in enumerate(self._submesh_devices):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t * 1000,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=device,
                )

                latents[device_idx], v = self._prediction_tracers[device_idx](
                    latents=latents[device_idx],
                    prompt=early_context[device_idx] if early else late_context[device_idx],
                    pooled=early_pooled[device_idx] if early else late_pooled[device_idx],
                    timestep=timestep,
                    submesh_idx=device_idx,
                    traced=traced,
                )
                velocity_pred.append(v)

                if self._cfg_enabled:
                    velocity_pred[device_idx] = self._combiner.combine(velocity_pred[device_idx], cfg_scale)

            for device_idx, device in enumerate(self._submesh_devices):
                ttnn.synchronize_device(device)  # Helps with accurate time profiling.
                latents[device_idx] = self._solvers[device_idx].step(
                    step=i, latent=latents[device_idx], velocity_pred=velocity_pred[device_idx]
                )

            on_event(SectionEnd(f"denoising_step_{i}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[0], traced=vae_traced)
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return images

    def _prediction(
        self,
        *,
        latents: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_idx: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        model_input = ttnn.concat([latents, latents]) if self._cfg_enabled and not self._cfg_parallel else latents

        velocity_pred = self._transformers[submesh_idx].forward(
            spatial=model_input,
            prompt=prompt,
            pooled=pooled,
            timestep=timestep,
        )

        # Make latents an output, because inputs are copied to the trace region before executing a
        # trace and might be overwritten during execution.
        return latents, velocity_pred

    def _random_latents(self, batch_size: int, seed: int | None) -> list[ttnn.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        shape = [batch_size, _LATENT_CHANNELS, self._height // _VAE_SCALE_FACTOR, self._width // _VAE_SCALE_FACTOR]

        # We let randn generate a permuted latent tensor in float32, so that the generated noise
        # matches the reference implementation.
        latents = torch.randn(shape, dtype=torch.float32).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)
        latents = self._transformers[0].patchify(latents)

        return [
            tensor.from_torch(latents, device=d, mesh_axes=[None, self._sp_axis, None]) for d in self._submesh_devices
        ]

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, traced: bool) -> list[Image.Image]:
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self._submesh_devices[0])

        tt_latents = self._ccl_managers[0].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = self._transformers[0].unpatchify(
            torch_latents,
            height=self._height // _VAE_SCALE_FACTOR,
            width=self._width // _VAE_SCALE_FACTOR,
        )

        with self._reshape_encoder_device():
            decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

    def _distribute_cfg(
        self, x: torch.Tensor, *, on_host: bool
    ) -> tuple[ttnn.Tensor] | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return one tensor per submesh from a conditioning batch."""
        match self._submesh_devices:
            case (device,):
                return (tensor.from_torch(x, device=device, on_host=on_host),)
            case (device1, device2):
                half = x.shape[0] // 2
                return (
                    tensor.from_torch(x[:half], device=device1, on_host=on_host),
                    tensor.from_torch(x[half:], device=device2, on_host=on_host),
                )
            case _:
                msg = "unsupported number of submeshes"
                raise ValueError(msg)


def _schedule(*, step_count: int, linear_quadratic_emulating_steps: int) -> tuple[list[float], list[float]]:
    """A slight variation of ``schedules.linear_quadratic``."""
    assert step_count % 2 == 0

    s = step_count
    n = linear_quadratic_emulating_steps
    a = s // 2 / n - 1

    sigmas1 = torch.linspace(1, 0, n + 1)[: s // 2]
    sigmas2 = torch.linspace(0, 1, s // 2 + 1).pow(2) * a - a

    sigmas = torch.concat([sigmas1, sigmas2])
    alphas = 1 - sigmas

    return sigmas.tolist(), alphas.tolist()
