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

from ...models.transformers.transformer_motif import MotifCheckpoint, MotifTransformer
from ...models.vae.vae_sd35 import VAEDecoderAdapter
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...solvers import EulerSolver
from ...utils import tensor
from ...utils.tracing import Tracer
from ..cfg import CFGCombiner
from ..events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from ..mesh import create_submeshes, reshape_device
from .text_encoder import TextEncoder

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from PIL import Image


_DEFAULT_PARALLELISM_BY_MESH_SHAPE: dict[tuple[int, int], dict] = {
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

    enable_t5_text_encoder: bool = True
    use_torch_t5_text_encoder: bool = False
    use_torch_clip_text_encoder: bool = False
    use_torch_vae: bool = False

    height: int = 1024
    width: int = 1024
    checkpoint_name: str = "Motif-Technologies/Motif-Image-6B-Preview"


class MotifPipeline:
    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: MotifPipelineConfig,
    ) -> None:
        self._config = config
        self._mesh_device = device
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._submesh_devices[0].shape}")
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self._submesh_devices
        ]

        self._combiner = CFGCombiner(tuple(self._submesh_devices))

        self.encoder_device = self._submesh_devices[0]
        original_encoder_mesh_shape = list(self.encoder_device.shape)
        original_encoder_mesh_shape[
            self._encoder_parallel_config.tensor_parallel.mesh_axis
        ] = self._encoder_parallel_config.tensor_parallel.factor
        original_encoder_mesh_shape[1 - self._encoder_parallel_config.tensor_parallel.mesh_axis] = (
            self.encoder_device.shape.mesh_size() // self._encoder_parallel_config.tensor_parallel.factor
        )
        self.encoder_mesh_shape = ttnn.MeshShape(*original_encoder_mesh_shape)
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = 0  # Use submesh 0 for VAE

        logger.info("loading models...")
        self._num_channels_latents = 16
        self._prompt_embedding_dim = MotifTransformer.ENCODED_TEXT_DIM
        self._patch_size = 2
        self._vae_scale_factor = 8

        logger.info("creating TT-NN transformer...")
        checkpoint = MotifCheckpoint(config.checkpoint_name)
        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = checkpoint.build(
                latents_height=config.height // self._vae_scale_factor,
                latents_width=config.width // self._vae_scale_factor,
                ccl_manager=self._ccl_managers[i],
                parallel_config=config.dit_parallel_config,
            )
            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._step_inner_tracers = [
            Tracer(self._step_inner, device=device, prep_run=False) for device in self._submesh_devices
        ]
        self._solvers = [EulerSolver() for _ in self._submesh_devices]

        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        with self._reshape_encoder_device():
            logger.info("creating TT-NN CLIP text encoder...")
            self._text_encoder = TextEncoder(
                ccl_manager=self._ccl_managers[0],
                parallel_config=config.encoder_parallel_config,
                enable_t5=config.enable_t5_text_encoder,
                use_torch_clip_encoder=config.use_torch_clip_text_encoder,
                use_torch_t5_encoder=config.use_torch_t5_text_encoder,
            )

            ttnn.synchronize_device(self.encoder_device)

            logger.info("creating TT-NN VAE decoder...")
            self._vae = VAEDecoderAdapter(
                checkpoint_name="stabilityai/stable-diffusion-3.5-large",
                parallel_config=self._vae_parallel_config,
                ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                use_torch=config.use_torch_vae,
                skip_shift=True,  # Motif intentionally omits the VAE shift.
            )
            ttnn.synchronize_device(self.encoder_device)

        self._allocate_persistent_buffers()

    def _reshape_encoder_device(self) -> AbstractContextManager[None]:
        return reshape_device(self.encoder_device, self.encoder_mesh_shape)

    @staticmethod
    def create_pipeline(
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
        checkpoint_name: str = "Motif-Technologies/Motif-Image-6B-Preview",
        model_checkpoint_path: str | None = None,
    ) -> MotifPipeline:
        """Factory that picks parallelism defaults based on the mesh shape."""
        if model_checkpoint_path is not None:
            checkpoint_name = model_checkpoint_path
            logger.warning("DEPRECATED: model_checkpoint_path is deprecated. Use checkpoint_name instead.")

        preset = _DEFAULT_PARALLELISM_BY_MESH_SHAPE.get(tuple(mesh_device.shape), {})
        cfg_factor, cfg_axis = dit_cfg or preset["cfg"]
        sp_factor, sp_axis = dit_sp or preset["sp"]
        tp_factor, tp_axis = dit_tp or preset["tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or preset["encoder_tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or preset["vae_tp"]
        num_links = num_links or preset["num_links"]

        config = MotifPipelineConfig(
            topology=topology,
            num_links=num_links,
            dit_parallel_config=DiTParallelConfig(
                cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
                tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
                sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
            ),
            encoder_parallel_config=EncoderParallelConfig(
                tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
            ),
            vae_parallel_config=VAEParallelConfig(
                tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
            ),
            enable_t5_text_encoder=enable_t5_text_encoder,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            use_torch_vae=use_torch_vae,
            height=height,
            width=width,
            checkpoint_name=checkpoint_name,
        )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Encoder parallel config: {config.encoder_parallel_config}")
        logger.info(f"VAE parallel config: {config.vae_parallel_config}")
        logger.info(f"T5 enabled: {enable_t5_text_encoder}")

        return MotifPipeline(device=mesh_device, config=config)

    def _allocate_persistent_buffers(self) -> None:
        """Allocate persistent buffers by running a pipeline pass without tracing.

        This is improtant, so they do not get allocated after trace capture, which would lead to
        them being overwritten during trace execution.
        """
        logger.info("Pipeline allocation run...")
        self.run_single_prompt(prompt="", num_inference_steps=2, traced=False)

    def run_single_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 40,
        cfg_scale: float = 5.0,
        seed: int | None = None,
        traced: bool = True,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        return self.__call__(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            traced=traced,
            vae_traced=vae_traced,
            encoder_traced=encoder_traced,
            on_event=on_event,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str | None],
        negative_prompt_2: list[str | None],
        negative_prompt_3: list[str | None],
        linear_quadratic_emulating_steps: int = 100,
        negative_strategy_switch_time: float = 0.85,
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        prompt_count = len(prompt_1)
        on_event = on_event if on_event is not None else null_callback

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        on_event(SectionStart("total"))
        cfg_enabled = cfg_scale > 1
        logger.info("encoding prompts...")

        on_event(SectionStart("encoder"))
        with self._reshape_encoder_device():
            (
                prompt_embeds1,
                pooled_prompt_embeds1,
                prompt_embeds2,
                pooled_prompt_embeds2,
            ) = self._text_encoder.encode_cfg(
                (prompt_1, prompt_2, prompt_3),
                (negative_prompt_1, negative_prompt_2, negative_prompt_3),
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=cfg_enabled,
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
        timesteps = [s * 1000 for s in sigmas[:-1]]

        logger.info("preparing latents...")

        if seed is not None:
            torch.manual_seed(seed)

        shape = [
            prompt_count * num_images_per_prompt,
            self._num_channels_latents,
            self._height // self._vae_scale_factor,
            self._width // self._vae_scale_factor,
        ]
        # We let randn generate a permuted latent tensor in float32, so that the generated noise
        # matches the reference implementation.
        latents = self.transformers[0].patchify(
            torch.randn(shape, dtype=torch.float32).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)
        )

        tt_prompt_embeds_list = []
        tt_prompt_embeds1_list = []
        tt_prompt_embeds2_list = []
        tt_pooled_prompt_embeds_list = []
        tt_pooled_prompt_embeds1_list = []
        tt_pooled_prompt_embeds2_list = []
        tt_latents_step_list = []
        for i, submesh_device in enumerate(self._submesh_devices):
            # Allocate tensors on the host to ensure that they do not get overwritten by trace
            # execution.
            tt_prompt_embeds1 = tensor.from_torch(
                prompt_embeds1[i : i + 1] if cfg_factor == 2 else prompt_embeds1,
                device=submesh_device,
                on_host=traced,
            )
            tt_prompt_embeds2 = tensor.from_torch(
                prompt_embeds2[i : i + 1] if cfg_factor == 2 else prompt_embeds2,
                device=submesh_device,
                on_host=traced,
            )
            tt_pooled_prompt_embeds1 = tensor.from_torch(
                pooled_prompt_embeds1[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds1,
                device=submesh_device,
                on_host=traced,
            )
            tt_pooled_prompt_embeds2 = tensor.from_torch(
                pooled_prompt_embeds2[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds2,
                device=submesh_device,
                on_host=traced,
            )

            tt_initial_latents = tensor.from_torch(latents, device=submesh_device, mesh_axes=[None, sp_axis, None])

            tt_prompt_embeds1_list.append(tt_prompt_embeds1)
            tt_prompt_embeds2_list.append(tt_prompt_embeds2)
            tt_pooled_prompt_embeds1_list.append(tt_pooled_prompt_embeds1)
            tt_pooled_prompt_embeds2_list.append(tt_pooled_prompt_embeds2)
            tt_latents_step_list.append(tt_initial_latents)
            del tt_initial_latents

        logger.info("denoising...")

        on_event(SectionStart("denoising"))
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{i}"))
            tt_timestep_list = []
            for submesh_nr, submesh_device in enumerate(self._submesh_devices):
                # Allocation on device is fine, because timesteps are not used after
                # trace execution, and can be overwritten during trace execution.
                tt_timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=submesh_device,
                )
                tt_timestep_list.append(tt_timestep)

            if t >= 1000 * negative_strategy_switch_time:
                tt_prompt_embeds_list = tt_prompt_embeds1_list
                tt_pooled_prompt_embeds_list = tt_pooled_prompt_embeds1_list
            else:
                tt_prompt_embeds_list = tt_prompt_embeds2_list
                tt_pooled_prompt_embeds_list = tt_pooled_prompt_embeds2_list

            tt_latents_step_list = self._step(
                timestep=tt_timestep_list,
                latents=tt_latents_step_list,
                cfg_enabled=cfg_enabled,
                prompt_embeds=tt_prompt_embeds_list,
                pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
                cfg_scale=cfg_scale,
                step_index=i,
                traced=traced,
            )
            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")

        on_event(SectionStart("vae"))
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self.vae_device)

        tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents_step_list[self.vae_submesh_idx],
            dim=1,
            mesh_axis=sp_axis,
            use_hyperparams=True,
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = self.transformers[0].unpatchify(
            torch_latents,
            height=self._height // self._vae_scale_factor,
            width=self._width // self._vae_scale_factor,
        )

        with self._reshape_encoder_device():
            decoded_output = self._vae.decode(torch_latents, traced=vae_traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return output

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_id: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        latent_input = (
            ttnn.concat([latent, latent]) if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1 else latent
        )

        noise_pred = self.transformers[submesh_id].forward(
            spatial=latent_input,
            prompt=prompt,
            pooled=pooled,
            timestep=timestep,
        )

        # Make latents an output, because inputs are copied to the trace region before executing a
        # trace and might be overwritten during execution.
        return latent, noise_pred

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],
        timestep: list[ttnn.Tensor],
        pooled_prompt_embeds: list[ttnn.Tensor],
        prompt_embeds: list[ttnn.Tensor],
        step_index: int,
        traced: bool,
    ) -> list[ttnn.Tensor]:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        latents_out = []
        noise_pred_list = []

        for submesh_id in range(len(self._submesh_devices)):
            inner = self._step_inner_tracers[submesh_id] if traced else self._step_inner

            latent, noise_pred = inner(
                cfg_enabled=cfg_enabled,
                latent=latents[submesh_id],
                prompt=prompt_embeds[submesh_id],
                pooled=pooled_prompt_embeds[submesh_id],
                timestep=timestep[submesh_id],
                submesh_id=submesh_id,
            )

            latents_out.append(latent)
            noise_pred_list.append(noise_pred)

        if cfg_enabled:
            for submesh_id in range(len(self._submesh_devices)):
                noise_pred_list[submesh_id] = self._combiner.combine(noise_pred_list[submesh_id], cfg_scale)

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
            latents_out[submesh_id] = self._solvers[submesh_id].step(
                step=step_index, latent=latents_out[submesh_id], velocity_pred=noise_pred_list[submesh_id]
            )

        return latents_out


def _schedule(*, step_count: int, linear_quadratic_emulating_steps: int) -> tuple[list[float], list[float]]:
    assert step_count % 2 == 0

    s = step_count
    n = linear_quadratic_emulating_steps
    a = s // 2 / n - 1

    sigmas1 = torch.linspace(1, 0, n + 1)[: s // 2]
    sigmas2 = torch.linspace(0, 1, s // 2 + 1).pow(2) * a - a

    sigmas = torch.concat([sigmas1, sigmas2])
    alphas = 1 - sigmas

    return sigmas.tolist(), alphas.tolist()
