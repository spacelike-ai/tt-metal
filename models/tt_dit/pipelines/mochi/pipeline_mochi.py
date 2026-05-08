# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import tqdm
from diffusers.video_processor import VideoProcessor
from loguru import logger

import ttnn

from models.tt_dit.models.transformers.transformer_mochi import MochiCheckpoint
from models.tt_dit.models.vae.vae_mochi import MochiVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, MochiVAEParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.mochi.text_encoder import TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver, schedules
from models.tt_dit.utils import cache
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_CHECKPOINT = "genmo/mochi-1-preview"

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp": (2, 0),
        "tp": (4, 1),
        "vae_mesh_shape": (1, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 1,
        "reload_dit_model": True,
    },
    (4, 8): {
        "sp": (8, 1),
        "tp": (4, 0),
        "vae_mesh_shape": (4, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 4,
        "reload_dit_model": False,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "sp": (2, 0),
        "tp": (2, 1),
        "vae_mesh_shape": (1, 4),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": True,
    },
    (2, 4): {
        "sp": (2, 0),
        "tp": (4, 1),
        "vae_mesh_shape": (2, 4),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": False,
    },
    (4, 8): {
        "sp": (8, 1),
        "tp": (4, 0),
        "vae_mesh_shape": (4, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": False,
    },
}


@dataclass(frozen=True, kw_only=True)
class MochiPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    vae_parallel_config: MochiVAEParallelConfig
    vae_mesh_shape: tuple[int, ...]

    use_reference_vae: bool
    force_zeros_for_empty_prompt: bool
    reload_dit_model: bool

    height: int
    width: int
    num_frames: int
    cfg_enabled: bool

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        vae_parallel_config: MochiVAEParallelConfig | None = None,
        vae_mesh_shape: tuple[int, ...] | None = None,
        use_reference_vae: bool = False,
        force_zeros_for_empty_prompt: bool = False,
        reload_dit_model: bool | None = None,
        height: int = 480,
        width: int = 848,
        num_frames: int = 168,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MochiPipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=preset["sp"], tp=preset["tp"])

        if vae_mesh_shape is None:
            vae_mesh_shape = preset["vae_mesh_shape"]

        if vae_parallel_config is None:
            vae_sp_axis = preset["vae_sp_axis"]
            vae_tp_axis = preset["vae_tp_axis"]
            w_factor = 1 if vae_mesh_shape[vae_sp_axis] == 1 else 2
            vae_parallel_config = MochiVAEParallelConfig.from_tuples(
                time=(vae_mesh_shape[vae_tp_axis], vae_tp_axis),
                h=(vae_mesh_shape[vae_sp_axis] // w_factor, vae_sp_axis),
                w=(w_factor, vae_sp_axis),
            )

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            vae_parallel_config=vae_parallel_config,
            vae_mesh_shape=tuple(vae_mesh_shape),
            use_reference_vae=use_reference_vae,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            reload_dit_model=reload_dit_model if reload_dit_model is not None else preset["reload_dit_model"],
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )


class MochiPipeline(PipelineAPIMixin):
    r"""
    The mochi pipeline for text-to-video generation.

    Reference: https://github.com/genmoai/models

    Args:
        transformer ([`MochiTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        vae ([`AutoencoderKLMochi`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 848,
        num_frames: int = 168,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MochiPipeline:
        config = MochiPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: MochiPipelineConfig,
    ) -> None:
        # TODO: determine these scaling factors from model parameters
        self.vae_spatial_scale_factor = 8
        self.vae_temporal_scale_factor = 6
        self.patch_size = 2

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_scale_factor)

        self.mesh_device = device
        self.vae_mesh_shape = config.vae_mesh_shape
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.num_links = config.num_links
        self.reload_dit_model = config.reload_dit_model  # Only required if VAE is memory-constrained.
        self._height = config.height
        self._width = config.width
        self._num_frames = config.num_frames
        self._cfg_enabled = config.cfg_enabled

        if self.reload_dit_model and not cache.cache_dir_is_set():
            msg = (
                "Cache must be enabled when DiT model reloading is enabled (reload_dit_model=True). "
                "Please set TT_DIT_CACHE_DIR environment variable to enable caching."
            )
            raise RuntimeError(msg)

        # Create CCL manager
        self.ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=config.topology,
        )

        # Create VAE CCL manager using the VAE mesh shape.
        with reshape_device(self.mesh_device, self.vae_mesh_shape):
            self.vae_ccl_manager = CCLManager(
                mesh_device=device,
                num_links=config.num_links,
                topology=ttnn.Topology.Linear,
            )

        self._solver = EulerSolver()

        # Load pretrained T5 text encoder and tokenizer (Torch)
        checkpoint_name = config.checkpoint_name
        self._text_encoder = TextEncoder(
            checkpoint_name=checkpoint_name,
            force_zeros_for_empty_prompt=config.force_zeros_for_empty_prompt,
        )

        # Load pretrained Mochi Transformer (TT)
        self._checkpoint = MochiCheckpoint(checkpoint_name)

        self.transformer = self._checkpoint.build(
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=True,
        )
        self._transformer_tracer = Tracer(self.transformer.forward, device=device, prep_run=False)

        self._checkpoint.load(
            self.transformer,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )

        with reshape_device(self.mesh_device, self.vae_mesh_shape):
            self._vae = MochiVAEDecoderAdapter(
                checkpoint_name=checkpoint_name,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                use_torch=config.use_reference_vae,
            )
            if not config.use_reference_vae:
                self._vae.reload_weights()

        logger.info("Pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, guidance_scale=2 if config.cfg_enabled else 1, traced=False)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
    ):
        height = height // self.vae_spatial_scale_factor
        width = width // self.vae_spatial_scale_factor
        num_frames = (num_frames - 1) // self.vae_temporal_scale_factor + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        return torch.randn(shape, dtype=torch.float32).to(dtype)


    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int = 64,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        seed: int = 0,
        max_sequence_length: int = 256,
        traced: bool = False,
        vae_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ):
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * len(prompts)

        vae_traced = vae_traced if vae_traced is not None else traced
        height = self._height
        width = self._width
        num_frames = self._num_frames

        if guidance_scale > 1 and not self._cfg_enabled:
            msg = "guidance_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        cfg_enabled = guidance_scale > 1.0
        batch_size = len(prompts)

        # 3. Prepare text embeddings
        on_event(SectionStart("encoder"))
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            cfg_enabled=cfg_enabled,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            disable_attention_mask=traced,
            on_event=on_event,
        )
        on_event(SectionEnd("encoder"))

        print(f"prompt_embeds.shape: {prompt_embeds.shape}")
        print(f"prompt_attention_mask.shape: {prompt_attention_mask.shape}")
        print(f"negative_prompt_embeds.shape: {negative_prompt_embeds.shape}")
        print(f"negative_prompt_attention_mask.shape: {negative_prompt_attention_mask.shape}")

        # 3b. If the transformer was destroyed, recreate it.
        if self.transformer is None:
            logger.info("Recreating MochiTransformer3DModel")
            self.transformer = self._checkpoint.build(
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                is_fsdp=True,
            )
            self._transformer_tracer = Tracer(
                self.transformer.forward, device=self.mesh_device, prep_run=True, clone_prep_inputs=False
            )

            logger.info("Loading MochiTransformer3DModel state_dict")
            self._checkpoint.load(
                self.transformer,
                mesh_device=self.mesh_device,
                parallel_config=self.parallel_config,
            )

        # 4. Prepare latent variables
        torch.manual_seed(seed)

        num_channels_latents = self._checkpoint.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
        )
        print(f"preparing latents with H: {height}, W: {width}, num_frames: {num_frames}")
        print(f"latents.shape: {latents.shape}")

        if cfg_enabled:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        sigmas, alphas = schedules.linear_quadratic(num_inference_steps, threshold_noise=0.025)
        sigmas, alphas = alphas, sigmas  # equivalent to diffuser's invert_sigmas=True
        self._solver.set_schedule(sigmas, alphas)
        timesteps = [s * 1000 for s in sigmas[:-1]]

        # 6. Denoising loop
        on_event(SectionStart("denoising"))
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = self._step(
                    step=i,
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    cfg_enabled=cfg_enabled,
                    guidance_scale=guidance_scale,
                    traced=traced,
                )

                progress_bar.update()
        on_event(SectionEnd("denoising"))

        return self._decode_latents(
            latents,
            vae_traced=vae_traced,
            on_event=on_event,
        )

    def synchronize_devices(self):
        ttnn.synchronize_device(self.mesh_device)

    def _decode_latents(
        self,
        latents: torch.Tensor,
        *,
        vae_traced: bool,
        on_event: PipelineEventCallback,
    ) -> torch.Tensor:
        # If the VAE is memory-constrained, free the transformer.
        if self.reload_dit_model:
            logger.info("Freeing MochiTransformer3DModel")
            self.transformer = None
            self._transformer_tracer.release_trace()
            self._transformer_tracer = None

        on_event(SectionStart("vae"))
        with reshape_device(self.mesh_device, self.vae_mesh_shape):
            video = self._vae.decode(latents, traced=vae_traced)
        on_event(SectionEnd("vae"))

        return self.video_processor.postprocess_video(video, output_type="pil")

    def _predict(
        self,
        *,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        timestep: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        traced: bool,
    ) -> torch.Tensor:
        assert self.transformer is not None
        assert self._transformer_tracer is not None

        B, C, T, H, W = spatial.shape

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.transformer.prepare_rope_features(T, H, W)
        temb_11BD, prompt_1BLP = self.transformer.prepare_timestep_text_features(
            timestep, prompt, prompt_attention_mask
        )
        spatial_1BNI, N = self.transformer.preprocess_spatial_input(spatial)

        forward = self._transformer_tracer if traced else self.transformer.forward

        proj_out_1BNI = forward(
            temb_11BD=temb_11BD,
            prompt_1BLP=prompt_1BLP,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            spatial_1BNI=spatial_1BNI,
            trans_mat=trans_mat,
            N=N,
        )

        out = self.transformer.postprocess_spatial_output(proj_out_1BNI, T, H, W, N)
        return out.to(torch.float32)

    def _step(
        self,
        *,
        step: int,
        t: float,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        cfg_enabled: bool,
        guidance_scale: float,
        traced: bool,
    ) -> torch.Tensor:
        latent_model_input = torch.cat([latents] * 2) if cfg_enabled else latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = torch.tensor(t).expand(latent_model_input.shape[0]).to(latents.dtype)

        noise_pred_uncond = self._predict(
            spatial=latent_model_input[:1],
            prompt=prompt_embeds[:1],
            timestep=timestep[:1],
            prompt_attention_mask=prompt_attention_mask[:1],
            traced=traced,
        )
        noise_pred_text = self._predict(
            spatial=latent_model_input[1:],
            prompt=prompt_embeds[1:],
            timestep=timestep[1:],
            prompt_attention_mask=prompt_attention_mask[1:],
            traced=traced,
        )

        assert cfg_enabled
        if cfg_enabled:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_dtype = latents.dtype
        latents = self._solver.step(step=step, latent=latents.to(torch.float32), velocity_pred=noise_pred)
        latents = latents.to(latents_dtype)

        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        return latents
