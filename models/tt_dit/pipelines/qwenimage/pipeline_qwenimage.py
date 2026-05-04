# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from loguru import logger

import ttnn

from models.tt_dit.encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageTransformer
from models.tt_dit.models.vae.vae_qwenimage import QwenImageVaeDecoder
from models.tt_dit.parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
    VAEParallelConfig,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg
from models.tt_dit.solvers import EulerSolver, schedules
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.padding import PaddingConfig
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from PIL import Image

    from models.perf.benchmarking_utils import BenchmarkProfiler

PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX = 34

_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image"

# The encoder is currently hardcoded to always be FSDP as it is the most memory efficient
# configuration with little to no performance penalty.
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": False,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": True,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
        "is_fsdp": False,
        "dynamic_load_encoder": False,
        "dynamic_load_vae": False,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": False,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": False,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
        "is_fsdp": False,
        "dynamic_load_encoder": False,
        "dynamic_load_vae": False,
    },
}


@dataclass(frozen=True, kw_only=True)
class QwenImagePipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool

    height: int
    width: int
    cfg_enabled: bool

    is_fsdp: bool
    dynamic_load_encoder: bool
    dynamic_load_vae: bool

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        is_fsdp: bool | None = None,
        dynamic_load_encoder: bool | None = None,
        dynamic_load_vae: bool | None = None,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig.from_tuple(preset["encoder_tp"])

        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig.from_tuple(preset["vae_tp"])

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            dynamic_load_encoder=(
                dynamic_load_encoder if dynamic_load_encoder is not None else preset["dynamic_load_encoder"]
            ),
            dynamic_load_vae=dynamic_load_vae if dynamic_load_vae is not None else preset["dynamic_load_vae"],
            checkpoint_name=checkpoint_name,
        )


class QwenImagePipeline:
    """
    QwenImagePipeline is a pipeline for generating images from text prompts.
    It uses a transformer to encode the text prompts and a VAE to decode the latent space.
    Dynamic loading is controlled by the initialization state. During inference, modules will be loaded/offloaded as needed.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipeline:
        config = QwenImagePipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: QwenImagePipelineConfig,
    ) -> None:
        if config.dynamic_load_encoder or config.dynamic_load_vae:
            assert (
                cache.cache_dir_is_set()
            ), "Dynamic loading of encoder or vae is enabled but the cache directory (env variable TT_DIT_CACHE_DIR) is not set."

        self._mesh_device = device
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled
        self._is_fsdp = config.is_fsdp
        self._checkpoint_name = config.checkpoint_name

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._submesh_devices[0].shape}")

        self._ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self._submesh_devices
        ]
        self._cfg_combiner = CFGCombiner(self._submesh_devices)

        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = len(self._submesh_devices) - self.encoder_submesh_idx - 1  # Use other submesh for VAE. 0

        self.encoder_device = self._submesh_devices[self.encoder_submesh_idx]
        self.vae_device = self._submesh_devices[self.vae_submesh_idx]

        self._wan_vae_parallel_config = self.get_wan_vae_parallel_config()

        self.encoder_mesh_shape = self.get_mesh_shape(
            self.encoder_device, self._encoder_parallel_config.tensor_parallel
        )
        self.vae_mesh_shape = self.get_mesh_shape(self.vae_device, self._vae_parallel_config.tensor_parallel)

        logger.info("loading models...")

        torch_transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
            self._checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()

        self._torch_vae = AutoencoderKLQwenImage.from_pretrained(self._checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLQwenImage)
        # Store VAE state dict for loading/reloading
        self._vae_state_dict = self._torch_vae.state_dict()

        self._num_channels_latents = 16
        self._patch_size = torch_transformer.config.patch_size
        self._vae_scale_factor = 8

        if torch_transformer.config.num_attention_heads % config.dit_parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                torch_transformer.config.num_attention_heads,
                torch_transformer.config.attention_head_dim,
                config.dit_parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self._transformer_state_dict = torch_transformer.state_dict()
        self._padding_config = padding_config
        self._pos_embed = torch_transformer.pos_embed

        # Initialize the transformers. Loading logic comes after.
        self.transformers = [
            QwenImageTransformer(
                patch_size=torch_transformer.config.patch_size,
                in_channels=torch_transformer.config.in_channels,
                num_layers=torch_transformer.config.num_layers,
                attention_head_dim=torch_transformer.config.attention_head_dim,
                num_attention_heads=torch_transformer.config.num_attention_heads,
                joint_attention_dim=torch_transformer.config.joint_attention_dim,
                out_channels=torch_transformer.config.out_channels,
                device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=self._parallel_config,
                padding_config=self._padding_config,
                is_fsdp=self._is_fsdp,
            )
            for i, submesh_device in enumerate(self._submesh_devices)
        ]
        self._step_inner_tracers = [
            Tracer(self._step_inner, device=device, prep_run=False) for device in self._submesh_devices
        ]
        self._solvers = [EulerSolver() for _ in self._submesh_devices]
        self._transformers_loaded = False

        # initialize text encoder. This will load the weights
        self._use_torch_text_encoder = config.use_torch_text_encoder
        with reshape_device(self.encoder_device, self.encoder_mesh_shape):
            logger.info("creating TT-NN text encoder (loading before transformers for memory efficiency)...")
            self._text_encoder = Qwen25VlTokenizerEncoderPair(
                self._checkpoint_name,
                tokenizer_subfolder="tokenizer",
                encoder_subfolder="text_encoder",
                device=self._submesh_devices[self.encoder_submesh_idx],
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=self._encoder_parallel_config,
                use_torch=config.use_torch_text_encoder,
                is_fsdp=True,  # Best configuration for wh t3k and galaxy
            )
        ttnn.synchronize_device(self.encoder_device)

        # Encoder is already loaded. Decide if we should also load the transformers.
        if (
            not config.dynamic_load_encoder or config.use_torch_text_encoder
        ):  # Implies we have enough space. VAE comes after denoising, so load all transformers now.
            self._load_transformers(self.encoder_submesh_idx)

        # Always load transformers for vae since it comes before VAE
        self._load_transformers(self.vae_submesh_idx)

        self._latents_scaling = 1.0 / torch.tensor(self._torch_vae.config.latents_std)
        self._latents_shift = torch.tensor(self._torch_vae.config.latents_mean)

        self._image_processor = VaeImageProcessor(vae_scale_factor=2 * self._vae_scale_factor)

        self._use_torch_vae_decoder = config.use_torch_vae_decoder

        if config.use_torch_vae_decoder:
            self._vae_decoder = None
            self._vae_decoder_tracer = None
        else:
            with reshape_device(self.vae_device, self.vae_mesh_shape):
                logger.info("creating TT-NN VAE decoder...")
                self._vae_decoder = QwenImageVaeDecoder(
                    base_dim=self._torch_vae.config.base_dim,
                    z_dim=self._torch_vae.config.z_dim,
                    dim_mult=self._torch_vae.config.dim_mult,
                    num_res_blocks=self._torch_vae.config.num_res_blocks,
                    temperal_downsample=self._torch_vae.config.temperal_downsample,
                    device=self.vae_device,
                    parallel_config=self._wan_vae_parallel_config,
                    ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                )
                self._vae_decoder_tracer = Tracer(self._vae_decoder.forward, device=self.vae_device, prep_run=False)
            ttnn.synchronize_device(self.vae_device)

            # Load VAE weights based on configuration
            if not config.dynamic_load_vae:
                self._vae_decoder.load_torch_state_dict(self._vae_state_dict)

        logger.info("Pipeline allocation run...")
        self.run_single_prompt(
            prompt="",
            num_inference_steps=2,
            seed=0,
            cfg_scale=2 if config.cfg_enabled else 1,
            traced=False,
        )

    def _load_transformers(self, idx) -> None:
        """Load transformer weights to device. Called lazily for device encoder path."""
        if self.transformers[idx].is_loaded():
            return

        cache.load_model(
            tt_model=self.transformers[idx],
            get_torch_state_dict=lambda: self._transformer_state_dict,
            model_name=self._checkpoint_name,
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._submesh_devices[idx].shape),
            is_fsdp=self._is_fsdp,
        )

        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_transformers(self, idx) -> None:
        """Deallocate transformer weights from device to free memory."""
        if not self.transformers[idx].is_loaded():
            return

        logger.info("deallocating transformer weights to free memory...")
        self.transformers[idx].deallocate_weights()
        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_vae(self) -> None:
        """Deallocate VAE decoder weights from device to free memory."""
        if self._use_torch_vae_decoder or not self._vae_decoder.is_loaded():
            return

        logger.info("deallocating VAE decoder weights to free memory...")
        self._vae_decoder.deallocate_weights()
        ttnn.synchronize_device(self.vae_device)

    def _reload_vae(self) -> None:
        """Load or reload VAE decoder weights to device."""
        if self._use_torch_vae_decoder or self._vae_decoder.is_loaded():
            return

        with reshape_device(self.vae_device, self.vae_mesh_shape):
            logger.info("loading VAE decoder weights to device...")
            self._vae_decoder.load_torch_state_dict(self._vae_state_dict)
        ttnn.synchronize_device(self.vae_device)

    @staticmethod
    def get_mesh_shape(mesh_device, parallel_factor):
        mesh_shape = list(mesh_device.shape)
        mesh_shape[parallel_factor.mesh_axis] = parallel_factor.factor
        mesh_shape[1 - parallel_factor.mesh_axis] = mesh_device.shape.mesh_size() // parallel_factor.factor
        return ttnn.MeshShape(tuple(mesh_shape))

    # TODO: Configure the correct parallel config
    def get_wan_vae_parallel_config(self):
        return VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=self.vae_device.shape[self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
            width_parallel=ParallelFactor(
                factor=self.vae_device.shape[1 - self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=1 - self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
        )

    def run_single_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 4.0,
        seed: int = 0,
        traced: bool = True,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        """Run inference for a single prompt. Convenience method for inference server."""
        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt],
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            traced=traced,
            vae_traced=vae_traced,
            encoder_traced=encoder_traced,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

    def prepare_encoder(self) -> None:
        """Prepare encoder for inference."""
        if not self._text_encoder.encoder_loaded():
            self._deallocate_transformers(self.encoder_submesh_idx)
            with reshape_device(self.encoder_device, self.encoder_mesh_shape):
                self._text_encoder.reload_encoder_weights()

    def prepare_transformers(self) -> None:
        if not self.transformers[self.encoder_submesh_idx].is_loaded():
            self._text_encoder.deallocate_encoder_weights()
            self._load_transformers(self.encoder_submesh_idx)

        if not self.transformers[self.vae_submesh_idx].is_loaded():
            self._deallocate_vae()
            self._load_transformers(self.vae_submesh_idx)

    def prepare_vae(self) -> None:
        if not self._vae_decoder.is_loaded():
            self._deallocate_transformers(self.vae_submesh_idx)
            with reshape_device(self.vae_device, self.vae_mesh_shape):
                self._reload_vae()

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        with profiler("total", profiler_iteration) if profiler else nullcontext():
            cfg_enabled = cfg_scale > 1
            logger.info("encoding prompts...")

            self.prepare_encoder()

            with profiler("encoder", profiler_iteration) if profiler else nullcontext():
                with reshape_device(self.encoder_device, self.encoder_mesh_shape):
                    prompt_embeds, prompt_mask = self._encode_prompts(
                        prompts=prompts,
                        negative_prompts=negative_prompts,
                        num_images_per_prompt=num_images_per_prompt,
                        cfg_enabled=cfg_enabled,
                        profiler=profiler,
                        profiler_iteration=profiler_iteration,
                        traced=encoder_traced,
                    )
            _, prompt_sequence_length, _ = prompt_embeds.shape

            self.prepare_transformers()

            logger.info("preparing timesteps...")

            mu = _calculate_shift(spatial_sequence_length, 256, 8192, 0.5, 0.9)
            sigmas, _ = schedules.shifted_linear(
                num_inference_steps, shift=math.exp(mu), sigma_small=1 / num_inference_steps
            )
            sigmas, alphas = _stretch_to_terminal(sigmas, terminal=0.02)
            for solver in self._solvers:
                solver.set_schedule(sigmas, alphas)
            timesteps = [s * 1000 for s in sigmas[:-1]]

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [
                transformer_batch_size,
                self._num_channels_latents,
                self._height // self._vae_scale_factor,
                self._width // self._vae_scale_factor,
            ]
            # We let randn generate a permuted latent tensor in float32, so that the generated noise
            # matches the reference implementation.
            latents = self.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))

            p = self._patch_size
            img_shapes = [[(1, latents_height // p, latents_width // p)]] * transformer_batch_size
            txt_seq_lens = [prompt_sequence_length] * transformer_batch_size
            spatial_rope, prompt_rope = self._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")

            spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
            spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
            prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
            prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

            tt_prompt_embeds_list = distribute_cfg(prompt_embeds, devices=self._submesh_devices, on_host=False)
            tt_latents_step_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []

            for i, submesh_device in enumerate(self._submesh_devices):
                tt_latents_step_list.append(
                    tensor.from_torch(latents, device=submesh_device, mesh_axes=[None, sp_axis, None])
                )
                tt_spatial_rope_cos_list.append(
                    tensor.from_torch(spatial_rope_cos, device=submesh_device, mesh_axes=[sp_axis, None])
                )
                tt_spatial_rope_sin_list.append(
                    tensor.from_torch(spatial_rope_sin, device=submesh_device, mesh_axes=[sp_axis, None])
                )
                tt_prompt_rope_cos_list.append(tensor.from_torch(prompt_rope_cos, device=submesh_device))
                tt_prompt_rope_sin_list.append(tensor.from_torch(prompt_rope_sin, device=submesh_device))

            logger.info("denoising...")

            with profiler("denoising", profiler_iteration) if profiler else nullcontext():
                for i, t in enumerate(tqdm.tqdm(timesteps)):
                    with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                        # Allocation on device is fine, because timesteps are not used after
                        # trace execution, and can be overwritten during trace execution.
                        tt_timestep_list = [
                            ttnn.full(
                                [1, 1],
                                fill_value=t,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.float32,
                                device=submesh_device,
                            )
                            for submesh_device in self._submesh_devices
                        ]

                        reuse_tensors = i > 0 and traced

                        tt_latents_step_list = self._step(
                            step_index=i,
                            timestep=tt_timestep_list,
                            latents=tt_latents_step_list,
                            prompt_embeds=None if reuse_tensors else tt_prompt_embeds_list,
                            spatial_rope_cos=None if reuse_tensors else tt_spatial_rope_cos_list,
                            spatial_rope_sin=None if reuse_tensors else tt_spatial_rope_sin_list,
                            prompt_rope_cos=None if reuse_tensors else tt_prompt_rope_cos_list,
                            prompt_rope_sin=None if reuse_tensors else tt_prompt_rope_sin_list,
                            spatial_sequence_length=spatial_sequence_length,
                            prompt_sequence_length=prompt_sequence_length,
                            cfg_scale=cfg_scale,
                            cfg_enabled=cfg_enabled,
                            profiler=profiler,
                            profiler_iteration=profiler_iteration,
                            traced=traced,
                        )

            logger.info("decoding image...")

            with profiler("vae", profiler_iteration) if profiler else nullcontext():
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
                    height=latents_height,
                    width=latents_width,
                )

                torch_latents = torch_latents / self._latents_scaling + self._latents_shift

                if self._vae_decoder is None:
                    torch_latents = torch_latents.permute(0, 3, 1, 2).unsqueeze(2)
                    with torch.no_grad():
                        decoded_output = self._torch_vae.decode(torch_latents).sample[:, :, 0]
                else:
                    self.prepare_vae()

                    with reshape_device(self.vae_device, self.vae_mesh_shape):
                        tt_latents, logical_h = self._vae_decoder.prepare_input(torch_latents)
                        vae_decode = self._vae_decoder_tracer if vae_traced else self._vae_decoder.forward
                        tt_decoded_output, logical_h = vae_decode(tt_latents, logical_h)
                        decoded_output = self._vae_decoder.postprocess_output(tt_decoded_output, logical_h)

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        timestep: ttnn.Tensor,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor | None,
        spatial_rope_cos: ttnn.Tensor | None,
        spatial_rope_sin: ttnn.Tensor | None,
        prompt_rope_cos: ttnn.Tensor | None,
        prompt_rope_sin: ttnn.Tensor | None,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        submesh_id: int,
    ) -> ttnn.Tensor:
        latent_input = (
            ttnn.concat([latent, latent]) if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1 else latent
        )

        noise_pred = self.transformers[submesh_id].forward(
            spatial=latent_input,
            prompt=prompt,
            timestep=timestep,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

        # Make latents an output, because inputs are copied to the trace region before executing a
        # trace and might be overwritten during execution.
        return latent, noise_pred

    def _step(
        self,
        *,
        step_index: int,
        timestep: list[ttnn.Tensor],
        latents: list[ttnn.Tensor],
        prompt_embeds: list[ttnn.Tensor] | None,
        spatial_rope_cos: list[ttnn.Tensor] | None,
        spatial_rope_sin: list[ttnn.Tensor] | None,
        prompt_rope_cos: list[ttnn.Tensor] | None,
        prompt_rope_sin: list[ttnn.Tensor] | None,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        cfg_enabled: bool,
        cfg_scale: float,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
        traced: bool,
    ) -> list[ttnn.Tensor]:
        latents_out = []
        noise_pred_list = []

        for submesh_id in range(len(self._submesh_devices)):
            inner = self._step_inner_tracers[submesh_id] if traced else self._step_inner

            latent, noise_pred = inner(
                cfg_enabled=cfg_enabled,
                timestep=timestep[submesh_id],
                latent=latents[submesh_id],
                prompt=prompt_embeds[submesh_id] if prompt_embeds is not None else None,
                spatial_rope_cos=spatial_rope_cos[submesh_id] if spatial_rope_cos is not None else None,
                spatial_rope_sin=spatial_rope_sin[submesh_id] if spatial_rope_sin is not None else None,
                prompt_rope_cos=prompt_rope_cos[submesh_id] if prompt_rope_cos is not None else None,
                prompt_rope_sin=prompt_rope_sin[submesh_id] if prompt_rope_sin is not None else None,
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=prompt_sequence_length,
                submesh_id=submesh_id,
            )

            latents_out.append(latent)
            noise_pred_list.append(noise_pred)

        if cfg_enabled:
            for i in range(len(noise_pred_list)):
                noise_pred_list[i] = self._cfg_combiner.combine(noise_pred_list[i], cfg_scale)

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)
            latents_out[submesh_id] = self._solvers[submesh_id].step(
                step=step_index, latent=latents_out[submesh_id], velocity_pred=noise_pred_list[submesh_id]
            )

        return latents_out

    def _encode_prompts(
        self,
        *,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_images_per_prompt: int,
        cfg_enabled: bool,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
        traced: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        # TODO: necessary?
        negative_prompts = [x if x is not None else "" for x in negative_prompts]

        if cfg_enabled:
            prompts = negative_prompts + prompts

        prompts = [PROMPT_TEMPLATE.format(e) for e in prompts]

        embeds, mask = self._text_encoder.encode(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=512 + PROMPT_DROP_IDX,
            enable_tracing=traced,
        )

        embeds[torch.logical_not(mask)] = 0.0

        return embeds[:, PROMPT_DROP_IDX:], mask[:, PROMPT_DROP_IDX:]

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _stretch_to_terminal(sigmas: list[float], terminal: float) -> tuple[list[float], list[float]]:
    inner = sigmas[:-1]
    one_minus = [1 - s for s in inner]
    scale = one_minus[-1] / (1 - terminal)
    sigmas = [1 - om / scale for om in one_minus]
    sigmas.append(0.0)
    alphas = [1 - s for s in sigmas]
    return sigmas, alphas
