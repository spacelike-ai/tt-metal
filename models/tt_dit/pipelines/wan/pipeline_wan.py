# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from __future__ import annotations

import html
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import ftfy
import regex as re
import torch
import tqdm
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from loguru import logger
from transformers import AutoTokenizer, UMT5EncoderModel

import ttnn

from ...encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from ...models.transformers.wan2_2.transformer_wan import WanCheckpoint, WanTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from ...pipelines.pipeline_api import PipelineAPIMixin
from ...solvers import UniPCSolver, UniPCVariant, schedules
from ...utils import cache, tensor
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import (
    fast_device_to_host,
    float32_tensor,
    float_to_uint8,
    float_to_unit_range,
    typed_tensor_2dshard,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_UNSET = object()  # sentinel for "use config default" in WanPipelineConfig.default

_DEFAULT_CHECKPOINT = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 1,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 4,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": True,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (1, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (2, 2): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (2, 4): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": False,
        "vae_t_chunk_size": 7,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,  # full-T
    },
    (4, 32): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,
        "sdpa_t_fracture_w_only": True,
    },
}


@dataclass(frozen=True, kw_only=True)
class WanPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    flow_shift: float
    boundary_ratio: float | None
    expand_timesteps: bool
    dynamic_load: bool
    is_fsdp: bool
    model_type: str
    vae_dtype: ttnn.DataType
    vae_t_chunk_size: int | None
    sdpa_t_fracture_w_only: bool

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
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        flow_shift: float = 12.0,
        boundary_ratio: float | None = 0.875,
        expand_timesteps: bool = False,
        dynamic_load: bool | None = None,
        is_fsdp: bool | None = None,
        model_type: str = "t2v",
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        vae_t_chunk_size: object = _UNSET,
        sdpa_t_fracture_w_only: bool | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> WanPipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None or vae_parallel_config is None or encoder_parallel_config is None:
            sp_axis = preset["sp_axis"]
            tp_axis = preset["tp_axis"]
            h_factor = tuple(mesh_shape)[tp_axis]
            w_factor = tuple(mesh_shape)[sp_axis]
            if dit_parallel_config is None:
                dit_parallel_config = DiTParallelConfig.from_tuples(
                    cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis)
                )
            if vae_parallel_config is None:
                vae_parallel_config = VaeHWParallelConfig.from_tuples(
                    height=(h_factor, tp_axis), width=(w_factor, sp_axis)
                )
            if encoder_parallel_config is None:
                encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

        if vae_t_chunk_size is _UNSET:
            vae_t_chunk_size = preset.get("vae_t_chunk_size", 1)

        return cls(
            topology=topology if topology is not None else preset["topology"],
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            flow_shift=flow_shift,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            dynamic_load=dynamic_load if dynamic_load is not None else preset["dynamic_load"],
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            model_type=model_type,
            vae_dtype=vae_dtype,
            vae_t_chunk_size=vae_t_chunk_size,
            sdpa_t_fracture_w_only=(
                sdpa_t_fracture_w_only
                if sdpa_t_fracture_w_only is not None
                else preset.get("sdpa_t_fracture_w_only", False)
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


@dataclass
class TransformerState:
    model: WanTransformer3DModel
    checkpoint: WanCheckpoint
    guidance_scale: float
    prompt_buffer: object = field(default=None)
    negative_prompt_buffer: object = field(default=None)


class WanPipeline(PipelineAPIMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    Args:
        mesh_device (`ttnn.MeshDevice`):
            The TT mesh device to run inference on.
        parallel_config (`DiTParallelConfig`):
            Parallelism configuration for the transformer.
        vae_parallel_config (`VaeHWParallelConfig`):
            Parallelism configuration for the VAE decoder.
        encoder_parallel_config (`EncoderParallelConfig`):
            Parallelism configuration for the text encoder.
        num_links (`int`):
            Number of links to use for CCL operations.
        checkpoint_name (`str`, *optional*, defaults to `"Wan-AI/Wan2.2-T2V-A14B-Diffusers"`):
            HuggingFace Hub repo ID to load model weights from.
        scheduler (`FlowMatchEulerDiscreteScheduler`, *optional*):
            Scheduler to use for denoising. Defaults to `UniPCMultistepScheduler` loaded from the checkpoint.
        boundary_ratio (`float`, *optional*, defaults to `0.875`):
            Ratio of total timesteps used as the boundary for switching between the two transformers in two-stage
            denoising. `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
        expand_timesteps (`bool`, *optional*, defaults to `False`):
            Whether to expand timesteps per-token for image-to-video (Wan2.2 TI2V) conditioning.
        dynamic_load (`bool`, *optional*, defaults to `False`):
            If `True`, model components are loaded/offloaded to device dynamically during inference.
        topology (`ttnn.Topology`, *optional*, defaults to `ttnn.Topology.Linear`):
            Fabric topology to use for CCL operations across devices.
        is_fsdp (`bool`, *optional*, defaults to `True`):
            Whether to use fully-sharded data parallelism for transformer weights.
        model_type (`str`, *optional*, defaults to `"t2v"`):
            Model variant identifier (e.g. `"t2v"` for text-to-video).
        vae_dtype (`ttnn.DataType`, *optional*, defaults to `ttnn.bfloat16`):
            Data type to use for VAE inference.
        vae_use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache VAE convolution programs across calls.
        sdpa_t_fracture_w_only (`bool`, *optional*, defaults to `False`):
            Whether to fracture SDPA only along the width dimension for temporal attention.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        pipeline_class: type[WanPipeline] | None = None,
    ) -> WanPipeline:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
        )
        pipeline_class_ = pipeline_class or cls
        return pipeline_class_(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
    ) -> None:
        self.checkpoint_name = config.checkpoint_name
        self.model_type = config.model_type
        self.vae_t_chunk_size = config.vae_t_chunk_size
        self._cfg_enabled = config.cfg_enabled
        self._height = config.height
        self._width = config.width
        self._num_frames = config.num_frames

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.checkpoint_name, subfolder="tokenizer", trust_remote_code=True
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            config.checkpoint_name, subfolder="text_encoder", trust_remote_code=True
        )
        self.vae = AutoencoderKLWan.from_pretrained(config.checkpoint_name, subfolder="vae", trust_remote_code=True)
        self._flow_shift = config.flow_shift
        self._checkpoint = WanCheckpoint(config.checkpoint_name, subfolder="transformer")
        self._checkpoint_2 = WanCheckpoint(config.checkpoint_name, subfolder="transformer_2")

        self.dit_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=config.topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=ttnn.Topology.Linear,  # NOTE: VAE always uses Linear topology. TODO: enable ring if given.
        )

        # See what options we have for topology. We should consider reusing CCL managers
        self.encoder_ccl_manager = self.vae_ccl_manager

        self.is_fsdp = config.is_fsdp
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.encoder_parallel_config = config.encoder_parallel_config
        self.mesh_device = device
        self.dynamic_load = config.dynamic_load

        # Load TT text encoder
        umt5_config = UMT5Config(
            vocab_size=self.text_encoder.config.vocab_size,
            embed_dim=self.text_encoder.config.d_model,
            ff_dim=self.text_encoder.config.d_ff,
            kv_dim=self.text_encoder.config.d_kv,
            num_heads=self.text_encoder.config.num_heads,
            num_hidden_layers=self.text_encoder.config.num_layers,
            max_prompt_length=512,  # TODO: Consider removing
            layer_norm_eps=self.text_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=self.text_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=self.text_encoder.config.relative_attention_max_distance,
        )

        self.tt_umt5_encoder = UMT5Encoder(
            config=umt5_config,
            mesh_device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            parallel_config=self.encoder_parallel_config,
        )

        self.transformer = self._checkpoint.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
        )

        self.transformer_2 = self._checkpoint_2.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
        )

        full_latent_T = (config.num_frames - 1) // 4 + 1
        decoder_t_chunk_size = full_latent_T if config.vae_t_chunk_size is None else config.vae_t_chunk_size

        self.tt_vae = WanDecoder(
            base_dim=self.vae.config.base_dim,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            out_channels=self.vae.config.out_channels,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
            dtype=config.vae_dtype,
            sdpa_t_fracture_w_only=config.sdpa_t_fracture_w_only,
            target_height=config.height,
            target_width=config.width,
            t_chunk_size=decoder_t_chunk_size,
            cached=(config.vae_t_chunk_size is not None),
        )

        self.transformer_states = [
            TransformerState(self.transformer, self._checkpoint, guidance_scale=4.0),
            TransformerState(self.transformer_2, self._checkpoint_2, guidance_scale=3.0),
        ]

        self._solver = UniPCSolver(order=2, variant=UniPCVariant.B2)

        if self.dynamic_load:
            # setup models that cannot be loaded together with the corresponding model.
            # The module loading utility will take care of the necessary unloading.
            if ttnn.device.is_blackhole():
                self.transformer.register_coresident_exclusions(self.transformer_2)
                self.transformer_2.register_coresident_exclusions(self.transformer)
            else:
                # WH T3K has tighter DRAM — include VAE in the unload chain so
                # transformers and VAE never coexist in DRAM across pipeline runs.
                self.transformer.register_coresident_exclusions(self.transformer_2, self.tt_vae)
                self.transformer_2.register_coresident_exclusions(self.transformer, self.tt_vae)
                self.tt_vae.register_coresident_exclusions(self.transformer, self.transformer_2)

        # Cache warmup: Load in reverse order of use to ensure the earliest required models stay loaded before call.
        self._prepare_transformer(1)
        self._prepare_transformer(0)
        self._prepare_text_encoder()
        self._prepare_vae()

        self._boundary_ratio = config.boundary_ratio
        self._expand_timesteps = config.expand_timesteps
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Precompute VAE latent normalization constants (avoids recreating every call)
        self._vae_latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        self._vae_latents_std = torch.tensor(self.vae.config.latents_std, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        # TODO: Reset buffers for change in resolution. Also reinitialize trace
        logger.info("Pipeline allocation run...")
        self(
            prompts=["warmup"],
            num_inference_steps=2,
            guidance_scale=2 if config.cfg_enabled else 1,
            guidance_scale_2=2 if config.cfg_enabled else 1,
        )

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        prompt_1BLP = tt_model.prepare_text_conditioning(prompt_embeds)
        if buffer is None or not traced:
            buffer = prompt_1BLP
        else:
            ttnn.copy(prompt_1BLP, buffer)
        return buffer

    def _prepare_text_encoder(self):
        cache.load_model(
            self.tt_umt5_encoder,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="text_encoder",
            parallel_config=self.encoder_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.text_encoder.state_dict(),
        )

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        state.checkpoint.load(
            state.model,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
        )

    def _prepare_vae(self):
        blocking_key = conv3d_blocking_hash(self.tt_vae)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache.load_model(
            self.tt_vae,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.vae.state_dict(),
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        # NOTE: while the reference impl does not pad to max_sequence_length, for some reason this seems to be necessary for correctness in this pipeline.
        # TODO: investigate
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # Shard on batch dimension. On non TP axis
        dims = [None, None]
        DP_axis = 1 - self.parallel_config.tensor_parallel.mesh_axis
        dims[DP_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims)
        tt_prompt = ttnn.from_torch(
            text_input_ids,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
        )

        tt_mask = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
        )

        prompt_embeds = self.tt_umt5_encoder(tt_prompt, attention_mask=tt_mask)[-1]

        # use the mask to zero out the padding tokens.
        prompt_embeds = prompt_embeds * ttnn.unsqueeze(tt_mask, -1)

        prompt_embeds = self.encoder_ccl_manager.all_gather(
            prompt_embeds, dim=0, mesh_axis=DP_axis, use_hyperparams=True
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = ttnn.repeat(prompt_embeds, (1, num_videos_per_prompt, 1))
        prompt_embeds_1BLP = ttnn.view(prompt_embeds, (1, batch_size * num_videos_per_prompt, seq_len, -1))
        return prompt_embeds_1BLP

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Batch encodes the prompt and negative prompt into text encoder hidden states..

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Setup batching variables
        all_input_prompts = []
        pos_prompt_end_idx = 0
        neg_prompt_end_idx = 0

        if prompt_embeds is None:
            all_input_prompts += prompt
            pos_prompt_end_idx = batch_size * num_videos_per_prompt

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            all_input_prompts += negative_prompt
            neg_prompt_end_idx = pos_prompt_end_idx + batch_size * num_videos_per_prompt

        # Add data to pad for size of device on mesh axis to ensure proper shadding on batch dimension.
        total_prompts = len(all_input_prompts)
        num_devices = self.mesh_device.shape[1 - self.parallel_config.tensor_parallel.mesh_axis]

        # Pad batch list of prompts to ensure proper sharding on batch dimension.
        all_input_prompts += [" "] * ((num_devices - (total_prompts % num_devices)) % num_devices)
        all_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=all_input_prompts,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # When CFG is enabled, we should be able to leave the shards on device.
        prompt_embeds = all_prompt_embeds[:, :pos_prompt_end_idx] if pos_prompt_end_idx > 0 else prompt_embeds
        negative_prompt_embeds = (
            all_prompt_embeds[:, pos_prompt_end_idx:neg_prompt_end_idx]
            if neg_prompt_end_idx > 0
            else negative_prompt_embeds
        )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self._boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    def get_model_input(self, latents, cond_latents):
        """
        Adapter function to enable I2V. For base T2V, just return the latents.
        """
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt=None,  # unused in T2V
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype), None

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = torch.randn(shape, dtype=torch.float32, device=torch.device(device))
        return latents, None

    @property
    def do_classifier_free_guidance(self):
        return self.transformer_states[0].guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        image_prompt=None,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_scale_2: Optional[float] = 3.0,
        num_videos_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        max_sequence_length: int = 512,
        traced: bool = False,
        on_event: PipelineEventCallback | None = None,
    ):
        on_event = on_event if on_event is not None else null_callback
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            seed (`int`, *optional*):
                A random generator seed to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `seed`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            The generated video frames.
        """

        negative_prompts = (
            negative_prompts if negative_prompts is not None else [self.DEFAULT_NEGATIVE_PROMPT] * len(prompts)
        )
        height = self._height
        width = self._width
        num_frames = self._num_frames

        if guidance_scale > 1 and not self._cfg_enabled:
            msg = "guidance_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)
        if guidance_scale_2 is not None and guidance_scale_2 > 1 and not self._cfg_enabled:
            msg = "guidance_scale_2 > 1 requires CFG to be enabled"
            raise ValueError(msg)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompts,
            negative_prompts,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self._boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self.transformer_states[0].guidance_scale = guidance_scale
        self.transformer_states[1].guidance_scale = guidance_scale_2

        # device = self._execution_device
        device = "cpu"

        # 2. Define call parameters
        if prompts is not None:
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        on_event(SectionStart("encoder"))
        with nullcontext():
            self._prepare_text_encoder()
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompts,
                negative_prompt=negative_prompts,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        on_event(SectionEnd("encoder"))

        # 4. Prepare schedule
        (sigmas, alphas) = schedules.shifted_linear(
            num_inference_steps, shift=self._flow_shift, sigma_small=0.001 + 0.999 / num_inference_steps
        )
        sigmas[0] -= 1e-6

        # diffusers uses float32 for the schedule
        sigmas = torch.tensor(sigmas, dtype=torch.float32).tolist()
        alphas = (1 - torch.tensor(sigmas, dtype=torch.float32)).tolist()

        timesteps = torch.tensor([s * 1000 for s in sigmas[:-1]])
        self._solver.set_schedule(sigmas, alphas)

        # 5. Prepare latent variables
        if seed is not None:
            torch.manual_seed(seed)

        on_event(SectionStart("prepare_latents"))
        with nullcontext():
            latents, cond_latents = self.prepare_latents(
                batch_size=batch_size * num_videos_per_prompt,
                image_prompt=image_prompt,
                num_channels_latents=self.vae.config.z_dim,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                latents=latents,
            )
        on_event(SectionEnd("prepare_latents"))

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        self._num_timesteps = len(timesteps)

        if self._boundary_ratio is not None:
            boundary_timestep = self._boundary_ratio * 1000
        else:
            boundary_timestep = -1  # Always use transformer (no transformer_2)

        on_event(SectionStart("denoising"))

        permuted_latent_tt = None
        rope_args = None

        latent_frames, latent_height, latent_width = latents.shape[2], latents.shape[3], latents.shape[4]
        prepared_prompts = [False, False]

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                warmup_t2 = i == 1 and len(timesteps) == 2  # Ensure transformer_2 is also warmed up

                # 0=> wan2.1 or high-noise stage in wan2.2 (transformer) | 1=> low-noise stage in wan2.2 (transformer_2)
                transformer_idx = 0 if (t >= boundary_timestep) and not warmup_t2 else 1
                self._prepare_transformer(transformer_idx)
                ts = self.transformer_states[transformer_idx]
                if not prepared_prompts[transformer_idx]:
                    # Prepare the text conditioning in an optional persistent buffer depending on traced
                    ts.prompt_buffer = self.prepare_text_conditioning(ts.model, prompt_embeds, ts.prompt_buffer, traced)
                    ts.negative_prompt_buffer = self.prepare_text_conditioning(
                        ts.model, negative_prompt_embeds, ts.negative_prompt_buffer, traced
                    )
                    prepared_prompts[transformer_idx] = True

                if permuted_latent_tt is None:
                    # First iteration, preprocess spatial input and prepare rope features
                    permuted_latent, patchified_seqlen = ts.model.preprocess_spatial_input_host(latents)

                    if cond_latents is not None:
                        cond_latents, _ = ts.model.preprocess_spatial_input_host(cond_latents)

                    rope_cos_1HND, rope_sin_1HND, trans_mat = ts.model.get_rope_features(latents)
                    rope_args = {
                        "rope_cos_1HND": rope_cos_1HND,
                        "rope_sin_1HND": rope_sin_1HND,
                        "trans_mat": trans_mat,
                    }

                    sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis
                    permuted_latent_tt = tensor.from_torch(
                        permuted_latent,
                        device=self.mesh_device,
                        mesh_axes=[None, None, sp_axis, None],
                        dtype=ttnn.float32,
                    )

                if self._expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                permuted_model_input = self.get_model_input(permuted_latent_tt, cond_latents)
                permuted_model_input = ttnn.typecast(permuted_model_input, ttnn.bfloat16)

                assert timestep.ndim == 1, "Wan2.2-T2V/I2V requires a 1D timestep tensor"
                timestep = float32_tensor(
                    timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=(None if traced else self.mesh_device)
                )

                permuted_noise_pred_tt = ts.model.combined_step(
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    spatial_1BNI=permuted_model_input,
                    prompt_1BLP=ts.prompt_buffer,
                    negative_prompt_1BLP=ts.negative_prompt_buffer,
                    N=patchified_seqlen,
                    timestep=timestep,
                    **rope_args,
                    guidance_scale=ts.guidance_scale,
                    traced=traced,
                    gather_output=False,
                )

                permuted_latent_tt = self._solver.step(
                    step=i,
                    latent=permuted_latent_tt,
                    velocity_pred=permuted_noise_pred_tt,
                )

                progress_bar.update()

        self._current_timestep = None

        sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis
        permuted_latent_tt = ts.model.ccl_manager.all_gather_persistent_buffer(
            permuted_latent_tt, dim=2, mesh_axis=sp_axis
        )
        permuted_latent = ttnn.to_torch(ttnn.get_device_tensors(permuted_latent_tt)[0])

        # Postprocess spatial output
        latents = ts.model.postprocess_spatial_output_host(
            permuted_latent, F=latent_frames, H=latent_height, W=latent_width, N=patchified_seqlen
        )

        on_event(SectionEnd("denoising"))
        on_event(SectionStart("vae"))

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = latents * self._vae_latents_std + self._vae_latents_mean

            tt_latents_BTHWC, logical_h = self.tt_vae.prepare_input(latents)

            tt_latents_BTHWC = typed_tensor_2dshard(
                tt_latents_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
                dtype=self.tt_vae.dtype,
            )
            self._prepare_vae()
            tt_video_BCTHW, new_logical_h = self.tt_vae(tt_latents_BTHWC, logical_h, t_chunk_size=self.vae_t_chunk_size)

            concat_dims = [None, None]
            concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
            concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
            d2h_permute = (0, 2, 3, 4, 1) if output_type in ("np", "uint8") else None

            if output_type == "uint8":
                pre_fn = float_to_uint8
            elif output_type == "np":
                pre_fn = float_to_unit_range
            else:
                pre_fn = None

            video_torch = fast_device_to_host(
                tt_video_BCTHW,
                self.mesh_device,
                concat_dims,
                ccl_manager=self.vae_ccl_manager,
                pre_transfer_fn=pre_fn,
                permute=d2h_permute,
            )

            if d2h_permute is not None:
                # Output is (B, T, H, W, C) — trim height in dim 2.
                video_torch = video_torch[:, :, :new_logical_h, :, :]
            else:
                # Output is (B, C, T, H, W) — trim height in dim 3.
                video_torch = video_torch[:, :, :, :new_logical_h, :]

            if output_type == "uint8":
                video = video_torch.numpy()
            elif output_type == "np":
                video = video_torch.float().numpy()
            else:
                video = self.video_processor.postprocess_video(video_torch, output_type=output_type)
        else:
            video = latents

        on_event(SectionEnd("vae"))

        return video

    def synchronize_devices(self):
        ttnn.synchronize_device(self.mesh_device)

    def release_traces(self):
        for model in (self.transformer, self.transformer_2):
            tracer = WanTransformer3DModel.combined_step._tracers.get(model)
            if tracer is not None:
                tracer.release_trace()
