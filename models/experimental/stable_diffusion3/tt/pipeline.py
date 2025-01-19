from __future__ import annotations

import time

import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import ttnn

from .transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters


class TtStableDiffusion3Pipeline:
    def __init__(self, *, checkpoint: str, device: ttnn.Device) -> None:
        self._device = device

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer_2")
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(checkpoint, subfolder="tokenizer_3")
        self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder")
        self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder_2")
        self._text_encoder_3 = T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder_3")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")
        self._vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
        torch_transformer = SD3Transformer2DModel.from_pretrained(
            checkpoint, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, CLIPTokenizer)
        assert isinstance(self._tokenizer_3, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_2, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_3, T5EncoderModel)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._vae, AutoencoderKL)
        assert isinstance(torch_transformer, SD3Transformer2DModel)

        logger.info("creating tt transformer...")
        parameters = TtSD3Transformer2DModelParameters.from_torch(torch_transformer.state_dict(), device=device)
        self._tt_transformer = TtSD3Transformer2DModel(
            parameters, num_attention_heads=torch_transformer.config.num_attention_heads
        )
        self._num_channels_latents = torch_transformer.config.in_channels
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim
        logger.info("done")

        self._block_out_channels = self._vae.config.block_out_channels
        self._vae_scaling_factor = self._vae.config.scaling_factor
        self._vae_shift_factor = self._vae.config.shift_factor

        self._vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

    def __call__(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str],
        negative_prompt_2: list[str],
        negative_prompt_3: list[str],
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        max_t5_sequence_length: int = 256,
        seed: int = 0,
    ) -> None:
        assert height % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert width % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert max_t5_sequence_length <= 512

        batch_size = len(prompt_1)
        do_classifier_free_guidance = guidance_scale > 1
        tokenizer_max_length = self._tokenizer_1.model_max_length
        latents_shape = (
            batch_size * num_images_per_prompt,
            self._num_channels_latents,
            height // self._vae_scale_factor,
            width // self._vae_scale_factor,
        )

        logger.info("warm up")
        prompt_embeds = torch.randn([2, 333, 4096])
        pooled_prompt_embeds = torch.randn([2, 2048])
        latents = torch.randn(latents_shape)
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        timestep = torch.tensor([500]).expand(latent_model_input.shape[0])
        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_latent_model_input = ttnn.from_torch(
            latent_model_input.permute([0, 2, 3, 1]),  # BCYX -> BYXC
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            dtype=ttnn.bfloat16,
        )
        tt_timestep = ttnn.from_torch(
            timestep.unsqueeze(1),
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            dtype=ttnn.float32,
        )
        self._tt_transformer(
            spatial=tt_latent_model_input,
            prompt_embed=tt_prompt_embeds,
            pooled_projection=tt_pooled_prompt_embeds,
            timestep=tt_timestep,
        )

        start_time = time.time()

        logger.info("encode prompts")

        logger.info("prompt 1")
        prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=prompt_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )

        logger.info("prompt 2")
        prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_2,
            text_encoder=self._text_encoder_2,
            tokenizer_max_length=tokenizer_max_length,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        logger.info("prompt 3")
        t5_prompt_embed = _get_t5_prompt_embeds(
            prompt=prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_t5_sequence_length,
            tokenizer=self._tokenizer_3,
            text_encoder=self._text_encoder_3,
            tokenizer_max_length=tokenizer_max_length,
            joint_attention_dim=self._joint_attention_dim,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance:
            logger.info("negative prompt 1")
            negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
            )
            logger.info("negative prompt 2")
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            logger.info("negative prompt 3")
            t5_negative_prompt_embed = _get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_t5_sequence_length,
                tokenizer=self._tokenizer_3,
                text_encoder=self._text_encoder_3,
                tokenizer_max_length=tokenizer_max_length,
                joint_attention_dim=self._joint_attention_dim,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (
                    0,
                    t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
                ),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        logger.info("prepare timesteps")

        self._scheduler.set_timesteps(num_inference_steps)
        timesteps = self._scheduler.timesteps

        logger.info("prepare latents")

        torch.manual_seed(seed)
        latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)

        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        logger.info("denoising loop")

        for t in tqdm.tqdm(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            timestep = t.expand(latent_model_input.shape[0])

            tt_latent_model_input = ttnn.from_torch(
                latent_model_input.permute([0, 2, 3, 1]),  # BCYX -> BYXC
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
                dtype=ttnn.bfloat16,
            )
            tt_timestep = ttnn.from_torch(
                timestep.unsqueeze(1),
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
                dtype=ttnn.float32,
            )

            tt_noise_pred = self._tt_transformer(
                spatial=tt_latent_model_input,
                prompt_embed=tt_prompt_embeds,
                pooled_projection=tt_pooled_prompt_embeds,
                timestep=tt_timestep,
            )
            noise_pred = ttnn.to_torch(tt_noise_pred).to(dtype=torch.float32)

            noise_pred = _reshape_noise_pred(
                noise_pred,
                height=latent_model_input.shape[-2],
                width=latent_model_input.shape[-1],
                patch_size=self._tt_transformer.patch_size,
            )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self._scheduler.step(
                noise_pred,  # type: ignore  # noqa: PGH003
                t,  # type: ignore  # noqa: PGH003
                latents,  # type: ignore  # noqa: PGH003
                return_dict=False,
            )[0]

        latents = (latents / self._vae_scaling_factor) + self._vae_shift_factor

        with torch.no_grad():
            image = self._vae.decode(latents, return_dict=False)[0]
            image = self._image_processor.postprocess(image, output_type="pt")
            assert isinstance(image, torch.Tensor)

        pil_images = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
        pil_image = pil_images[0]

        runtime = time.time() - start_time
        logger.info(f"runtime: {runtime}")

        pil_image.save("sd3.png")


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    clip_skip: int | None = None,
    device: torch.device | None = None,
    num_images_per_prompt: int,
    prompt: list[str],
    text_encoder: CLIPTextModelWithProjection,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
):
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    joint_attention_dim: int,
    max_sequence_length: int,
    num_images_per_prompt: int,
    text_encoder: T5EncoderModel | None,
    tokenizer_max_length: int,
    tokenizer: T5TokenizerFast,
) -> torch.Tensor:
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None:
        return torch.zeros(
            (
                batch_size * num_images_per_prompt,
                tokenizer_max_length,
                joint_attention_dim,
            ),
            device=device,
            dtype=dtype,
        )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _reshape_noise_pred(
    noise_pred: torch.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    patch_count_y = height // patch_size
    patch_count_x = width // patch_size

    shape1 = (
        noise_pred.shape[0],
        patch_count_y,
        patch_count_x,
        patch_size,
        patch_size,
        -1,
    )

    shape2 = (
        noise_pred.shape[0],
        -1,
        patch_count_y * patch_size,
        patch_count_x * patch_size,
    )

    noise_pred = noise_pred.reshape(shape1)
    noise_pred = torch.einsum("nhwpqc->nchpwq", noise_pred)
    return noise_pred.reshape(shape2)
