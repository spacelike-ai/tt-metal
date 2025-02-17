# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-FileCopyrightText: Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ..reference.transformer import FluxTransformer2DModel
from .t5_encoder import TtT5Encoder, TtT5EncoderParameters
from .transformer import TtFluxTransformer2DModel, TtFluxTransformer2DModelParameters


class TtFluxPipeline:
    def __init__(self, *, checkpoint: str, device: ttnn.Device) -> None:
        self._device = device

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self._tokenizer_2 = T5TokenizerFast.from_pretrained(checkpoint, subfolder="tokenizer_2")
        self._text_encoder_1 = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
        # torch_text_encoder_2 = T5EncoderModel.from_pretrained(
        #     checkpoint, subfolder="text_encoder_2"
        # )  # TODO: change to bfloat16!
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")
        self._vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
        torch_transformer = FluxTransformer2DModel.from_pretrained(
            checkpoint,
            subfolder="transformer",
            torch_dtype=torch.float32,  # TODO: change to bfloat16!
        )

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModel)
        # assert isinstance(torch_text_encoder_2, T5EncoderModel)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._vae, AutoencoderKL)
        # assert isinstance(torch_transformer, FluxTransformer2DModel)

        self._text_encoder_1.eval()
        self._vae.eval()
        torch_transformer.eval()

        logger.info("creating TT-NN transformer...")

        # parameters = TtFluxTransformer2DModelParameters.from_torch(
        #     torch_transformer.state_dict(), device=device, dtype=ttnn.bfloat16
        # )
        # self._tt_transformer = TtFluxTransformer2DModel(
        #     parameters, num_attention_heads=torch_transformer.config.num_attention_heads
        # )
        # self._tt_transformer = torch_transformer
        self._transformer = torch_transformer
        self._num_channels_latents = torch_transformer.config.in_channels // 4

        self._vae_scaling_factor = self._vae.config.scaling_factor
        self._vae_shift_factor = self._vae.config.shift_factor

        self._vae_scale_factor = 2 ** len(self._vae.config.block_out_channels)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        logger.info("creating TT-NN text encoder...")

        # parameters = TtT5EncoderParameters.from_torch(
        #     torch_text_encoder_2.state_dict(), device=device, dtype=ttnn.bfloat16
        # )
        # self._text_encoder_2 = TtT5Encoder(
        #     parameters,
        #     num_heads=torch_text_encoder_2.config.num_heads,
        #     relative_attention_num_buckets=torch_text_encoder_2.config.relative_attention_num_buckets,
        #     relative_attention_max_distance=torch_text_encoder_2.config.relative_attention_max_distance,
        #     layer_norm_epsilon=torch_text_encoder_2.config.layer_norm_epsilon,
        # )
        # self._text_encoder_2 = torch_text_encoder_2

    def prepare(
        self,
        *,
        prompt_count: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        max_t5_sequence_length: int = 512,
    ) -> None:
        self._prepared_prompt_count = prompt_count
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_max_t5_sequence_length = max_t5_sequence_length

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=[""] * prompt_count,
            prompt_2=[""] * prompt_count,
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
        )

        # latents_shape = (
        #     prompt_count * num_images_per_prompt,
        #     height // self._vae_scale_factor,
        #     width // self._vae_scale_factor,
        #     self._num_channels_latents,
        # )

        # tt_prompt_embeds = ttnn.from_torch(
        #     prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        # )
        # tt_pooled_prompt_embeds = ttnn.from_torch(
        #     pooled_prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        # )
        # tt_timestep = ttnn.allocate_tensor_on_device([1, 1], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self._device)
        # tt_sigma_difference = ttnn.allocate_tensor_on_device([1, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)
        # tt_latents = ttnn.allocate_tensor_on_device(latents_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)

        # # cache
        # self._step(
        #     timestep=tt_timestep,
        #     latents=tt_latents,
        #     prompt_embeds=tt_prompt_embeds,
        #     pooled_prompt_embeds=tt_pooled_prompt_embeds,
        #     sigma_difference=tt_sigma_difference,
        # )

        # # trace
        # tid = ttnn.begin_trace_capture(self._device)
        # self._step(
        #     timestep=tt_timestep,
        #     latents=tt_latents,
        #     prompt_embeds=tt_prompt_embeds,
        #     pooled_prompt_embeds=tt_pooled_prompt_embeds,
        #     sigma_difference=tt_sigma_difference,
        # )
        # ttnn.end_trace_capture(self._device, tid)

        # self._trace = PipelineTrace(
        #     tid=tid,
        #     spatial_input_output=tt_latents,
        #     prompt_input=tt_prompt_embeds,
        #     pooled_projection_input=tt_pooled_prompt_embeds,
        #     timestep_input=tt_timestep,
        #     sigma_difference_input=tt_sigma_difference,
        # )

    def __call__(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        num_inference_steps: int = 40,
        seed: int | None = None,
    ) -> None:
        start_time = time.time()

        prompt_count = self._prepared_prompt_count
        num_images_per_prompt = self._prepared_num_images_per_prompt
        width = self._prepared_width
        height = self._prepared_height
        max_t5_sequence_length = self._prepared_max_t5_sequence_length

        # assert height % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        # assert width % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert max_t5_sequence_length <= 512
        assert prompt_count == len(prompt_1)
        assert prompt_count == len(prompt_2)

        logger.info("encoding prompts...")

        prompt_encoding_start_time = time.time()
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=prompt_1,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
        )
        prompt_encoding_end_time = time.time()

        # torch.save(prompt_embeds, "prompt_embeds.pt")
        # torch.save(pooled_prompt_embeds, "pooled_prompt_embeds.pt")

        logger.info("preparing timesteps...")

        sigmas = np.linspace(
            1.0,
            1 / num_inference_steps,
            num_inference_steps,
        )

        self._scheduler.set_timesteps(sigmas=sigmas)  # type: ignore  # noqa: PGH003
        timesteps = self._scheduler.timesteps

        logger.info("preparing latents...")

        latents_height = height // self._vae_scale_factor
        latents_width = width // self._vae_scale_factor

        unpacked_latents_shape = [
            prompt_count * num_images_per_prompt,
            self._num_channels_latents,
            latents_height * 2,
            latents_width * 2,
        ]

        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(unpacked_latents_shape, dtype=prompt_embeds.dtype)

        latents = _pack_latents(
            latents,
            prompt_count * num_images_per_prompt,
            self._num_channels_latents,
            latents_height,
            latents_width,
        )

        logger.info("preparing embeddings...")
        text_ids = torch.zeros([prompt_embeds.shape[1], 3], dtype=self._text_encoder_1.dtype)

        image_ids = _latent_image_ids(height=latents_height, width=latents_width).to(dtype=prompt_embeds.dtype)

        ids = torch.cat((text_ids, image_ids), dim=0)
        image_rotary_emb = self._transformer.pos_embed(ids)

        tt_prompt_embeds = ttnn.from_torch(prompt_embeds, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_pooled_prompt_embeds = ttnn.from_torch(pooled_prompt_embeds, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_initial_latents = ttnn.from_torch(latents, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_image_rotary_emb = (
            ttnn.from_torch(image_rotary_emb[0], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(image_rotary_emb[1], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        )

        logger.info("denoising...")
        denoising_start_time = time.time()

        # ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace.prompt_input)
        # ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace.pooled_projection_input)
        # ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace.spatial_input_output)
        # ttnn.copy_host_to_device_tensor(tt_image_rotary_emb[0], self._trace.image_rotary_emb_input[0])
        # ttnn.copy_host_to_device_tensor(tt_image_rotary_emb[1], self._trace.image_rotary_emb_input[1])

        for i, t in enumerate(tqdm.tqdm(timesteps)):
            tt_timestep = ttnn.full([1, 1], fill_value=t, dtype=ttnn.float32)

            sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]
            # tt_sigma_difference = ttnn.full(
            #     [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            # )

            # ttnn.copy_host_to_device_tensor(tt_timestep, self._trace.timestep_input)
            # ttnn.copy_host_to_device_tensor(tt_sigma_difference, self._trace.sigma_difference_input)

            # self._trace.execute()

            # tt_noise_pred = self._transformer(
            #     spatial=self._trace.spatial_input_output,
            #     timestep=self._trace.timestep_input,
            #     pooled_projections=self._trace.pooled_projection_input,
            #     prompt_embed=self._trace.prompt_input,
            #     image_rotary_emb=self._trace.image_rotary_emb_input,
            # )

            # noise_pred = ttnn.to_torch(tt_noise_pred)

            with torch.no_grad():
                noise_pred = self._transformer(
                    spatial=latents,
                    timestep=ttnn.to_torch(tt_timestep).squeeze(0),
                    pooled_projections=pooled_prompt_embeds,
                    prompt_embed=prompt_embeds,
                    image_rotary_emb=image_rotary_emb,
                )

                latents += sigma_difference * noise_pred

            # tt_latents = ttnn.from_torch(latents, device=self._device, layout=ttnn.TILE_LAYOUT)
            # ttnn.copy_host_to_device_tensor(tt_latents, self._trace.spatial_input_output)

        denoising_end_time = time.time()

        logger.info("decoding image...")

        image_decoding_start_time = time.time()

        # latents = ttnn.to_torch(self._trace.spatial_input_output).to(torch.float32)
        latents = _unpack_latents(latents, height, width, self._vae_scale_factor)
        latents = latents / self._vae_scaling_factor + self._vae_shift_factor

        with torch.no_grad():
            image = self._vae.decoder(latents)
            image = self._image_processor.postprocess(image, output_type="pt")
            assert isinstance(image, torch.Tensor)

        image_decoding_end_time = time.time()

        output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        end_time = time.time()

        logger.info(f"prompt encoding duration: {prompt_encoding_end_time - prompt_encoding_start_time}")
        logger.info(f"denoising duration: {denoising_end_time - denoising_start_time}")
        logger.info(f"image decoding duration: {image_decoding_end_time - image_decoding_start_time}")
        logger.info(f"total runtime: {end_time - start_time}")

        return output

    # def _step(
    #     self,
    #     *,
    #     do_classifier_free_guidance: bool,
    #     guidance_scale: float,
    #     latents: ttnn.Tensor,
    #     timestep: ttnn.Tensor,
    #     pooled_prompt_embeds: ttnn.Tensor,
    #     prompt_embeds: ttnn.Tensor,
    #     sigma_difference: ttnn.Tensor,
    # ) -> None:
    #     latent_model_input = ttnn.concat([latents, latents]) if do_classifier_free_guidance else latents
    #     batch_size = latent_model_input.shape[0]

    #     timestep = ttnn.repeat(timestep, ttnn.Shape([batch_size, 1]))
    #     timestep = ttnn.to_layout(timestep, ttnn.TILE_LAYOUT)

    #     noise_pred = self._tt_transformer(
    #         spatial=latent_model_input,
    #         prompt=prompt_embeds,
    #         pooled_projection=pooled_prompt_embeds,
    #         timestep=timestep,
    #     )

    #     noise_pred = _reshape_noise_pred(
    #         noise_pred,
    #         height=latent_model_input.shape[-3],
    #         width=latent_model_input.shape[-2],
    #         patch_size=self._tt_transformer.patch_size,
    #     )

    #     if do_classifier_free_guidance:
    #         split_pos = noise_pred.shape[0] // 2
    #         uncond = noise_pred[0:split_pos]
    #         cond = noise_pred[split_pos:]
    #         noise_pred = uncond + guidance_scale * (cond - uncond)

    #     ttnn.add_(latents, sigma_difference * noise_pred)

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        num_images_per_prompt: int,
        max_t5_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeds = torch.load("prompt_embeds.pt")
        pooled_prompt_embeds = torch.load("pooled_prompt_embeds.pt")

        # tokenizer_max_length = self._tokenizer_1.model_max_length

        # pooled_prompt_embeds = _get_clip_prompt_embeds(
        #     prompt=prompt_1,
        #     num_images_per_prompt=num_images_per_prompt,
        #     tokenizer=self._tokenizer_1,
        #     text_encoder=self._text_encoder_1,
        #     tokenizer_max_length=tokenizer_max_length,
        # )

        # prompt_embeds = _get_t5_prompt_embeds(
        #     device=self._device,
        #     prompt=prompt_2,
        #     num_images_per_prompt=num_images_per_prompt,
        #     max_sequence_length=max_t5_sequence_length,
        #     tokenizer=self._tokenizer_2,
        #     text_encoder=self._text_encoder_2,
        #     # tokenizer_max_length=tokenizer_max_length,
        # )

        return prompt_embeds, pooled_prompt_embeds


@dataclass
class PipelineTrace:
    spatial_input_output: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    tid: int

    def execute(self) -> None:
        ttnn.execute_trace(self.spatial_input_output.device(), self.tid)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _get_clip_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
) -> torch.Tensor:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("CLIP input text was truncated")

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    return prompt_embeds.view(prompt_count * num_images_per_prompt, -1)


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# def _get_t5_prompt_embeds(
#     prompt: list[str],
#     *,
#     torch_device: torch.device | None = None,
#     device: ttnn.Device,
#     joint_attention_dim: int,
#     max_sequence_length: int,
#     num_images_per_prompt: int,
#     text_encoder: TtT5Encoder | None,
#     tokenizer_max_length: int,
#     tokenizer: T5TokenizerFast,
# ) -> torch.Tensor:
#     prompt = [prompt] if isinstance(prompt, str) else prompt
#     prompt_count = len(prompt)

#     if text_encoder is None:
#         return torch.zeros(
#             (
#                 prompt_count * num_images_per_prompt,
#                 tokenizer_max_length,
#                 joint_attention_dim,
#             ),
#             device=torch_device,
#             dtype=torch.bfloat16,
#         )

#     text_inputs = tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=max_sequence_length,
#         truncation=True,
#         add_special_tokens=True,
#         return_tensors="pt",
#     )
#     text_input_ids = text_inputs.input_ids
#     untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

#     if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
#         removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
#         logger.warning(
#             "The following part of your input was truncated because `max_sequence_length` is set to "
#             f" {max_sequence_length} tokens: {removed_text}"
#         )

#     tt_text_input_ids = ttnn.from_torch(text_input_ids, device=device, layout=ttnn.TILE_LAYOUT)
#     tt_prompt_embeds = text_encoder(tt_text_input_ids)
#     prompt_embeds = ttnn.to_torch(tt_prompt_embeds)

#     prompt_embeds = prompt_embeds.to(device=torch_device)

#     _, seq_len, _ = prompt_embeds.shape

#     # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
#     prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#     return prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)


def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    max_sequence_length: int,
    device: ttnn.device,
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
) -> torch.Tensor:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("T5 input text was truncated")

    prompt_embeds = text_encoder.forward(text_input_ids)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # B, C, H * P, W * Q -> B, H * W, C * P * Q
    latents = latents.view(batch_size, num_channels_latents, height, 2, width, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height) * (width), num_channels_latents * 4)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    # B, H * W, C * P * Q -> B, C, H * P, W * Q
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    return latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)


# def _reshape_noise_pred(
#     noise_pred: ttnn.Tensor,
#     *,
#     height: int,
#     width: int,
#     patch_size: int,
# ) -> ttnn.Tensor:
#     patch_count_y = height // patch_size
#     patch_count_x = width // patch_size

#     shape1 = (
#         noise_pred.shape[0] * patch_count_y,
#         patch_count_x,
#         patch_size,
#         -1,
#     )

#     shape2 = (
#         noise_pred.shape[0],
#         patch_count_y * patch_size,
#         patch_count_x * patch_size,
#         -1,
#     )

#     noise_pred = noise_pred.reshape(shape1)
#     noise_pred = ttnn.transpose(noise_pred, 1, 2)
#     return noise_pred.reshape(shape2)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _latent_image_ids(*, height: int, width: int) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    return latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)
