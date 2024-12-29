from __future__ import annotations

import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import ttnn

from ..reference.transformer import SD3Transformer2DModel
from ..tt.transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters

CHECKPOINT = "stabilityai/stable-diffusion-3.5-medium"


def test_transformer(*, device: ttnn.Device):
    model_input = {
        "prompt_1": ["cat"],
        "prompt_2": ["cat"],
        "prompt_3": ["cat"],
        "negative_prompt_1": ["mouse"],
        "negative_prompt_2": ["mouse"],
        "negative_prompt_3": ["mouse"],
        "width": 512,  # default = 1024, works with 512
        "height": 512,  # default = 1024
        "num_inference_steps": 5,  # default = 50
        "guidance_scale": 7.0,
        "num_images_per_prompt": 1,
        "max_t5_sequence_length": 256,
        "seed": 1,
    }

    logger.info("loading models")
    tokenizer_1 = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer_2")
    tokenizer_3 = T5TokenizerFast.from_pretrained(CHECKPOINT, subfolder="tokenizer_3")
    text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(CHECKPOINT, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(CHECKPOINT, subfolder="text_encoder_2")
    text_encoder_3 = T5EncoderModel.from_pretrained(CHECKPOINT, subfolder="text_encoder_3")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CHECKPOINT, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(CHECKPOINT, subfolder="vae")
    logger.info("loading transformer")
    torch_transformer = SD3Transformer2DModel.from_pretrained(
        CHECKPOINT, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    logger.info("loading done")

    assert isinstance(torch_transformer, SD3Transformer2DModel)
    assert isinstance(tokenizer_1, CLIPTokenizer)
    assert isinstance(tokenizer_2, CLIPTokenizer)
    assert isinstance(tokenizer_3, T5TokenizerFast)
    assert isinstance(text_encoder_1, CLIPTextModelWithProjection)
    assert isinstance(text_encoder_2, CLIPTextModelWithProjection)
    assert isinstance(text_encoder_3, T5EncoderModel)
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert isinstance(vae, AutoencoderKL)

    logger.info("creating tt transformer")
    parameters = TtSD3Transformer2DModelParameters.from_torch(torch_transformer.state_dict(), device=device)
    tt_transformer = TtSD3Transformer2DModel(
        parameters, num_attention_heads=torch_transformer.config.num_attention_heads
    )
    num_channels_latents = torch_transformer.in_channels
    logger.info("done")

    block_out_channels = vae.config.block_out_channels  # type: ignore  # noqa: PGH003
    vae_scaling_factor = vae.config.scaling_factor  # type: ignore  # noqa: PGH003
    vae_shift_factor = vae.config.shift_factor  # type: ignore  # noqa: PGH003

    vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    assert model_input["height"] % (vae_scale_factor * tt_transformer.patch_size) == 0
    assert model_input["width"] % (vae_scale_factor * tt_transformer.patch_size) == 0
    assert model_input["max_t5_sequence_length"] <= 512  # noqa: PLR2004

    batch_size = len(model_input["prompt_1"])
    do_classifier_free_guidance = model_input["guidance_scale"] > 1
    tokenizer_max_length = tokenizer_1.model_max_length

    logger.info("encode prompts")

    logger.info("prompt 1")
    prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
        prompt=model_input["prompt_1"],
        num_images_per_prompt=model_input["num_images_per_prompt"],
        tokenizer=tokenizer_1,
        text_encoder=text_encoder_1,
        tokenizer_max_length=tokenizer_max_length,
    )

    logger.info("prompt 2")
    prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
        prompt=model_input["prompt_2"],
        num_images_per_prompt=model_input["num_images_per_prompt"],
        tokenizer=tokenizer_2,
        text_encoder=text_encoder_2,
        tokenizer_max_length=tokenizer_max_length,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    logger.info("prompt 3")
    t5_prompt_embed = _get_t5_prompt_embeds(
        prompt=model_input["prompt_3"],
        num_images_per_prompt=model_input["num_images_per_prompt"],
        max_sequence_length=model_input["max_t5_sequence_length"],
        tokenizer=tokenizer_3,
        text_encoder=text_encoder_3,
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
            prompt=model_input["negative_prompt_1"],
            num_images_per_prompt=model_input["num_images_per_prompt"],
            tokenizer=tokenizer_1,
            text_encoder=text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )
        logger.info("negative prompt 2")
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=model_input["negative_prompt_2"],
            num_images_per_prompt=model_input["num_images_per_prompt"],
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            tokenizer_max_length=tokenizer_max_length,
        )
        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        logger.info("negative prompt 3")
        t5_negative_prompt_embed = _get_t5_prompt_embeds(
            prompt=model_input["negative_prompt_3"],
            num_images_per_prompt=model_input["num_images_per_prompt"],
            max_sequence_length=model_input["max_t5_sequence_length"],
            tokenizer=tokenizer_3,
            text_encoder=text_encoder_3,
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

    scheduler.set_timesteps(model_input["num_inference_steps"])
    timesteps = scheduler.timesteps

    logger.info("prepare latents")

    latents_shape = (
        batch_size * model_input["num_images_per_prompt"],
        num_channels_latents,
        model_input["height"] // vae_scale_factor,
        model_input["width"] // vae_scale_factor,
    )
    torch.manual_seed(model_input["seed"])
    latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)

    tt_prompt_embeds = ttnn.from_torch(prompt_embeds, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_pooled_prompt_embeds = ttnn.from_torch(
        pooled_prompt_embeds, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    logger.info("denoising loop")

    for t in tqdm.tqdm(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        timestep = t.expand(latent_model_input.shape[0])

        tt_latent_model_input = ttnn.from_torch(latent_model_input, device=device, dtype=ttnn.bfloat16)

        tt_noise_pred = tt_transformer(
            spatial=tt_latent_model_input,
            prompt_embed=tt_prompt_embeds,
            pooled_projection=tt_pooled_prompt_embeds,
            torch_timestep=timestep,
        )
        noise_pred = ttnn.to_torch(tt_noise_pred)

        patch_count_y = latent_model_input.shape[-2] // tt_transformer.patch_size
        patch_count_x = latent_model_input.shape[-1] // tt_transformer.patch_size

        noise_pred = noise_pred.reshape(
            (
                noise_pred.shape[0],
                patch_count_y,
                patch_count_x,
                tt_transformer.patch_size,
                tt_transformer.patch_size,
                -1,
            )
        )
        noise_pred = torch.einsum("nhwpqc->nchpwq", noise_pred)
        noise_pred = noise_pred.reshape(latent_model_input.shape)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + model_input["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(
            noise_pred,  # type: ignore  # noqa: PGH003
            t,  # type: ignore  # noqa: PGH003
            latents,  # type: ignore  # noqa: PGH003
            return_dict=False,
        )[0]

    latents = (latents / vae_scaling_factor) + vae_shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pt")
    assert isinstance(image, torch.Tensor)

    pil_images = image_processor.numpy_to_pil(image_processor.pt_to_numpy(image))
    pil_image = pil_images[0]
    pil_image.save("sd3.png")


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModelWithProjection,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(prompt_count * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    max_sequence_length: int,
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
) -> torch.Tensor:
    prompt_count = len(prompt)

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

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("CLIP input text was truncated")

    prompt_embeds = text_encoder.forward(text_input_ids)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)
