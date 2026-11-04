# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import os

import numpy as np
import pytest
import torch
import transformers
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole

from ....models.transformers.transformer_flux3 import Flux3Checkpoint
from ....pipelines.flux3.pipeline_flux3 import Flux3Pipeline, Flux3PipelineConfig
from ....pipelines.flux3.text_encoder import reference_prompt_embeds
from ....utils.check import assert_quality
from .device_params import line_params_flux3_traced
from .reference_flux3 import Flux3Transformer2DModel

CHECKPOINT = "black-forest-labs/flux-3-image-klein"

PROMPTS = [
    "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
    "A storefront sign that reads 'FLUX.3 ON TENSTORRENT' in bold red letters, photographed at golden hour",
]
EDIT_PROMPT = "Make the letters on the sign blue and add snow on the ledge"


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(2, 2), line_params_flux3_traced],
        [(2, 4), line_params_flux3_traced],
        [(4, 8), line_params_flux3_traced],
    ],
    ids=["2x2", "2x4", "4x8"],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height"), [(1024, 1024)])
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("traced", [True, False], ids=["traced", "not_traced"])
def test_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    traced: bool,
    no_prompt: bool,
    model_location_generator,  # noqa: ANN001
) -> None:
    config = Flux3PipelineConfig.default(
        mesh_shape=tuple(mesh_device.shape),
        checkpoint_name=model_location_generator(CHECKPOINT),
        resolutions=[(width, height)],
        # Fewer prompt lengths to warm up.
        max_prompt_length=160,
    )
    pipeline = Flux3Pipeline(mesh_device=mesh_device, config=config, trace_warmup=traced)

    arch = "bh" if is_blackhole() else "wh"
    mesh_tag = "x".join(str(s) for s in mesh_device.shape)
    filename_prefix = f"flux3_klein_{arch}_{width}_{height}_{mesh_tag}"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int, images: list[Image.Image] | None = None) -> Image.Image:
        image = pipeline(prompts=[prompt], seed=seed, traced=traced, images=images)[0]
        output_filename = f"{filename_prefix}_{number}.png"
        image.save(output_filename)
        logger.info(f"image saved as {output_filename}")
        return image

    if no_prompt:
        image = None
        for i, prompt in enumerate(PROMPTS):
            image = run(prompt=prompt, number=i, seed=0)
        assert image is not None
        run(prompt=EDIT_PROMPT, number=len(PROMPTS), seed=0, images=[image])
    else:
        prompt = PROMPTS[0]
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(2, 2), line_params_flux3_traced],
        [(2, 4), line_params_flux3_traced],
    ],
    ids=["2x2", "2x4"],
    indirect=True,
)
# The step-distilled checkpoint is made for 16 steps; with fewer, both garble fine detail differently.
@pytest.mark.parametrize("num_inference_steps", [16])
# The torch reference runs on the CPU.
@pytest.mark.timeout(3600)
def test_pipeline_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps: int,
    model_location_generator,  # noqa: ANN001
) -> None:
    """Compare a generation with the torch reference of diffusers' distilled pipeline, untraced."""
    width = height = 1024
    prompt = PROMPTS[1]
    seed = 42
    name = model_location_generator(CHECKPOINT)

    config = Flux3PipelineConfig.default(
        mesh_shape=tuple(mesh_device.shape),
        checkpoint_name=name,
        resolutions=[(width, height)],
        max_prompt_length=160,
        max_references=0,
    )
    pipeline = Flux3Pipeline(mesh_device=mesh_device, config=config, trace_warmup=False)
    tt_image = pipeline(prompts=[prompt], num_inference_steps=num_inference_steps, seed=seed, traced=False)[0]
    tt_image.save(f"flux3_klein_accuracy_{num_inference_steps}_tt.png")

    torch_image = _torch_reference(name, prompt, num_inference_steps=num_inference_steps, seed=seed, size=width)
    torch_image.save(f"flux3_klein_accuracy_{num_inference_steps}_torch.png")

    assert_quality(
        torch.from_numpy(np.asarray(torch_image)).float(),
        torch.from_numpy(np.asarray(tt_image)).float(),
        pcc=0.9,
    )


@torch.no_grad()
def _torch_reference(
    name: str,
    prompt: str,
    *,
    num_inference_steps: int,
    seed: int,
    size: int,
    reference: Image.Image | None = None,
) -> Image.Image:
    """diffusers ``Flux3DistilledModularPipeline`` written out, with an optional square reference image
    of the output size."""
    processor = transformers.AutoProcessor.from_pretrained(name, subfolder="text_encoder")
    text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    prompt_embeds = reference_prompt_embeds(prompt, processor=processor, model=text_encoder)
    del text_encoder

    transformer = Flux3Transformer2DModel.from_pretrained(name, subfolder="transformer", torch_dtype=torch.bfloat16)
    assert isinstance(transformer, Flux3Transformer2DModel)
    transformer.eval()

    tokens = size // 16
    vae = AutoencoderKLFlux2.from_pretrained(name, subfolder="vae")
    assert isinstance(vae, AutoencoderKLFlux2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config["batch_norm_eps"])

    reference_inputs = {}
    if reference is not None:
        assert reference.size == (size, size)
        pixels = torch.from_numpy(np.asarray(reference.convert("RGB"))).float().permute(2, 0, 1)[None] / 127.5 - 1.0
        reference_latents = vae.encode(pixels).latent_dist.mode()
        # Patchify: (B, C, H, W) -> (B, C * 4, H / 2, W / 2).
        b, c, h, w = reference_latents.shape
        reference_latents = reference_latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        reference_latents = (reference_latents.reshape(b, c * 4, h // 2, w // 2) - mean) / std
        reference_inputs["image_cond_hidden_states"] = reference_latents.flatten(2).transpose(1, 2).to(torch.bfloat16)
        reference_inputs["image_cond_ids"] = Flux3Checkpoint.image_ids(height=tokens, width=tokens, index=-1)

    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn([1, 128, tokens, tokens], generator=generator, dtype=torch.float32).to(torch.bfloat16)
    latents = latents.flatten(2).transpose(1, 2)
    img_ids = Flux3Checkpoint.image_ids(height=tokens, width=tokens)
    txt_ids = Flux3Checkpoint.text_ids(prompt_embeds.shape[1])

    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
    scheduler.set_begin_index(0)
    timesteps = scheduler.timesteps
    assert isinstance(timesteps, torch.Tensor)

    for i, t in enumerate(timesteps):
        logger.info(f"torch reference step {i}...")
        next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t)
        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=(t / 1000).expand(1).to(torch.bfloat16),
            target_timestep=(next_t / 1000).expand(1).to(torch.bfloat16),
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
            **reference_inputs,
        )[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents.view(1, tokens, tokens, -1).permute(0, 3, 1, 2).float()
    latents = latents * std + mean
    # Unpatchify: (B, C * 4, H, W) -> (B, C, H * 2, W * 2).
    batch_size, channels, h, w = latents.shape
    latents = latents.reshape(batch_size, channels // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, channels // 4, h * 2, w * 2)
    image = vae.decode(latents, return_dict=False)[0]
    image = ((image[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(image.numpy())
