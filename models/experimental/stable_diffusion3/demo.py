from __future__ import annotations

import pytest
import torch

import ttnn

from .tt import TtStableDiffusion3Pipeline


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_sd3(
    *,
    device: ttnn.Device,
    use_program_cache: None,
):
    pipeline = TtStableDiffusion3Pipeline(checkpoint="stabilityai/stable-diffusion-3.5-medium", device=device)

    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )
    negative_prompt = ""

    pipeline(
        prompt_1=[prompt],
        prompt_2=[prompt],
        prompt_3=[prompt],
        negative_prompt_1=[negative_prompt],
        negative_prompt_2=[negative_prompt],
        negative_prompt_3=[negative_prompt],
        width=1024,
        height=1024,
        num_inference_steps=40,
        seed=0,
    )
