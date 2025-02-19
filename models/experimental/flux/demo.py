# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .tt import TtFluxPipeline

if TYPE_CHECKING:
    import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15210496}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_sd3(
    *,
    device: ttnn.Device,
) -> None:
    pipeline = TtFluxPipeline(checkpoint="black-forest-labs/FLUX.1-schnell", device=device)

    pipeline.prepare(width=1024, height=1024, prompt_count=1, num_images_per_prompt=1)

    prompt = "A luxury sports car."

    while True:
        new_prompt = input("Enter the input prompt, or q to exit: ")
        if new_prompt:
            prompt = new_prompt
        if prompt == "q":
            break

        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=4,
            seed=0,
        )

        images[0].save("flux_1024.png")
