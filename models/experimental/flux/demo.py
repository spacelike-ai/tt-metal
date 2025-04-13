# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn
from models.experimental.flux.tt import FluxPipeline


def main() -> None:
    device_count = ttnn.get_num_devices()

    mesh_shape = ttnn.MeshShape(1, min(2, device_count))
    use_torch_encoder = device_count == 1

    mesh_device = ttnn.open_mesh_device(
        mesh_shape,
        l1_small_size=8192,
        trace_region_size=15210496,
    )
    for device in mesh_device.get_devices():
        ttnn.enable_program_cache(device)

    pipeline = FluxPipeline(
        checkpoint="black-forest-labs/FLUX.1-schnell",
        device=mesh_device,
        use_torch_encoder=use_torch_encoder,
    )

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


if __name__ == "__main__":
    main()
