# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn
from models.experimental.flux.tt import FluxPipeline


def run(
    *,
    mesh_width: int,
    mesh_height: int,
    num_images_per_prompt: int,
    use_torch_encoder: bool,
) -> None:
    assert num_images_per_prompt % mesh_height == 0

    # ruff: noqa: T201
    print(f"mesh_width = {mesh_width}")
    print(f"mesh_height = {mesh_height}")
    print(f"num_images_per_prompt = {num_images_per_prompt}")
    print(f"use_torch_encoder = {use_torch_encoder}")

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(mesh_height, mesh_width),
        l1_small_size=8192,
        trace_region_size=15210496,
    )
    for device in mesh_device.get_devices():
        ttnn.enable_program_cache(device)

    device.enable_async(True)  # noqa: FBT003

    pipeline = FluxPipeline(
        checkpoint="black-forest-labs/FLUX.1-schnell",
        device=mesh_device,
        use_torch_encoder=use_torch_encoder,
    )

    pipeline.prepare(
        width=1024,
        height=1024,
        prompt_count=1,
        num_images_per_prompt=num_images_per_prompt,
    )

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

        for i, image in enumerate(images, start=1):
            image.save(f"flux_1024_{i}.png")


def main() -> None:
    device_count = ttnn.get_num_devices()

    mesh_width = 1 if device_count == 1 else 2
    mesh_height = device_count // mesh_width

    run(
        mesh_width=mesh_width,
        mesh_height=mesh_height,
        num_images_per_prompt=mesh_height,
        # use_torch_encoder=mesh_width == 1,
        use_torch_encoder=True,
    )


if __name__ == "__main__":
    main()
