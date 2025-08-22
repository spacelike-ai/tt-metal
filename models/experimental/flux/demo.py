# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import itertools

import ttnn
from models.experimental.flux import FluxPipeline


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

    if ttnn.get_num_devices() > 1:
        is_blackhole = ttnn.get_arch_name() == "blackhole"
        dispatch_core_axis = ttnn.DispatchCoreAxis.COL if is_blackhole else ttnn.DispatchCoreAxis.ROW
        dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH, dispatch_core_axis)
    else:
        dispatch_core_config = None

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(mesh_height, mesh_width),
        l1_small_size=8192,
        trace_region_size=15210496,
        dispatch_core_config=dispatch_core_config,
    )
    for device in mesh_device.get_devices():
        ttnn.enable_program_cache(device)

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

    for iteration in itertools.count(start=1):
        new_prompt = input("Enter the input prompt, or q to exit: ")
        if new_prompt:
            prompt = new_prompt
        if prompt == "q":
            break

        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=4,
            seed=iteration,
        )

        for i, image in enumerate(images, start=1):
            image.save(f"flux_1024_{i}.png")


def main() -> None:
    parser = argparse.ArgumentParser(prog="FLUX.1 demo")
    parser.add_argument("--batch-size", type=int, default=1, help="corresponds to the mesh height")
    parser.add_argument("--mesh-width", type=int, help="parallelization of the model weights")
    parser.add_argument("--encode-on-device", action="store_true", help="run T5 on the device instead of the CPU")
    args = parser.parse_args()

    device_count = ttnn.get_num_devices()

    mesh_height = args.batch_size
    mesh_width = args.mesh_width if args.mesh_width is not None else device_count // mesh_height

    run(
        mesh_width=mesh_width,
        mesh_height=mesh_height,
        num_images_per_prompt=mesh_height,
        use_torch_encoder=not args.encode_on_device,
    )


if __name__ == "__main__":
    main()
