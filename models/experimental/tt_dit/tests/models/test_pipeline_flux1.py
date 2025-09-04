# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    ("model_variant", "width", "height", "num_inference_steps"),
    [
        ("schnell", 1024, 1024, 4),
    ],
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp"),
    [
        ((2, 2), (2, 0), (2, 1)),
        ((1, 4), (1, 0), (4, 1)),
        ((4, 4), (4, 0), (4, 1)),
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "t3k_cfg2_sp1_tp4",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
@pytest.mark.parametrize("traced", [True, False], ids=["yes_traced", "no_traced"])
def test_flux1_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    model_variant: str,
    width: int,
    height: int,
    num_inference_steps: int,
    sp: tuple[int, int],
    tp: tuple[int, int],
    no_prompt: bool,
    model_location_generator,
    traced: bool,
) -> None:
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    logger.info(f"Mesh device shape: {mesh_device.shape}")
    logger.info(f"Parallel config: {parallel_config}")
    # logger.info(f"T5 enabled: {enable_t5_text_encoder}")

    # Create timing collector
    timing_collector = TimingCollector()

    # Create pipeline
    pipeline = Flux1Pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-{model_variant}"),
        mesh_device=mesh_device,
        enable_t5_text_encoder=True,
        use_torch_t5_text_encoder=True,
        parallel_config=parallel_config,
    )

    # Set timing collector
    pipeline.timing_collector = timing_collector

    # Prepare pipeline
    pipeline.prepare(batch_size=1, width=width, height=height)

    # Define test prompt
    prompt = "A luxury sports car."

    if no_prompt:
        # Run single generation
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=num_inference_steps,
            seed=1,
            traced=traced,
        )

        # Save image
        output_filename = f"flux_new_{width}_{height}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        # Print timing information
        timing_data = timing_collector.get_timing_data()
        logger.info(f"CLIP encoding time: {timing_data.clip_encoding_time:.2f}s")
        logger.info(f"T5 encoding time: {timing_data.t5_encoding_time:.2f}s")
        logger.info(f"Total encoding time: {timing_data.total_encoding_time:.2f}s")
        logger.info(f"VAE decoding time: {timing_data.vae_decoding_time:.2f}s")
        logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")
        if timing_data.denoising_step_times:
            avg_step_time = sum(timing_data.denoising_step_times) / len(timing_data.denoising_step_times)
            logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

    else:
        # Interactive demo
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break

            images = pipeline(
                prompt_1=[prompt],
                prompt_2=[prompt],
                num_inference_steps=num_inference_steps,
                seed=1,
                traced=traced,
            )

            output_filename = f"flux_{width}_{height}_{i}.png"
            images[0].save(output_filename)
            logger.info(f"Image saved as {output_filename}")
