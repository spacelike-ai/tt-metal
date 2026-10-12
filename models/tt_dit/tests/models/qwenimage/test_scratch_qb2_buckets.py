# Temporary: check that the resolutions and spatial buckets fit, and measure the trace region they
# need (QI-27, QI-29). Not for commit.

import time

import numpy as np
import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig
from models.tt_dit.utils.test import line_params_req_exact_devices

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."

# Each spatial bucket with the resolutions that fall into it.
BUCKETS = {
    4096: [(1024, 1024)],
    6144: [(1664, 928), (928, 1664)],
    6528: [(1472, 1120), (1120, 1472)],
    6656: [(1600, 1056), (1056, 1600)],
    7168: [(1344, 1344)],
}


def _log_memory(pipeline: QwenImagePipeline, label: str) -> None:
    for i, device in enumerate(pipeline._devices):
        view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        logger.info(
            f"DRAM {label} submesh {i}: allocated {view.total_bytes_allocated_per_bank / 2**30:.2f} GiB/bank, "
            f"free {view.total_bytes_free_per_bank / 2**30:.2f} GiB/bank, "
            f"largest free {view.largest_contiguous_bytes_free_per_bank / 2**30:.2f} GiB/bank, "
            f"{view.num_banks} banks"
        )
        trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
        logger.info(
            f"TRACE {label} submesh {i}: allocated {trace.total_bytes_allocated_per_bank * trace.num_banks / 1e6:.1f} MB "
            f"({trace.total_bytes_allocated_per_bank} B/bank x {trace.num_banks} banks), "
            f"free {trace.total_bytes_free_per_bank * trace.num_banks / 1e6:.1f} MB"
        )


def _device_params(trace_region_size: int) -> dict:
    return {**line_params_req_exact_devices, "l1_small_size": 32768, "trace_region_size": trace_region_size}


CASES = [
    pytest.param((2, 2), _device_params(1_000_000_000), None, False, True, id="2x2-preset"),
    # BH LoudBox and BH/WH Galaxy, with each preset's resolutions. WH chips have less DRAM.
    pytest.param(
        (2, 4), _device_params(1_000_000_000 if is_blackhole() else 700_000_000), None, False, True, id="2x4-preset"
    ),
    pytest.param(
        (4, 8), _device_params(1_000_000_000 if is_blackhole() else 700_000_000), None, False, True, id="4x8-preset"
    ),
    *(
        pytest.param((2, 4), _device_params(250_000_000), bucket, False, False, id=f"2x4-bucket{bucket}")
        for bucket in BUCKETS
    ),
    pytest.param((2, 4), _device_params(250_000_000), 4096, True, False, id="2x4-bucket4096-torchenc"),
    *(
        pytest.param((2, 4), _device_params(200_000_000), bucket, True, False, id=f"2x4-bucket{bucket}-torchenc")
        for bucket in list(BUCKETS)[1:]
    ),
]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "bucket", "torch_text_encoder", "generate"),
    CASES,
    indirect=["mesh_device", "device_params"],
)
def test_qb2_buckets(
    *, mesh_device: ttnn.MeshDevice, bucket: int | None, torch_text_encoder: bool, generate: bool
) -> None:
    extra = {}
    if bucket is not None:
        extra = {"resolutions": BUCKETS[bucket], "spatial_sequence_length_buckets": [bucket]}
    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape, use_torch_text_encoder=torch_text_encoder, **extra
    )
    logger.info(f"resolutions {config.resolutions}, spatial buckets {config.spatial_sequence_length_buckets}")

    start = time.perf_counter()
    pipeline = QwenImagePipeline(device=mesh_device, config=config)
    logger.info(f"construction took {time.perf_counter() - start:.0f} s")
    _log_memory(pipeline, f"after warm-up bucket={bucket} torch_text_encoder={torch_text_encoder}")

    if not generate:
        return

    for width, height in config.resolutions:
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=0, width=width, height=height)
        elapsed = time.perf_counter() - start
        image.save(f"qwenimage_qb2_{width}x{height}.png")
        pixels = np.asarray(image, dtype=np.float32)
        logger.info(f"QB2 {width}x{height}: {elapsed:.1f} s, mean {pixels.mean():.1f}, std {pixels.std():.1f}")
        assert image.size == (width, height)
