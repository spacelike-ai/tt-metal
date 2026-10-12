# Temporary: narrow down the WH Galaxy hang (QI-29), traced. "preset" runs under the watcher,
# "only_1024" leaves most of the DRAM free, "sp_axis1" moves the ring attention to the other mesh
# axis. Not for commit.

import time

import pytest
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.pipelines.cfg import submesh_shape
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

from .device_params import MESH_DEVICE_PARAMS

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("variant", ["preset", "only_1024", "sp_axis1"])
def test_galaxy_hang2(*, mesh_device: ttnn.MeshDevice, variant: str) -> None:
    preset = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape)
    options = {}
    if variant == "only_1024":
        options["resolutions"] = [(1024, 1024)]
    if variant == "sp_axis1":
        cfg = preset.dit_parallel_config.cfg_parallel
        shape = submesh_shape(preset.dit_parallel_config)
        rows, cols = shape[0], shape[1]
        options["dit_parallel_config"] = DiTParallelConfig.from_tuples(
            cfg=(cfg.factor, cfg.mesh_axis), sp=(cols, 1), tp=(rows, 0)
        )
    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape, **options)
    logger.info(f"variant {variant}: {config.dit_parallel_config}, resolutions {config.resolutions}")
    pipeline = QwenImagePipeline(device=mesh_device, config=config)

    for i in range(3):
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=i, width=1024, height=1024)
        logger.info(f"generation {i}: {time.perf_counter() - start:.1f} s")
        image.save(f"qwenimage_galaxy_hang2_{variant}_{i}.png")
