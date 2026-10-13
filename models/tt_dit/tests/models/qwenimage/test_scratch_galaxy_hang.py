# Temporary: narrow down the WH Galaxy hang (QI-29) with the preset, traced. Each variant changes
# one thing: the number of links, or a host synchronize after every trace replay. Not for commit.

import time
from typing import Any

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig
from models.tt_dit.utils import tracing

from .device_params import MESH_DEVICE_PARAMS

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    ("num_links", "sync_each_replay"),
    [(1, False), (2, False), (None, True)],
    ids=["links1", "links2", "sync_each_replay"],
)
def test_galaxy_hang(
    *,
    mesh_device: ttnn.MeshDevice,
    num_links: int | None,
    sync_each_replay: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sync_each_replay:
        execute = tracing.Tracer._execute

        def execute_and_sync(self: tracing.Tracer, args: tuple, kwargs: dict, **options: Any) -> None:
            execute(self, args, kwargs, **options)
            for d in self._devices:
                ttnn.synchronize_device(d)

        monkeypatch.setattr(tracing.Tracer, "_execute", execute_and_sync)

    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape, num_links=num_links)
    logger.info(f"num_links {config.num_links}, sync_each_replay {sync_each_replay}")
    pipeline = QwenImagePipeline(device=mesh_device, config=config)

    for i in range(3):
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=i, width=1024, height=1024)
        logger.info(f"generation {i}: {time.perf_counter() - start:.1f} s")
        image.save(f"qwenimage_galaxy_hang_{i}.png")
