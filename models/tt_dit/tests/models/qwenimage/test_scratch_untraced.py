# Temporary: generate with the mesh's preset without capturing any trace, to tell whether the WH
# Galaxy hang needs tracing (QI-29). Not for commit.

import time

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

from .device_params import MESH_DEVICE_PARAMS

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
def test_untraced(*, mesh_device: ttnn.MeshDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    warm_up = QwenImagePipeline._warm_up

    def warm_up_untraced(self: QwenImagePipeline, *, traced: bool) -> None:
        if not traced:
            warm_up(self, traced=False)

    monkeypatch.setattr(QwenImagePipeline, "_warm_up", warm_up_untraced)

    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape, resolutions=[(1024, 1024)])
    pipeline = QwenImagePipeline(device=mesh_device, config=config)

    for i in range(2):
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=i, width=1024, height=1024, traced=False)
        logger.info(f"untraced generation {i}: {time.perf_counter() - start:.1f} s")
        image.save(f"qwenimage_untraced_{i}.png")
