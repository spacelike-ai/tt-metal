# Temporary: test whether the WH Galaxy hang (QI-29) needs the Galaxy's 8x9 matmul grid. tt-triage
# found one core of a 2D-mcast matmul with fused bias stuck in the padding variant of its kernels;
# on a T3K the grid is 8x8. Clamps the grid tt_dit's matmuls use to 8x8. Not for commit.

import time

import pytest
from loguru import logger

import ttnn
from models.tt_dit.blocks import attention_opt
from models.tt_dit.layers import linear
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from models.tt_dit.utils import matmul

from .device_params import MESH_DEVICE_PARAMS

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
def test_grid8x8(*, mesh_device: ttnn.MeshDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    get_matmul_core_grid = matmul.get_matmul_core_grid

    def get_matmul_core_grid_8x8(device: ttnn.MeshDevice) -> ttnn.CoreCoord:
        grid = get_matmul_core_grid(device)
        return ttnn.CoreCoord(min(grid.x, 8), min(grid.y, 8))

    for module in (matmul, linear, attention_opt):
        monkeypatch.setattr(module, "get_matmul_core_grid", get_matmul_core_grid_8x8)

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)

    for i in range(3):
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=i, width=1024, height=1024)
        logger.info(f"generation {i}: {time.perf_counter() - start:.1f} s")
        image.save(f"qwenimage_grid8x8_{i}.png")
