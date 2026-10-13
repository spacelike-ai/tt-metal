import base64
import io
from typing import Any

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from models.tt_dit.utils import tracing

from .device_params import EDIT_MESH_DEVICE_PARAMS
from .test_pipeline_qwenimage import DOG_IMAGE_PATH


def _thumbnail(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.resize((160, 160)).save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode()


def _warm_copy(value: Any, *, path_label: str) -> Any:
    del path_label
    if isinstance(value, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(value):
        tmp = ttnn.clone(value)
        ttnn.copy(value, tmp)
        ttnn.deallocate(tmp)
    return value


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "checkpoint_name", ["Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2511"], ids=["2509", "2511"]
)
@pytest.mark.parametrize("mode", ["warm_copy", "untraced"])
def test_scratch_edit_plus_bh(
    *, mesh_device: ttnn.MeshDevice, checkpoint_name: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    traced = mode != "untraced"
    if mode == "untraced":
        # Without the capture run, nothing is traced and the pipeline can run untraced.
        warm_up = QwenImagePipeline._warm_up

        def untraced_warm_up(self: QwenImagePipeline, *, traced: bool) -> None:
            if not traced:
                warm_up(self, traced=False)

        monkeypatch.setattr(QwenImagePipeline, "_warm_up", untraced_warm_up)
    else:
        # Compiles the copies into the captured inputs before each capture, so that their programs
        # don't land in memory a trace writes to on replay.
        capture = tracing.Tracer._capture

        def capture_after_warm_copy(self: tracing.Tracer, args: tuple, kwargs: dict, **options: Any) -> None:
            tracing._tree_map(_warm_copy, args, path_label="args")
            tracing._tree_map(_warm_copy, kwargs, path_label="kwargs")
            capture(self, args, kwargs, **options)

        monkeypatch.setattr(tracing.Tracer, "_capture", capture_after_warm_copy)

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name=checkpoint_name)
    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    before = np.asarray(image.resize((1024, 1024)), dtype=np.float32)

    (edited,) = pipeline(
        prompts=["Make the dog wear a red knitted scarf."],
        num_inference_steps=50,
        seed=0,
        image=image,
        traced=traced,
        encoder_traced=traced,
        vae_traced=traced,
    )
    diff = np.abs(np.asarray(edited, dtype=np.float32) - before).mean()
    logger.info(f"RESULT {mode} mean absolute change {diff:.2f}")
    logger.info(f"THUMB {mode} {_thumbnail(edited)}")
