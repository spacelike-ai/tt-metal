import base64
import io

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

from .device_params import EDIT_MESH_DEVICE_PARAMS
from .test_pipeline_qwenimage import DOG_IMAGE_PATH


def _thumbnail(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.resize((160, 160)).save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("checkpoint_name", ["Qwen/Qwen-Image-Edit-2509"])
def test_scratch_edit_plus_bh(*, mesh_device: ttnn.MeshDevice, checkpoint_name: str) -> None:
    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name=checkpoint_name)
    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    before = np.asarray(image.resize((1024, 1024)), dtype=np.float32)
    logger.info(f"THUMB input {_thumbnail(image)}")

    for traced in (True, False):
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
        logger.info(f"RESULT traced={traced} mean absolute change {diff:.2f}")
        logger.info(f"THUMB traced={traced} {_thumbnail(edited)}")
