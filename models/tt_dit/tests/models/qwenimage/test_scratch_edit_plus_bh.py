import base64
import io
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig
from models.tt_dit.utils import tensor, tracing

from .device_params import EDIT_MESH_DEVICE_PARAMS
from .test_pipeline_qwenimage import DOG_IMAGE_PATH


def _thumbnail(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.resize((160, 160)).save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode()


def _compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.float().flatten(), b.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if a.std() > 0 and b.std() > 0 else float("nan")
    logger.info(
        f"PROBE {name}: pcc {pcc:.5f} max abs diff {(a - b).abs().max().item():.4f} norms {a.norm():.2f} {b.norm():.2f}"
    )


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("variant", ["probe", "single_bucket", "always_copy"])
def test_scratch_edit_plus_bh(*, mesh_device: ttnn.MeshDevice, variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_name = "Qwen/Qwen-Image-Edit-2509"
    if variant == "always_copy":

        def update_input(self: tracing.Tracer, prev: Any, new: Any, *, path_label: str) -> None:
            if isinstance(new, ttnn.Tensor) and new.device() is not None:
                if new is not prev:
                    ttnn.copy(new, prev)
                return
            update(self, prev, new, path_label=path_label)

        update = tracing.Tracer._update_input
        monkeypatch.setattr(tracing.Tracer, "_update_input", update_input)

    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape,
        checkpoint_name=checkpoint_name,
        max_images=1 if variant == "single_bucket" else None,
    )
    pipeline = QwenImagePipeline(device=mesh_device, config=config)

    contexts: list[torch.Tensor] = []
    encode = pipeline._text_encoder.encode_cfg

    def encode_cfg(*args: Any, **kwargs: Any) -> Any:
        result = encode(*args, **kwargs)
        contexts[:] = [c.clone() for c in result]
        return result

    monkeypatch.setattr(pipeline._text_encoder, "encode_cfg", encode_cfg)

    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    before = np.asarray(image.resize((1024, 1024)), dtype=np.float32)
    (edited,) = pipeline(
        prompts=["Make the dog wear a red knitted scarf."], num_inference_steps=50, seed=0, image=image
    )
    diff = np.abs(np.asarray(edited, dtype=np.float32) - before).mean()
    logger.info(f"RESULT {variant} mean absolute change {diff:.2f}")
    logger.info(f"THUMB {variant} {_thumbnail(edited)}")

    logger.info(f"PROBE context shapes {[tuple(c.shape) for c in contexts]}")
    if len(contexts) == 1:
        contexts = list(contexts[0].split(1))
    _compare("uncond vs cond context", contexts[0], contexts[-1])
    for (idx, length, spatial), tracer in pipeline._tracers.items():
        if not tracer.trace_captured or spatial != 8320 or length != contexts[0].shape[1]:
            continue
        prompt = tensor.to_torch(tracer.inputs["prompt"])
        logger.info(f"PROBE tracer {(idx, length, spatial)} prompt input shape {tuple(prompt.shape)}")
        for j, context in enumerate(contexts):
            _compare(f"tracer {idx} prompt input vs context {j}", prompt[:1], context[:1])
