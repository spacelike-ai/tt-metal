# Temporary: narrow down the tiled reshape hang (QIE-14) with the real pipeline. The patch merger's
# reshape is switched back to the tiled one that hung (before f002807). Each step logs before it
# runs, so a hang shows where. Run with TT_METAL_OPERATION_TIMEOUT_SECONDS=5. Not for commit.
#
# "calls": two edit calls, the original hang.
# "vision": one edit call, then only the vision tower.
# "merger": one edit call, then only the merger on its recorded input.
# "merger_only_reshape": one edit call, then only the tiled reshape on the merger's recorded input.
# "no_call": no edit call, the vision tower twice after the warm-up.

from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.encoders.qwen25vl import vision_qwen25vl
from models.tt_dit.encoders.qwen25vl.vision_qwen25vl import Qwen25VlVisionPatchMerger
from models.tt_dit.encoders.vision import row_parallel_forward
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

from .device_params import EDIT_MESH_DEVICE_PARAMS

DOG_IMAGE_PATH = Path(__file__).resolve().parents[4] / "demos" / "multimodal" / "gemma3" / "dog.jpg"
PROMPT = "Make the dog wear a red knitted scarf."

_recorded: dict[str, torch.Tensor] = {}


def _tiled_forward(self: Qwen25VlVisionPatchMerger, x: ttnn.Tensor) -> ttnn.Tensor:
    x = ttnn.typecast(self.norm.forward(x), ttnn.bfloat16)
    if "input" not in _recorded:
        _recorded["input"] = ttnn.to_torch(ttnn.get_device_tensors(x)[0])
    logger.info(f"merger: tiled reshape {tuple(x.shape)} -> (-1, {self.merged_size})")
    x = ttnn.reshape(x, (-1, self.merged_size))
    return row_parallel_forward(self.linear_fc2, ttnn.gelu(self.linear_fc1.forward(x)), self._p)


def _sync(device: ttnn.MeshDevice, label: str) -> None:
    ttnn.synchronize_device(device)
    logger.info(f"{label}: done")


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", ["calls", "vision", "merger", "merger_only_reshape", "no_call"])
def test_reshape_hang2(
    *, mesh_device: ttnn.MeshDevice, mode: str, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
    monkeypatch.setattr(vision_qwen25vl.Qwen25VlVisionPatchMerger, "forward", _tiled_forward)

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name="Qwen/Qwen-Image-Edit-2511")
    encoder = pipeline._text_encoder
    device = encoder._device
    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    logger.info("pipeline created")

    def edit(label: str) -> None:
        logger.info(f"{label}: edit call")
        pipeline(prompts=[PROMPT], num_inference_steps=50, seed=0, image=image)
        logger.info(f"{label}: edit call done")

    def vision(label: str) -> None:
        logger.info(f"{label}: vision tower")
        encoder._encode_images([image])
        _sync(device, label)

    if mode == "no_call":
        vision("first")
        vision("second")
        return

    edit("first")
    if mode == "calls":
        edit("second")
    elif mode == "vision":
        vision("second")
    else:
        merger = encoder._vision.merger
        x = ttnn.from_torch(
            _recorded["input"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        logger.info(f"second: {mode} on {tuple(x.shape)}")
        if mode == "merger":
            row_parallel_forward(merger.linear_fc2, ttnn.gelu(merger.linear_fc1.forward(ttnn.reshape(x, (-1, merger.merged_size)))), merger._p)
        else:
            ttnn.reshape(x, (-1, merger.merged_size))
        _sync(device, "second")
