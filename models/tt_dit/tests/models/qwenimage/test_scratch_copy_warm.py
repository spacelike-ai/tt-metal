# Temporary: test whether the WH Galaxy hang (QI-29) comes from the programs of the Tracer's input
# copies, which are first compiled after a capture, so their buffers can lie where a trace writes.
# Compiles the copy for every tensor input before each capture. Not for commit.

import time
from typing import Any

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from models.tt_dit.utils import tracing
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING, TraceAllocationTracker

from .device_params import MESH_DEVICE_PARAMS

PROMPT = "A ginger cat sitting on a wooden bench in front of a red brick wall."


def _warm_copy(value: Any, *, path_label: str) -> Any:
    del path_label
    if isinstance(value, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(value):
        tmp = ttnn.clone(value)
        ttnn.copy(value, tmp)
        ttnn.deallocate(tmp)
    return value


def _acknowledge(value: Any, *, path_label: str) -> Any:
    del path_label
    if isinstance(value, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(value) and value.is_allocated():
        TraceAllocationTracker.acknowledge_corruptible(value)
    return value


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("warm_copy", [True, False], ids=["warm_copy", "no_warm_copy"])
def test_copy_warm(*, mesh_device: ttnn.MeshDevice, warm_copy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if TRACE_ALLOC_TRACKING:
        # The new inputs are only the sources of the copies into the captured inputs, so trace
        # allocation tracking can report what else lives across a replay.
        execute = tracing.Tracer._execute

        def execute_acknowledging_inputs(self: tracing.Tracer, args: tuple, kwargs: dict, **options: Any) -> None:
            tracing._tree_map(_acknowledge, args, path_label="args")
            tracing._tree_map(_acknowledge, kwargs, path_label="kwargs")
            execute(self, args, kwargs, **options)

        monkeypatch.setattr(tracing.Tracer, "_execute", execute_acknowledging_inputs)

    if warm_copy:
        capture = tracing.Tracer._capture

        def capture_after_warm_copy(self: tracing.Tracer, args: tuple, kwargs: dict, **options: Any) -> None:
            tracing._tree_map(_warm_copy, args, path_label="args")
            tracing._tree_map(_warm_copy, kwargs, path_label="kwargs")
            capture(self, args, kwargs, **options)

        monkeypatch.setattr(tracing.Tracer, "_capture", capture_after_warm_copy)

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)

    for i in range(2):
        start = time.perf_counter()
        (image,) = pipeline(prompts=[PROMPT], num_inference_steps=50, seed=i, width=1024, height=1024)
        logger.info(f"generation {i}: {time.perf_counter() - start:.1f} s")
        image.save(f"qwenimage_copy_warm_{warm_copy}_{i}.png")
