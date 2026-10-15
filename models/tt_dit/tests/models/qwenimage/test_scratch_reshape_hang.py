# Temporary: minimal repro for the tiled reshape hang after a trace replay (QIE-14). The Qwen2.5-VL
# patch merger's tiled reshape (tokens, 1280) -> (tokens / 4, 5120) hung on the second edit call on
# QB2 and BH Galaxy, after a DiT trace had been captured and replayed. Not for commit.
#
# Each step logs before it runs, so a hang shows where. Run with TT_METAL_OPERATION_TIMEOUT_SECONDS=5.

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.pipelines.cfg import create_submeshes
from models.tt_dit.utils.tracing import Tracer

from .device_params import MESH_DEVICE_PARAMS

HIDDEN = 1280
MERGED = HIDDEN * 4


def _reshape(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.reshape(x, (-1, MERGED))


def _churn(x: ttnn.Tensor, w: ttnn.Tensor, *, layers: int) -> ttnn.Tensor:
    # Allocates and frees many large intermediates, so the trace's buffers cover most of the DRAM
    # that is free at capture time.
    for _ in range(layers):
        y = ttnn.matmul(x, w)
        z = ttnn.gelu(y)
        x = ttnn.add(z, x)
        ttnn.deallocate(y)
        ttnn.deallocate(z)
    return x


def _check(device: ttnn.MeshDevice, torch_x: torch.Tensor, label: str) -> None:
    logger.info(f"{label}: reshape on {device.shape}")
    x = ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    y = _reshape(x)
    ttnn.synchronize_device(device)
    out = ttnn.to_torch(ttnn.get_device_tensors(y)[0])
    ok = torch.equal(out, torch_x.reshape(-1, MERGED))
    logger.info(f"{label}: done, matches torch: {ok}")
    ttnn.deallocate(x)
    ttnn.deallocate(y)
    assert ok


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("tokens", [5476, 4096], ids=["tokens5476", "tokens4096"])
@pytest.mark.parametrize("where", ["same", "other", "parent"])
def test_reshape_hang(*, mesh_device: ttnn.MeshDevice, tokens: int, where: str) -> None:
    # The QB2 preset's layout: two CFG submeshes. The vision tower runs on the first one, the DiT
    # traces on both.
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    config = DiTParallelConfig.from_tuples(cfg=(2, 0), sp=(1, 0), tp=(cols, 1)) if rows == 2 else None
    assert config is not None, "only meshes with 2 rows"
    submeshes = create_submeshes(mesh_device, config)

    reshape_device = mesh_device if where == "parent" else submeshes[0]
    trace_devices = {"same": [submeshes[0]], "other": [submeshes[1]], "parent": submeshes}[where]

    torch.manual_seed(0)
    torch_x = torch.randn(tokens, HIDDEN).bfloat16()

    _check(reshape_device, torch_x, "before trace")

    dim = 4096
    tracers = []
    for i, d in enumerate(trace_devices):
        mapper = ttnn.ReplicateTensorToMesh(d)
        x = ttnn.from_torch(torch.randn(8192, dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, mesh_mapper=mapper)
        w = ttnn.from_torch(torch.randn(dim, dim) / dim**0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, mesh_mapper=mapper)
        tracer = Tracer(lambda x, w: _churn(x, w, layers=8), device=d)
        logger.info(f"capturing trace {i}")
        tracer(x, w)
        tracers.append((tracer, x, w))
    for n in range(3):
        for i, (tracer, x, w) in enumerate(tracers):
            logger.info(f"replaying trace {i}, {n}")
            tracer(x, w)
    for d in trace_devices:
        ttnn.synchronize_device(d)

    _check(reshape_device, torch_x, "after trace")
