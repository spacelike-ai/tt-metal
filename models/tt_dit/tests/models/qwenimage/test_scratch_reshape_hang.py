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
@pytest.mark.parametrize("where", ["same", "other"])
@pytest.mark.parametrize("order", ["reshape_first", "trace_first"])
def test_reshape_hang(*, mesh_device: ttnn.MeshDevice, tokens: int, where: str, order: str) -> None:
    # The QB2 preset's layout: two CFG submeshes. The vision tower runs on the first one, the DiT
    # traces on both. "trace_first" runs the first reshape, which allocates the op-owned mapping
    # tensor, after the capture, as the pipeline did before it warmed up the vision tower.
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    assert rows == 2, "only meshes with 2 rows"
    config = DiTParallelConfig.from_tuples(cfg=(2, 0), sp=(1, 0), tp=(cols, 1))
    submeshes = create_submeshes(mesh_device, config)
    reshape_device = submeshes[0]
    trace_device = submeshes[0] if where == "same" else submeshes[-1]

    torch.manual_seed(0)
    torch_x = torch.randn(tokens, HIDDEN).bfloat16()

    if order == "reshape_first":
        _check(reshape_device, torch_x, "first reshape")

    dim = 4096
    mapper = ttnn.ReplicateTensorToMesh(trace_device)
    x = ttnn.from_torch(
        torch.randn(8192, dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=trace_device, mesh_mapper=mapper
    )
    w = ttnn.from_torch(
        torch.randn(dim, dim) / dim**0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=trace_device,
        mesh_mapper=mapper,
    )
    tracer = Tracer(lambda x, w: _churn(x, w, layers=8), device=trace_device)
    logger.info("capturing trace")
    tracer(x, w)
    ttnn.synchronize_device(trace_device)

    if order == "trace_first":
        _check(reshape_device, torch_x, "first reshape")

    for n in range(3):
        logger.info(f"replaying trace {n}")
        tracer(x, w)
    ttnn.synchronize_device(trace_device)

    _check(reshape_device, torch_x, "second reshape")


