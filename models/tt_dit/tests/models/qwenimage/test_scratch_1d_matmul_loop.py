# Temporary: try to reproduce the WH Galaxy hang (QI-29) outside the pipeline. Loops each small-M
# 1D-mcast linear with bias that Qwen-Image runs, on the device's full (8x9 on a Galaxy) grid,
# traced, and logs the shape before each so a hang names it. Not for commit.

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.linear import MATH_FIDELITY, _linear
from models.tt_dit.utils.test import line_params_req_exact_devices

# (M, K, N) from the Qwen-Image WH Galaxy preset's "1D matmul: no swept config" log lines.
SHAPES = [
    (32, 256, 3072),
    (32, 3072, 3072),
    (32, 3072, 4608),
    (32, 3072, 6144),
    (128, 3072, 768),
    (128, 3072, 2304),
    (128, 3072, 3072),
    (128, 3584, 768),
]
OPS_PER_TRACE = 50
REPLAYS = 4000


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param(shape, {**line_params_req_exact_devices, "trace_region_size": 50_000_000}, id=f"{shape[0]}x{shape[1]}")
        for shape in [(2, 4), (4, 8)]
    ],
    indirect=True,
)
@pytest.mark.parametrize("clamp", [False, True], ids=["full_grid", "grid8x8"])
def test_1d_matmul_loop(*, mesh_device: ttnn.MeshDevice, clamp: bool) -> None:
    # Unset clamps every row count to 8x8 in `_linear`; "none" matches no row count.
    if clamp:
        os.environ.pop("TT_DIT_WH_1D_CLAMP_ROWS", None)
    else:
        os.environ["TT_DIT_WH_1D_CLAMP_ROWS"] = "none"
    logger.info(f"grid {mesh_device.compute_with_storage_grid_size()}")
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=MATH_FIDELITY[ttnn.bfloat16],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def to_device(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    for m, k, n in SHAPES:
        x = to_device(torch.randn(1, 1, m, k))
        weight = to_device(torch.randn(k, n) / k**0.5)
        bias = to_device(torch.randn(1, n))

        def run() -> None:
            for _ in range(OPS_PER_TRACE):
                ttnn.deallocate(
                    _linear(
                        x, weight, bias=bias, mesh_device=mesh_device, compute_kernel_config=compute_config, dtype=None
                    )
                )

        run()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        logger.info(f"shape {(m, k, n)}: {REPLAYS} replays of {OPS_PER_TRACE} ops")
        for i in range(REPLAYS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if i % 50 == 49:
                ttnn.synchronize_device(mesh_device)
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace_id)
        logger.info(f"shape {(m, k, n)}: done")
