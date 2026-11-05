# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image with and without DBCache, benchmarked like ``test_performance_qwenimage.py``.

Each test runs once per variant: ``plain`` builds the pipeline without DBCache, ``dbcache`` with
``DEFAULT_DBCACHE``. Each variant opens its own device and builds its own pipeline, with the same
device parameters, cases, unmeasured run and timed runs, and the prompts, seeds and photos of our
NVIDIA benchmark, on which DBCache's skipped steps depend. Select both variants in one
pytest call (``-k <mesh>``) so they run on the same machine; the second prints the speedups and the
PSNR of each case's DBCache image against the plain one.
"""

import statistics

import numpy as np
import pytest
from PIL import Image

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    DEFAULT_DBCACHE,
    QwenImagePipeline,
    QwenImagePipelineConfig,
)

from .device_params import DBCACHE_MESH_DEVICE_PARAMS, EDIT_DBCACHE_MESH_DEVICE_PARAMS
from .test_performance_qwenimage import (
    BENCHMARK_RUNS_PER_CASE,
    EDIT_PROMPT,
    EDIT_SEED,
    PROMPT,
    SEED,
    Case,
    benchmark_cases,
    durations,
    edit_cases,
    print_report,
    run_cases,
)

VARIANTS = [pytest.param(False, id="plain"), pytest.param(True, id="dbcache")]

# (test, mesh shape) -> variant -> case label -> (mean total, mean denoising, last image)
_RESULTS: dict[tuple[str, tuple[int, int]], dict[bool, dict[str, tuple[float, float, Image.Image]]]] = {}


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")


def benchmark(
    *,
    test: str,
    mesh_device: ttnn.MeshDevice,
    config: QwenImagePipelineConfig,
    cases: list[Case],
    prompt: str,
    seed: int,
    use_dbcache: bool,
) -> None:
    pipeline = QwenImagePipeline(device=mesh_device, config=config)
    variant = "dbcache" if use_dbcache else "plain"
    profiler, iterations, last_images = run_cases(
        pipeline,
        cases,
        prompt=prompt,
        seed=seed,
        file_prefix=f"{test}_{variant}",
        runs_per_case=BENCHMARK_RUNS_PER_CASE,
    )
    print_report(
        title=f"{test.upper()} PERFORMANCE RESULTS, {variant.upper()}",
        profiler=profiler,
        iterations=iterations,
        cases=cases,
        config=config,
        mesh_device=mesh_device,
    )
    pipeline.synchronize_devices()

    results = _RESULTS.setdefault((test, tuple(mesh_device.shape)), {})
    results[use_dbcache] = {
        case.label: (
            statistics.mean(durations(profiler, "run", iterations[case.label])),
            statistics.mean(durations(profiler, "denoising", iterations[case.label])),
            last_images[case.label],
        )
        for case in cases
    }
    if len(results) < 2:
        return

    print("\n" + "=" * 100)
    print(f"{test.upper()}: DBCACHE VS. PLAIN, mean of {BENCHMARK_RUNS_PER_CASE} runs each")
    print("=" * 100)
    print(
        f"{'Case':25} | {'Plain':>8} | {'DBCache':>8} | {'Speedup':>7} | "
        f"{'Denoising plain':>15} | {'DBCache':>8} | {'Speedup':>7} | {'PSNR':>7}"
    )
    for case in cases:
        plain_total, plain_denoising, plain_image = results[False][case.label]
        cached_total, cached_denoising, cached_image = results[True][case.label]
        print(
            f"{case.label:25} | {plain_total:7.2f}s | {cached_total:7.2f}s | {plain_total / cached_total:6.2f}x | "
            f"{plain_denoising:14.2f}s | {cached_denoising:7.2f}s | {plain_denoising / cached_denoising:6.2f}x | "
            f"{psnr(plain_image, cached_image):4.1f} dB"
        )
    print("=" * 100)


@pytest.mark.parametrize("use_dbcache", VARIANTS)
@pytest.mark.parametrize(("mesh_device", "device_params"), DBCACHE_MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_dbcache_performance(
    *, mesh_device: ttnn.MeshDevice, use_dbcache: bool, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark resolutions and inpainting, as in ``test_qwenimage_pipeline_performance``."""
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape, dbcache=DEFAULT_DBCACHE if use_dbcache else None
    )
    benchmark(
        test="qwenimage",
        mesh_device=mesh_device,
        config=config,
        cases=benchmark_cases(config),
        prompt=PROMPT,
        seed=SEED,
        use_dbcache=use_dbcache,
    )


@pytest.mark.parametrize("use_dbcache", VARIANTS)
@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_DBCACHE_MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_edit_dbcache_performance(
    *, mesh_device: ttnn.MeshDevice, use_dbcache: bool, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen-Image-Edit-2511 with the cases of ``test_qwenimage_edit_pipeline_performance``."""
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape,
        checkpoint_name="Qwen/Qwen-Image-Edit-2511",
        dbcache=DEFAULT_DBCACHE if use_dbcache else None,
    )
    benchmark(
        test="qwenimage_edit_2511",
        mesh_device=mesh_device,
        config=config,
        cases=edit_cases(),
        prompt=EDIT_PROMPT,
        seed=EDIT_SEED,
        use_dbcache=use_dbcache,
    )
