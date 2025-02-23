from __future__ import annotations

from typing import Any

import pytest
import torch
import ttnn

from ..reference import FluxTransformer2DModel


@pytest.fixture
def device_params(request: pytest.FixtureRequest) -> dict(str, Any):
    return getattr(request, "param", {})


@pytest.fixture
def device_type(request: pytest.FixtureRequest) -> type[ttnn.Device | ttnn.MeshDevice]:
    return getattr(request, "param", ttnn.Device)


@pytest.fixture
def program_cache_enabled(request: pytest.FixtureRequest) -> bool:
    return getattr(request, "param", False)


@pytest.fixture
def device(
    *, device_params: dict(str, Any), device_type: type[ttnn.Device | ttnn.MeshDevice], program_cache_enabled: bool
) -> None:
    if device_type is ttnn.Device:
        device = ttnn.CreateDevice(0, **device_params)
        if program_cache_enabled:
            ttnn.enable_program_cache(device)

        yield device

        ttnn.CloseDevice(device)
    elif device_type is ttnn.MeshDevice:
        if ttnn.get_num_devices() < 2:
            pytest.skip("Machine has only one device")

        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **device_params)
        if program_cache_enabled:
            for device in mesh_device.get_devices():
                ttnn.enable_program_cache(device)

        yield mesh_device

        ttnn.close_mesh_device(mesh_device)
    else:
        msg = f"Unsupported device type: {device_type}"
        raise ValueError(msg)


@pytest.fixture(scope="module")
def parent_torch_model() -> FluxTransformer2DModel:
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def pytest_make_parametrize_id(
    config: pytest.Config,  # noqa: ARG001
    val: Any,  # noqa: ANN401
    argname: str,
) -> str | None:
    if argname == "program_cache_enabled":
        return "with_cache" if val else "without_cache"
    if argname == "device_type":
        return "normal_device" if val == ttnn.Device else "mesh_device" if val == ttnn.MeshDevice else None
    return None
