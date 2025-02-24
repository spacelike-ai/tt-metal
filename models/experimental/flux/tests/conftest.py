from __future__ import annotations

import pytest
import torch

from ..reference import FluxTransformer2DModel


@pytest.fixture(scope="module")
def parent_torch_model() -> FluxTransformer2DModel:
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model
