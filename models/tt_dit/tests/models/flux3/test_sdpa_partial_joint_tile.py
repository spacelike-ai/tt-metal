# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ring joint SDPA with a sharded joint stream that ends in a partial tile.

FLUX.3 pads its prompt to whole tiles per device and passes the true length as ``logical_l``. When
that length is not a multiple of the tile size, ring joint SDPA computes some output rows wrongly:
a few rows have errors of more than 100 %, while the padding values themselves do not matter. FLUX.3
works around it by passing the prompt as the ring stream instead; see ``Flux3Block.forward``.

The shapes are those of FLUX.3 at 1024 × 1024 on a 2 × 4 Wormhole mesh: 4096 image tokens, 24 heads
of size 128, SP 4 and TP 2, and the attention's chunk sizes.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ....tests.unit import test_ring_joint_attention as helper

# The largest per-row RMSE of an output, relative to the expected standard deviation. Correct
# results stay below 0.15; the broken rows exceed 1.
MAX_ROW_ERROR = 0.3

KNOWN_BAD = pytest.mark.xfail(reason="ring joint SDPA garbles rows when the joint stream ends in a partial tile")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
    ids=["line"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    ("padded_l", "logical_l"),
    [
        pytest.param(128, 80, marks=KNOWN_BAD),
        pytest.param(128, 96),
        pytest.param(128, 112, marks=KNOWN_BAD),
        pytest.param(128, 128),
    ],
)
def test_sdpa_partial_joint_tile(
    mesh_device: ttnn.MeshDevice, padded_l: int, logical_l: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []

    def comp_pcc(actual: torch.Tensor, expected: torch.Tensor, threshold: float) -> tuple[bool, str]:  # noqa: ARG001
        captured.append((actual.float(), expected.float()))
        return True, "checked below"

    monkeypatch.setattr(helper, "comp_pcc", comp_pcc)
    torch.manual_seed(0)
    helper.run_ring_joint_sdpa_sharded_prompt(
        mesh_device,
        b=1,
        nh=24,
        base_seq_len=4096,
        padded_seq_len=4096,
        padded_joint_seq_len=padded_l,
        d=128,
        rp_axis=1,
        rp_factor=4,
        up_axis=0,
        q_chunk_size=256,
        k_chunk_size=256,
        num_links=1,
        logical_l=logical_l,
    )

    errors = {}
    for name, (actual, expected) in zip(["spatial", "joint"], captured, strict=True):
        row_error = (actual - expected).pow(2).mean(dim=(0, 1, 3)).sqrt() / expected.std()
        errors[name] = row_error.max().item()
    assert all(e < MAX_ROW_ERROR for e in errors.values()), f"largest relative row errors: {errors}"
