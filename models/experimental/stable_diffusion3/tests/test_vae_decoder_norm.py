# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from ..tt.utils import allocate_tensor_on_device_like
from ..tt.vae_decoder import TtGroupNorm, TtGroupNormParameters


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16384}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
def test_group_norm(*, device: ttnn.Device) -> None:
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    batch_size = 1
    channels = 512
    height = 16
    width = 16
    group_count = 32

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    torch_model.eval()

    parameters = TtGroupNormParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn_dtype)
    tt_model = TtGroupNorm(parameters, num_groups=group_count, eps=torch_model.eps)

    torch.manual_seed(0)

    inp = torch.randn([batch_size, channels, height, width], dtype=torch_dtype)

    tt_inp_host = ttnn.from_torch(inp.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)

    with torch.no_grad():
        out = torch_model(inp)

    tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=device)

    # # cache
    # tt_model(tt_inp)

    # # trace
    # tid = ttnn.begin_trace_capture(device)
    # tt_out = tt_model(tt_inp)
    # ttnn.end_trace_capture(device, tid)

    # # execute
    # ttnn.copy_host_to_device_tensor(tt_inp_host, tt_inp)
    # ttnn.execute_trace(device, tid)

    ttnn.copy_host_to_device_tensor(tt_inp_host, tt_inp)
    tt_out = tt_model(tt_inp)

    tt_out_torch = ttnn.to_torch(tt_out).permute(0, 3, 1, 2)

    mse = torch.nn.functional.mse_loss(
        out.to(dtype=torch.float32),
        tt_out_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"latent mse: {mse:.6f}")
    assert_with_pcc(out, tt_out_torch, pcc=0.999_900)
