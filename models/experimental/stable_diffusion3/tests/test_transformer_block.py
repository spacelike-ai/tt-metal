import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference import SD3Transformer2DModel
from ..reference.transformer_block import TransformerBlock
from ..tt.transformer_block import TtTransformerBlock, TtTransformerBlockParameters
from ..tt.utils import allocate_tensor_on_device_like


@pytest.mark.parametrize(
    "block_index, batch_size, spatial_sequence_length, prompt_sequence_length",
    [
        (0, 2, 1024, 333),
        (23, 2, 1024, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
def test_transformer_block(
    *,
    device: ttnn.Device,
    use_program_cache: None,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
):
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

    parameters = TtTransformerBlockParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn_dtype)
    tt_model = TtTransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 1536

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim), dtype=torch_dtype)
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim), dtype=torch_dtype)
    time = torch.randn((batch_size, embedding_dim))

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)
    tt_time_host = ttnn.from_torch(time.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    tt_time = allocate_tensor_on_device_like(tt_time_host, device=device)

    # cache
    tt_model(spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time)

    # trace
    tid = ttnn.begin_trace_capture(device)
    tt_spatial_output, tt_prompt_output = tt_model(spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time)
    ttnn.end_trace_capture(device, tid)

    # execute
    ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
    ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
    ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
    ttnn.execute_trace(device, tid)

    assert (prompt_output is None) == (tt_prompt_output is None)

    tt_prompt_output_torch = ttnn.to_torch(tt_prompt_output) if tt_prompt_output is not None else None
    tt_spatial_output_torch = ttnn.to_torch(tt_spatial_output)

    if prompt_output is not None and tt_prompt_output_torch is not None:
        mse = torch.nn.functional.mse_loss(
            prompt_output.to(dtype=torch.float32),
            tt_prompt_output_torch.to(dtype=torch.float32),
        ).item()
        logger.info(f"prompt mse: {mse:.6f}")
        assert_with_pcc(prompt_output, tt_prompt_output_torch, pcc=0.995)

    assert spatial_output.shape == tt_spatial_output_torch.shape
    mse = torch.nn.functional.mse_loss(
        spatial_output.to(dtype=torch.float32),
        tt_spatial_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"spatial mse: {mse:.6f}")
    assert_with_pcc(spatial_output, tt_spatial_output_torch, pcc=0.995)
