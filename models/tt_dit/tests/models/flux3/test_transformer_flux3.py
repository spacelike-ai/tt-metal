# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_flux3 import Flux3Checkpoint, interleave_shards, padded_prompt_length
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from .device_params import line_params_flux3, ring_params_flux3
from .reference_flux3 import Flux3Transformer2DModel

CHECKPOINT = "black-forest-labs/flux-3-image-klein"


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, topology, num_links, device_params",
    [
        [(2, 2), 0, 1, ttnn.Topology.Linear, 2, line_params_flux3],
        [(2, 4), 1, 0, ttnn.Topology.Linear, 1, line_params_flux3],
        [(2, 4), 0, 1, ttnn.Topology.Linear, 1, line_params_flux3],
        [(4, 8), 0, 1, ttnn.Topology.Ring, 2, ring_params_flux3],
    ],
    ids=["2x2sp0tp1", "2x4sp1tp0", "2x4sp0tp1", "4x8sp0tp1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("num_layers", "num_single_layers"),
    [
        pytest.param(1, 2, id="few_blocks"),
        pytest.param(None, None, id="all_blocks"),
    ],
)
# 640 tokens are whole tiles on up to four devices, so they need no padding.
@pytest.mark.parametrize("prompt_length", [80, 400, 640])
@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "edit"])
def test_transformer(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    num_layers: int | None,
    num_single_layers: int | None,
    prompt_length: int,
    edit: bool,
    model_location_generator,  # noqa: ANN001
) -> None:
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    # 1024² output, 512² reference.
    height, width = 64, 64
    reference_height, reference_width = 32, 32

    name = model_location_generator(CHECKPOINT)
    checkpoint = Flux3Checkpoint(name)
    config = checkpoint.config
    num_layers = config["num_layers"] if num_layers is None else num_layers
    num_single_layers = config["num_single_layers"] if num_single_layers is None else num_single_layers

    torch_model = Flux3Transformer2DModel.from_pretrained(
        name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
    )
    assert isinstance(torch_model, Flux3Transformer2DModel)
    torch_model.eval()
    assert len(torch_model.single_transformer_blocks) == num_single_layers

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    tt_model = checkpoint.build(
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        get_torch_state_dict=torch_model.state_dict,
    )

    torch.manual_seed(0)
    spatial = torch.randn([1, height * width, config["in_channels"]])
    prompt = torch.randn([1, prompt_length, config["joint_attention_dim"]])
    reference = torch.randn([1, reference_height * reference_width, config["cond_in_channels"]]) if edit else None
    timestep = 0.7
    target_timestep = 0.55 if checkpoint.flow_map else None

    image_ids = checkpoint.image_ids(height=height, width=width)
    text_ids = checkpoint.text_ids(prompt_length)
    reference_ids = checkpoint.image_ids(height=reference_height, width=reference_width, index=-1) if edit else None

    logger.info("running the torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial.to(torch.bfloat16),
            encoder_hidden_states=prompt.to(torch.bfloat16),
            timestep=torch.tensor([timestep]),
            img_ids=image_ids,
            txt_ids=text_ids,
            image_cond_hidden_states=reference.to(torch.bfloat16) if reference is not None else None,
            image_cond_ids=reference_ids,
            target_timestep=torch.tensor([target_timestep]) if target_timestep is not None else None,
        ).sample.float()

    logger.info("running the TT model...")
    padded_length = padded_prompt_length(prompt_length, sp_factor=sp_factor)
    seq_axes = [None, sp_axis, None]
    tt_prompt = tensor.from_torch(
        torch.nn.functional.pad(prompt, (0, 0, 0, padded_length - prompt_length)),
        device=mesh_device,
        mesh_axes=seq_axes,
    )

    inputs = {}
    if reference is not None:
        assert reference_ids is not None
        inputs["reference"] = tensor.from_torch(reference, device=mesh_device, mesh_axes=seq_axes)
        inputs["reference_rope"] = checkpoint.rope_tensor(reference_ids, device=mesh_device, sp_axis=sp_axis)
        inputs["joint_rope"] = checkpoint.rope_tensor(
            interleave_shards(image_ids, reference_ids, sp_factor), device=mesh_device, sp_axis=sp_axis
        )
        inputs["reference_sequence_length"] = reference.shape[1]

    tt_output = tt_model.forward(
        spatial=tensor.from_torch(spatial, device=mesh_device, mesh_axes=seq_axes),
        prompt=tt_model.context_embedder(tt_prompt),
        timestep=_timestep_tensor(timestep, device=mesh_device),
        target_timestep=_timestep_tensor(target_timestep, device=mesh_device) if target_timestep is not None else None,
        spatial_rope=checkpoint.rope_tensor(image_ids, device=mesh_device, sp_axis=sp_axis),
        prompt_rope=checkpoint.rope_tensor(text_ids, device=mesh_device, sp_axis=sp_axis, padded_length=padded_length),
        spatial_sequence_length=spatial.shape[1],
        prompt_sequence_length=prompt_length,
        **inputs,
    )
    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=seq_axes)

    assert_quality(torch_output, tt_output_torch, pcc=0.99, relative_rmse=0.15)


def _timestep_tensor(t: float, *, device: ttnn.MeshDevice) -> ttnn.Tensor:
    return tensor.from_torch(torch.full([1, 1], t * 1000), dtype=ttnn.float32, device=device)
