# Scratch: sweeps conv3d blockings for the (1, 3, 3) convs of the Qwen-Image and FIBO VAEs at 1024²,
# on one device with the per-device shape of a split mesh. Inputs include the 1-pixel halo, so the
# conv itself pads nothing.

import inspect
import textwrap

import pytest

import ttnn
from models.tt_dit.tests.models.wan2_2 import bruteforce_conv3d_sweep as sweep
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, run_sweep

_OUT = "/tmp/claude-1000/-home-friedrich-tenstorrent/da7563d8-39d7-4864-bdbe-0c9834ee8b5e/scratchpad/conv3d_sweep"

# (name, C_in (aligned), C_out, resolution) at the largest resolution each channel pair runs at.
_QWEN_LAYERS = [
    ("dec_conv_in", 32, 384, 128),
    ("dec_up1_res", 384, 384, 256),
    ("dec_up1_upsample", 384, 192, 512),
    ("dec_up1_res0", 192, 384, 256),
    ("dec_up2_res", 192, 192, 512),
    ("dec_up2_upsample", 192, 96, 1024),
    ("dec_up3_res", 96, 96, 1024),
    ("dec_conv_out", 96, 3, 1024),
    ("enc_conv_in", 32, 96, 1024),
    ("enc_down1_res0", 96, 192, 512),
    ("enc_conv_out", 384, 16, 128),
]

_FIBO_LAYERS = [
    ("fibo_dec_conv_in", 64, 1024, 64),
    ("fibo_dec_1024", 1024, 1024, 256),
    ("fibo_dec_1024_512", 1024, 512, 256),
    ("fibo_dec_512", 512, 512, 512),
    ("fibo_dec_512_256", 512, 256, 512),
    ("fibo_dec_256", 256, 256, 512),
    ("fibo_dec_conv_out", 256, 12, 512),
    ("fibo_enc_conv_in", 32, 160, 512),
    ("fibo_enc_160", 160, 160, 512),
    ("fibo_enc_160_320", 160, 320, 256),
    ("fibo_enc_320", 320, 320, 256),
    ("fibo_enc_320_640", 320, 640, 128),
    ("fibo_enc_640", 640, 640, 128),
    ("fibo_enc_conv_out", 640, 48, 64),
]

# (arch, model, H split, W split) of the VAE submesh in each pipeline preset.
_SPLITS = [
    ("wh", "qwen", 4, 2),  # T3K
    ("wh", "qwen", 4, 4),  # WH Galaxy
    ("wh", "fibo", 2, 2),  # T3K
    ("bh", "qwen", 2, 1),  # QB2
    ("bh", "qwen", 4, 1),  # BH LoudBox
    ("bh", "qwen", 4, 4),  # BH Galaxy
    ("bh", "fibo", 2, 1),  # QB2
    ("bh", "fibo", 4, 4),  # BH Galaxy
]

_CASES = [
    pytest.param(arch, h_factor, w_factor, *layer, id=f"{arch}{h_factor}x{w_factor}-{layer[0]}")
    for arch, model, h_factor, w_factor in _SPLITS
    for layer in (_QWEN_LAYERS if model == "qwen" else _FIBO_LAYERS)
]


def _build_all_blockings_keeping_small_cout():  # noqa: ANN202
    """Returns the tool's build_all_blockings, but only dropping small C_out blocks on large tensors
    when a valid larger one remains. Otherwise C_out 160 and 320 get no candidates at all."""
    source = textwrap.dedent(inspect.getsource(sweep.build_all_blockings))
    old = """    if tensor_vol > 2000 and padded_cout > 64:
        couts = [c for c in couts if c >= 64]
    if tensor_vol > 10000 and padded_cout > 96:
        couts = [c for c in couts if c >= 96]
"""
    new = """    for min_vol, min_cout in [(2000, 64), (10000, 96)]:
        large = [c for c in couts if c >= min_cout]
        if tensor_vol > min_vol and padded_cout > min_cout and any(valid_matmul_subblock(c) for c in large):
            couts = large
"""
    assert old in source
    namespace = dict(vars(sweep))
    exec(source.replace(old, new), namespace)  # noqa: S102
    return namespace["build_all_blockings"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], ids=["trace"], indirect=True)
@pytest.mark.parametrize(("arch", "h_factor", "w_factor", "name", "c_in", "c_out", "resolution"), _CASES)
def test_sweep(
    *,
    mesh_device: ttnn.MeshDevice,
    arch: str,
    h_factor: int,
    w_factor: int,
    name: str,
    c_in: int,
    c_out: int,
    resolution: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if arch != ("bh" if ttnn.device.is_blackhole() else "wh"):
        pytest.skip("other arch")
    monkeypatch.setattr(sweep, "build_all_blockings", _build_all_blockings_keeping_small_cout())
    run_sweep(
        mesh_device,
        c_in,
        c_out,
        (1, 3, 3),
        1,
        resolution // h_factor + 2,
        resolution // w_factor + 2,
        f"{_OUT}/{arch}{h_factor}x{w_factor}_{name}_{c_in}x{c_out}.json",
        h_factor=h_factor,
        w_factor=w_factor,
        grid_size=mesh_device.compute_with_storage_grid_size(),
        max_combos=150,
    )
