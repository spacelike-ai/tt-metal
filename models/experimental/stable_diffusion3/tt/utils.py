from loguru import logger

import ttnn


def increase_to_nearest_multiple(x, factor):
    """Return smallest multiple of `factor` bigger or equal to `x`."""
    return (x + factor - 1) // factor * factor


def untilize(t: ttnn.Tensor) -> ttnn.Tensor:
    end = [x - 1 for x in t.shape]
    return ttnn.untilize_with_unpadding(t, output_tensor_end=end)


def tilize(t: ttnn.Tensor) -> ttnn.Tensor:
    if t.dtype != ttnn.bfloat16:
        logger.warning("tilize_with_val_padding expects bfloat16 input")

    [*n, h, w] = list(t.shape)
    shape = [*n, increase_to_nearest_multiple(h, 32), increase_to_nearest_multiple(w, 32)]
    return ttnn.tilize_with_val_padding(t, output_tensor_shape=shape, pad_value=0.0)


def allocate_tensor_on_device_like(t: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device)
