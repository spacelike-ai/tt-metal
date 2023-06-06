from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}")
sys.path.append(f"{f}/../../../..")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import pytest

import tt_lib
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
    tt2torch_tensor,
)

from mnist import *

_batch_size = 1


def run_mnist_inference(pcc, PERF_CNT=1):
    # Initialize the device
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        torch_ConvNet, state_dict = load_torch("convnet_mnist.pt")
        test_dataset, test_loader = prep_data()
        first_input, label = next(iter(test_loader))

        tt_convnet = TtConvNet(device, host, state_dict)

        profiler.enable()

        # Run one input through the network

        profiler.start("\nExec time of reference model")
        pytorch_out = torch_ConvNet(first_input)
        profiler.end("\nExec time of reference model")

        tt_image = tt_lib.tensor.Tensor(
            first_input.reshape(-1).tolist(),
            first_input.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        profiler.start("\nExecution time of tt_mnist first run")
        tt_out = tt_convnet(tt_image)
        profiler.end("\nExecution time of tt_mnist first run")

        enable_compile_cache()

        logger.info(f"\nRunning the tt_mnist model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_mnist model")
            tt_output = tt_convnet(tt_image)
            profiler.end("\nAverage execution time of tt_mnist model")

        tt_output = tt2torch_tensor(tt_output, host)

        logger.info(
            f"Sample Correct Output from the batch: {pytorch_out.topk(1).indices[0]}"
        )
        logger.info(
            f"Sample Predicted Output from the batch: {tt_output.topk(1).indices[0]}\n"
        )

        pcc_passing, pcc_output = comp_pcc(pytorch_out, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        profiler.print()

    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, iter",
    ((0.99, 2),),
)
def test_mnist_inference(pcc, iter):
    disable_compile_cache()
    run_mnist_inference(pcc, iter)


if __name__ == "__main__":
    run_mnist_inference(0.99, 2)
