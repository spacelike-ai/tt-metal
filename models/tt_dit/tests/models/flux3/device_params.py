# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared ``device_params`` dicts for the FLUX.3 tests.

Each row asks for exactly the mesh it names, so one command self-selects the rows of the machine.
"""

from ....utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

# The VAE uses conv2d, which needs L1_SMALL buffers.
L1_SMALL_SIZE = 65536
# The pipeline traces the denoising step, once per image size, reference and prompt length.
TRACE_REGION_SIZE = 250_000_000

line_params_flux3 = {**line_params_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}
ring_params_flux3 = {**ring_params_8k_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}

line_params_flux3_traced = {**line_params_flux3, "trace_region_size": TRACE_REGION_SIZE}
ring_params_flux3_traced = {**ring_params_flux3, "trace_region_size": TRACE_REGION_SIZE}
