# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from .base import Solver

if TYPE_CHECKING:
    from collections.abc import Sequence


class EulerSolver(Solver):
    def step(
        self,
        *,
        step: int,
        latent: ttnn.Tensor,
        sigmas: Sequence[float],
        alphas: Sequence[float],
        clean_pred: ttnn.Tensor,
    ) -> ttnn.Tensor:
        del alphas

        sigma_curr = sigmas[step]
        sigma_next = sigmas[step + 1]

        if sigma_curr == 0:
            msg = "current sigma is zero; schedule should not step from a fully denoised state"
            raise ValueError(msg)

        f = sigma_next / sigma_curr
        return f * latent + (1 - f) * clean_pred
