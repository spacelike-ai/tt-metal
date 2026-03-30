# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Sequence


class Solver(ABC):
    @abstractmethod
    def step(
        self,
        *,
        step: int,
        latent: ttnn.Tensor,
        sigmas: Sequence[float],
        alphas: Sequence[float],
        velocity_pred: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Advance the latent one step toward the clean data.

        Args:
            step: Current step index into the sigmas/alphas schedule.
            latent: Noisy latent at the current step.
            sigmas: Full noise schedule (length = num_steps + 1).
            alphas: Full signal schedule (length = num_steps + 1).
            velocity_pred: Predicted velocity at the current step.

        Returns:
            The predicted latent at the next step.
        """
