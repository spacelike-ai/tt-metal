# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Union

import torch
import ttnn

from ..utils import tensor
from .base import Solver

if TYPE_CHECKING:
    from collections.abc import Sequence

    Tensor = Union[torch.Tensor, ttnn.Tensor]


@dataclass(frozen=True)
class _State:
    clean_preds: tuple[Tensor, ...]
    corrected: Tensor


class UniPCVariant(Enum):
    B1 = "b1"  # B(h) = h
    B2 = "b2"  # B(h) = 1 - e^(-h)

    def b(self, h: float) -> float:
        if self is UniPCVariant.B1:
            return h
        return 1 - math.exp(-h)


class UniPCSolver(Solver):
    def __init__(self, *, order: int, variant: UniPCVariant) -> None:
        if order not in (1, 2):
            msg = "only order 1 and 2 are supported"
            raise ValueError(msg)

        self.order = order
        self.variant = variant
        self._state = None

    def step(
        self,
        *,
        step: int,
        latent: Tensor,
        sigmas: Sequence[float],
        alphas: Sequence[float],
        clean_pred: Tensor,
    ) -> Tensor:
        state = self._state or _State(
            tuple(_empty_like(latent) for _ in range(self.order)),
            _empty_like(latent),
        )

        if step != 0:
            corrected = self._correct(
                order=_taper(self.order, step - 1, len(sigmas) - 1),
                latent=state.corrected,
                step=step - 1,
                sigmas=sigmas,
                alphas=alphas,
                clean_preds=(*state.clean_preds, clean_pred),
            )
        else:
            corrected = latent

        del latent

        _copy(corrected, state.corrected)
        del corrected

        _copy(clean_pred, state.clean_preds[0])
        clean_preds = (*state.clean_preds[1:], state.clean_preds[0])
        del clean_pred

        predicted = self._predict(
            order=_taper(self.order, step, len(sigmas) - 1),
            latent=state.corrected,
            step=step,
            sigmas=sigmas,
            alphas=alphas,
            clean_preds=clean_preds,
        )

        self._state = _State(clean_preds, state.corrected)
        return predicted

    def _predict(
        self,
        *,
        order: int,
        latent: Tensor,
        step: int,
        sigmas: Sequence[float],
        alphas: Sequence[float],
        clean_preds: Sequence[Tensor],
    ) -> Tensor:
        sigma_curr, sigma_next = sigmas[step : step + 2]
        alpha_curr, alpha_next = alphas[step : step + 2]

        lam_curr = _log_div(alpha_curr, sigma_curr)
        lam_next = _log_div(alpha_next, sigma_next)
        h = lam_next - lam_curr

        coeff_latent = sigma_next / sigma_curr
        coeff_curr = alpha_next * (1 - math.exp(-h))

        if order == 1:
            return coeff_latent * latent + coeff_curr * clean_preds[-1]

        lam_prev = _log_div(alphas[step - 1], sigmas[step - 1])
        r = (lam_prev - lam_curr) / h
        w = alpha_next * self.variant.b(h) * 0.5 / r

        return coeff_latent * latent + (coeff_curr - w) * clean_preds[-1] + w * clean_preds[-2]

    def _correct(
        self,
        *,
        order: int,
        latent: Tensor,
        step: int,
        sigmas: Sequence[float],
        alphas: Sequence[float],
        clean_preds: Sequence[Tensor],
    ) -> Tensor:
        sigma_curr, sigma_next = sigmas[step : step + 2]
        alpha_curr, alpha_next = alphas[step : step + 2]

        lam_curr = _log_div(alpha_curr, sigma_curr)
        lam_next = _log_div(alpha_next, sigma_next)
        h = lam_next - lam_curr
        exp_neg_h = math.exp(-h)

        coeff_latent = sigma_next / sigma_curr
        coeff_clean = alpha_next * (1 - exp_neg_h)

        if order == 1:
            # UniC-1: c=0.5, r=1
            w = alpha_next * self.variant.b(h) * 0.5
            return coeff_latent * latent + (coeff_clean - w) * clean_preds[-2] + w * clean_preds[-1]

        # UniC-2: solve 2x2 system
        lam_prev = _log_div(alphas[step - 1], sigmas[step - 1])
        r_1 = (lam_prev - lam_curr) / h

        g1 = (h - 1 + exp_neg_h) / h**2
        g2 = (h**2 - 2 * h + 2 - 2 * exp_neg_h) / h**2

        det = h * (1 - r_1)
        c_1 = (h * g1 - g2) / det
        c_2 = (g2 - r_1 * h * g1) / det

        w_prev = alpha_next * h * c_1 / r_1
        w_pred = alpha_next * h * c_2

        return (
            coeff_latent * latent
            + w_prev * clean_preds[-3]
            + (coeff_clean - w_prev - w_pred) * clean_preds[-2]
            + w_pred * clean_preds[-1]
        )


def _empty_like(x: Tensor) -> Tensor:
    if isinstance(x, torch.Tensor):
        return torch.empty_like(x)
    return tensor.empty_like(x)


def _copy(src: Tensor, dst: Tensor) -> None:
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    else:
        ttnn.copy(src, dst)


def _taper(order: int, step: int, num_steps: int) -> int:
    return min(order, step + 1, num_steps - step)


def _log_div(alpha: float, sigma: float) -> float:
    eps = 1e-6
    return math.log(max(alpha, eps) / max(sigma, eps))
