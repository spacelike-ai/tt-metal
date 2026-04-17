# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionStart:
    name: str


@dataclass(frozen=True)
class SectionEnd:
    name: str


@dataclass(frozen=True)
class DenoiseStep:
    step: int
    total: int
    sigma: float


PipelineEvent = SectionStart | SectionEnd | DenoiseStep
PipelineEventCallback = Callable[[PipelineEvent], None]


def null_callback(_event: PipelineEvent) -> None:
    pass
