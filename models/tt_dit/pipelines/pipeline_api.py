# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    from .events import PipelineEventCallback


class _Pipeline(Protocol):
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int = ...,
        seed: int | None = None,
        traced: bool = ...,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        ...


class PipelineAPIMixin:
    def run_single_prompt(
        self: _Pipeline,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        kwargs = {}

        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps

        if traced is not None:
            kwargs["traced"] = traced

        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt is not None else None,
            seed=seed,
            on_event=on_event,
            **kwargs,
        )
