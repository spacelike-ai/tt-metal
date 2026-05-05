# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .events import PipelineEventCallback


_R = TypeVar("_R", covariant=True)


class _Pipeline(Protocol[_R]):
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int = ...,
        seed: int | None = None,
        traced: bool = ...,
        on_event: PipelineEventCallback | None = None,
    ) -> _R:
        ...


class PipelineAPIMixin:
    def run_single_prompt(
        self: _Pipeline[_R],
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> _R:
        kwargs = {}

        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps

        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt is not None else None,
            seed=seed,
            traced=True,
            on_event=on_event,
            **kwargs,
        )
