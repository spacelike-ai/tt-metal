from __future__ import annotations

from typing import Any


def substate(state: dict[str, Any], key: str) -> dict[str, Any]:
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def substate_exists(state: dict[str, Any], key: str) -> bool:
    prefix = f"{key}."

    for k in state:
        if k.startswith(prefix):
            return True

    return False
