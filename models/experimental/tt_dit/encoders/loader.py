# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .rope import RopeConfig
from .transformer import Transformer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    model_desc = json.loads(Path(f"models/{name}.json").read_text())
    tokenizer_desc = model_desc["tokenizer"]

    tokenizer_type = _get_class_by_name(tokenizer_desc["reference_implementation"])
    tokenizer = tokenizer_type.from_pretrained(
        tokenizer_desc["repo_id"],
        subfolder=tokenizer_desc["folder"],
        revision=tokenizer_desc["revision"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_transformer_models(name: str, *, dtype: torch.dtype | None) -> tuple[PreTrainedModel, Transformer]:
    model_desc = json.loads(Path(f"models/{name}.json").read_text())
    text_encoder_desc = model_desc["text_encoder"]

    reference_model_type = _get_class_by_name(text_encoder_desc["reference_implementation"])
    reference_model = reference_model_type.from_pretrained(
        text_encoder_desc["repo_id"],
        subfolder=text_encoder_desc["folder"],
        revision=text_encoder_desc["revision"],
        device_map="auto",
        dtype=dtype,
    )
    reference_model.eval()

    with torch.device("meta"):
        model = Transformer(
            embed_size=model_desc["embed_size"],
            ff_size=model_desc["ff_size"],
            head_size=model_desc["head_size"],
            norm_eps=model_desc["norm_eps"],
            num_heads=model_desc["num_heads"],
            num_kv_heads=model_desc["num_kv_heads"],
            num_layers=model_desc["num_layers"],
            attn_qkv_bias=model_desc["attn_qkv_bias"],
            attn_out_bias=model_desc["attn_out_bias"],
            vocab_size=model_desc["vocab_size"],
            rope_config=RopeConfig.from_dict(model_desc["rope"]),
        )

    state_dict = _convert_state_dict(reference_model.state_dict(), model_desc.get("rename"), model_desc.get("remove"))
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    return reference_model, model


def _convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    rename: Sequence[Sequence[str]] | None,
    remove: Sequence[str] | None,
) -> dict[str, torch.Tensor]:
    in_ = dict(state_dict)
    out = {}

    compiled = [(re.compile(pattern), template) for (pattern, template) in rename or []]

    for k in list(in_):
        transformed = False
        for pattern, t in compiled:
            new_k, count = pattern.subn(t, k, count=1)
            if count == 1:
                if transformed:
                    msg = f"multiple renames for key: {k}"
                    raise RuntimeError(msg)
                if new_k in out:
                    msg = f"key collision: {new_k}"
                    raise RuntimeError(msg)
                out[new_k] = in_.pop(k)
                transformed = True

        for pattern in remove or []:
            if re.search(pattern, k):
                if transformed:
                    msg = f"multiple renames/removes for key: {k}"
                    raise RuntimeError(msg)
                in_.pop(k)
                transformed = True

    if in_:
        warnings.warn(f"unprocessed keys remain: {', '.join(in_.keys())}", stacklevel=2)

    return {**in_, **out}


def _get_class_by_name(name: str) -> Any:
    module_name, class_name = name.rsplit(".", maxsplit=1)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        msg = f"not a type: {name}"
        raise TypeError(msg)

    return cls
