# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import Embedding, Linear, Module, ModuleList, RMSNorm

from .rope import RopeConfig, RotaryEmbedding

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


@dataclass
class GenerationOutput:
    tokens: torch.Tensor
    logits: list[torch.Tensor] | None


class Transformer(Module):
    def __init__(
        self,
        *,
        embed_size: int,
        ff_size: int,
        head_size: int,
        norm_eps: float,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        attn_qkv_bias: bool,
        attn_out_bias: bool,
        vocab_size: int,
        rope_config: RopeConfig,
    ) -> None:
        super().__init__()

        self.pos_embedding = RotaryEmbedding(head_size=head_size, config=rope_config)

        self.token_embedding = Embedding(vocab_size, embed_size)
        self.layers = ModuleList(
            DecoderLayer(
                head_size=head_size,
                embed_size=embed_size,
                ff_size=ff_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                norm_eps=norm_eps,
                attn_qkv_bias=attn_qkv_bias,
                attn_out_bias=attn_out_bias,
                cache_id=i,
            )
            for i in range(num_layers)
        )
        self.final_norm = RMSNorm(embed_size, eps=norm_eps)
        self.final_linear = Linear(embed_size, vocab_size, bias=False)

        self.embed_size = embed_size  # ty:ignore
        self.ff_size = ff_size  # ty:ignore
        self.head_size = head_size  # ty:ignore
        self.norm_eps = norm_eps  # ty:ignore
        self.num_heads = num_heads  # ty:ignore
        self.num_kv_heads = num_kv_heads  # ty:ignore
        self.num_layers = num_layers  # ty:ignore
        self.attn_qkv_bias = attn_qkv_bias  # ty:ignore
        self.attn_out_bias = attn_out_bias  # ty:ignore
        self.vocab_size = vocab_size  # ty:ignore
        self.rope_config = rope_config  # ty:ignore

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        pos_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        device = tokens.device
        dtype = self.token_embedding.weight.dtype

        start_pos = cache.sequence_position if cache is not None else 0

        if mask is not None:
            assert mask.shape == (batch_size, start_pos + seq_len)

        if mask is None and start_pos != 0:
            mask = torch.ones([batch_size, seq_len], dtype=torch.bool, device=device)

        if pos_embeds is None:
            pos = _make_positions(
                start=start_pos,
                sequence_length=seq_len,
                mask=mask,
                device=device,
            )
            pos_embeds = self.pos_embedding.forward(pos, dtype=dtype)

        if mask is not None:
            mask = _make_causal_mask(mask, query_length=seq_len)

        x = self.token_embedding.forward(tokens)

        for decoder_layer in self.layers:
            x = decoder_layer.forward(
                x,
                mask=mask,
                pos_embeds=pos_embeds,
                cache=cache,
            )

        x = self.final_norm.forward(x)
        x = self.final_linear.forward(x)

        return x

    def generate(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        max_length: int,
        eos_tokens: int | Sequence[int] | None,
        top_k: int | None = None,
        top_p: float = 1,
        temperature: float = 1,
        use_cache: bool = True,
        return_logits: bool = False,
    ) -> GenerationOutput:
        # The original Llama implementation starts generation after the shortest
        # input, thereby overwriting any padding tokens that are on the right,
        # resuing that space. We use a slightly simpler approach and start
        # generation at the end of the inputs, which is also what the transformers
        # library does.

        batch_size, input_length = tokens.shape
        device = tokens.device
        dtype = self.token_embedding.weight.dtype

        if mask is not None:
            assert mask.shape == tokens.shape

            mask = mask.bool()
            mask = torch.nn.functional.pad(mask, [0, max_length - input_length], value=1)

        if eos_tokens is not None:
            if isinstance(eos_tokens, int):
                eos_tokens = [eos_tokens]
            elif len(eos_tokens) == 0:
                eos_tokens = None

        eos_token_tensor = torch.tensor(eos_tokens, device=device, dtype=tokens.dtype) if eos_tokens else None

        positions = _make_positions(start=0, sequence_length=max_length, mask=mask, device=device)
        cos, sin = self.pos_embedding.forward(positions, dtype=dtype)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cache = Cache() if use_cache else None
        prev_pos = 0

        logits = [] if return_logits else None

        for pos in range(input_length, max_length):
            current_logits = self.forward(
                tokens=tokens[:, prev_pos:],
                mask=mask[:, :pos] if mask is not None else None,
                pos_embeds=(cos[:, prev_pos:pos], sin[:, prev_pos:pos]),
                cache=cache,
            )[:, -1:, :]

            prob = torch.softmax(current_logits / temperature, 2)
            new_tokens = _sample(prob, top_k=top_k, top_p=top_p).squeeze(1)

            tokens = torch.concat([tokens, new_tokens], dim=1)

            if logits is not None:
                logits.append(current_logits.squeeze(1))

            if eos_token_tensor is not None:
                finished |= (new_tokens == eos_token_tensor).any(dim=1)
                if finished.all():
                    break

            if cache is not None:
                prev_pos = pos

        return GenerationOutput(tokens=tokens, logits=logits)


class DecoderLayer(Module):
    def __init__(
        self,
        *,
        head_size: int,
        embed_size: int,
        num_heads: int,
        num_kv_heads: int,
        ff_size: int,
        norm_eps: float,
        attn_qkv_bias: bool,
        attn_out_bias: bool,
        cache_id: Hashable,
    ) -> None:
        super().__init__()

        self.attn = Attention(
            head_size=head_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=attn_qkv_bias,
            out_bias=attn_out_bias,
            cache_id=cache_id,
        )
        self.ff = FeedForward(embed_size=embed_size, hidden_size=ff_size)
        self.attn_norm = RMSNorm(embed_size, eps=norm_eps)
        self.ff_norm = RMSNorm(embed_size, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        pos_embeds: tuple[torch.Tensor, torch.Tensor],
        cache: Cache | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.attn_norm.forward(x)
        x = self.attn.forward(x, mask=mask, pos_embeds=pos_embeds, cache=cache)
        x = x + residual

        residual = x
        x = self.ff_norm.forward(x)
        x = self.ff.forward(x)
        x = x + residual

        return x


class Attention(Module):
    def __init__(
        self,
        *,
        head_size: int,
        embed_size: int,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool,
        out_bias: bool,
        cache_id: Hashable,
    ) -> None:
        super().__init__()

        self.q_proj = Linear(embed_size, num_heads * head_size, bias=qkv_bias)
        self.k_proj = Linear(embed_size, num_kv_heads * head_size, bias=qkv_bias)
        self.v_proj = Linear(embed_size, num_kv_heads * head_size, bias=qkv_bias)
        self.o_proj = Linear(num_heads * head_size, embed_size, bias=out_bias)

        self._num_heads = num_heads  # ty:ignore
        self._num_kv_heads = num_kv_heads  # ty:ignore
        self._cache_id = cache_id  # ty:ignore

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        pos_embeds: tuple[torch.Tensor, torch.Tensor],
        cache: Cache | None = None,
    ) -> torch.Tensor:
        assert x.ndim == 3

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.unflatten(2, [self._num_heads, -1]).transpose(1, 2)
        k = k.unflatten(2, [self._num_kv_heads, -1]).transpose(1, 2)
        v = v.unflatten(2, [self._num_kv_heads, -1]).transpose(1, 2)

        cos, sin = pos_embeds
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self._cache_id, k, v)

        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            enable_gqa=True,
        )

        x = x.transpose(1, 2).flatten(2, 3)

        return self.o_proj.forward(x)


class FeedForward(Module):
    def __init__(self, embed_size: int, hidden_size: int) -> None:
        super().__init__()

        self.gate = Linear(embed_size, hidden_size, bias=False)
        self.linear_in = Linear(embed_size, hidden_size, bias=False)
        self.linear_out = Linear(hidden_size, embed_size, bias=False)
        self._act_fn = torch.nn.functional.silu  # ty:ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act_fn(self.gate.forward(x)) * self.linear_in.forward(x)
        return self.linear_out(x)


class Cache:
    def __init__(self) -> None:
        self.k_cache = {}
        self.v_cache = {}

        self._sequence_position = 0

    def update(self, cache_id: Hashable, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_id in self.k_cache:
            k = self.k_cache[cache_id] = torch.cat([self.k_cache[cache_id], k], dim=2)
            v = self.v_cache[cache_id] = torch.cat([self.v_cache[cache_id], v], dim=2)
        else:
            self.k_cache[cache_id] = k
            self.v_cache[cache_id] = v

        self._sequence_position = k.size(2)

        return k, v

    @property
    def sequence_position(self) -> int:
        return self._sequence_position


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    assert cos.ndim == 3
    assert sin.ndim == 3

    assert cos.shape == sin.shape
    assert cos.shape == (x.shape[0], *x.shape[2:]) or cos.shape == (1, *x.shape[2:])

    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.concat([torch.neg(x2), x1], dim=-1)


def _make_causal_mask(mask: torch.Tensor, *, query_length: int) -> torch.Tensor:
    batch_size, kv_length = mask.shape

    mask = mask.bool()
    mask = mask.view([batch_size, 1, 1, kv_length])
    mask = mask.expand([batch_size, 1, query_length, kv_length])

    return torch.tril(mask, diagonal=kv_length - query_length)


def _make_positions(
    *,
    start: int,
    sequence_length: int,
    mask: torch.Tensor | None,
    device: torch.DeviceLikeType | None,  # ty:ignore
) -> torch.Tensor:
    if mask is None:
        return torch.arange(start, start + sequence_length, dtype=torch.int64, device=device).unsqueeze(0)

    assert mask.ndim == 2
    assert mask.size(1) == start + sequence_length

    # Since RoPE is invariant under position shifts this is only needed if the
    # mask is not contiguous, i.e., has masked tokens in the middle. For
    # continuous masks we could just return a fixed sequence as above.
    return mask.long().cumsum(1)[:, start:] - 1


def _sample(prob: torch.Tensor, *, top_k: int | None = None, top_p: float = 1, num_samples: int = 1) -> torch.Tensor:
    assert 0 < top_p <= 1

    if top_k is None:
        top_k = prob.size(-1)
    else:
        assert top_k > 0
        top_k = min(top_k, prob.size(-1))

    output_shape = [*prob.shape[:-1], num_samples]
    prob = prob.reshape(-1, prob.shape[-1]).float()

    values, indices = torch.topk(prob, k=top_k, dim=-1)
    values = values / values.sum(dim=1, keepdim=True)

    ignore = values.cumsum(1) - values >= top_p
    values[ignore] = 0

    picked = torch.multinomial(values, num_samples=num_samples, replacement=True)
    return torch.gather(indices, 1, picked).view(output_shape)
