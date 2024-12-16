from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .linear import TtLinear, TtLinearParameters
from .substate import substate


@dataclass
class TtRmsNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtRmsNormParameters:
        return cls(
            weight=ttnn.from_torch(
                state["weight"],
                dtype=dtype,
                device=device,
            )
        )


@dataclass
class TtLayerNormParameters:
    weight: ttnn.Tensor | None
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtRmsNormParameters:
        torch_weight = state["weight"]
        torch_bias = state["bias"]

        return cls(
            weight=ttnn.from_torch(
                torch_weight,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if torch_weight is not None
            else None,
            bias=ttnn.from_torch(
                torch_bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
            if torch_bias is not None
            else None,
        )


@dataclass
class TtAdaLayerNormParameters:
    linear: TtLinearParameters
    norm: TtLayerNormParameters


class TtSD35AdaLayerNormZeroX:
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, 9 * embedding_dim)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def __call__(
        self, hidden_states: torch.Tensor, *, emb: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        emb = self.linear(self.silu(emb))

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = emb.chunk(9, dim=1)

        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_hidden_states2,
            gate_msa2,
        )


class TtAdaLayerNormZero:
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def __call__(
        self, x: torch.Tensor, *, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TtAdaLayerNormContinuous:
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(conditioning_embedding_dim, embedding_dim * 2)
        self.norm = torch.nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def __call__(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class TtRmsNorm:
    def __init__(self, parameters: TtRmsNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = ttnn.to_torch(parameters.weight)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        torch_x = ttnn.to_torch(x)

        variance = torch_x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        torch_x = torch_x * torch.rsqrt(variance + self._eps) * self._weight

        return ttnn.from_torch(torch_x, layout=x.layout, dtype=x.dtype, device=x.device())


class TtLayerNorm:
    def __init__(self, parameters: TtLayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self._weight, bias=self._bias, epsilon=self._eps)
