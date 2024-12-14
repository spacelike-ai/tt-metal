import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/normalization.py
class SD35AdaLayerNormZeroX(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, 9 * embedding_dim)
        self.norm = torch.nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=1e-6
        )

    def forward(
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
        hidden_states = (
            norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        norm_hidden_states2 = (
            norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        )

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_hidden_states2,
            gate_msa2,
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/normalization.py
class AdaLayerNormZero(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = torch.nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=1e-6
        )

    def forward(
        self, x: torch.Tensor, *, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/normalization.py
class AdaLayerNormContinuous(torch.nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(conditioning_embedding_dim, embedding_dim * 2)
        self.norm = torch.nn.LayerNorm(
            embedding_dim, eps=1e-6, elementwise_affine=False
        )

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/normalization.py
class RmsNorm(torch.nn.Module):
    def __init__(self, *, dim: int, eps: float) -> None:
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(self.weight.dtype) * self.weight
