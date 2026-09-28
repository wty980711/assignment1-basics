import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.d_k = d_k

        positions = torch.arange(max_seq_len, device=device)
        pairs = torch.arange(d_k // 2, device=device)
        frequencies = 1 / (theta ** (2 * pairs / d_k))
        angles = positions[:, None] * frequencies[None, :]

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions].to(dtype=x.dtype)
        sin = self.sin_cache[token_positions].to(dtype=x.dtype)

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        p1 = x_pairs[..., 0]
        p2 = x_pairs[..., 1]

        p1_rotated = cos * p1 - sin * p2
        p2_rotated = sin * p1 + cos * p2

        return torch.stack((p1_rotated, p2_rotated), dim=-1).flatten(start_dim=-2) 
