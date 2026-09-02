import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                d_model,
                device=device,
                dtype=dtype
            )
        )

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight.to(torch.float32)).to(input_dtype)
