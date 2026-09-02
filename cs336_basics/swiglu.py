import torch
from torch import nn
from jaxtyping import Float

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]):
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        w1x = x_float @ self.w1.to(torch.float32).T
        silu = w1x / (1 + torch.exp(-w1x))
        w3x = x_float @ self.w3.to(torch.float32).T
        glu = silu * w3x
        swiglu = glu @ self.w2.to(torch.float32).T

        return swiglu.to(input_dtype)
