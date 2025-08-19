from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
        
    
class MLPwithSwiGLU(nn.Module):
    """
    MLPwithSwiGLU: Multi-Layer Perceptron with SwiGLU activation.

    This module implements a feed-forward block commonly used in transformer architectures,
    utilizing the SwiGLU activation function for improved expressiveness and efficiency.

    Args:
        dim (int): Input and output feature dimension.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4 * dim (rounded for GLU).
        dropout (float, optional): Dropout probability applied after output projection. Default: 0.0.

    Attributes:
        w1 (nn.Linear): First linear projection layer.
        w2 (nn.Linear): Output projection layer.
        w3 (nn.Linear): Gating projection layer for GLU.
        dropout (nn.Dropout): Dropout layer applied to output.
        w2.SCALE_INIT (float): Custom attribute for initialization scaling (if used in downstream code).
        mlp_bias (bool): Whether to use bias in MLP layers.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, mlp_dropout: Optional[float] = 0.0, mlp_bias: bool = True) -> None:
        """
        Initializes the MLPwithSwiGLU module.

        Sets up linear projections and dropout. Adjusts hidden_dim for GLU splitting and parameter efficiency.
        """
        super().__init__()
        
        # Default hidden dimension is 4x input dim, commonly used in transformers
        hidden_dim = hidden_dim or dim * 4

        # Adjust hidden_dim for GLU splitting (2/3 of hidden for better parameter efficiency)
        hidden_dim = int(2 * hidden_dim / 3)
        # Round hidden_dim to nearest multiple of 256 for hardware efficiency
        hidden_dim = (hidden_dim + 255) // 256 * 256

        # First projection layer (input to hidden)
        self.w1 = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        # Output projection layer (hidden to output)
        self.w2 = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        # Gating projection layer for GLU (input to hidden)
        self.w3 = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        # Dropout layer applied after output projection
        self.dropout = nn.Dropout(mlp_dropout)

        # Custom attribute for initialization scaling (if used in downstream code)
        self.w2.SCALE_INIT = 1


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLPwithSwiGLU.

        Applies SwiGLU activation and dropout:
            output = Dropout(w2(SwiGLU(w1(x), w3(x))))
            where SwiGLU(a, b) = SiLU(a) * b

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim] or [*, dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, dim] or [*, dim].
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))