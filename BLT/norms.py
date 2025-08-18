from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization.

    This module implements RMSNorm, a normalization technique that normalizes inputs
    using the root mean square of the elements in the last dimension, followed by a learnable
    scale (weight) and bias. RMSNorm is commonly used in transformer architectures for
    improved stability and performance.

    Args:
        dim (int): The dimension of the input tensor to be normalized (usually last dimension).
        eps (float, optional): Small epsilon value added for numerical stability. Default: 1e-8.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape [dim].
        bias (nn.Parameter): Learnable bias parameter of shape [dim].
        eps (float): Epsilon value for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): Epsilon value to avoid division by zero. Default: 1e-8.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale and bias parameters for affine transformation after normalization
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the RMS normalization of the input tensor.

        The input is normalized by dividing by the root mean square (RMS) of its elements
        along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim].

        Returns:
            torch.Tensor: RMS-normalized tensor of the same shape as input.
        """
        # Compute mean of squares along last dimension, add epsilon for stability
        # Normalize input by RMS
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Applies RMS normalization, then a learnable scale and bias.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim].

        Returns:
            torch.Tensor: Output tensor after RMS normalization and affine transformation.
        """
        # Normalize in float32 for numerical stability, then cast back to input dtype
        # Apply learnable scale and bias (affine transformation)
        return self._norm(x.float()).type_as(x) * self.weight + self.bias