from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the frequency bands
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache for cos and sin embeddings
        self._build_cache()
    
    def _build_cache(self):
        seq_len = self.max_position_embeddings
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        Args:
            x: input tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        return self.apply_rotary_pos_emb(x, self.cos_cached[:seq_len], self.sin_cached[:seq_len])
    
    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input tensor"""
        # Split the last dimension into two halves for rotation
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        ], dim=-1)
        
        return rotated