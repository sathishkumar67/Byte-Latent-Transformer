from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional



class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    This module applies rotary position embeddings to input tensors, enabling models to encode
    relative positional information in a way that is efficient and compatible with attention mechanisms.

    Args:
        dim (int): Dimension of the embedding (typically head_dim).
        max_position_embeddings (int, optional): Maximum sequence length for which to cache embeddings. Default: 2048.
        base (int, optional): Base for computing rotary frequencies. Default: 10000.

    Attributes:
        dim (int): Embedding dimension.
        max_position_embeddings (int): Maximum sequence length for cached embeddings.
        base (int): Base for rotary frequency computation.
        inv_freq (torch.Tensor): Precomputed inverse frequencies for rotary embedding.
        cos_cached (torch.Tensor): Cached cosine values for all positions.
        sin_cached (torch.Tensor): Cached sine values for all positions.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Initializes the RotaryPositionEmbedding module.

        Precomputes inverse frequencies and caches cosine/sine embeddings for efficient application.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the frequency bands for rotary embedding
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build and cache cosine and sine embeddings for all positions
        self._build_cache()
    
    def _build_cache(self):
        """
        Precompute and cache cosine and sine embeddings for all positions up to max_position_embeddings.
        """
        seq_len = self.max_position_embeddings
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)  # Position indices
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)     # Compute frequencies for each position
        emb = torch.cat((freqs, freqs), dim=-1)               # Duplicate frequencies for rotation
        
        # Cache cosine and sine values for all positions
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, head_dim].
            seq_len (Optional[int]): Sequence length to use for embeddings. If None, uses x.shape[1].

        Returns:
            torch.Tensor: Tensor with rotary position embedding applied, same shape as input.
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Select cached cos/sin embeddings for the current sequence length
        return self.apply_rotary_pos_emb(x, self.cos_cached[:seq_len], self.sin_cached[:seq_len])
    
    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, head_dim].
            cos (torch.Tensor): Cosine embedding tensor of shape [seq_len, head_dim].
            sin (torch.Tensor): Sine embedding tensor of shape [seq_len, head_dim].

        Returns:
            torch.Tensor: Tensor with rotary position embedding applied.
        """
        # Split the last dimension into two halves for rotation
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Reshape cos and sin for broadcasting: [1, seq_len, 1, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Apply rotation to each half and concatenate
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        ], dim=-1)
        
        return rotated
    

# class RotaryEmbedding(nn.Module):
#     """Rotary Position Embedding"""
    
#     def __init__(self, dim: int, theta: float = 10000.0):
#         super().__init__()
#         self.dim = dim
#         self.theta = theta
        
#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         seq_len: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         device = q.device
#         dtype = q.dtype
        
#         # Create position indices
#         pos = torch.arange(seq_len, device=device, dtype=dtype)
        
#         # Create frequency bands
#         freq_seq = torch.arange(0, self.dim, 2, device=device, dtype=dtype)
#         inv_freq = 1.0 / (self.theta ** (freq_seq / self.dim))
        
#         # Compute rotary embeddings
#         sincos = torch.einsum('i,j->ij', pos, inv_freq)
#         sin, cos = sincos.sin(), sincos.cos()
        
#         # Apply rotary embeddings
#         q_rot = self._apply_rotary(q, sin, cos)
#         k_rot = self._apply_rotary(k, sin, cos)
        
#         return q_rot, k_rot
    
#     def _apply_rotary(
#         self,
#         x: torch.Tensor,
#         sin: torch.Tensor,
#         cos: torch.Tensor,
#     ) -> torch.Tensor:
#         # x shape: (batch, heads, seq_len, dim)
#         x1, x2 = x[..., ::2], x[..., 1::2]
        
#         # Apply rotation
#         y1 = x1 * cos - x2 * sin
#         y2 = x1 * sin + x2 * cos
        
#         # Interleave back
#         y = torch.stack([y1, y2], dim=-1).flatten(-2)
#         return y    


# class OptimizedRotaryEmbedding(nn.Module):
#     """
#     Highly optimized RoPE with caching and correct interleaved pattern
#     """
#     def __init__(
#         self, 
#         dim: int, 
#         max_position_embeddings: int = 8192,
#         base: float = 10000.0,
#         device: str = None,
#         cache_on_init: bool = True
#     ):
#         super().__init__()
#         self.dim = dim
#         self.base = base
#         self.max_position_embeddings = max_position_embeddings
        
#         # Create frequency bands (only computed once)
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
#         self.register_buffer("inv_freq", inv_freq)
        
#         # Cache management
#         self._seq_len_cached = -1
#         self._cos_cached = None
#         self._sin_cached = None
        
#         # Optionally pre-build cache
#         if cache_on_init:
#             self._update_cache(max_position_embeddings, device=device)
    
#     def _update_cache(self, seq_len: int, device=None):
#         """Update the cache to support seq_len positions"""
#         if seq_len > self._seq_len_cached:
#             self._seq_len_cached = seq_len
            
#             # Build position indices
#             t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            
#             # Compute frequencies: [seq_len, dim/2]
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            
#             # Cache cos and sin: [seq_len, dim/2]
#             self._cos_cached = freqs.cos()
#             self._sin_cached = freqs.sin()
    
#     def apply_rotary_emb(
#         self,
#         xq: torch.Tensor,
#         xk: torch.Tensor,
#         start_pos: int = 0
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Apply rotary embeddings to query and key tensors
        
#         Args:
#             xq: Query tensor [batch, heads, seq_len, head_dim]
#             xk: Key tensor [batch, heads, seq_len, head_dim]
#             start_pos: Starting position for caching (useful for generation)
#         """
#         seq_len = xq.shape[2]
        
#         # Update cache if needed
#         if self._cos_cached is None or start_pos + seq_len > self._seq_len_cached:
#             self._update_cache(start_pos + seq_len, device=xq.device)
        
#         # Get cached values for current positions
#         cos = self._cos_cached[start_pos:start_pos + seq_len]
#         sin = self._sin_cached[start_pos:start_pos + seq_len]
        
#         # Apply rotary embedding using complex number formulation (most efficient)
#         return self._apply_rotary_pos_emb(xq, cos, sin), \
#                self._apply_rotary_pos_emb(xk, cos, sin)
    
#     def _apply_rotary_pos_emb(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Apply rotary position embedding using interleaved complex representation
#         This is the most efficient implementation
#         """
#         # x: [batch, heads, seq_len, head_dim]
#         # cos, sin: [seq_len, head_dim/2]
        
#         # Reshape x to separate real and imaginary parts (interleaved)
#         x_reshape = x.reshape(*x.shape[:-1], -1, 2)  # [..., head_dim/2, 2]
#         x_real = x_reshape[..., 0]  # Even indices
#         x_imag = x_reshape[..., 1]  # Odd indices
        
#         # Reshape cos and sin for broadcasting
#         cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
#         sin = sin.unsqueeze(0).unsqueeze(0)
        
#         # Apply rotation using complex multiplication
#         # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
#         out_real = x_real * cos - x_imag * sin
#         out_imag = x_real * sin + x_imag * cos
        
#         # Interleave back
#         out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
        
#         return out.type_as(x)

#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         start_pos: int = 0
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Forward pass interface"""
#         return self.apply_rotary_emb(q, k, start_pos)


# # Even more optimized version using complex tensors (if supported)
# class ComplexRotaryEmbedding(nn.Module):
#     """
#     Ultra-optimized RoPE using PyTorch complex tensor operations
#     """
#     def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
#         super().__init__()
#         self.dim = dim
#         self.base = base
        
#         # Precompute complex frequencies
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
        
#         # Cache complex exponentials
#         t = torch.arange(max_position_embeddings).float()
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
        
#         # Store as complex tensor
#         self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))
    
#     def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Convert to complex
#         xq_complex = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
#         xk_complex = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        
#         # Get frequencies for sequence length
#         seq_len = xq.shape[2]
#         freqs_cis = self.freqs_cis[:seq_len]
        
#         # Apply rotation (complex multiplication)
#         xq_out = torch.view_as_real(xq_complex * freqs_cis.unsqueeze(0).unsqueeze(0)).flatten(-2)
#         xk_out = torch.view_as_real(xk_complex * freqs_cis.unsqueeze(0).unsqueeze(0)).flatten(-2)
        
#         return xq_out.type_as(xq), xk_out.type_as(xk)
    

# #     For Production/Inference: Use Implementation 1 or the Optimized version (with caching)
# # For Research/Experimentation: Use Implementation 2 (more flexible)
# # For Maximum Performance: Use the Complex tensor version if your hardware supports it efficiently