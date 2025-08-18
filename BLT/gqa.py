import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    GroupedQueryAttention: Highly optimized multi-head attention with grouped queries.

    This module implements Grouped Query Attention (GQA), which allows multiple query heads to share key-value heads,
    reducing memory and computation costs. It uses fused QKV projections for efficient memory access and supports
    causal and masked attention.

    Args:
        d_model (int): Dimensionality of input and output features.
        n_heads (int, optional): Number of query heads. Default is 8.
        n_kv_heads (Optional[int], optional): Number of key-value heads. If None, defaults to n_heads.
        dropout (float, optional): Dropout probability for attention weights. Default is 0.1.
        bias (bool, optional): If True, adds bias to linear layers. Default is False.

    Attributes:
        d_model (int): Model dimensionality.
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of key-value heads.
        n_groups (int): Number of query heads per key-value head.
        d_head (int): Dimensionality per attention head.
        dropout_p (float): Dropout probability.
        qkv_proj (nn.Linear): Fused linear layer for QKV projection.
        out_proj (nn.Linear): Output projection layer.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        # If n_kv_heads is not specified, use n_heads (standard multi-head attention)
        if n_kv_heads is None:
            n_kv_heads = n_heads
            
        # Ensure dimensions are compatible with number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # Number of query heads per key-value head
        self.d_head = d_model // n_heads       # Dimension per head
        self.dropout_p = dropout
        
        # Fused QKV projection: projects input to concatenated Q, K, V for efficiency
        # Output dimension: [d_model + 2 * n_kv_heads * d_head]
        self.qkv_proj = nn.Linear(
            d_model, 
            d_model + 2 * self.n_kv_heads * self.d_head, 
            bias=bias
        )
        # Output projection: maps attention output back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Grouped Query Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            mask (Optional[torch.Tensor]): Optional attention mask of shape [batch_size, 1, seq_len, seq_len].
            is_causal (bool, optional): If True, applies causal masking for autoregressive decoding.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.shape
        
        # Fused QKV computation: project input to concatenated Q, K, V
        # qkv shape: [batch_size, seq_len, d_model + 2 * n_kv_heads * d_head]
        qkv = self.qkv_proj(x)
        
        # Split concatenated QKV into separate tensors
        # queries: [batch_size, seq_len, d_model]
        # keys:    [batch_size, seq_len, n_kv_heads * d_head]
        # values:  [batch_size, seq_len, n_kv_heads * d_head]
        q_size = self.d_model
        k_size = self.n_kv_heads * self.d_head
        v_size = self.n_kv_heads * self.d_head
        
        queries, keys, values = torch.split(
            qkv, [q_size, k_size, v_size], dim=-1
        )
        
        # Reshape queries for multi-head attention
        # queries: [batch_size, n_heads, seq_len, d_head]
        queries = queries.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # keys: [batch_size, n_kv_heads, seq_len, d_head]
        keys = keys.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        # values: [batch_size, n_kv_heads, seq_len, d_head]
        values = values.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # Efficiently repeat key-value heads for grouped queries if needed
        # This allows multiple query heads to share the same key-value heads
        if self.n_groups > 1:
            keys = keys.repeat_interleave(self.n_groups, dim=1)
            values = values.repeat_interleave(self.n_groups, dim=1)
        
        # Compute scaled dot-product attention using PyTorch's optimized function
        # Output shape: [batch_size, n_heads, seq_len, d_head]
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and mask is None,
        )
        
        # Reshape output back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # Final output projection to d_model dimension
        output = self.out_proj(output)
        
        return output