import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    GQA interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):
    - MHA: n_kv_heads = n_heads
    - MQA: n_kv_heads = 1
    - GQA: 1 < n_kv_heads < n_heads
    
    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (must divide n_heads evenly)
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        
        # Default to standard MHA if n_kv_heads not specified
        if n_kv_heads is None:
            n_kv_heads = n_heads
            
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Grouped Query Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len, seq_len) or
                  (batch_size, 1, seq_len, seq_len)
            is_causal: Whether to apply causal masking
            need_weights: Whether to return attention weights
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights (if need_weights=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, and values
        queries = self.q_proj(x)  # (batch_size, seq_len, d_model)
        keys = self.k_proj(x)      # (batch_size, seq_len, n_kv_heads * d_head)
        values = self.v_proj(x)    # (batch_size, seq_len, n_kv_heads * d_head)
        
        # Reshape and transpose for attention computation
        queries = queries.view(batch_size, seq_len, self.n_heads, self.d_head)
        queries = queries.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)
        
        keys = keys.view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        keys = keys.transpose(1, 2)  # (batch_size, n_kv_heads, seq_len, d_head)
        
        values = values.view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        values = values.transpose(1, 2)  # (batch_size, n_kv_heads, seq_len, d_head)
        
        # Repeat keys and values for each group
        if self.n_groups > 1:
            keys = self._repeat_kv(keys, self.n_groups)
            values = self._repeat_kv(values, self.n_groups)
        
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        # attn_scores shape: (batch_size, n_heads, seq_len, seq_len)
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask
        
        # Apply attention mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add heads dimension
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, values)
        # output shape: (batch_size, n_heads, seq_len, d_head)
        
        # Reshape back to (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        return output, None
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads to match number of query heads.
        
        Args:
            x: Tensor of shape (batch_size, n_kv_heads, seq_len, d_head)
            n_rep: Number of repetitions per kv head
            
        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_head)
        """
        batch_size, n_kv_heads, seq_len, d_head = x.shape
        if n_rep == 1:
            return x
        
        # Expand adds new dimensions, repeat copies the values
        x = x.unsqueeze(2)  # (batch_size, n_kv_heads, 1, seq_len, d_head)
        x = x.expand(batch_size, n_kv_heads, n_rep, seq_len, d_head)
        x = x.reshape(batch_size, n_kv_heads * n_rep, seq_len, d_head)
        return x