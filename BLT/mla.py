from __future__ import annotations
from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryPositionEmbedding


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA)
    ---------------------------------
    Implements a memory-efficient multi-head attention mechanism for transformers.
    Compresses key-value (KV) states using low-rank projections, reducing cache size and memory usage.
    Decompresses KV for attention computation, supports rotary positional encoding (RoPE), and caching.

    Args:
        hidden_size (int): Dimensionality of input hidden states.
        num_heads (int): Number of attention heads.
        kv_lora_rank (int): Rank for low-rank KV compression.
        qk_rope_head_dim (int): Head dimension for RoPE (rotary positional encoding) part of Q/K.
        v_head_dim (int): Head dimension for values.
        qk_nope_head_dim (int): Head dimension for non-RoPE part of Q/K.
        max_position_embeddings (int): Maximum sequence length for positional encoding.
        rope_base (int): Base for rotary positional encoding.
        dropout (float): Dropout probability for attention weights.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        max_position_embeddings: int = 2048,
        rope_base: int = 10000,
        dropout: float = 0.0,
        training: bool = True
    ):
        """
        Initialize the MultiHeadLatentAttention module.
        Sets up all linear projections, rotary positional embedding, and dropout.
        """
        super().__init__()

        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.training = training

        # Calculate head dimensions
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # Query head dimension
        self.head_dim = self.q_head_dim + v_head_dim           # Total head dimension

        # Linear layer to project input to queries
        self.q_proj = nn.Linear(hidden_size, num_heads * self.q_head_dim, bias=False)

        # Linear layer to compress input for KV using low-rank decomposition
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)

        # Linear layers to decompress compressed KV into keys and values
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.q_head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * self.v_head_dim, bias=False)

        # Output projection to map attention output back to hidden size
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=False)

        # Rotary positional embedding for RoPE part of Q/K
        self.rotary_emb = RotaryPositionEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base
        )

        # Store dropout probability for scaled_dot_product_attention
        self.dropout_p = dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-head Latent Attention (MLA).

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Optional mask tensor of shape [batch_size, 1, seq_len, seq_len].
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional tuple containing past compressed KV states for caching.
            use_cache (bool): If True, returns the compressed KV for future caching.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - output: Tensor of shape [batch_size, seq_len, hidden_size] (final attention output).
                - present_key_value: Tuple containing compressed KV for cache (if use_cache is True).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project hidden states to queries
        queries = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * q_head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.q_head_dim)

        # Compress hidden states for KV using low-rank projection
        compressed_kv = self.kv_down_proj(hidden_states)  # [batch_size, seq_len, kv_lora_rank]

        # Concatenate past compressed KV if provided (for caching)
        if past_key_value is not None:
            compressed_kv = torch.cat([past_key_value[0], compressed_kv], dim=1)

        # Decompress compressed KV to keys and values
        keys = self.k_up_proj(compressed_kv)    # [batch_size, kv_seq_len, num_heads * q_head_dim]
        values = self.v_up_proj(compressed_kv)  # [batch_size, kv_seq_len, num_heads * v_head_dim]

        # Reshape keys and values for multi-head attention
        keys = keys.view(batch_size, -1, self.num_heads, self.q_head_dim)
        values = values.view(batch_size, -1, self.num_heads, self.v_head_dim)

        kv_seq_len = keys.shape[1]

        # Split queries and keys into non-RoPE and RoPE parts
        q_nope, q_rope = torch.split(
            queries, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_nope, k_rope = torch.split(
            keys, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Apply rotary positional encoding (RoPE) to RoPE parts
        q_rope = self.rotary_emb(q_rope, seq_len)
        k_rope = self.rotary_emb(k_rope, kv_seq_len)

        # Concatenate non-RoPE and RoPE parts back together
        queries = torch.cat([q_nope, q_rope], dim=-1)
        keys = torch.cat([k_nope, k_rope], dim=-1)

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Use scaled_dot_product_attention for efficient attention computation
        # This function internally handles scaling, masking, softmax, and dropout
        attention_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.q_head_dim)
        )

        # Reshape and project output back to hidden size
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attention_output)

        # Prepare present_key_value for cache if requested
        present_key_value = None
        if use_cache:
            present_key_value = (compressed_kv,)  # Only store compressed representation

        return output, present_key_value
    

class MLABlock(nn.Module):
    """
    MLABlock: Transformer block using Multi-Head Latent Attention (MLA).

    This block consists of:
      - Layer normalization before attention
      - MLA attention mechanism
      - Residual connection and dropout after attention
      - Layer normalization before feed-forward network
      - Feed-forward network (MLP) with GELU activation and dropout
      - Residual connection after feed-forward

    Args:
        hidden_size (int): Dimensionality of input hidden states.
        num_heads (int): Number of attention heads.
        kv_lora_rank (int, optional): Rank for low-rank KV compression. Default: 512.
        intermediate_size (int, optional): Size of intermediate layer in MLP. Default: 4 * hidden_size.
        dropout (float, optional): Dropout probability. Default: 0.1.
        layer_norm_eps (float, optional): Epsilon for layer normalization. Default: 1e-5.
        max_position_embeddings (int, optional): Maximum sequence length for positional encoding. Default: 2048.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        intermediate_size: int = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048
    ):
        """
        Initializes the MLABlock.

        Sets up layer normalization, MLA attention, feed-forward network, and dropout.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or 4 * hidden_size
        
        # Layer normalization before attention
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # Layer normalization before feed-forward
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # MLA attention mechanism
        self.attention = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Feed-forward network (MLP) with GELU activation and dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Dropout after attention
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        """
        Forward pass for MLABlock.

        Applies layer normalization, MLA attention, residual connection, dropout,
        feed-forward network, and returns output and present_key_value for caching.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Optional mask tensor for attention.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached KV states.
            use_cache (bool): If True, returns present_key_value for caching.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - hidden_states: Output tensor after attention and feed-forward.
                - present_key_value: Cached compressed KV states (if use_cache is True).
        """
        # Save residual for attention block
        residual = hidden_states
        # Apply layer normalization before attention
        hidden_states = self.input_layernorm(hidden_states)
        # MLA attention mechanism
        attention_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # Add residual and apply dropout after attention
        hidden_states = residual + self.dropout(attention_output)
        
        # Save residual for feed-forward block
        residual = hidden_states
        # Apply layer normalization before feed-forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Feed-forward network with residual connection
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states, present_key_value