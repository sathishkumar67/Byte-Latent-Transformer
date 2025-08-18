from __future__ import annotations
from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryPositionEmbedding


class MultiHeadLatentAttentionWithGQAFused(nn.Module):
    """
    MultiHeadLatentAttentionWithGQAFused: Optimized MLA with Grouped Query Attention and Fused QKV Projection.

    This module implements a memory- and compute-efficient multi-head attention mechanism for transformers,
    combining low-rank compression of key-value states (MLA) with grouped query attention (GQA) and a single
    fused linear projection for queries and compressed KV. It supports rotary positional encoding (RoPE),
    caching for autoregressive decoding, and efficient handling of grouped queries.

    Args:
        hidden_size (int): Dimensionality of input hidden states.
        num_heads (int): Number of query heads.
        n_kv_heads (Optional[int], optional): Number of key-value heads. If None, defaults to num_heads.
        kv_lora_rank (int, optional): Rank for low-rank KV compression. Default: 512.
        qk_rope_head_dim (int, optional): Head dimension for RoPE part of Q/K. Default: 64.
        v_head_dim (int, optional): Head dimension for values. Default: 128.
        qk_nope_head_dim (int, optional): Head dimension for non-RoPE part of Q/K. Default: 128.
        max_position_embeddings (int, optional): Maximum sequence length for positional encoding. Default: 2048.
        rope_base (int, optional): Base for rotary positional encoding. Default: 10000.
        dropout (float, optional): Dropout probability for attention weights. Default: 0.0.
        bias (bool, optional): If True, adds bias to linear layers. Default: False.

    Attributes:
        hidden_size (int): Model dimensionality.
        num_heads (int): Number of query heads.
        n_kv_heads (int): Number of key-value heads.
        n_groups (int): Number of query heads per key-value head.
        kv_lora_rank (int): Rank for low-rank KV compression.
        qk_rope_head_dim (int): RoPE dimension for Q/K.
        qk_nope_head_dim (int): Non-RoPE dimension for Q/K.
        v_head_dim (int): Value head dimension.
        q_head_dim (int): Total query head dimension.
        kv_head_dim (int): Total key head dimension.
        qkv_proj (nn.Linear): Fused linear layer for Q and compressed KV.
        k_up_proj (nn.Linear): Decompression layer for keys.
        v_up_proj (nn.Linear): Decompression layer for values.
        o_proj (nn.Linear): Output projection layer.
        rotary_emb (RotaryPositionEmbedding): RoPE module.
        dropout_p (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_kv_heads: Optional[int] = None,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        max_position_embeddings: int = 2048,
        rope_base: int = 10000,
        dropout: float = 0.0,
        bias: bool = False
    ):
        """
        Initialize the MultiHeadLatentAttentionWithGQAFused module.

        Sets up fused QKV projection, decompression layers for keys/values, RoPE, and output projection.
        """
        super().__init__()

        # Default to standard multi-head attention if n_kv_heads is not specified
        if n_kv_heads is None:
            n_kv_heads = num_heads

        # Condition checking
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % n_kv_heads == 0, "num_heads must be divisible by n_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = num_heads // n_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout_p = dropout

        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # Query head dimension
        self.kv_head_dim = self.q_head_dim                     # Key head dimension

        # Fused projection for queries and compressed KV
        # Output: [queries | compressed_kv]
        self.qkv_proj = nn.Linear(
            hidden_size,
            num_heads * self.q_head_dim + kv_lora_rank,
            bias=bias
        )

        # Decompression layers for keys and values (for n_kv_heads)
        self.k_up_proj = nn.Linear(kv_lora_rank, n_kv_heads * self.kv_head_dim, bias=bias)
        self.v_up_proj = nn.Linear(kv_lora_rank, n_kv_heads * self.v_head_dim, bias=bias)

        # Output projection to map attention output back to hidden size
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=bias)

        # Rotary positional embedding for RoPE part of Q/K
        self.rotary_emb = RotaryPositionEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for MultiHeadLatentAttentionWithGQAFused.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Optional mask tensor for attention.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached compressed KV states.
            use_cache (bool, optional): If True, returns present_key_value for caching.
            is_causal (bool, optional): If True, applies causal masking for autoregressive decoding.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - output: Tensor of shape [batch_size, seq_len, hidden_size] (final attention output).
                - present_key_value: Tuple containing compressed KV for cache (if use_cache is True).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection: project input to queries and compressed KV
        # qkv shape: [batch_size, seq_len, num_heads * q_head_dim + kv_lora_rank]
        qkv = self.qkv_proj(hidden_states)

        # Split into queries and compressed KV
        q_size = self.num_heads * self.q_head_dim
        queries, compressed_kv = torch.split(qkv, [q_size, self.kv_lora_rank], dim=-1)

        # Reshape queries for multi-head attention
        # queries: [batch_size, seq_len, num_heads, q_head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.q_head_dim)

        # Handle caching: concatenate past compressed KV if provided
        if past_key_value is not None:
            compressed_kv = torch.cat([past_key_value[0], compressed_kv], dim=1)

        # Decompress compressed KV to keys and values
        # keys: [batch_size, kv_seq_len, n_kv_heads * kv_head_dim]
        # values: [batch_size, kv_seq_len, n_kv_heads * v_head_dim]
        keys, values = self.k_up_proj(compressed_kv), self.v_up_proj(compressed_kv)

        # Reshape keys and values for multi-head attention
        # keys: [batch_size, kv_seq_len, n_kv_heads, kv_head_dim]
        # values: [batch_size, kv_seq_len, n_kv_heads, v_head_dim]
        keys = keys.view(batch_size, -1, self.n_kv_heads, self.kv_head_dim)
        values = values.view(batch_size, -1, self.n_kv_heads, self.v_head_dim)

        kv_seq_len = keys.shape[1]

        # Split queries and keys into non-RoPE and RoPE parts
        q_nope, q_rope = torch.split(queries, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, k_rope = torch.split(keys, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply rotary positional encoding (RoPE) to RoPE parts
        q_rope = self.rotary_emb(q_rope, seq_len)
        k_rope = self.rotary_emb(k_rope, kv_seq_len)

        # Concatenate non-RoPE and RoPE parts back together
        queries = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)  # [batch, num_heads, seq_len, q_head_dim]
        keys = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)     # [batch, n_kv_heads, kv_seq_len, kv_head_dim]
        values = values.transpose(1, 2)                                # [batch, n_kv_heads, kv_seq_len, v_head_dim]

        # Repeat KV for grouped queries if needed
        # This allows multiple query heads to share the same key-value heads
        if self.n_groups > 1:
            keys = keys.repeat_interleave(self.n_groups, dim=1)
            values = values.repeat_interleave(self.n_groups, dim=1)

        # Compute scaled dot-product attention using PyTorch's optimized function
        # Output shape: [batch_size, num_heads, seq_len, v_head_dim]
        output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p,
            is_causal=is_causal and attention_mask is None,
            scale=1.0 / math.sqrt(self.q_head_dim)
        )

        # Reshape and project output back to hidden size
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(output)

        # Prepare present_key_value for cache if requested
        present_key_value = (compressed_kv,) if use_cache else None

        return output, present_key_value    






# class MultiHeadLatentAttention(nn.Module):
#     """
#     Multi-head Latent Attention (MLA)
#     ---------------------------------
#     Implements a memory-efficient multi-head attention mechanism for transformers.
#     Compresses key-value (KV) states using low-rank projections, reducing cache size and memory usage.
#     Decompresses KV for attention computation, supports rotary positional encoding (RoPE), and caching.

#     Args:
#         hidden_size (int): Dimensionality of input hidden states.
#         num_heads (int): Number of attention heads.
#         kv_lora_rank (int): Rank for low-rank KV compression.
#         qk_rope_head_dim (int): Head dimension for RoPE (rotary positional encoding) part of Q/K.
#         v_head_dim (int): Head dimension for values.
#         qk_nope_head_dim (int): Head dimension for non-RoPE part of Q/K.
#         max_position_embeddings (int): Maximum sequence length for positional encoding.
#         rope_base (int): Base for rotary positional encoding.
#         dropout (float): Dropout probability for attention weights.
#     """
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         kv_lora_rank: int = 512,
#         qk_rope_head_dim: int = 64,
#         v_head_dim: int = 128,
#         qk_nope_head_dim: int = 128,
#         max_position_embeddings: int = 2048,
#         rope_base: int = 10000,
#         dropout: float = 0.0,
#         training: bool = True
#     ):
#         """
#         Initialize the MultiHeadLatentAttention module.
#         Sets up all linear projections, rotary positional embedding, and dropout.
#         """
#         super().__init__()

#         # Store configuration parameters
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.kv_lora_rank = kv_lora_rank
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.v_head_dim = v_head_dim
#         self.training = training

#         # Calculate head dimensions
#         self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # Query head dimension
#         self.head_dim = self.q_head_dim + v_head_dim           # Total head dimension

#         # Linear layer to project input to queries
#         self.q_proj = nn.Linear(hidden_size, num_heads * self.q_head_dim, bias=False)

#         # Linear layer to compress input for KV using low-rank decomposition
#         self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)

#         # Linear layers to decompress compressed KV into keys and values
#         self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.q_head_dim, bias=False)
#         self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * self.v_head_dim, bias=False)

#         # Output projection to map attention output back to hidden size
#         self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=False)

#         # Rotary positional embedding for RoPE part of Q/K
#         self.rotary_emb = RotaryPositionEmbedding(
#             qk_rope_head_dim,
#             max_position_embeddings=max_position_embeddings,
#             base=rope_base
#         )

#         # Store dropout probability for scaled_dot_product_attention
#         self.dropout_p = dropout
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         use_cache: bool = False
#     ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#         """
#         Forward pass for Multi-head Latent Attention (MLA).

#         Args:
#             hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
#             attention_mask (Optional[torch.Tensor]): Optional mask tensor of shape [batch_size, 1, seq_len, seq_len].
#             past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional tuple containing past compressed KV states for caching.
#             use_cache (bool): If True, returns the compressed KV for future caching.

#         Returns:
#             Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#                 - output: Tensor of shape [batch_size, seq_len, hidden_size] (final attention output).
#                 - present_key_value: Tuple containing compressed KV for cache (if use_cache is True).
#         """
#         batch_size, seq_len, _ = hidden_states.shape

#         # Project hidden states to queries
#         queries = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * q_head_dim]
#         queries = queries.view(batch_size, seq_len, self.num_heads, self.q_head_dim)

#         # Compress hidden states for KV using low-rank projection
#         compressed_kv = self.kv_down_proj(hidden_states)  # [batch_size, seq_len, kv_lora_rank]

#         # Concatenate past compressed KV if provided (for caching)
#         if past_key_value is not None:
#             compressed_kv = torch.cat([past_key_value[0], compressed_kv], dim=1)

#         # Decompress compressed KV to keys and values
#         keys = self.k_up_proj(compressed_kv)    # [batch_size, kv_seq_len, num_heads * q_head_dim]
#         values = self.v_up_proj(compressed_kv)  # [batch_size, kv_seq_len, num_heads * v_head_dim]

#         # Reshape keys and values for multi-head attention
#         keys = keys.view(batch_size, -1, self.num_heads, self.q_head_dim)
#         values = values.view(batch_size, -1, self.num_heads, self.v_head_dim)

#         kv_seq_len = keys.shape[1]

#         # Split queries and keys into non-RoPE and RoPE parts
#         q_nope, q_rope = torch.split(
#             queries, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
#         )
#         k_nope, k_rope = torch.split(
#             keys, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
#         )

#         # Apply rotary positional encoding (RoPE) to RoPE parts
#         q_rope = self.rotary_emb(q_rope, seq_len)
#         k_rope = self.rotary_emb(k_rope, kv_seq_len)

#         # Concatenate non-RoPE and RoPE parts back together
#         queries = torch.cat([q_nope, q_rope], dim=-1)
#         keys = torch.cat([k_nope, k_rope], dim=-1)

#         # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
#         queries = queries.transpose(1, 2)
#         keys = keys.transpose(1, 2)
#         values = values.transpose(1, 2)

#         # Use scaled_dot_product_attention for efficient attention computation
#         # This function internally handles scaling, masking, softmax, and dropout
#         attention_output = F.scaled_dot_product_attention(
#             queries,
#             keys,
#             values,
#             attn_mask=attention_mask,
#             dropout_p=self.dropout_p if self.training else 0.0,
#             scale=1.0 / math.sqrt(self.q_head_dim)
#         )

#         # Reshape and project output back to hidden size
#         attention_output = attention_output.transpose(1, 2).contiguous()
#         attention_output = attention_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
#         output = self.o_proj(attention_output)

#         # Prepare present_key_value for cache if requested
#         present_key_value = None
#         if use_cache:
#             present_key_value = (compressed_kv,)  # Only store compressed representation

#         return output, present_key_value
    

# class MLABlock(nn.Module):
#     """
#     MLABlock: Transformer block using Multi-Head Latent Attention (MLA).

#     This block consists of:
#       - Layer normalization before attention
#       - MLA attention mechanism
#       - Residual connection and dropout after attention
#       - Layer normalization before feed-forward network
#       - Feed-forward network (MLP) with GELU activation and dropout
#       - Residual connection after feed-forward

#     Args:
#         hidden_size (int): Dimensionality of input hidden states.
#         num_heads (int): Number of attention heads.
#         kv_lora_rank (int, optional): Rank for low-rank KV compression. Default: 512.
#         intermediate_size (int, optional): Size of intermediate layer in MLP. Default: 4 * hidden_size.
#         dropout (float, optional): Dropout probability. Default: 0.1.
#         layer_norm_eps (float, optional): Epsilon for layer normalization. Default: 1e-5.
#         max_position_embeddings (int, optional): Maximum sequence length for positional encoding. Default: 2048.
#     """
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         kv_lora_rank: int = 512,
#         intermediate_size: int = None,
#         dropout: float = 0.1,
#         layer_norm_eps: float = 1e-5,
#         max_position_embeddings: int = 2048
#     ):
#         """
#         Initializes the MLABlock.

#         Sets up layer normalization, MLA attention, feed-forward network, and dropout.
#         """
#         super().__init__()
        
#         self.hidden_size = hidden_size
#         intermediate_size = intermediate_size or 4 * hidden_size
        
#         # Layer normalization before attention
#         self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         # Layer normalization before feed-forward
#         self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
#         # MLA attention mechanism
#         self.attention = MultiHeadLatentAttention(
#             hidden_size=hidden_size,
#             num_heads=num_heads,
#             kv_lora_rank=kv_lora_rank,
#             max_position_embeddings=max_position_embeddings,
#             dropout=dropout
#         )
        
#         # Feed-forward network (MLP) with GELU activation and dropout
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, intermediate_size),
#             nn.GELU(),
#             nn.Linear(intermediate_size, hidden_size),
#             nn.Dropout(dropout)
#         )
        
#         # Dropout after attention
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         use_cache: bool = False
#     ):
#         """
#         Forward pass for MLABlock.

#         Applies layer normalization, MLA attention, residual connection, dropout,
#         feed-forward network, and returns output and present_key_value for caching.

#         Args:
#             hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
#             attention_mask (Optional[torch.Tensor]): Optional mask tensor for attention.
#             past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached KV states.
#             use_cache (bool): If True, returns present_key_value for caching.

#         Returns:
#             Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#                 - hidden_states: Output tensor after attention and feed-forward.
#                 - present_key_value: Cached compressed KV states (if use_cache is True).
#         """
#         # Save residual for attention block
#         residual = hidden_states
#         # Apply layer normalization before attention
#         hidden_states = self.input_layernorm(hidden_states)
#         # MLA attention mechanism
#         attention_output, present_key_value = self.attention(
#             hidden_states,
#             attention_mask=attention_mask,
#             past_key_value=past_key_value,
#             use_cache=use_cache
#         )
#         # Add residual and apply dropout after attention
#         hidden_states = residual + self.dropout(attention_output)
        
#         # Save residual for feed-forward block
#         residual = hidden_states
#         # Apply layer normalization before feed-forward
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         # Feed-forward network with residual connection
#         hidden_states = residual + self.mlp(hidden_states)
        
#         return hidden_states, present_key_value
