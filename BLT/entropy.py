from __future__ import annotations
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm
from .mlp import MLPwithSwiGLU
from .attention import MultiHeadLatentAttentionWithGQAFused
from typing import Optional



@dataclass
class EntropyConfig:
    # Attention hyperparameters
	hidden_size: int = 512
	num_heads: int = 8
	n_kv_heads: Optional[int] = None
	kv_lora_rank: int = 512
	qk_rope_head_dim: int = 64
	v_head_dim: int = 128
	qk_nope_head_dim: int = 128
	max_position_embeddings: int = 2048
	rope_base: int = 10000
	attn_dropout: float = 0.0
	attn_bias: bool = False

	# MLP hyperparameters
	mlp_hidden_dim: Optional[int] = None  # If None, will be set to 4 * hidden_size
	mlp_dropout: float = 0.0
	mlp_bias: bool = True

	# RMSNorm hyperparameters
	rmsnorm_eps: float = 1e-8

	# RotaryPositionEmbedding hyperparameters
	rotary_max_position_embeddings: int = 2048
	rotary_base: int = 10000
	

class EntropyBlock(nn.Module):
	def __init__(self, config: EntropyConfig):
		super().__init__()
		self.attention = MultiHeadLatentAttentionWithGQAFused(
			hidden_size=config.hidden_size,
			num_heads=config.num_heads,
			n_kv_heads=config.n_kv_heads,
			kv_lora_rank=config.kv_lora_rank,
			qk_rope_head_dim=config.qk_rope_head_dim,
			v_head_dim=config.v_head_dim,
			qk_nope_head_dim=config.qk_nope_head_dim,
			max_position_embeddings=config.max_position_embeddings,
			rope_base=config.rope_base,
			attn_dropout=config.attn_dropout,
			attn_bias=config.attn_bias
        )
		self.mlp = MLPwithSwiGLU(
			dim=config.hidden_size,
            hidden_dim=config.mlp_hidden_dim,
			mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias
        )
		self.rmsnorm = RMSNorm(
			dim=config.hidden_size,
			eps=config.rmsnorm_eps
        )