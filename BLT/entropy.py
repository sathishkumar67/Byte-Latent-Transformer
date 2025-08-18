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
	dropout: float = 0.0
	bias: bool = False

	# MLP hyperparameters
	mlp_hidden_dim: Optional[int] = None  # If None, will be set to 4 * hidden_size
	mlp_dropout: float = 0.0

	# RMSNorm hyperparameters
	rmsnorm_eps: float = 1e-8

	# RotaryPositionEmbedding hyperparameters
	rotary_max_position_embeddings: int = 2048
	rotary_base: int = 10000