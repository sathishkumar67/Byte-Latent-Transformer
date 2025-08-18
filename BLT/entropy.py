from __future__ import annotations
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm
from .mlp import MLPwithSwiGLU
from .attention import MultiHeadLatentAttentionWithGQAFused


@dataclass
class EntropyConfig:
    pass

