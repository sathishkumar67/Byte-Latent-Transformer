from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from BLT.norms import RMSNorm
from BLT.mlp import MLPwithSwiGLU
from BLT.attention import MultiHeadLatentAttentionWithGQAFused
from typing import Optional, Tuple



# Configuration for the EntropyBlock
@dataclass
class EntropyConfig:
	# model hyperparameters
	vocab_size: int = 256
	num_layers: int = 8
	head_bias: bool = False

    # Attention hyperparameters
	hidden_size: int = 512
	num_heads: int = 8
	n_kv_heads: Optional[int] = None
	kv_lora_rank: int = 256
	qk_rope_head_dim: int = 64
	v_head_dim: int = 128
	qk_nope_head_dim: int = 128
	max_position_embeddings: int = 4096
	rope_base: int = 10000
	attn_dropout: float = 0.0
	attn_bias: bool = False

	# MLP hyperparameters
	mlp_hidden_dim: Optional[int] = None  # If None, will be set to 4 * hidden_size
	mlp_dropout: float = 0.0
	mlp_bias: bool = False

	# RMSNorm hyperparameters
	rmsnorm_eps: float = 1e-8

	# RotaryPositionEmbedding hyperparameters
	rotary_max_position_embeddings: int = 4096
	rotary_base: int = 10000



# EntropyBlock
class EntropyBlock(nn.Module):
    """
    EntropyBlock: Transformer block combining MultiHeadLatentAttentionWithGQAFused, RMSNorm, and MLPwithSwiGLU.

    This block consists of:
      - Pre-attention RMSNorm normalization
      - Multi-head latent attention with grouped query and fused QKV projection
      - Residual connection after attention
      - Post-attention RMSNorm normalization
      - Feed-forward MLP with SwiGLU activation
      - Residual connection after MLP

    Args:
        config (EntropyConfig): Configuration dataclass containing all hyperparameters for attention, MLP, and normalization.

    Attributes:
        attention (MultiHeadLatentAttentionWithGQAFused): Multi-head latent attention module.
        mlp (MLPwithSwiGLU): Feed-forward network with SwiGLU activation.
        rmsnorm_1 (RMSNorm): RMSNorm layer before attention.
        rmsnorm_2 (RMSNorm): RMSNorm layer before MLP.
        config (EntropyConfig): Configuration object.
    """

    def __init__(self, config: EntropyConfig) -> None:
        """
        Initializes the EntropyBlock.

        Sets up attention, MLP, and normalization layers using the provided configuration.
        """
        super().__init__()
        self.config = config

        # Multi-head latent attention with grouped query and fused QKV projection
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

        # Feed-forward network with SwiGLU activation
        self.mlp = MLPwithSwiGLU(
            dim=config.hidden_size,
            hidden_dim=config.mlp_hidden_dim,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias
        )

        # RMSNorm layers for pre-attention and pre-MLP normalization
        self.rmsnorm_1 = RMSNorm(
            dim=config.hidden_size,
            eps=config.rmsnorm_eps
        )
        self.rmsnorm_2 = RMSNorm(
            dim=config.hidden_size,
            eps=config.rmsnorm_eps
        )


    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass for EntropyBlock.

        Applies RMSNorm, attention, and MLP with residual connections.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Optional attention mask for attention module.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached key/value states for attention.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # Pre-attention normalization and attention block with residual connection
        hidden_states = hidden_states + self.attention(
            self.rmsnorm_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )[0]  # attention returns a tuple (output, present_key_value), we only need output here

        # Pre-MLP normalization and MLP block with residual connection
        hidden_states = hidden_states + self.mlp(self.rmsnorm_2(hidden_states))

        return hidden_states
    


# EntropyModel
class EntropyModel(nn.Module):
    """
    EntropyModel: Transformer-based model for byte-level sequence modeling.

    This model consists of:
      - Token embedding layer for input tokens
      - Multiple stacked EntropyBlock transformer blocks
      - Final RMSNorm normalization
      - Output linear layer (head) for logits
      - Weight sharing between embedding and output head

    Args:
        config (EntropyConfig): Configuration dataclass containing all hyperparameters for embeddings, attention, MLP, normalization, and output.

    Attributes:
        config (EntropyConfig): Model configuration.
        entropy_block (nn.ModuleDict): Contains token embedding, stacked blocks, and final RMSNorm.
        head (nn.Linear): Output projection layer mapping hidden states to vocabulary logits.
    """

    def __init__(self, config: EntropyConfig) -> None:
        """
        Initializes the EntropyModel.

        Sets up token embedding, transformer blocks, final normalization, and output head.
        Implements weight sharing between embedding and output head for parameter efficiency.
        """
        super().__init__()

        self.config = config

        # ModuleDict for embedding, blocks, and final normalization
        self.entropy_block = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.hidden_size),  # Token embedding
            blocks = nn.ModuleList([EntropyBlock(config) for _ in range(config.num_layers)]),  # Stacked transformer blocks
            rms_final = RMSNorm(config.hidden_size)  # Final normalization
        ))

        # Output projection layer (head) for logits
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.head_bias)

        # Weight sharing: output head uses embedding weights
        self.head.weight = self.entropy_block.token_embedding.weight


    def forward(
            self,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Forward pass for EntropyModel.

        Processes input tokens through embedding, stacked transformer blocks, final normalization, and output head.
        Optionally computes cross-entropy loss if targets are provided.

        Args:
            inputs (torch.Tensor): Input tensor of token indices [batch_size, seq_len].
            targets (Optional[torch.Tensor]): Target tensor for loss computation [batch_size, seq_len].
            attention_mask (Optional[torch.Tensor]): Optional attention mask for transformer blocks.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional cached key/value states for attention.

        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size].
            If targets is provided, also returns cross-entropy loss.
        """
        _, T = inputs.shape

        # Ensure input sequence length does not exceed maximum allowed
        assert T <= self.config.max_position_embeddings, "Input sequence length exceeds maximum position length"

        # Token embedding
        x = self.entropy_block.token_embedding(inputs)

        # Pass through each transformer block
        for block in self.entropy_block.blocks:
            x = block(x, attention_mask=attention_mask, past_key_value=past_key_value)

        # Final normalization and output projection
        x = self.head(self.entropy_block.rms_final(x))

        # If targets are provided, compute cross-entropy loss
        if targets is not None:
            # Flatten logits and targets for loss computation
            return x, F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        
        # Otherwise, return logits only
        return x
    


# EntropyWrapper: PyTorch Lightning wrapper for the EntropyModel
class EntropyWrapper(L.LightningModule):
    """
    EntropyWrapper: PyTorch Lightning module for training the EntropyModel.

    This wrapper integrates the EntropyModel with PyTorch Lightning, handling training steps,
    optimizer configuration, and logging. It enables efficient training and monitoring of
    the model using Lightning's features.

    Args:
        config (EntropyConfig): Configuration dataclass containing all model hyperparameters.
        model (EntropyModel): The main transformer model to be trained.

    Attributes:
        config (EntropyConfig): Model configuration.
        model (EntropyModel): The transformer model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
    """

    def __init__(self, config: EntropyConfig, model: EntropyModel) -> None:
        """
        Initializes the EntropyWrapper Lightning module.

        Args:
            config (EntropyConfig): Model configuration.
            model (EntropyModel): The transformer model to be wrapped.
        """
        super().__init__()
        self.config = config
        self.model = model
        # Configure optimizer for the model
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.

        Sets the model to training mode, computes loss, performs optimizer step, and logs training loss.

        Args:
            batch (tuple): A batch of (inputs, targets) from the dataloader.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        self.model.train()  # Ensure model is in training mode
        optimizer = self.optimizers()  # Get optimizer from Lightning
        optimizer.zero_grad()  # Zero gradients before backward pass

        inputs, targets = batch
        _, loss = self.model(inputs, targets)  # Forward pass and compute loss
        self.log("Train_Loss", loss, prog_bar=True)  # Log training loss to progress bar

        return loss

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for model parameters.
        """
        optimizer = configure_optimizer(self.model)
        return optimizer
    



def configure_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Configures the AdamW optimizer for the EntropyModel.

    This function groups model parameters into those that should receive weight decay
    (typically 2D tensors such as weights in linear layers and embeddings) and those that should not
    (typically biases and normalization parameters). It then creates an AdamW optimizer with
    appropriate weight decay settings for each group.

    Args:
        model (nn.Module): The model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    # Collect all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Group parameters: 2D tensors (weights) get weight decay, others (biases, norms) do not
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    # Create AdamW optimizer, using fused implementation if available for speed
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ],
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True
    )
    return optimizer