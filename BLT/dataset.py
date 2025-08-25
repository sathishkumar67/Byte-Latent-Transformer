from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    TokenDataset: Dataset for sequential token blocks for language modeling.

    This dataset splits a long sequence of token IDs into blocks suitable for training
    autoregressive models. Each item returns a tuple of (input, target) tensors, where
    the target is the input sequence shifted by one position.

    Args:
        block_size (int): Length of each block (sequence) returned.
        input_ids (list[int] | np.ndarray): List or array of token IDs.

    Attributes:
        block_size (int): Length of each block.
        input_ids (list[int] | np.ndarray): Sequence of token IDs.
    """

    def __init__(self, block_size: int, input_ids: list[int] | np.ndarray) -> None:
        """
        Initializes the TokenDataset.

        Args:
            block_size (int): Length of each block to return.
            input_ids (list[int] | np.ndarray): Sequence of token IDs.
        """
        self.block_size = block_size
        self.input_ids = input_ids

    def __len__(self) -> int:
        """
        Returns the number of blocks in the dataset.

        Returns:
            int: Number of blocks (each block is block_size long).
        """
        # Subtract 1 so targets can be shifted by one position
        return (len(self.input_ids) - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (input, target) tensors for the given block index.

        Args:
            idx (int): Index of the block.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input tensor of shape [block_size]
                - target tensor of shape [block_size], shifted by one position
        """
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        # Input: block_size tokens
        input_tensor = torch.tensor(self.input_ids[start_idx:end_idx], dtype=torch.long)
        # Target: next block_size tokens (shifted by one)
        target_tensor = torch.tensor(self.input_ids[start_idx+1:end_idx+1], dtype=torch.long)
        return input_tensor, target_tensor