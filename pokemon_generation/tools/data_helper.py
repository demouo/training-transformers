# data_helper.py

## Define Dataset
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class PixelSequenceDataset(Dataset):
    def __init__(self, data: List[List[int]], mode: str = "train"):
        """
        A dataset class for handling pixel sequence.

        Args:
            data (List[List[int]]): A list of sequence, where each sequence is a list of intergers.
            mode (str): The mode of operation, either 'train', 'dev', or 'test'.
                - "train": Returns (input ids, labels) where input ids are sequence[:-1] and labels are sequence[1:].
                - "dev": Returns (input ids, labels) where input ids are sequence[:-160] and labels are sequence[-160:].
                - "test": Returns only input ids, as labels are not available.
        """
        self.data = data
        self.mode = mode

    def __len__(self) -> int:
        """Returns the total number of sequence in the dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Fetches a sequence from the dataset and process it based on the mode.

        Args:
            idx (int): The idex of the sequence.

        Returns:
            - If mode == "train": Tuple[torch.Tensor, torch.Tensor] -> (input ids, lebels).
            - If mode == "dev": Tuple[torch.Tensor, torch.Tensor] -> (input ids, lebels).
            - If mode == "test": torch.Tensor -> input ids.
        """
        sequence = self.data[idx]

        if self.mode == "train":
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
            labels = torch.tensor(sequence[1:], dtype=torch.long)
            return input_ids, labels

        elif self.mode == "dev":
            # Make the first 80% of a sequence as the input ids
            input_ids = torch.tensor(sequence[:-160], dtype=torch.long)
            labels = torch.tensor(sequence[-160:], dtype=torch.long)
            return input_ids, labels

        elif self.mode == "test":
            input_ids = torch.tensor(sequence, dtype=torch.long)
            return input_ids

        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Choose from 'train', 'dev' or 'test'."
            )
