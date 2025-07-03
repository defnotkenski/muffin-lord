import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RaceAwareDataset(Dataset):
    def __init__(self, races_data, continuous_cols, categorical_cols, target_col, max_horses=16):
        """
        Args:
            races_data: List of race dictionaries from prepare_race_grouped_data()
            continuous_cols: List of continuous feature column names
            categorical_cols: List of categorical feature column names
            target_col: Target column name
            max_horses: Maximum horses per race for padding
        """
        self.races = races_data
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.max_horses = max_horses

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]
        horses_df = race["horses"]  # DataFrame of all horses in this race
        num_horses = len(horses_df)

        # Extract features
        continuous_features = horses_df[self.continuous_cols].values.astype(np.float32)
        categorical_features = horses_df[self.categorical_cols].values
        targets = horses_df[self.target_col].values.astype(np.int64)

        # Create attention mask (1 for real horses, 0 for padding)
        attention_mask = np.ones(num_horses, dtype=np.float32)

        return {
            "continuous_features": continuous_features,  # Shape: (num_horses, num_continuous)
            "categorical_features": categorical_features,  # Shape: (num_horses, num_categorical)
            "targets": targets,  # Shape: (num_horses,)
            "attention_mask": attention_mask,  # Shape: (num_horses,)
            "num_horses": num_horses,
            "race_id": race["race_id"],
        }


def race_collate_fn(batch, max_horses=16):
    """
    Collate function to pad races to consistent size and create proper tensors.

    Args:
        batch: List of race samples from RaceAwareDataset
        max_horses: Maximum horses per race for padding
    """
    batch_size = len(batch)

    # Get dimensions from first sample
    num_continuous = batch[0]["continuous_features"].shape[1]
    num_categorical = batch[0]["categorical_features"].shape[1]

    # Initialize padded tensors
    continuous_batch = torch.zeros(batch_size, max_horses, num_continuous, dtype=torch.float32)
    categorical_batch = torch.zeros(batch_size, max_horses, num_categorical, dtype=torch.long)
    targets_batch = torch.zeros(batch_size, max_horses, dtype=torch.long)
    attention_mask_batch = torch.zeros(batch_size, max_horses, dtype=torch.float32)

    # Collect metadata
    race_ids = []
    num_horses_list = []

    for i, race_sample in enumerate(batch):
        num_horses = race_sample["num_horses"]
        num_horses_list.append(num_horses)
        race_ids.append(race_sample["race_id"])

        # Ensure we don't exceed max_horses
        horses_to_use = min(num_horses, max_horses)

        # Fill tensors with actual data
        continuous_batch[i, :horses_to_use] = torch.from_numpy(race_sample["continuous_features"][:horses_to_use])
        categorical_batch[i, :horses_to_use] = torch.from_numpy(race_sample["categorical_features"][:horses_to_use])
        targets_batch[i, :horses_to_use] = torch.from_numpy(race_sample["targets"][:horses_to_use])
        attention_mask_batch[i, :horses_to_use] = torch.from_numpy(race_sample["attention_mask"][:horses_to_use])

    return {
        "continuous": continuous_batch,  # Shape: (batch_size, max_horses, num_continuous)
        "categorical": categorical_batch,  # Shape: (batch_size, max_horses, num_categorical)
        "target": targets_batch,  # Shape: (batch_size, max_horses)
        "attention_mask": attention_mask_batch,  # Shape: (batch_size, max_horses)
        "num_horses": num_horses_list,  # List of actual horse counts
        "race_ids": race_ids,  # List of race identifiers
    }
