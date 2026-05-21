"""
Cached-features dataset for fast action-expert-only training.

Loads pre-computed backbone features from .npz files (produced by
``robots/metaworld/scripts/cache_features.py``) instead of running
the 5B-param backbone on every training step.

Each sample contains:
  - obs_features: [S, hidden_size]  — backbone output (pre-computed)
  - state:        [state_dim]
  - actions:      [horizon, action_dim]
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CachedFeatureDataset(Dataset):
    """Load pre-computed backbone features from .npz files."""

    def __init__(self, cache_dir: str, train: bool = True):
        self.cache_dir = Path(cache_dir)

        # Load metadata
        meta_path = self.cache_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.sample_paths = sorted(self.cache_dir.glob("sample_*.npz"))
        if not self.sample_paths:
            raise FileNotFoundError(
                f"No sample_*.npz files in {cache_dir}. "
                "Run cache_features.py first."
            )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.sample_paths[idx])
        return {
            "obs_features": torch.from_numpy(data["obs_features"]),  # [S, hidden]
            "state": torch.from_numpy(data["state"]),                # [state_dim]
            "actions": torch.from_numpy(data["actions"]),            # [H, action_dim]
        }


def cached_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack cached-feature samples into a batch."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def build_cached_dataloaders(
    cache_dir: str,
    batch_size: int = 8,
    train_split: float = 0.9,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from cached backbone features.

    Splits samples (not episodes) into train/val by index.
    """
    full_ds = CachedFeatureDataset(cache_dir)
    n = len(full_ds)
    n_train = max(1, int(n * train_split))

    train_ds = torch.utils.data.Subset(full_ds, range(n_train))
    val_ds = torch.utils.data.Subset(full_ds, range(n_train, n))
    if len(val_ds) == 0:
        val_ds = torch.utils.data.Subset(full_ds, range(n_train - 1, n_train))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=cached_collate_fn, pin_memory=pin,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=cached_collate_fn, pin_memory=pin,
        num_workers=num_workers,
    )
    return train_loader, val_loader
