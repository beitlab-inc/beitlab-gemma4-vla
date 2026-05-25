"""
Per-dim normalisation statistics for state and action vectors.

`DatasetStats` is computed once over a training DataLoader, stored next to
the model weights as ``normalization.pt``, and applied symmetrically at
training (normalize) and inference (denormalize) time so the network only
ever sees zero-mean / unit-std actions.

The on-disk format is intentionally a flat dict with ``state_mean``,
``state_std``, ``action_mean``, ``action_std``, ``normalize`` keys — the
existing eval script (`robots/metaworld/scripts/test.py`) already reads
that shape.
"""

import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

STATS_FILENAME = "normalization.pt"

logger = logging.getLogger(__name__)


class DatasetStats:
    """Per-dim mean / std for state and action vectors.

    All tensors are stored on CPU as fp32; callers move them onto the
    appropriate device when applying.
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        enabled: bool = True,
    ):
        self.state_mean = state_mean.float().cpu()
        self.state_std = state_std.float().cpu().clamp_min(1e-6)
        self.action_mean = action_mean.float().cpu()
        self.action_std = action_std.float().cpu().clamp_min(1e-6)
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def compute_from_loader(
        cls,
        loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> "DatasetStats":
        """Streaming mean/variance over a DataLoader's `state` and `actions`."""
        state_sum = state_sq = None
        action_sum = action_sq = None
        n_state = n_action = 0

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            s = batch["state"].float()
            a = batch["actions"].float()
            s_flat = s.reshape(-1, s.shape[-1])
            a_flat = a.reshape(-1, a.shape[-1])

            if state_sum is None:
                state_sum = torch.zeros(s_flat.shape[-1], dtype=torch.float64)
                state_sq = torch.zeros_like(state_sum)
                action_sum = torch.zeros(a_flat.shape[-1], dtype=torch.float64)
                action_sq = torch.zeros_like(action_sum)

            state_sum += s_flat.double().sum(dim=0)
            state_sq += (s_flat.double() ** 2).sum(dim=0)
            n_state += s_flat.shape[0]
            action_sum += a_flat.double().sum(dim=0)
            action_sq += (a_flat.double() ** 2).sum(dim=0)
            n_action += a_flat.shape[0]

        if n_state == 0 or n_action == 0:
            raise ValueError("DatasetStats.compute_from_loader saw no samples.")

        state_mean = (state_sum / n_state).float()
        state_var = (state_sq / n_state - state_mean.double() ** 2).clamp_min(0).float()
        action_mean = (action_sum / n_action).float()
        action_var = (action_sq / n_action - action_mean.double() ** 2).clamp_min(0).float()

        return cls(
            state_mean=state_mean,
            state_std=state_var.sqrt(),
            action_mean=action_mean,
            action_std=action_var.sqrt(),
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, checkpoint_dir: str) -> str:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, STATS_FILENAME)
        torch.save(
            {
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "normalize": self.enabled,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, checkpoint_dir: str) -> Optional["DatasetStats"]:
        path = os.path.join(checkpoint_dir, STATS_FILENAME)
        if not os.path.exists(path):
            return None
        try:
            d = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            state_mean=d["state_mean"],
            state_std=d["state_std"],
            action_mean=d["action_mean"],
            action_std=d["action_std"],
            enabled=bool(d.get("normalize", True)),
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return state
        d = state.shape[-1]
        mean = self.state_mean[:d].to(state.device, state.dtype)
        std = self.state_std[:d].to(state.device, state.dtype)
        return (state - mean) / std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return actions
        d = actions.shape[-1]
        mean = self.action_mean[:d].to(actions.device, actions.dtype)
        std = self.action_std[:d].to(actions.device, actions.dtype)
        return (actions - mean) / std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return actions
        d = actions.shape[-1]
        mean = self.action_mean[:d].to(actions.device, actions.dtype)
        std = self.action_std[:d].to(actions.device, actions.dtype)
        return actions * std + mean


def maybe_compute_and_save_stats(
    cfg,
    loader: DataLoader,
    checkpoint_dir: str,
    max_batches: Optional[int] = None,
) -> Optional[DatasetStats]:
    """Compute stats from `loader` and save them when the training config asks for it."""
    tr = cfg.training
    if not getattr(tr, "normalize_stats", False):
        return None
    logger.info("Computing dataset normalisation stats…")
    stats = DatasetStats.compute_from_loader(loader, max_batches=max_batches)
    path = stats.save(checkpoint_dir)
    logger.info(f"Saved normalisation stats to {path}")
    return stats
