"""
Pre-compute backbone features for all training samples.

Since the backbone is frozen in action-expert-only training, its output
is deterministic.  Running it once and caching the features eliminates
the expensive 5B-param forward pass from every training step.

Speedup: ~100-200x per step (backbone forward ≈ 20s → cached load ≈ 0.1s).

Usage::

    uv run python -m robots.metaworld.scripts.cache_features \
        --config robots/metaworld/configs/metaworld_push_m3.yaml \
        --data-dir data/metaworld_demos \
        --output-dir data/metaworld_features
"""

import argparse
import os
import time
import logging

import torch
import numpy as np

from gemma4_vla.config import Gemma4VLAConfig
from gemma4_vla.train import load_config_from_yaml
from gemma4_vla.model import Gemma4VLA

logger = logging.getLogger(__name__)


def cache_features(
    cfg: Gemma4VLAConfig,
    data_dir: str,
    output_dir: str,
    instruction: str = "push the object to the goal",
    batch_size: int = 1,
):
    """
    Extract backbone features for every training sample and save to disk.

    Produces one .npz file per sample containing:
      - obs_features: [S, hidden_size]  (backbone output)
      - state:        [state_dim]
      - actions:      [horizon, action_dim]
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model (backbone only needed for feature extraction)
    logger.info("Loading Gemma4VLA for feature extraction...")
    model = Gemma4VLA(cfg)
    try:
        model = model.to(device)
    except NotImplementedError:
        model.obs_proj = model.obs_proj.to(device)
        model.action_expert = model.action_expert.to(device)

    model.eval()
    logger.info(f"Model loaded on {device}. Hidden size: {model.backbone.hidden_size}")

    # Build dataset (reuse MetaWorld adapter)
    from robots.metaworld.dataset import MetaWorldHDF5Dataset
    from gemma4_vla.dataset import collate_fn
    from torch.utils.data import DataLoader

    processor = model.backbone.processor
    dataset = MetaWorldHDF5Dataset(
        data_dir, cfg, processor=processor, train=True, instruction=instruction,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,  # single-threaded for HDF5
    )

    logger.info(f"Dataset: {len(dataset)} samples from {data_dir}")
    logger.info(f"Output:  {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    n_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_dev = {k: v.to(device) for k, v in batch.items()}

            # Run backbone (the expensive part — done only once)
            obs_features = model.backbone(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                pixel_values=batch_dev.get("pixel_values"),
                image_position_ids=batch_dev.get("image_position_ids"),
                mm_token_type_ids=batch_dev.get("mm_token_type_ids"),
            )  # [B, S, hidden_size]

            # Save each sample in the batch
            B = obs_features.shape[0]
            for i in range(B):
                sample_idx = batch_idx * batch_size + i
                np.savez_compressed(
                    os.path.join(output_dir, f"sample_{sample_idx:06d}.npz"),
                    obs_features=obs_features[i].cpu().float().numpy(),
                    state=batch["state"][i].numpy(),
                    actions=batch["actions"][i].numpy(),
                )
                n_saved += 1

            # Progress
            elapsed = time.time() - t0
            samples_per_sec = n_saved / max(elapsed, 1e-6)
            remaining = (len(dataset) - n_saved) / max(samples_per_sec, 1e-6)
            if (batch_idx + 1) % 10 == 0 or n_saved == len(dataset):
                logger.info(
                    f"  [{n_saved:5d}/{len(dataset)}]  "
                    f"{samples_per_sec:.1f} samples/s  "
                    f"~{remaining / 60:.0f} min remaining"
                )

    total_time = time.time() - t0
    logger.info(
        f"Done. Cached {n_saved} samples in {total_time:.0f}s "
        f"({total_time / 60:.1f} min)"
    )

    # Save metadata
    import json
    meta = {
        "model_name": cfg.backbone.model_name,
        "hidden_size": model.backbone.hidden_size,
        "n_samples": n_saved,
        "data_dir": data_dir,
        "instruction": instruction,
        "cache_time_s": total_time,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {output_dir}/metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute backbone features for cached training"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--data-dir", type=str, default="data/metaworld_demos")
    parser.add_argument("--output-dir", type=str, default="data/metaworld_features")
    parser.add_argument("--instruction", type=str, default="push the object to the goal")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    cache_features(cfg, args.data_dir, args.output_dir, args.instruction)


if __name__ == "__main__":
    main()
