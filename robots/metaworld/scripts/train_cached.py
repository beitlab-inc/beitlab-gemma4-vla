"""
Fast action-expert training using pre-computed backbone features.

This is ~100-200x faster than standard training because it skips the
5B-param backbone forward pass entirely.  The backbone features are
loaded from .npz files produced by cache_features.py.

Two-step workflow::

    # Step 1: Cache features (one-time, ~15 min on M3)
    uv run python -m robots.metaworld.scripts.cache_features \
        --config robots/metaworld/configs/metaworld_push_m3.yaml \
        --data-dir data/metaworld_demos \
        --output-dir data/metaworld_features

    # Step 2: Train action expert (fast, ~5 min on M3)
    uv run python -m robots.metaworld.scripts.train_cached \
        --config robots/metaworld/configs/metaworld_push_m3.yaml \
        --cache-dir data/metaworld_features \
        --mlflow
"""

import argparse
import json
import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW

from gemma4_vla.config import Gemma4VLAConfig
from gemma4_vla.train import (
    load_config_from_yaml,
    build_scheduler,
    append_metric,
)
from gemma4_vla.model import Gemma4VLA
from gemma4_vla.observability import MlflowRun, ensure_parent
from robots.metaworld.cached_dataset import build_cached_dataloaders

logger = logging.getLogger(__name__)


def train_cached(
    cfg: Gemma4VLAConfig,
    cache_dir: str,
    metrics_path: str = None,
    mlflow_run: MlflowRun = None,
):
    """Train action expert only, using pre-computed backbone features."""
    if mlflow_run is None:
        mlflow_run = MlflowRun(enabled=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tr = cfg.training

    # --- Model (action expert + obs_proj only, no backbone needed) ---
    logger.info("Initialising Gemma4VLA...")
    model = Gemma4VLA(cfg)
    try:
        model = model.to(device)
    except NotImplementedError:
        model.obs_proj = model.obs_proj.to(device)
        model.action_expert = model.action_expert.to(device)

    # Count only action expert + obs_proj params
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Trainable params: {trainable / 1e6:.1f} M (action expert + obs_proj)")

    # --- Data ---
    train_loader, val_loader = build_cached_dataloaders(
        cache_dir,
        batch_size=tr.batch_size,
        num_workers=tr.num_workers,
    )
    logger.info(
        f"Cached dataset: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val samples"
    )

    # --- Optimizer (action expert params only) ---
    expert_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        expert_params,
        lr=tr.learning_rate,
        weight_decay=tr.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, tr.warmup_steps, tr.max_steps)

    # --- Training loop ---
    model.train()
    step = 0
    epoch = 0
    best_val_loss = float("inf")
    micro_step = 0
    running_loss = 0.0
    last_loss_val = 0.0

    os.makedirs(tr.output_dir, exist_ok=True)
    logger.info(
        f"Starting cached training: {tr.max_steps} steps, "
        f"batch={tr.batch_size} x {tr.grad_accum_steps} accum, "
        f"lr={tr.learning_rate}, device={device}"
    )
    t0 = time.time()

    append_metric(metrics_path, {
        "type": "train_start",
        "mode": "cached",
        "cache_dir": cache_dir,
        "batch_size": tr.batch_size,
        "grad_accum_steps": tr.grad_accum_steps,
        "max_steps": tr.max_steps,
        "learning_rate": tr.learning_rate,
    })

    while step < tr.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= tr.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward — uses cached obs_features, skips backbone
            out = model.compute_loss_cached(batch)
            raw_loss = out["loss"]
            loss = raw_loss / tr.grad_accum_steps

            loss.backward()
            micro_step += 1
            running_loss += raw_loss.item()

            if micro_step % tr.grad_accum_steps != 0:
                continue

            nn.utils.clip_grad_norm_(expert_params, tr.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            last_loss_val = running_loss / tr.grad_accum_steps
            running_loss = 0.0

            # --- Logging ---
            if step % tr.log_every_n_steps == 0:
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                steps_per_sec = step / max(elapsed, 1e-6)
                eta_min = (tr.max_steps - step) / max(steps_per_sec, 1e-6) / 60
                logger.info(
                    f"step {step:6d}/{tr.max_steps}  "
                    f"loss={last_loss_val:.4f}  "
                    f"lr={lr:.2e}  "
                    f"{steps_per_sec:.1f} steps/s  "
                    f"ETA {eta_min:.0f}min"
                )
                append_metric(metrics_path, {
                    "type": "step",
                    "step": step,
                    "loss": last_loss_val,
                    "lr": lr,
                    "elapsed_s": elapsed,
                })
                mlflow_run.log_metrics(
                    {"train/loss": last_loss_val, "train/lr": lr},
                    step=step,
                )

            # --- Validation ---
            if step % tr.eval_every_n_steps == 0:
                val_loss = evaluate_cached(model, val_loader, device)
                logger.info(f"  val_loss={val_loss:.4f}")
                model.train()
                mlflow_run.log_metrics({"val/loss": val_loss}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(os.path.join(tr.output_dir, "best"))

            # --- Checkpoint ---
            if step % tr.save_every_n_steps == 0:
                ckpt_dir = os.path.join(tr.output_dir, f"step_{step:07d}")
                model.save_pretrained(ckpt_dir)
                logger.info(f"  checkpoint → {ckpt_dir}")

    # Final save
    final_dir = os.path.join(tr.output_dir, "final")
    model.save_pretrained(final_dir)
    total_time = time.time() - t0
    logger.info(
        f"Training complete in {total_time:.0f}s ({total_time / 60:.1f} min). "
        f"Best val loss: {best_val_loss:.4f}"
    )
    append_metric(metrics_path, {
        "type": "train_end",
        "total_time_s": total_time,
        "best_val_loss": best_val_loss,
    })
    mlflow_run.log_metrics({
        "train/final_loss": last_loss_val,
        "train/best_val_loss": best_val_loss if best_val_loss < float("inf") else 0.0,
        "train/total_time_s": total_time,
    })
    return model


@torch.no_grad()
def evaluate_cached(model, val_loader, device, max_batches=50):
    """Evaluate using cached features."""
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.compute_loss_cached(batch)
        total_loss += out["loss"].item()
        n += 1
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train action expert using cached backbone features"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="data/metaworld_features")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-cached")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-log-artifacts", action="store_true")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    mlflow_run = MlflowRun(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ).start({
        "script": "train_cached",
        "config": args.config,
        "cache_dir": args.cache_dir,
        "max_steps": cfg.training.max_steps,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
    })

    model = train_cached(
        cfg, args.cache_dir,
        metrics_path=args.metrics_path,
        mlflow_run=mlflow_run,
    )

    if args.mlflow_log_artifacts:
        if args.metrics_path:
            mlflow_run.log_artifact(args.metrics_path, artifact_path="metrics")
        final_dir = os.path.join(cfg.training.output_dir, "final")
        if os.path.isdir(final_dir):
            mlflow_run.log_artifact(final_dir, artifact_path="checkpoints")
    mlflow_run.end()


if __name__ == "__main__":
    main()
