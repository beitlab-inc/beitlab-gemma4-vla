"""
Training script for Gemma4VLA.

Two training strategies are supported:

  1. Action-expert-only (recommended start point):
     Backbone is frozen / LoRA-adapted.  Only the action expert trains.
     Fast convergence, low VRAM.

  2. Full fine-tuning:
     All parameters train.  Requires a large robot dataset and ≥40 GB VRAM
     (or gradient checkpointing + 8-bit/4-bit quantisation).

Run with::

    uv run python -m gemma4_vla.train --config robots/metaworld/configs/metaworld_push.yaml

Or from Python::

    from gemma4_vla.train import train
    from gemma4_vla.config import metaworld_push_config
    train(metaworld_push_config())
"""

import json
import os
import time
import logging
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .config import Gemma4VLAConfig, metaworld_push_config
from .model import Gemma4VLA
from .dataset import RandomDemoDataset, collate_fn
from .observability import MlflowRun, ensure_parent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL metrics helper
# ---------------------------------------------------------------------------

def append_metric(path, record):
    """Append a JSON record to a JSONL file."""
    if path is None:
        return
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine schedule with linear warmup."""
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-7,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def build_optimizer(model: Gemma4VLA, cfg: Gemma4VLAConfig):
    """
    Build an AdamW optimiser with separate learning rates for:
      - backbone parameters   → lr * backbone_lr_multiplier
      - action expert / other → lr
    """
    tr = cfg.training

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters()
                    if id(p) not in backbone_ids and p.requires_grad]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad],
         "lr": tr.learning_rate * tr.backbone_lr_multiplier,
         "name": "backbone"},
        {"params": other_params,
         "lr": tr.learning_rate,
         "name": "action_expert"},
    ]
    # Remove empty groups
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = AdamW(
        param_groups,
        lr=tr.learning_rate,
        weight_decay=tr.weight_decay,
        betas=(0.9, 0.95),
    )
    return optimizer


def load_config_from_yaml(config_path: str) -> Gemma4VLAConfig:
    """Load a full nested config tree from YAML overrides."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load YAML configs. Install with: uv add PyYAML"
        ) from e

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        return Gemma4VLAConfig.from_dict(raw)
    except KeyError as e:
        raise ValueError(f"Invalid config file '{config_path}': {e}") from e


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: Gemma4VLAConfig,
    model: Optional[Gemma4VLA] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    metrics_path: Optional[str] = None,
    mlflow_run: Optional[MlflowRun] = None,
):
    """
    Train a Gemma4VLA model.

    Args:
        cfg:          Config object.
        model:        Pre-instantiated model (created fresh if None).
        train_loader: Pre-built DataLoader (built from cfg if None).
        val_loader:   Pre-built DataLoader (built from cfg if None).
        metrics_path: Optional JSONL path for training metrics.
        mlflow_run:   Optional MlflowRun instance for experiment tracking.
    """
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

    # --- Model ---
    if model is None:
        logger.info("Initialising Gemma4VLA...")
        model = Gemma4VLA(cfg)
        try:
            model = model.to(device)
        except NotImplementedError:
            # Backbone loaded with device_map="auto"; move only non-backbone parts
            model.obs_proj = model.obs_proj.to(device)
            model.action_expert = model.action_expert.to(device)

    # --- Gradient checkpointing (saves memory, trades compute) ---
    if tr.gradient_checkpointing:
        if hasattr(model.backbone.model, "gradient_checkpointing_enable"):
            model.backbone.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on backbone")

    total_params = model.num_parameters()
    trainable_params = model.num_parameters(trainable_only=True)
    logger.info(
        f"Model params: {total_params / 1e6:.1f} M total, "
        f"{trainable_params / 1e6:.1f} M trainable"
    )
    if tr.grad_accum_steps > 1:
        logger.info(
            f"Gradient accumulation: {tr.grad_accum_steps} micro-steps, "
            f"effective batch = {tr.batch_size} × {tr.grad_accum_steps} "
            f"= {tr.batch_size * tr.grad_accum_steps}"
        )

    # --- DataLoaders ---
    # Robot-specific dataset adapters (e.g. robots/metaworld/dataset.py) should
    # build DataLoaders and pass them here.  If none are provided we fall back
    # to a synthetic RandomDemoDataset for smoke-testing the training loop.
    if train_loader is None or val_loader is None:
        logger.warning(
            "No DataLoaders provided. Using RandomDemoDataset for testing. "
            "Pass train_loader/val_loader from your robot-specific adapter."
        )
        demo_train = RandomDemoDataset(
            cfg, n_samples=512, processor=model.backbone.processor
        )
        demo_val = RandomDemoDataset(
            cfg, n_samples=64, processor=model.backbone.processor
        )
        train_loader = DataLoader(
            demo_train, batch_size=tr.batch_size, shuffle=True,
            collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            demo_val, batch_size=tr.batch_size, shuffle=False,
            collate_fn=collate_fn, pin_memory=True
        )

    # --- Optimiser & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, tr.warmup_steps, tr.max_steps)

    # --- Mixed precision ---
    use_amp = tr.mixed_precision in ("fp16", "bf16") and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if tr.mixed_precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=(tr.mixed_precision == "fp16" and device.type == "cuda"))

    # --- Training loop ---
    model.train()
    step = 0
    epoch = 0
    best_val_loss = float("inf")
    os.makedirs(tr.output_dir, exist_ok=True)

    logger.info("Starting training…")
    t0 = time.time()

    append_metric(metrics_path, {
        "type": "train_start",
        "dataset_root": tr.dataset_root,
        "batch_size": tr.batch_size,
        "max_steps": tr.max_steps,
        "learning_rate": tr.learning_rate,
        "mixed_precision": tr.mixed_precision,
        "config": cfg.to_dict(),
    })

    micro_step = 0
    running_loss = 0.0
    last_loss_val = 0.0

    while step < tr.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= tr.max_steps:
                break

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + loss (scale for gradient accumulation)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.compute_loss(batch)
                raw_loss = out["loss"]
                loss = raw_loss / tr.grad_accum_steps

            # Backward (accumulate gradients)
            scaler.scale(loss).backward()
            micro_step += 1
            running_loss += raw_loss.item()

            # Only step optimiser after accumulating grad_accum_steps micro-batches
            if micro_step % tr.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            last_loss_val = running_loss / tr.grad_accum_steps
            running_loss = 0.0

            # --- Logging ---
            if step % tr.log_every_n_steps == 0:
                elapsed = time.time() - t0
                lr_ae = optimizer.param_groups[-1]["lr"]
                logger.info(
                    f"step {step:6d}/{tr.max_steps}  "
                    f"loss={last_loss_val:.4f}  "
                    f"lr={lr_ae:.2e}  "
                    f"elapsed={elapsed:.0f}s"
                )
                append_metric(metrics_path, {
                    "type": "step",
                    "step": step,
                    "loss": last_loss_val,
                    "lr": lr_ae,
                    "elapsed_s": elapsed,
                })
                mlflow_run.log_metrics(
                    {"train/loss": last_loss_val, "train/lr": lr_ae},
                    step=step,
                )

            # --- Validation ---
            if step % tr.eval_every_n_steps == 0:
                val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
                logger.info(f"  val_loss={val_loss:.4f}")
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(os.path.join(tr.output_dir, "best"))

            # --- Checkpoint ---
            if step % tr.save_every_n_steps == 0:
                ckpt_dir = os.path.join(tr.output_dir, f"step_{step:07d}")
                model.save_pretrained(ckpt_dir)
                logger.info(f"  Saved checkpoint → {ckpt_dir}")

    # Final save
    final_dir = os.path.join(tr.output_dir, "final")
    model.save_pretrained(final_dir)
    total_time = time.time() - t0
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    append_metric(metrics_path, {
        "type": "train_end",
        "checkpoint_path": final_dir,
        "total_time_s": total_time,
        "best_val_loss": best_val_loss,
    })
    mlflow_run.log_metrics({
        "train/final_loss": last_loss_val,
        "train/best_val_loss": best_val_loss if best_val_loss < float("inf") else 0.0,
        "train/total_time_s": total_time,
    })
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: Gemma4VLA,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 50,
) -> float:
    """Run validation loop and return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model.compute_loss(batch)
        total_loss += out["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Gemma4VLA")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional; uses metaworld_push_config() if not provided)",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Optional JSONL path for training metrics")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log training params, metrics, and optional artifacts to MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-train",
                        help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                        help="Optional MLflow run name")
    parser.add_argument("--mlflow-log-artifacts", action="store_true",
                        help="Upload checkpoint and metrics JSONL as MLflow artifacts")
    args = parser.parse_args()

    # Load config
    if args.config and args.config.endswith((".yaml", ".yml")):
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = metaworld_push_config()

    # Apply CLI overrides
    if args.data_root:
        cfg.training.dataset_root = args.data_root
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
    ).start(
        {
            "script": "gemma4_vla.train",
            "config": args.config or "metaworld_push_config()",
            "data_root": cfg.training.dataset_root,
            "output_dir": cfg.training.output_dir,
            "max_steps": cfg.training.max_steps,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "mixed_precision": cfg.training.mixed_precision,
        }
    )

    model = train(cfg, metrics_path=args.metrics_path, mlflow_run=mlflow_run)

    if args.mlflow_log_artifacts:
        if args.metrics_path:
            mlflow_run.log_artifact(args.metrics_path, artifact_path="metrics")
        final_dir = os.path.join(cfg.training.output_dir, "final")
        if os.path.isdir(final_dir):
            mlflow_run.log_artifact(final_dir, artifact_path="checkpoints")
    mlflow_run.end()


if __name__ == "__main__":
    main()
