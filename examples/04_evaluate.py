"""
Example 04 — Evaluate a trained Gemma4VLA policy.

Runs the model on a held-out validation set and reports:
  - Flow matching loss
  - Action prediction error (L1 / L2 per joint)
  - Inference latency and throughput (Hz)

For sim evaluation with gymnasium / gym environments, see the optional
section at the bottom.

Usage:
    uv run python examples/04_evaluate.py --checkpoint checkpoints/best
    uv run python examples/04_evaluate.py --checkpoint checkpoints/best --split val --num_batches 100
"""

import argparse
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_action_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute action prediction metrics.

    Args:
        pred:   Predicted actions   [B, H, D].
        target: Ground-truth actions [B, H, D].

    Returns:
        Dict of scalar metrics.
    """
    with torch.no_grad():
        l1 = (pred - target).abs().mean().item()
        l2 = ((pred - target) ** 2).mean().sqrt().item()

        # Per-joint L2 (useful for debugging specific DOF)
        per_joint_l2 = ((pred - target) ** 2).mean(dim=(0, 1)).sqrt()  # [D]

        # Trajectory similarity (cosine over flattened H*D)
        p_flat = pred.flatten(1)    # [B, H*D]
        t_flat = target.flatten(1)
        cos_sim = torch.nn.functional.cosine_similarity(p_flat, t_flat, dim=-1).mean().item()

    metrics = {
        "l1_error": l1,
        "l2_error": l2,
        "cosine_similarity": cos_sim,
    }
    for i, v in enumerate(per_joint_l2.tolist()):
        metrics[f"l2_joint_{i}"] = v
    return metrics


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
    num_inference_steps: int = 10,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the model on a DataLoader.

    Returns aggregated metrics dict.
    """
    model.eval()
    all_metrics = defaultdict(list)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # ── Flow matching loss (teacher-forced) ──────────────────────────────
        out = model.compute_loss(batch)
        all_metrics["flow_matching_loss"].append(out["loss"].item())

        # ── Action prediction error (free-running inference) ─────────────────
        obs = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "state")}
        pred_actions = model.predict_action(obs, num_steps=num_inference_steps)

        target_actions = batch["actions"][:, :pred_actions.shape[1], :]
        act_metrics = compute_action_metrics(pred_actions, target_actions)
        for k, v in act_metrics.items():
            all_metrics[k].append(v)

        if verbose and i % 10 == 0:
            print(f"  batch {i:3d}/{num_batches}  "
                  f"loss={out['loss'].item():.4f}  "
                  f"l2={act_metrics['l2_error']:.4f}")

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------

def benchmark_latency(
    runner,
    num_runs: int = 30,
    num_inference_steps: int = 10,
) -> Dict[str, float]:
    """Measure inference latency in milliseconds."""
    cfg = runner.cfg
    device = runner.device

    obs = {
        "images": [
            np.random.randint(0, 256, (cfg.vision.image_size, cfg.vision.image_size, 3), dtype=np.uint8)
            for _ in range(cfg.vision.num_cameras)
        ],
        "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
        "instruction": "Pick up the object.",
    }

    # Warmup
    for _ in range(3):
        runner.predict(obs, num_inference_steps=num_inference_steps)

    # Measure
    if str(device) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        runner.predict(obs, num_inference_steps=num_inference_steps)
        if str(device) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    p95_ms = float(np.percentile(times, 95))

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p95_ms": p95_ms,
        "hz": 1000.0 / mean_ms,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float], title: str = "Results"):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<35s} {v:.4f}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma4VLA")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root (uses config value if not set)")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of validation batches to evaluate")
    parser.add_argument("--inference_steps", type=int, default=10,
                        help="Denoising steps per inference call")
    parser.add_argument("--no_latency", action="store_true",
                        help="Skip latency benchmarking")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from gemma4_vla import Gemma4VLA, PolicyRunner
    from gemma4_vla.dataset import build_dataloaders, RandomDemoDataset, collate_fn

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Gemma4VLA.from_pretrained(args.checkpoint)
    model = model.to(args.device)

    cfg = model.cfg
    if args.data_root:
        cfg.training.dataset_root = args.data_root

    runner = PolicyRunner(model, device=args.device)

    # ── Build validation loader ───────────────────────────────────────────────
    try:
        _, val_loader = build_dataloaders(cfg, model.backbone.processor)
    except FileNotFoundError:
        print("Dataset not found — using RandomDemoDataset for evaluation demo")
        demo_val = RandomDemoDataset(cfg, n_samples=200, processor=model.backbone.processor)
        val_loader = DataLoader(demo_val, batch_size=cfg.training.batch_size,
                                collate_fn=collate_fn)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating on validation set ({args.num_batches} batches)…")
    metrics = evaluate_dataset(
        model=model,
        loader=val_loader,
        device=torch.device(args.device),
        num_batches=args.num_batches,
        num_inference_steps=args.inference_steps,
    )
    print_metrics(metrics, title="Validation Metrics")

    # ── Latency ───────────────────────────────────────────────────────────────
    if not args.no_latency:
        print("\nBenchmarking inference latency…")
        lat = benchmark_latency(runner, num_inference_steps=args.inference_steps)
        print_metrics(lat, title=f"Inference Latency ({args.inference_steps} steps)")
        print(f"\n  Control frequency: {lat['hz']:.1f} Hz")
        if lat["hz"] >= 50:
            print("  ✓ Fast enough for 50 Hz real-time control")
        elif lat["hz"] >= 10:
            print("  ⚠ Suitable for 10 Hz control; reduce inference_steps for 50 Hz")
        else:
            print("  ✗ Too slow for real-time; consider quantisation or fewer steps")


if __name__ == "__main__":
    main()
