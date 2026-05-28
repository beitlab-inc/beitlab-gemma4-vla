"""
Train Gemma4VLA on MetaWorld demonstration data.

Uses the MetaWorld HDF5 dataset adapter and the core training loop::

    uv run python -m robots.metaworld.scripts.train \\
        --config robots/metaworld/configs/metaworld_push.yaml \\
        --data-dir data/metaworld_demos
"""

import argparse
import os

from gemma4_vla.config import metaworld_push_config
from gemma4_vla.model import Gemma4VLA
from gemma4_vla.train import train, load_config_from_yaml
from gemma4_vla.observability import MlflowRun
from gemma4_vla.machine_config import MachineProfile, EnvironmentConfig, apply_machine_defaults
from robots.metaworld.dataset import build_metaworld_dataloaders


def _is_rank_0() -> bool:
    """torchrun sets RANK on every process; non-distributed has no RANK."""
    return int(os.environ.get("RANK", "0")) == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gemma4VLA on MetaWorld data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (defaults to metaworld_push_config())")
    parser.add_argument("--data-dir", type=str, default="data/metaworld_demos",
                        help="Path to HDF5 episodes directory")
    parser.add_argument("--instruction", type=str, default="push the object to the goal",
                        help="Language instruction for all episodes")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--normalize-stats", action="store_true",
                        help="Compute dataset-fit normalisation stats once before training, "
                             "normalise inputs during training, and save normalization.pt next "
                             "to the checkpoint so inference can denormalise symmetrically.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a previous checkpoint directory (containing config.json "
                             "and weights.pt) to initialise model weights from. Used to chain "
                             "Stage 1 (expert-only) into Stage 2 (LoRA) of the two-stage recipe. "
                             "Matching parameter names are loaded with strict=False so a "
                             "Stage-1 checkpoint can seed a Stage-2 config that adds LoRA.")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Optional JSONL path for training metrics")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log training to MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-metaworld")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-log-artifacts", action="store_true")
    parser.add_argument("--mlflow-system-metrics", action="store_true",
                        help="Log CPU/RAM/GPU/disk metrics to MLflow (requires pynvml for GPU)")
    parser.add_argument("--mlflow-system-metrics-interval", type=float, default=10.0,
                        help="Seconds between system-metric samples (default: 10s)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect machine type and environment
    machine = MachineProfile()
    env_config = EnvironmentConfig()

    if args.config and args.config.endswith((".yaml", ".yml")):
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = metaworld_push_config()

    # Apply machine-specific defaults (can be overridden by CLI args)
    apply_machine_defaults(cfg, machine)

    # CLI args override everything
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    elif not args.output_dir:
        cfg.training.output_dir = env_config.output_dir

    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.normalize_stats:
        cfg.training.normalize_stats = True

    if not args.data_dir or args.data_dir == "data/metaworld_demos":
        args.data_dir = env_config.data_dir

    # Use env-config MLflow settings unless explicitly provided
    mlflow_enabled = args.mlflow or env_config.mlflow_enabled
    mlflow_tracking_uri = args.mlflow_tracking_uri or env_config.mlflow_tracking_uri

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg.backbone.model_name, trust_remote_code=True
    )

    train_loader, val_loader = build_metaworld_dataloaders(
        cfg, data_dir=args.data_dir, instruction=args.instruction,
        processor=processor,
    )

    # Only rank 0 talks to MLflow — other ranks get a disabled no-op.
    mlflow_run = MlflowRun(
        enabled=mlflow_enabled and _is_rank_0(),
        tracking_uri=mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        system_metrics=args.mlflow_system_metrics,
        system_metrics_interval=args.mlflow_system_metrics_interval,
    ).start({
        "script": "robots.metaworld.scripts.train",
        "config": args.config or "metaworld_push_config()",
        "machine_type": machine.machine_type,
        "data_dir": args.data_dir,
        "output_dir": cfg.training.output_dir,
        "max_steps": cfg.training.max_steps,
        "batch_size": cfg.training.batch_size,
    })

    init_model = None
    if args.init_from:
        if _is_rank_0():
            print(f"Initialising model from checkpoint: {args.init_from}")
        init_model = Gemma4VLA.from_pretrained(args.init_from, cfg=cfg)

    model = train(
        cfg,
        model=init_model,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics_path=args.metrics_path,
        mlflow_run=mlflow_run,
    )

    if args.mlflow_log_artifacts and _is_rank_0():
        if args.metrics_path:
            mlflow_run.log_artifact(args.metrics_path, artifact_path="metrics")
        final_dir = os.path.join(cfg.training.output_dir, "final")
        if os.path.isdir(final_dir):
            mlflow_run.log_artifact(final_dir, artifact_path="checkpoints")
    mlflow_run.end()


if __name__ == "__main__":
    main()
