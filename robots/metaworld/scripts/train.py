"""
Train Gemma4VLA on MetaWorld demonstration data.

Uses the MetaWorld HDF5 dataset adapter and the core training loop::

    uv run python -m robots.metaworld.scripts.train \\
        --config robots/metaworld/configs/metaworld_push.yaml \\
        --data-dir data/metaworld_demos
"""

import argparse

from gemma4_vla.config import metaworld_push_config
from gemma4_vla.train import train, load_config_from_yaml
from gemma4_vla.observability import MlflowRun
from robots.metaworld.dataset import build_metaworld_dataloaders


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
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Optional JSONL path for training metrics")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log training to MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-metaworld")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-log-artifacts", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and args.config.endswith((".yaml", ".yml")):
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = metaworld_push_config()

    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg.backbone.model_name, trust_remote_code=True
    )

    train_loader, val_loader = build_metaworld_dataloaders(
        cfg, data_dir=args.data_dir, instruction=args.instruction,
        processor=processor,
    )

    mlflow_run = MlflowRun(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ).start({
        "script": "robots.metaworld.scripts.train",
        "config": args.config or "metaworld_push_config()",
        "data_dir": args.data_dir,
        "output_dir": cfg.training.output_dir,
        "max_steps": cfg.training.max_steps,
        "batch_size": cfg.training.batch_size,
    })

    model = train(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics_path=args.metrics_path,
        mlflow_run=mlflow_run,
    )

    if args.mlflow_log_artifacts:
        import os
        if args.metrics_path:
            mlflow_run.log_artifact(args.metrics_path, artifact_path="metrics")
        final_dir = os.path.join(cfg.training.output_dir, "final")
        if os.path.isdir(final_dir):
            mlflow_run.log_artifact(final_dir, artifact_path="checkpoints")
    mlflow_run.end()


if __name__ == "__main__":
    main()
