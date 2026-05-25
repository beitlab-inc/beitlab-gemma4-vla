"""Test Gemma4VLA policy on Meta-World MT1 environments."""

import os
import argparse
import json
import numpy as np
import torch
import imageio.v2 as imageio

from robots.metaworld.env import MetaWorldMT1Wrapper
from gemma4_vla.inference import PolicyRunner
from gemma4_vla.model import Gemma4VLA, load_config_artifact
from gemma4_vla.config import Gemma4VLAConfig
from gemma4_vla.observability import MlflowRun, RerunLogger, ensure_parent
from gemma4_vla.stats import DatasetStats


def parse_args():
    parser = argparse.ArgumentParser(description="Test Gemma4VLA on Meta-World MT1")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best",
                        help="Path to trained Gemma4VLA checkpoint directory")
    parser.add_argument("--env-name", type=str, default="push-v3",
                        help="Meta-World MT1 task name, e.g. push-v3, reach-v3, pick-place-v3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum steps per episode")
    parser.add_argument("--camera-name", type=str, default="topview",
                        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--instruction", type=str, default="push the object to the goal",
                        help="Language instruction passed to the VLA")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--save-video", action="store_true", help="Save each episode as an MP4 video")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Optional JSON path for evaluation metrics and generated actions")
    parser.add_argument("--save-flow-trace", action="store_true",
                        help="Include every ODE integration step in --metrics-path output.")
    parser.add_argument("--rerun-mode", choices=["off", "spawn", "save", "connect"], default="off",
                        help="Rerun telemetry mode.")
    parser.add_argument("--rerun-path", type=str, default="rerun/test.rrd",
                        help="Rerun .rrd output path when --rerun-mode save is used")
    parser.add_argument("--rerun-connect-url", type=str, default=None,
                        help="Optional Rerun gRPC URL when --rerun-mode connect is used")
    parser.add_argument("--rerun-log-every", type=int, default=1,
                        help="Log every N environment steps to Rerun")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log evaluation params, metrics, and optional artifacts to MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-eval",
                        help="MLflow experiment name for evaluation")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Optional MLflow run name")
    parser.add_argument("--mlflow-log-artifacts", action="store_true",
                        help="Upload metrics JSON and saved videos as MLflow artifacts")
    return parser.parse_args()


def load_normalization_stats(checkpoint_path, device):
    """Load normalization stats from checkpoint if present.

    PolicyRunner already applies these on every predict() call when present,
    so this helper exists only to surface the loaded stats in MLflow params
    and the per-step metrics file for downstream analysis.
    """
    stats = DatasetStats.load(checkpoint_path)
    if stats is not None and stats.enabled:
        return {
            "state_mean": stats.state_mean.to(device),
            "state_std": stats.state_std.to(device),
            "action_mean": stats.action_mean.to(device),
            "action_std": stats.action_std.to(device),
            "normalize": True,
        }
    return {
        "state_mean": torch.zeros(1),
        "state_std": torch.ones(1),
        "action_mean": torch.zeros(1),
        "action_std": torch.ones(1),
        "normalize": False,
    }


def summarize_vectors(vectors):
    if not vectors:
        return {"count": 0}
    arr = np.asarray(vectors, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[test] Loading checkpoint from {args.checkpoint}")
    runner = PolicyRunner.from_pretrained(args.checkpoint, device=device)
    norm = load_normalization_stats(args.checkpoint, device)

    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    print(f"[test] Meta-World MT1 env: {args.env_name}")
    print(f"[test] camera={args.camera_name}, normalize={norm['normalize']}")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    rerun = RerunLogger(
        mode=args.rerun_mode,
        app_id=f"gemma4_vla_test_{args.env_name}",
        save_path=args.rerun_path,
        connect_url=args.rerun_connect_url,
    ).start()
    mlflow_run = MlflowRun(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    ).start(
        {
            "script": "scripts.test",
            "checkpoint": args.checkpoint,
            "env_name": args.env_name,
            "camera_name": args.camera_name,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "instruction": args.instruction,
            "device": str(device),
            "normalize": norm["normalize"],
            "save_flow_trace": args.save_flow_trace,
            "rerun_mode": args.rerun_mode,
        }
    )

    metrics = {
        "checkpoint": args.checkpoint,
        "env_name": args.env_name,
        "camera_name": args.camera_name,
        "instruction": args.instruction,
        "episodes_requested": args.episodes,
        "max_steps": args.max_steps,
        "normalize": norm["normalize"],
        "save_flow_trace": args.save_flow_trace,
        "episodes": [],
    }
    all_model_actions = []
    all_env_actions = []
    episode_rewards = []
    episode_successes = []
    global_step = 0
    saved_video_paths = []

    for ep in range(args.episodes):
        img, state, info = env.reset()
        step = 0
        ep_reward = 0.0
        ep_steps = []
        frames = [img.copy()]
        done = False

        while not done and step < args.max_steps:
            obs_img = img
            obs_state = state

            obs = {
                "images": [img],
                "state": state,
                "instruction": args.instruction,
            }

            with torch.no_grad():
                actions = runner.predict(obs)

            # `runner.predict` already denormalises with the checkpoint's stats
            # (if present), so the value here is in env space — only clipping
            # remains. Keep the three-step debug fields so analysis tooling
            # downstream still sees model/unclipped/clipped views.
            action_model_np = actions[0]
            action_env_np = action_model_np[:env.action_dim].copy()
            action_np = np.clip(action_env_np, env.action_low, env.action_high)

            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            step += 1
            global_step += 1
            success = int(info.get("success", 0))

            all_model_actions.append(action_model_np.tolist())
            all_env_actions.append(action_np.tolist())

            step_record = {
                "step": step,
                "reward": float(reward),
                "done": bool(done),
                "success": success,
                "action_model": action_model_np.tolist(),
                "action_env_unclipped": action_env_np.tolist(),
                "action_env": action_np.tolist(),
            }
            ep_steps.append(step_record)

            if rerun.enabled and (global_step % max(args.rerun_log_every, 1) == 0):
                rerun.set_step(global_step, episode=ep + 1)
                rerun.log_image("test/camera/image", obs_img)
                rerun.log_vector("test/robot/state", obs_state, dim_name="state")
                rerun.log_vector("test/policy/action_model", action_model_np, dim_name="action")
                rerun.log_vector("test/policy/action_env_unclipped", action_env_np, dim_name="action")
                rerun.log_vector("test/policy/action_env", action_np, dim_name="action")
                rerun.log_scalar("test/metrics/reward", reward)
                rerun.log_scalar("test/metrics/success", success)
                rerun.log_text("test/instruction", args.instruction)

            frames.append(img.copy())

        ep_success = int(info.get("success", 0))
        episode_rewards.append(float(ep_reward))
        episode_successes.append(ep_success)
        mlflow_run.log_metrics(
            {
                "eval/episode_reward": ep_reward,
                "eval/episode_steps": step,
                "eval/episode_success": ep_success,
            },
            step=ep + 1,
        )
        metrics["episodes"].append(
            {
                "episode": ep + 1,
                "reward": float(ep_reward),
                "steps": step,
                "success": ep_success,
                "steps_detail": ep_steps,
            }
        )

        print(
            f"[test] Episode {ep+1}/{args.episodes}: "
            f"reward={ep_reward:.3f}, steps={step}, success={ep_success}"
        )

        if args.save_video:
            video_path = os.path.join(args.video_dir, f"{args.env_name}_ep{ep+1:03d}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {video_path}")
            saved_video_paths.append(video_path)

    env.close()
    metrics["summary"] = {
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "action_model": summarize_vectors(all_model_actions),
        "action_env": summarize_vectors(all_env_actions),
    }
    print(
        "[test] Summary: "
        f"avg_reward={metrics['summary']['avg_reward']:.3f}, "
        f"success_rate={metrics['summary']['success_rate']:.3f}"
    )
    mlflow_run.log_metrics(
        {
            "eval/avg_reward": metrics["summary"]["avg_reward"],
            "eval/success_rate": metrics["summary"]["success_rate"],
        }
    )
    if args.metrics_path:
        ensure_parent(args.metrics_path)
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[test] Saved metrics to {args.metrics_path}")
    if args.mlflow_log_artifacts:
        if args.metrics_path:
            mlflow_run.log_artifact(args.metrics_path, artifact_path="metrics")
        for video_path in saved_video_paths:
            mlflow_run.log_artifact(video_path, artifact_path="videos")
    mlflow_run.end()
    print("[test] Done.")


if __name__ == "__main__":
    main()
