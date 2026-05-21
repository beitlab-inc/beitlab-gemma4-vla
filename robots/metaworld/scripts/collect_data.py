"""Collect demonstration data from Meta-World MT1 environments using expert policies."""

import os
import argparse
import time
import numpy as np
import h5py
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
from gemma4_vla.observability import MlflowRun, RerunLogger, ensure_parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="push-v3")
    parser.add_argument("--camera-name", type=str, default="topview",
                        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="data/metaworld_demos")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep between steps for visualization (seconds)")
    parser.add_argument("--instruction", type=str, default="push the object to the goal",
                        help="Fixed instruction for all episodes")
    parser.add_argument("--rerun-mode", choices=["off", "spawn", "save", "connect"], default="off",
                        help="Rerun telemetry mode. Use spawn for local viewer, save for .rrd, connect for existing viewer.")
    parser.add_argument("--rerun-path", type=str, default="rerun/collect_data.rrd",
                        help="Rerun .rrd output path when --rerun-mode save is used")
    parser.add_argument("--rerun-connect-url", type=str, default=None,
                        help="Optional Rerun gRPC URL when --rerun-mode connect is used")
    parser.add_argument("--rerun-log-every", type=int, default=1,
                        help="Log every N environment steps to Rerun")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log collection metadata and summary metrics to MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking URI, defaults to MLFLOW_TRACKING_URI or http://127.0.0.1:5001")
    parser.add_argument("--mlflow-experiment", type=str, default="gemma4-vla-data",
                        help="MLflow experiment name for data collection")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                        help="Optional MLflow run name")
    parser.add_argument("--mlflow-log-dataset-artifact", action="store_true",
                        help="Upload the generated dataset directory as an MLflow artifact")
    return parser.parse_args()


def extract_state(obs):
    return np.asarray(obs, dtype=np.float32).ravel()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rerun = RerunLogger(
        mode=args.rerun_mode,
        app_id=f"gemma4_vla_collect_{args.env_name}",
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
            "script": "scripts.collect_data",
            "env_name": args.env_name,
            "camera_name": args.camera_name,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "instruction": args.instruction,
            "output_dir": args.output_dir,
            "rerun_mode": args.rerun_mode,
        }
    )

    env = gym.make(
        "Meta-World/MT1",
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[args.env_name]()

    instruction = args.instruction
    global_step = 0
    total_transitions = 0
    episode_steps = []
    episode_successes = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        ep_images = []
        ep_states = []
        ep_actions = []

        # Log initial reset frame so the timeline starts from step 0
        if rerun.enabled:
            rerun.set_step(global_step, episode=ep + 1)
            init_img = env.render()
            init_state = extract_state(obs)
            rerun.log_image("collect/camera/image", init_img)
            rerun.log_vector_series("collect/robot/state", init_state)
            rerun.log_scalar("collect/reward", 0.0)
            rerun.log_scalar("collect/success", 0)
            rerun.log_text("collect/instruction", instruction)

        while not done and steps < args.max_steps:
            action = policy.get_action(obs)
            img = env.render()
            state = extract_state(obs)

            ep_images.append(img.copy())
            ep_states.append(state.copy())
            ep_actions.append(np.asarray(action, dtype=np.float32).copy())

            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1
            global_step += 1

            success = int(info.get("success", 0))
            if rerun.enabled and (global_step % max(args.rerun_log_every, 1) == 0):
                rerun.set_step(global_step, episode=ep + 1)
                rerun.log_image("collect/camera/image", img)
                rerun.log_vector_series("collect/robot/state", state)
                rerun.log_vector_series("collect/expert/action", action)
                rerun.log_scalar("collect/reward", reward)
                rerun.log_scalar("collect/success", success)
                rerun.log_text("collect/instruction", instruction)

            if args.sleep > 0:
                time.sleep(args.sleep)

        ep_success = int(info.get('success', 0))
        episode_steps.append(steps)
        episode_successes.append(ep_success)
        total_transitions += steps
        mlflow_run.log_metrics(
            {
                "collect/episode_steps": steps,
                "collect/episode_success": ep_success,
            },
            step=ep + 1,
        )

        ep_path = os.path.join(args.output_dir, f"episode_{ep:06d}.hdf5")
        with h5py.File(ep_path, "w") as f:
            img_grp = f.create_group(f"observation/images/{args.camera_name}")
            img_grp.create_dataset("data", data=np.stack(ep_images, axis=0), dtype=np.uint8)
            f.create_dataset("observation/state", data=np.stack(ep_states, axis=0), dtype=np.float32)
            f.create_dataset("action", data=np.stack(ep_actions, axis=0), dtype=np.float32)
            f.attrs["language_instruction"] = instruction

        print(f"Episode {ep+1}/{args.episodes} finished after {steps} steps, success={ep_success}")

    env.close()

    print(f"Saved {args.episodes} episodes to {args.output_dir}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Success rate: {float(np.mean(episode_successes)) if episode_successes else 0.0:.3f}")
    mlflow_run.log_metrics(
        {
            "collect/transitions": total_transitions,
            "collect/success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
            "collect/avg_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        }
    )
    if args.mlflow_log_dataset_artifact:
        mlflow_run.log_artifact(args.output_dir, artifact_path="datasets")
    mlflow_run.end()


if __name__ == "__main__":
    main()
