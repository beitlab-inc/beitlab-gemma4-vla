"""Collect demonstration data from Meta-World MT1 environments using expert policies."""

import os
import json
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
                        help="Single Meta-World camera (used if --cameras is not given). "
                             "Options: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera list to record per step "
                             "(e.g. 'topview,corner,gripperPOV'). Overrides --camera-name.")
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


ARM_JOINTS = ["right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]
GRIPPER_JOINTS = ["r_close", "l_close"]

# Named breakdown of the 39-dim MetaWorld observation. Skips the prev-frame
# stack (dims 18:36) and the unused second-object slots (dims 11:18) for the
# single-object MT1 tasks like push-v3.
STATE_LAYOUT = {
    "tcp/x": 0, "tcp/y": 1, "tcp/z": 2,
    "gripper": 3,
    "obj_pos/x": 4, "obj_pos/y": 5, "obj_pos/z": 6,
    "obj_quat/w": 7, "obj_quat/x": 8, "obj_quat/y": 9, "obj_quat/z": 10,
    "goal/x": 36, "goal/y": 37, "goal/z": 38,
}
ACTION_NAMES = ["dx", "dy", "dz", "gripper"]


def log_named_state(rerun, state, prefix="collect/robot/state"):
    if not rerun.enabled:
        return
    for name, idx in STATE_LAYOUT.items():
        if idx < len(state):
            rerun.log_scalar(f"{prefix}/{name}", float(state[idx]))


def log_named_action(rerun, action, prefix="collect/expert/action"):
    if not rerun.enabled:
        return
    a = np.asarray(action).ravel()
    for i, name in enumerate(ACTION_NAMES):
        if i < len(a):
            rerun.log_scalar(f"{prefix}/{name}", float(a[i]))


def log_robot_motors(rerun, env):
    if not rerun.enabled:
        return
    data = env.unwrapped.data
    qpos, qvel = data.qpos, data.qvel
    for i, name in enumerate(ARM_JOINTS):
        rerun.log_scalar(f"collect/robot/arm/{name}/qpos", qpos[i])
        rerun.log_scalar(f"collect/robot/arm/{name}/qvel", qvel[i])
        rerun.log_scalar(f"collect/robot/arm/{name}/qfrc", data.qfrc_actuator[i])
    for j, name in enumerate(GRIPPER_JOINTS):
        rerun.log_scalar(f"collect/robot/gripper/{name}/qpos", qpos[7 + j])
        rerun.log_scalar(f"collect/robot/gripper/{name}/qvel", qvel[7 + j])
    for k in range(data.ctrl.shape[0]):
        rerun.log_scalar(f"collect/robot/actuator/{k}/ctrl", data.ctrl[k])
        rerun.log_scalar(f"collect/robot/actuator/{k}/force", data.actuator_force[k])
    # Mocap target (IK target the arm is chasing)
    hand_quat = data.body("hand").xquat
    for ax, v in zip("wxyz", hand_quat):
        rerun.log_scalar(f"collect/robot/tcp_quat/{ax}", float(v))


def render_cameras(env, camera_ids):
    """Render one frame per camera by mutating the renderer's camera_id."""
    renderer = env.unwrapped.mujoco_renderer
    out = {}
    for name, cid in camera_ids.items():
        renderer.camera_id = cid
        out[name] = env.render().copy()
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else [args.camera_name]

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
            "cameras": ",".join(cameras),
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
        camera_name=cameras[0],
    )

    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[args.env_name]()

    mj_model = env.unwrapped.model
    camera_ids = {name: mj_model.camera(name).id for name in cameras}

    instruction = args.instruction
    global_step = 0
    total_transitions = 0
    episode_steps = []
    episode_successes = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        ep_images = {cam: [] for cam in cameras}
        ep_states = []
        ep_actions = []
        ep_tcp_quat = []
        ep_joint_pos = []
        ep_rewards = []
        ep_successes = []

        # Log initial reset frame so the timeline starts from step 0
        if rerun.enabled:
            rerun.set_step(global_step, episode=ep + 1)
            init_imgs = render_cameras(env, camera_ids)
            init_state = extract_state(obs)
            for cam, img in init_imgs.items():
                rerun.log_image(f"collect/camera/{cam}", img)
            log_named_state(rerun, init_state)
            rerun.log_scalar("collect/reward", 0.0)
            rerun.log_scalar("collect/success", 0)
            rerun.log_text("collect/instruction", instruction)
            log_robot_motors(rerun, env)

        while not done and steps < args.max_steps:
            action = policy.get_action(obs)
            imgs = render_cameras(env, camera_ids)
            state = extract_state(obs)

            mj_data = env.unwrapped.data
            tcp_quat = np.asarray(mj_data.body("hand").xquat, dtype=np.float32).copy()
            joint_pos = np.asarray(mj_data.qpos[:9], dtype=np.float32).copy()

            for cam in cameras:
                ep_images[cam].append(imgs[cam])
            ep_states.append(state.copy())
            ep_actions.append(np.asarray(action, dtype=np.float32).copy())
            ep_tcp_quat.append(tcp_quat)
            ep_joint_pos.append(joint_pos)

            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1
            global_step += 1

            success = int(info.get("success", 0))
            ep_rewards.append(float(reward))
            ep_successes.append(success)

            if rerun.enabled and (global_step % max(args.rerun_log_every, 1) == 0):
                rerun.set_step(global_step, episode=ep + 1)
                for cam, img in imgs.items():
                    rerun.log_image(f"collect/camera/{cam}", img)
                log_named_state(rerun, state)
                log_named_action(rerun, action)
                rerun.log_scalar("collect/reward", reward)
                rerun.log_scalar("collect/success", success)
                rerun.log_text("collect/instruction", instruction)
                log_robot_motors(rerun, env)

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
            for cam, frames in ep_images.items():
                g = f.create_group(f"observation/images/{cam}")
                g.create_dataset("data", data=np.stack(frames, axis=0), dtype=np.uint8)
            f.create_dataset("observation/state", data=np.stack(ep_states, axis=0), dtype=np.float32)
            f.create_dataset("observation/tcp_quat", data=np.stack(ep_tcp_quat, axis=0), dtype=np.float32)
            f.create_dataset("observation/joint_pos", data=np.stack(ep_joint_pos, axis=0), dtype=np.float32)
            f.create_dataset("action", data=np.stack(ep_actions, axis=0), dtype=np.float32)
            f.create_dataset("reward", data=np.asarray(ep_rewards, dtype=np.float32))
            f.create_dataset("success", data=np.asarray(ep_successes, dtype=np.uint8))
            f.attrs["language_instruction"] = instruction
            f.attrs["env_name"] = args.env_name
            f.attrs["seed"] = int(args.seed + ep)
            f.attrs["episode_index"] = int(ep)
            f.attrs["episode_success"] = ep_success
            f.attrs["camera_names"] = json.dumps(cameras)
            f.attrs["state_layout"] = json.dumps(STATE_LAYOUT)
            f.attrs["action_names"] = json.dumps(ACTION_NAMES)

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
