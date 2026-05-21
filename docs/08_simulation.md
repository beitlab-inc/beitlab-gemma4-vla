# Simulation Pipeline — MetaWorld + MuJoCo + Gymnasium

This document explains how Gemma4VLA integrates with robotic simulation
environments for training and evaluating vision-language-action policies.

---

## Overview

Gemma4VLA uses [Meta-World](https://meta-world.github.io/) as its primary
simulation benchmark. Meta-World provides 50 robotic manipulation tasks built
on [MuJoCo](https://mujoco.org/) physics, exposed through the
[Gymnasium](https://gymnasium.farama.org/) API.

The simulation pipeline has four stages:

```
Collect ──► Train ──► Evaluate ──► Analyze
(expert)    (model)   (simulation)  (metrics/video)
```

---

## Architecture

### How the model interacts with the simulation

```
                    ┌─────────────────────────┐
                    │   MetaWorldMT1Wrapper    │
                    │                         │
                    │  MuJoCo physics engine   │
                    │  Gymnasium API           │
                    │  MetaWorld task logic     │
                    └────────┬────────────────┘
                             │
                   reset()   │   step(action)
                             │
              ┌──────────────▼──────────────┐
              │                             │
    image [H,W,3] uint8       state [39] float32
              │                             │
              └──────────────┬──────────────┘
                             │
                   ┌─────────▼─────────┐
                   │   PolicyRunner     │
                   │                   │
                   │ _preprocess():    │
                   │  image → 224x224  │
                   │  text → tokens    │
                   │  state → tensor   │
                   └─────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Gemma4VLA Model        │
              │                             │
              │ Gemma 4 Backbone:           │
              │  camera image + "push the   │
              │  object to the goal"        │
              │       │                     │
              │       ▼ obs_features        │
              │                             │
              │ Action Expert:              │
              │  state + noise → velocity   │
              │  × cross-attention to obs   │
              │       │                     │
              │       ▼                     │
              │ Flow Matching ODE:          │
              │  Euler integration t:0→1    │
              │  10 steps → action chunk    │
              └──────────────┬──────────────┘
                             │
                   actions [50, 4]
                   take actions[0]
                             │
                   ┌─────────▼─────────┐
                   │   env.step(a)      │
                   │   → next obs       │
                   │   → reward         │
                   │   → done/success   │
                   └───────────────────┘
```

### Why 50 actions but only 1 is used?

Gemma4VLA predicts a full 50-step action horizon (1 second at 50 Hz), but
MetaWorld's control loop takes a single action per step. The simplest
approach is to use `actions[0]` and re-predict every step. This is
intentional — the 50-step prediction acts as implicit planning, and the
model can adjust its plan at every step based on new observations.

For efficiency, `PolicyRunner.stream(obs, replan_every=N)` buffers the
prediction and only re-plans every N steps:

```python
# Replan every 10 steps (5x faster, slight quality tradeoff)
for action in runner.stream(obs, replan_every=10):
    image, state, reward, done, info = env.step(action)
    obs = {"images": [image], "state": state, "instruction": "..."}
```

---

## MetaWorld environment details

### MetaWorldMT1Wrapper

The wrapper (`src/gemma4_vla/envs/metaworld_env.py`) simplifies the
MetaWorld / Gymnasium API into a flat interface:

| Method | Inputs | Outputs |
|--------|--------|---------|
| `reset(seed=None)` | — | `(image, state, info)` |
| `step(action)` | `[action_dim]` float32 | `(image, state, reward, done, info)` |
| `close()` | — | — |

**Properties:**

| Property | MetaWorld value | Description |
|----------|----------------|-------------|
| `state_dim` | 39 | Flattened observation (joint positions, velocities, object pose, goal) |
| `action_dim` | 4 | End-effector XYZ + gripper |
| `obs_shape` | (480, 480, 3) | RGB camera image |
| `action_low` | [-1, -1, -1, -1] | Action space lower bound |
| `action_high` | [+1, +1, +1, +1] | Action space upper bound |

### Supported tasks

| Task | Description | Expert success rate |
|------|-------------|-------------------|
| `push-v3` | Push object to goal position | ~100% |
| `reach-v3` | Move end-effector to goal | ~100% |
| `pick-place-v3` | Pick up object, place at goal | ~90% |

### Camera views

MetaWorld supports 7 camera viewpoints:

| Camera name | Description |
|-------------|-------------|
| `corner` | Diagonal view from front-left |
| `corner2` | Diagonal view from front-right |
| `corner3` | Diagonal view from back-left |
| `corner4` | Diagonal view from back-right |
| `topview` | Bird's-eye view |
| `behindGripper` | Behind the gripper |
| `gripperPOV` | First-person gripper view |

**Recommendation:** Use `corner` for training — it provides the most
informative view of both the robot arm and the workspace.

### State observation breakdown

MetaWorld's 39-dimensional state vector contains:

| Indices | Content |
|---------|---------|
| 0–2 | End-effector XYZ position |
| 3 | Gripper opening |
| 4–6 | End-effector XYZ velocity (unused by some tasks) |
| 7–9 | Object XYZ position |
| 10–13 | Object quaternion |
| 14–17 | Object XYZ velocity + angular velocity (partial) |
| 18+ | Goal position + task-specific features |

---

## Data collection

### Expert policies

MetaWorld provides scripted expert policies for each task via
`metaworld.policies.ENV_POLICY_MAP`. These are deterministic controllers
(not learned) that achieve near-perfect success rates.

### HDF5 episode format

Each episode is saved as a separate HDF5 file:

```
episode_000000.hdf5
├── observation/
│   ├── images/
│   │   └── <camera_name>/
│   │       └── data    [T, 480, 480, 3]  uint8
│   └── state           [T, 39]           float32
├── action              [T, 4]            float32
└── attrs:
    └── language_instruction: "push the object to the goal"
```

This format is read by `robots.metaworld.dataset.MetaWorldHDF5Dataset` for
training.

### CLI reference

```bash
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 \
    --camera-name corner \
    --episodes 100 \
    --max-steps 150 \
    --instruction "push the object to the goal" \
    --output-dir data/push_demos \
    --rerun-mode spawn \       # optional: live visualization
    --mlflow                    # optional: experiment tracking
```

---

## Training on simulation data

### Config preset

`configs/metaworld_push.yaml` provides a ready-made config for MetaWorld:

```yaml
robot:
  name: "metaworld-push"
  state_dim: 39       # MetaWorld observation size
  action_dim: 4       # MetaWorld action size
  max_state_dim: 39   # no padding needed (larger than default 18)

vision:
  num_cameras: 1      # single camera in MetaWorld

training:
  dataset_root: "./data/metaworld_demos"
```

### Two-stage recipe

```bash
# Stage 1: action expert warm-up (backbone frozen)
uv run python -m gemma4_vla.train \
    --config configs/metaworld_push.yaml \
    --max_steps 2000 \
    --metrics-path metrics/stage1.jsonl

# Stage 2: joint fine-tuning (LoRA unfrozen)
uv run python -m gemma4_vla.train \
    --config configs/metaworld_push.yaml \
    --max_steps 50000 \
    --metrics-path metrics/stage2.jsonl
```

### Normalization

The test script supports optional per-DOF action/state normalization. If a
`normalization.pt` file is present in the checkpoint directory, the test
script loads it and denormalizes model outputs before sending to the
environment:

```
action_env = action_model * action_std + action_mean
action_env = clip(action_env, env.action_low, env.action_high)
```

---

## Evaluation

### Core loop

```python
runner = PolicyRunner.from_pretrained("checkpoints/best", device="cuda")
env = MetaWorldMT1Wrapper("push-v3", camera_name="corner")

for ep in range(num_episodes):
    image, state, info = env.reset()
    done = False

    while not done and step < max_steps:
        obs = {"images": [image], "state": state, "instruction": "..."}
        actions = runner.predict(obs)  # [50, 4]
        action = actions[0]            # first action from horizon

        image, state, reward, done, info = env.step(action)
        success = info.get("success", 0)
```

### Metrics output

The evaluation JSON includes:

```json
{
  "checkpoint": "checkpoints/best",
  "env_name": "push-v3",
  "episodes": [
    {
      "episode": 1,
      "reward": 45.2,
      "steps": 87,
      "success": 1,
      "steps_detail": [
        {
          "step": 1,
          "reward": 0.5,
          "action_model": [0.1, -0.2, 0.05, 0.8],
          "action_env": [0.1, -0.2, 0.05, 0.8]
        }
      ]
    }
  ],
  "summary": {
    "avg_reward": 42.1,
    "success_rate": 0.8
  }
}
```

### Video recording

Each episode can be saved as an MP4 video at 20 FPS:

```bash
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/best \
    --save-video --video-dir videos
```

Output: `videos/push-v3_ep001.mp4`, `videos/push-v3_ep002.mp4`, etc.

---

## Observability during simulation

### Rerun

Rerun provides a real-time timeline view of the simulation. Each step logs:

| Channel | Type | Content |
|---------|------|---------|
| `test/camera/image` | Image | RGB camera frame |
| `test/robot/state` | Tensor | 39-dim state vector |
| `test/policy/action_model` | Tensor | Raw model output |
| `test/policy/action_env` | Tensor | Clipped env action |
| `test/metrics/reward` | Scalar | Step reward |
| `test/metrics/success` | Scalar | Task success flag |
| `test/instruction` | Text | Language instruction |

### MLflow

MLflow tracks experiment-level data:

| What | Where | When |
|------|-------|------|
| Params | MLflow params | Run start |
| Per-episode reward/steps/success | MLflow metrics | After each episode |
| Summary avg_reward/success_rate | MLflow metrics | Run end |
| Videos + metrics JSON | MLflow artifacts | Run end (if `--mlflow-log-artifacts`) |

### Docker MLflow stack

```bash
docker compose -f docker-compose.mlflow.yml up -d
# PostgreSQL backend + MLflow UI at http://localhost:5001
```
