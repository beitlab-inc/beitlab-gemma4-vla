# Gemma4VLA — Vision-Language-Action Model

**A BeitLab Robotics Research Initiative** 🤖

A PyTorch implementation of a **pi0-style VLA** (Vision-Language-Action model)
that replaces the PaliGemma backbone with **Google Gemma 4**.

> **What is a VLA?**  
> A Vision-Language-Action (VLA) model is a robot foundation model that
> reads camera images + a language instruction and predicts motor commands.
> It bridges the gap between large vision-language models (trained on internet
> data) and physical robot control.

---

## About BeitLab 🏭

**BeitLab Robotics** is committed to democratizing robot learning through open-source research and accessible tools. 

This project demonstrates that **cutting-edge VLA training is now possible on edge hardware** — specifically the **Jetson Thor**, a powerful yet cost-effective GPU for robotics applications. We've engineered this pipeline to be:

- 🚀 **Hardware-agnostic**: Automatic detection and optimization for Jetson Thor, cloud GPUs (A100/H100/L40S), and desktop setups
- 🔓 **Fully open-source**: All code, configs, and documentation freely available
- 📊 **Production-ready**: Integrated observability (Rerun + MLflow), distributed training (DDP), and multi-machine support
- 🤝 **Community-driven**: Built for researchers, engineers, and hobbyists

### Key Achievement

Train **Gemma 4 Vision-Language-Action models on Jetson Thor's shared 12GB VRAM** with:
- Automatic batch-size scaling (2–16 per hardware tier)
- Gradient checkpointing + LoRA for memory efficiency
- Real-time telemetry via Rerun
- GPU metrics tracking via MLflow
- Multi-camera data collection with semantic state logging

This is the foundation for on-device robot learning without cloud dependency.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Observation                                  │
│                                                                         │
│   Camera 0  ──┐                                                         │
│   Camera 1  ──┤──► Gemma 4 Vision Tower (SigLIP2)                      │
│   Camera N  ──┘          +                                              │
│                    Language tokens  ──► Gemma 4 Tokenizer               │
│                                              │                          │
│                               Gemma 4 Backbone (E2B–31B)                │
│                           (hybrid local + global attention)             │
│                                              │                          │
│                              obs_features  [B, S, D]                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  cross-attention
┌──────────────────────────────────▼──────────────────────────────────────┐
│                          Action Expert (~300M)                          │
│                                                                         │
│   proprioceptive state ──► state_token                                  │
│   noisy actions        ──► action_tokens  [B, H, D]                    │
│   noise level t        ──► sinusoidal embedding                         │
│                                    │                                    │
│                   Action Expert Transformer (8 layers)                  │
│                         (cross-attends to obs)                          │
│                                    │                                    │
│                         velocity  [B, H, action_dim]                   │
└─────────────────────────────────────────────────────────────────────────┘

Training:   Conditional Flow Matching (OT path)
            loss = ||v_θ(x_t, obs, t) − u(x_t|x_1)||²

Inference:  Euler ODE integration,  t: 0 → 1
            x_{t+dt} = x_t + v_θ(x_t, obs, t) · dt
```

### Key differences from pi0

| Component | pi0 (original) | Gemma4VLA (this repo) |
|-----------|---------------|----------------------|
| Backbone | PaliGemma 3B (SigLIP + Gemma 2B) | **Gemma 4** (E2B – 31B) |
| Vision | SigLIP ViT-So400M | Gemma 4 built-in vision tower |
| Context window | ~2K tokens | **256K tokens** |
| Attention | Bidirectional prefix + causal | **Hybrid local + global** |
| Fine-tuning | Full | **LoRA** (parameter-efficient) |
| Framework | JAX | **PyTorch** |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/beitlab-inc/beitlab-gemma4-vla
cd beitlab-gemma4-vla

# Install with uv (creates .venv automatically)
uv sync

# Install optional extras
uv sync --extra quant          # 4-bit quantisation (bitsandbytes)
uv sync --extra metaworld      # MetaWorld simulation (MuJoCo + Gymnasium)
uv sync --extra observability  # Rerun + MLflow

# Or install all extras at once
uv sync --all-extras
```

**Accept the Gemma 4 license** on Hugging Face and log in:

```bash
uv run huggingface-cli login
```

### Hardware requirements

| Config | VRAM | Throughput |
|--------|------|-----------|
| Gemma 4 E2B + LoRA | 12 GB | ~40 Hz |
| Gemma 4 E4B + LoRA | 20 GB | ~25 Hz |
| Gemma 4 E2B + 4-bit quant | 8 GB | ~30 Hz |
| Gemma 4 31B + LoRA | 55 GB | ~8 Hz |

### Dependencies

Core model: `torch`, `transformers`, `peft`, `h5py`, `numpy`

Simulation: `gymnasium`, `mujoco`, `metaworld`

Observability: `rerun-sdk`, `mlflow`

Video/image: `imageio`, `imageio[ffmpeg]`, `opencv-python`

---

## Quick start

### 1. Run inference

```python
import numpy as np
from gemma4_vla import Gemma4VLA, PolicyRunner, metaworld_push_config

# Load a trained model
runner = PolicyRunner.from_pretrained("checkpoints/best", device="cuda")

# Or start fresh (random weights — for testing only)
cfg    = metaworld_push_config()
model  = Gemma4VLA(cfg)
runner = PolicyRunner(model, device="cuda")

# Prepare an observation
obs = {
    "images":      [camera.capture()],          # list of uint8 [H, W, 3] arrays
    "state":       robot.get_joint_positions(),  # float32 [state_dim]
    "instruction": "Pick up the red cube.",
}

# Predict the next 50 action steps (1 second at 50 Hz)
actions = runner.predict(obs)  # [50, 6]

# Execute step by step with temporal action chunking
for action in runner.stream(obs, replan_every=25):
    robot.apply(action)
```

### 2. Train from scratch

```python
from gemma4_vla import Gemma4VLA, metaworld_push_config
from gemma4_vla.train import train

cfg = metaworld_push_config()
cfg.training.dataset_root = "./data/metaworld_demos"
cfg.training.output_dir   = "./checkpoints/metaworld_push"
cfg.training.max_steps    = 50_000
cfg.training.batch_size   = 16

train(cfg)
```

### 3. Train on MetaWorld data

```bash
# Collect expert demonstrations
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 --episodes 50

# Train on collected data
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --data-dir data/metaworld_demos

# Evaluate the trained checkpoint
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push/best \
    --env-name push-v3 --episodes 10 --save-video
```

---

## Robot simulation with MetaWorld

Gemma4VLA includes a full simulation pipeline for training and evaluating
policies in [Meta-World](https://meta-world.github.io/) robotic manipulation
tasks using MuJoCo physics and Gymnasium environments.

### How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│  MetaWorld MT1 Environment  (MuJoCo physics + Gymnasium API)       │
│                                                                     │
│  env.reset() ──► (camera_image, robot_state, info)                 │
│  env.step(action) ──► (camera_image, robot_state, reward, done)    │
│                                                                     │
│  Supported tasks: push-v3, reach-v3, pick-place-v3                 │
│  Camera views:  corner, corner2, topview, behindGripper, ...       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Gemma4VLA PolicyRunner                                            │
│                                                                     │
│  obs = {"images": [img], "state": state, "instruction": "..."}    │
│  actions = runner.predict(obs)   # [50, action_dim]                │
│  action = actions[0]             # take first step                 │
│                                                                     │
│  Gemma 4 backbone encodes image + text ──►                         │
│    ActionExpert denoises actions via flow matching ──►              │
│      50-step action horizon                                        │
└─────────────────────────────────────────────────────────────────────┘
```

The `MetaWorldMT1Wrapper` bridges MetaWorld into a simple
`(image, state) → action → (image, state, reward, done)` loop. The Gemma 4
backbone processes the raw camera image (480x480 uint8, resized internally to
224x224) and the text instruction directly — no custom tokenizer or image
preprocessing is needed on the user side.

### Full pipeline

The complete workflow from data collection to evaluation:

#### Step 1 — Collect expert demonstrations

Meta-World provides scripted expert policies for each task. The collection
script runs these experts to gather training data in HDF5 format:

```bash
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 \
    --camera-name corner \
    --episodes 100 \
    --instruction "push the object to the goal" \
    --output-dir data/push_demos
```

This produces one HDF5 file per episode:

```
data/push_demos/
  episode_000000.hdf5    # observation/images/corner, observation/state, action
  episode_000001.hdf5
  ...
```

#### Step 2 — Train the model

```bash
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --data-dir data/push_demos \
    --max-steps 50000 \
    --metrics-path metrics/train.jsonl \
    --mlflow --mlflow-experiment gemma4-vla-push
```

The MetaWorld config (`robots/metaworld/configs/metaworld_push.yaml`) sets `state_dim=39` and
`action_dim=4` to match MetaWorld's observation and action spaces.

Training follows the recommended two-stage approach:

1. **Stage 1** (freeze backbone, train action expert, 2K steps, LR=5e-4)
2. **Stage 2** (unfreeze LoRA adapters, joint fine-tuning, 50K steps, LR=2e-4)

#### Step 3 — Evaluate in simulation

```bash
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push/best \
    --env-name push-v3 \
    --camera-name corner \
    --instruction "push the object to the goal" \
    --episodes 10 \
    --save-video --video-dir videos \
    --metrics-path metrics/eval.json
```

At each simulation step, the model receives the current camera image and robot
state, predicts a 50-step action horizon, and executes the first action. The
evaluation script tracks reward, success rate, and per-step action data.

#### Step 4 — Inspect collected data

```bash
uv run python -m robots.metaworld.scripts.inspect_dataset \
    --dataset-dir data/push_demos \
    --episode-index 0 \
    --video-output videos/preview.mp4
```

### MetaWorld environment wrapper

```python
from robots.metaworld import MetaWorldMT1Wrapper

env = MetaWorldMT1Wrapper(
    env_name="push-v3",      # push-v3, reach-v3, pick-place-v3
    camera_name="corner",    # corner, topview, behindGripper, ...
    seed=42,
)

image, state, info = env.reset()  # [H,W,3] uint8, [39] float32
image, state, reward, done, info = env.step(action)  # [4] float32

# Properties
env.state_dim     # 39
env.action_dim    # 4
env.obs_shape     # (480, 480, 3)
env.action_low    # [-1, -1, -1, -1]
env.action_high   # [+1, +1, +1, +1]
```

### Using PolicyRunner with MetaWorld

```python
from gemma4_vla import PolicyRunner
from robots.metaworld import MetaWorldMT1Wrapper

runner = PolicyRunner.from_pretrained("checkpoints/best", device="cuda")
env = MetaWorldMT1Wrapper("push-v3", camera_name="corner")

image, state, info = env.reset()
done = False

while not done:
    obs = {
        "images": [image],
        "state": state,
        "instruction": "push the object to the goal",
    }
    actions = runner.predict(obs)  # [50, 4]
    action = actions[0]            # take first action from horizon

    image, state, reward, done, info = env.step(action)
```

---

## Observability (Rerun + MLflow)

Every pipeline script supports optional real-time visualization with
[Rerun](https://rerun.io/) and experiment tracking with
[MLflow](https://mlflow.org/).

### Rerun — real-time visualization

Rerun provides a live 3D/timeline viewer for camera images, robot state,
actions, rewards, and flow matching traces.

```bash
# Launch local viewer
uv run python -m robots.metaworld.scripts.test --checkpoint checkpoints/best \
    --rerun-mode spawn --episodes 3

# Save to .rrd file for later replay
uv run python -m robots.metaworld.scripts.test --checkpoint checkpoints/best \
    --rerun-mode save --rerun-path rerun/eval.rrd

# Stream to existing viewer
uv run python -m robots.metaworld.scripts.test --checkpoint checkpoints/best \
    --rerun-mode connect --rerun-connect-url 127.0.0.1:9876
```

Data logged per step: camera image, state vector, action vectors (model-space,
env-space, clipped), reward, success flag, instruction text.

### MLflow — experiment tracking

MLflow tracks hyperparameters, training/eval metrics, and artifacts (videos,
checkpoints, metrics files).

```bash
# Start MLflow server (Postgres-backed, persistent)
docker compose -f docker-compose.mlflow.yml up -d
# UI at http://localhost:5001

# Train with MLflow logging
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --mlflow --mlflow-experiment gemma4-vla-push \
    --mlflow-log-artifacts

# Evaluate with MLflow logging
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/best \
    --mlflow --mlflow-experiment gemma4-vla-eval \
    --mlflow-log-artifacts
```

### JSONL / JSON metrics

All scripts also support structured metrics files that can be parsed with
`jq`, `pandas`, or any JSON tooling:

```bash
# Training: JSONL with train_start, per-step loss/lr, train_end
uv run python -m gemma4_vla.train --metrics-path metrics/train.jsonl ...

# Evaluation: JSON with per-step actions, per-episode reward/success, summary
uv run python -m robots.metaworld.scripts.test --metrics-path metrics/eval.json ...
```

### Observability module

```python
from gemma4_vla import RerunLogger, MlflowRun

# Rerun
rerun = RerunLogger(mode="spawn", app_id="my_experiment").start()
rerun.set_step(42, episode=1)
rerun.log_image("camera/rgb", image)
rerun.log_scalar("metrics/reward", 0.95)

# MLflow
mlflow = MlflowRun(enabled=True, experiment_name="my-exp").start(params)
mlflow.log_metric("loss", 0.01, step=100)
mlflow.log_artifact("checkpoints/best")
mlflow.end()
```

---

## Monitoring workflow on a remote server

This section covers the typical "headless Jetson Thor + laptop viewer" setup
where the **Thor runs collect / train / test** and a **second machine** (laptop
or workstation) is used to inspect Rerun traces and the MLflow UI through SSH.

The pattern is the same in every case: **Thor writes artefacts to disk**
(`.rrd` files for Rerun, MLflow runs into Postgres + `mlflow-artifacts` volume),
and the **viewer machine reaches them over SSH** — either by port-forwarding
the running MLflow server, or by replaying / streaming the saved `.rrd` files.

> Throughout this section we assume:
> - **thor** is the Jetson Thor (training machine), reachable as
>   `user@thor.local` over SSH
> - **laptop** is your viewer machine (Rerun + browser)
> - the repo is checked out at `~/beitlab-gemma4-vla` on both machines

---

### 1. On the Jetson Thor — collect / train / test

All three commands write `.rrd` files into `rerun/` and (for train/test) log
metrics + artefacts to the locally-running MLflow container.

**Start the MLflow server once** (Postgres-backed, persistent):

```bash
# On thor
cd ~/beitlab-gemma4-vla
docker compose -f docker-compose.mlflow.yml up -d
# MLflow UI is now served on http://thor:5001
```

**Collect demonstrations** — save a Rerun trace of the scripted expert:

```bash
# On thor
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 --camera-name corner \
    --episodes 100 \
    --instruction "push the object to the goal" \
    --output-dir data/push_demos \
    --rerun-mode save \
    --rerun-path rerun/collect_push.rrd
```

**Train** with MLflow logging to the local server:

```bash
# On thor
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --data-dir data/push_demos \
    --metrics-path metrics/train.jsonl \
    --mlflow \
    --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-push \
    --mlflow-log-artifacts \
    --mlflow-system-metrics
```

**Evaluate** and save a validation Rerun trace + MLflow run:

```bash
# On thor
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push/best \
    --env-name push-v3 --camera-name corner \
    --episodes 10 \
    --rerun-mode save \
    --rerun-path rerun/eval_push.rrd \
    --mlflow \
    --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-eval \
    --mlflow-log-artifacts
```

After these runs, on the Thor you have:

```
~/beitlab-gemma4-vla/rerun/collect_push.rrd   # scripted-expert episodes
~/beitlab-gemma4-vla/rerun/eval_push.rrd      # validation rollouts
~/beitlab-gemma4-vla/checkpoints/...          # trained checkpoint
# plus the MLflow server on thor:5001 (train + eval runs)
```

---

### 2. On the viewer machine — watch over SSH

**a) MLflow UI in your browser via SSH port-forward.**
Forward the Thor's MLflow port to localhost on the laptop:

```bash
# On laptop
ssh -N -L 5001:127.0.0.1:5001 user@thor.local
# Then open http://localhost:5001 — you'll see both training and eval runs
```

Leave the tunnel running while you browse runs, compare metrics, and download
artefacts (checkpoints, metrics JSON, videos) directly from the UI.

**b) Replay a saved `.rrd` from the Thor without copying it.**
The cleanest path is to mount the Thor's `rerun/` directory over SSHFS so the
local Rerun viewer can open remote files transparently:

```bash
# On laptop (one-time)
sshfs user@thor.local:/home/user/beitlab-gemma4-vla/rerun ~/thor-rerun

# Replay the validation rollouts in the Rerun viewer
uv run rerun ~/thor-rerun/eval_push.rrd

# Or replay the collection trace
uv run rerun ~/thor-rerun/collect_push.rrd
```

If SSHFS isn't available, just `scp` the file once and open it locally:

```bash
# On laptop
scp user@thor.local:~/beitlab-gemma4-vla/rerun/eval_push.rrd ./eval_push.rrd
uv run rerun ./eval_push.rrd
```

**c) Stream a *live* run from Thor to the laptop viewer.**
Use `--rerun-mode connect` on the Thor instead of `save`, and forward Rerun's
gRPC port (9876) from the laptop to the Thor:

```bash
# On laptop — open viewer first and forward port
uv run rerun &                                       # opens the Rerun viewer
ssh -N -R 9876:127.0.0.1:9876 user@thor.local       # reverse-tunnel viewer to thor
```

```bash
# On thor — point the run at the tunnel
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push/best \
    --env-name push-v3 --episodes 5 \
    --rerun-mode connect \
    --rerun-connect-url 127.0.0.1:9876
```

Each step is streamed in real time to the viewer on the laptop. Use `save` for
durable runs you want to revisit; use `connect` when you want to babysit a run
as it executes.

---

### 3. Quick reference

| Goal | Where you run it | Command |
|------|------------------|---------|
| Save a Rerun trace during data collection | thor | `collect_data ... --rerun-mode save --rerun-path rerun/collect.rrd` |
| Save a Rerun trace during evaluation | thor | `test ... --rerun-mode save --rerun-path rerun/eval.rrd` |
| Log training to MLflow | thor | `train ... --mlflow --mlflow-tracking-uri http://127.0.0.1:5001` |
| View MLflow runs remotely | laptop | `ssh -N -L 5001:127.0.0.1:5001 user@thor` → `http://localhost:5001` |
| Replay a stored `.rrd` from Thor | laptop | `sshfs` mount, then `uv run rerun <path/to/file.rrd>` |
| Stream a live run from Thor | laptop + thor | reverse-forward `9876`, then `--rerun-mode connect --rerun-connect-url 127.0.0.1:9876` |

---

## Configuration

Gemma4VLA uses dataclass-based configuration.  Pre-built configs:

```python
from gemma4_vla import metaworld_push_config

cfg = metaworld_push_config()   # MetaWorld push-v3 (state_dim=39, action_dim=4)
```

Or build a custom config for your robot:

```python
from gemma4_vla.config import (
    Gemma4VLAConfig, VisionConfig, BackboneConfig,
    ActionExpertConfig, FlowMatchingConfig, RobotConfig, TrainingConfig
)

cfg = Gemma4VLAConfig(
    vision=VisionConfig(num_cameras=2, image_size=224),
    backbone=BackboneConfig(
        model_name="google/gemma-4-E4B-it",
        use_lora=True,
        lora_rank=32,
    ),
    action_expert=ActionExpertConfig(hidden_size=1024, num_layers=8),
    flow_matching=FlowMatchingConfig(action_horizon=50, num_inference_steps=10),
    robot=RobotConfig(name="my_arm", state_dim=7, action_dim=7),
)
```

You can also drive training from YAML:

```bash
uv run python -m gemma4_vla.train --config robots/metaworld/configs/metaworld_push.yaml
```

The CLI now applies the full nested config tree from YAML, not just the
`training` section.

### Important config fields

| Field | Default | Description |
|-------|---------|-------------|
| `backbone.model_name` | `google/gemma-4-E2B-it` | Gemma 4 model ID |
| `backbone.use_lora` | `True` | LoRA for efficient fine-tuning |
| `backbone.lora_rank` | `16` | LoRA rank (higher = more capacity) |
| `robot.state_dim` | `6` | Proprioceptive state size |
| `robot.action_dim` | `6` | Motor command dimension |
| `flow_matching.action_horizon` | `50` | Steps predicted per inference call |
| `flow_matching.num_inference_steps` | `10` | Denoising steps (more = higher quality) |
| `vision.num_cameras` | `2` | Number of RGB cameras |

---

## Checkpoints

`save_pretrained()` writes checkpoints as:

```text
checkpoint_dir/
  config.json         # canonical config artifact
  config.pt           # compatibility copy containing a plain dict
  weights.pt          # model state_dict
  normalization.pt    # (optional) per-dim state / action stats — see below
```

`from_pretrained()` reads `config.json` when present and still supports
older checkpoints that only contain the legacy `config.pt`. If
`normalization.pt` exists in the same directory it's loaded too, and
`PolicyRunner.predict()` will denormalise predicted actions automatically.

### Dataset-fit normalisation

Pass `--normalize-stats` to the metaworld train CLI (or set
`cfg.training.normalize_stats = True`) to:

1. Stream a single pass over the training loader (capped at
   `cfg.training.normalize_stats_batches`) and compute per-dim mean / std
   for both `state` and `actions`.
2. Save them next to the checkpoint as `normalization.pt`.
3. Normalise inputs inside the dataset's `__getitem__` so the network
   trains on zero-mean / unit-std vectors.
4. Denormalise predicted actions in `PolicyRunner.predict()` symmetrically.

```python
from gemma4_vla import DatasetStats

stats = DatasetStats.load("checkpoints/best")
if stats is not None and stats.enabled:
    print("Action mean:", stats.action_mean)
```

---

## Data format

Gemma4VLA uses a per-episode **HDF5** format.  Each episode is stored as
a single `.hdf5` file:

```
data/my_robot/
  episode_000000.hdf5   ← one trajectory per file
  episode_000001.hdf5
  ...
```

Inside each file:

```
observation/
  images/
    top/        [T, H, W, 3]   uint8  ← top camera
    wrist/      [T, H, W, 3]   uint8  ← wrist camera
  state         [T, state_dim] float32
action          [T, action_dim] float32
language_instruction  ← string attribute or dataset
```

To convert your own data:

```python
import h5py, numpy as np

with h5py.File("data/my_robot/episode_000000.hdf5", "w") as f:
    f.attrs["language_instruction"] = "Place the cup on the coaster."

    obs = f.create_group("observation")
    imgs = obs.create_group("images")
    imgs.create_dataset("top",   data=my_images)   # [T, H, W, 3] uint8
    obs.create_dataset("state",  data=my_states)   # [T, state_dim]
    f.create_dataset("action",   data=my_actions)  # [T, action_dim]
```

---

## Training recipe

### Recommended two-stage approach (matches pi0)

```
Stage 1 (2 000 – 5 000 steps)
  - Freeze Gemma 4 backbone
  - Train action expert from scratch
  - High LR (5e-4), fast convergence
  - Goal: learn action dynamics and noise schedule

Stage 2 (10 000 – 100 000 steps)
  - Unfreeze LoRA adapters on backbone
  - Train jointly with lower LR (2e-4)
  - backbone LR = 10× smaller than action expert
  - Goal: align language-vision-action representations
```

```python
from gemma4_vla import Gemma4VLA, metaworld_push_config
from gemma4_vla.train import train

cfg = metaworld_push_config()
model = Gemma4VLA(cfg)

# Stage 1 — action expert warm-up
model.freeze_backbone()
cfg.training.max_steps = 2_000
cfg.training.learning_rate = 5e-4
train(cfg, model=model)

# Stage 2 — joint fine-tuning
model.unfreeze_backbone()
cfg.training.max_steps = 52_000
cfg.training.learning_rate = 2e-4
cfg.training.backbone_lr_multiplier = 0.1
train(cfg, model=model)
```

### Training on multiple GPUs (DDP)

```bash
# 4 GPUs
uv run torchrun --nproc_per_node=4 -m gemma4_vla.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --data_root ./data/metaworld_demos \
    --output_dir checkpoints/metaworld_push_ddp
```

The training loop auto-detects `RANK` / `WORLD_SIZE` / `LOCAL_RANK` set by
`torchrun`, initialises NCCL, rebuilds the loaders with `DistributedSampler`,
wraps the model in `DistributedDataParallel`, and restricts logging /
MLflow / checkpoint writes to rank 0. The backbone's `device_map="auto"`
auto-shard is disabled in distributed mode so each rank holds its own model
copy. See [MULTI_MACHINE_TRAINING.md](MULTI_MACHINE_TRAINING.md) for details.

---

## Flow matching details

Gemma4VLA uses **conditional flow matching** with optimal-transport (OT) paths,
identical to the formulation in pi0.

**Training** (one step):
```
1. Sample clean action x_1  (from dataset)
2. Sample noise      x_0  ~ N(0, I)
3. Sample noise level t   ~ Uniform(0, 1)
4. Interpolate:  x_t = (1 − (1 − σ_min)·t)·x_0 + t·x_1
5. Target vel:   u   = x_1 − (1 − σ_min)·x_0
6. Predict:      v_θ = ActionExpert(x_t, obs, t)
7. Loss:         L   = ||v_θ − u||²
```

**Inference** (Euler integration, t: 0 → 1):
```
x_0 ~ N(0, I)          # start from noise
for t in linspace(0, 1, num_steps):
    v = model(x_t, obs, t)
    x_{t+dt} = x_t + v · dt
return x_1             # clean predicted action
```

The number of inference steps trades quality vs. speed:

| Steps | Quality | Latency (2B, A100) |
|-------|---------|-------------------|
| 5 | Good | ~12 ms (83 Hz) |
| 10 | Better | ~22 ms (45 Hz) |
| 25 | Best | ~55 ms (18 Hz) |
| 50 | Marginal gain | ~110 ms (9 Hz) |

---

## Cross-embodiment training

Gemma4VLA follows pi0's zero-padding strategy for cross-embodiment.
State/action vectors are padded to `max_state_dim` dimensions, allowing
a single model to control different robots without architecture changes.

Currently only MetaWorld is implemented (`state_dim=39`, `action_dim=4`).
Future robot adapters will be added under `robots/<name>/`, and
cross-embodiment training will work by mixing episode HDF5 files from
different adapters.

---

## Memory-efficient training

**LoRA** (default, recommended):
```python
cfg.backbone.use_lora = True
cfg.backbone.lora_rank = 16       # ~10 M extra parameters
```

**4-bit quantisation** (requires `bitsandbytes`):
```bash
uv add bitsandbytes
```
```python
# Edit model.py: pass quantization_config=bnb_cfg to from_pretrained
```

**Gradient checkpointing**:
```python
model.backbone.model.gradient_checkpointing_enable()
```

---

## API reference

### `Gemma4VLA`

```python
class Gemma4VLA(nn.Module):
    def compute_loss(self, batch) -> Dict[str, Tensor]
    def predict_action(self, obs, num_steps=None, use_rk4=False) -> Tensor
    def freeze_backbone(self)
    def unfreeze_backbone(self)
    def num_parameters(self, trainable_only=False) -> int
    def save_pretrained(self, path: str)

    @classmethod
    def from_pretrained(cls, path: str, cfg=None) -> "Gemma4VLA"
```

`batch["pixel_values"]` and `obs["pixel_values"]` use the canonical internal
shape `[B, num_cameras, 3, H, W]`. The backbone flattens the batch and camera
axes internally before calling Gemma 4.

### `PolicyRunner`

```python
class PolicyRunner:
    def predict(self, obs: Dict, num_inference_steps=None) -> np.ndarray
    def stream(self, obs: Dict, replan_every=None) -> Iterator[np.ndarray]

    @classmethod
    def from_pretrained(cls, path: str, device="cuda") -> "PolicyRunner"
```

### Flow matching utilities

```python
from gemma4_vla.flow_matching import (
    ot_flow_interpolate,    # compute noisy actions + target velocity
    flow_matching_loss,     # MSE loss with optional mask
    euler_integration,      # fast inference integration
    rk4_integration,        # high-quality inference integration
    SinusoidalEmbedding,    # noise level encoder
)
```

### `MetaWorldMT1Wrapper`

```python
class MetaWorldMT1Wrapper:
    def __init__(self, env_name, seed=42, render_mode="rgb_array", camera_name="topview")
    def reset(self, seed=None) -> (image, state, info)
    def step(self, action) -> (image, state, reward, done, info)
    def close(self)
```

### `RerunLogger`

```python
class RerunLogger:
    def __init__(self, mode="off", app_id="gemma4_vla", save_path=None, connect_url=None)
    def start(self) -> self
    def set_step(self, step, episode=None)
    def log_image(self, path, image)
    def log_scalar(self, path, value)
    def log_vector(self, path, vector, dim_name="dim")
    def log_tensor(self, path, tensor, dim_names=None)
    def log_text(self, path, text)
```

### `MlflowRun`

```python
class MlflowRun:
    def __init__(self, enabled=False, tracking_uri=None, experiment_name=..., run_name=None)
    def start(self, params=None) -> self
    def log_params(self, params)
    def log_metric(self, key, value, step=None)
    def log_metrics(self, metrics, step=None)
    def log_artifact(self, path, artifact_path=None)
    def end(self)
```

---

## Roadmap

- [x] Sim evaluation hooks (MuJoCo / MetaWorld / Gymnasium)
- [x] Real-time visualisation (Rerun) and experiment tracking (MLflow)
- [x] Dataset-fit per-dim state / action normalisation (`DatasetStats`,
      `normalization.pt`, applied at train + inference)
- [x] DDP training via `torchrun` (DistributedSampler, rank-0-gated
      logging / saves, NCCL teardown)
- [x] Vision-tower assertion at backbone load (no silent text-only
      fallback)
- [ ] FAST tokenisation variant (discrete action tokens via DCT + BPE)
- [ ] π0.5-style open-world generalisation
- [ ] IsaacGym / Genesis sim environments
- [ ] RLDS / Open X-Embodiment dataset adapter
- [ ] Automatic mixed-precision inference on edge hardware
- [ ] `LeRobotDataset` adapter + on-robot policy server (see
      [ROADMAP.md](ROADMAP.md) §2)
- [ ] TensorRT / FP8 export for Jetson Thor (see [ROADMAP.md](ROADMAP.md) §3)

---

## Citation

If you use this codebase, please cite the original pi0 paper, Gemma 4, and this work:

```bibtex
@software{beitlab_gemma4_vla_2025,
  title   = {Gemma4VLA: Open-Source Vision-Language-Action Models on Edge Hardware},
  author  = {BeitLab Robotics},
  url     = {https://github.com/beitlab-inc/beitlab-gemma4-vla},
  year    = {2025},
}

@article{black2024pi0,
  title   = {π0: A Vision-Language-Action Flow Model for General Robot Control},
  author  = {Black, Kevin and Brown, Noah and others},
  journal = {arXiv preprint arXiv:2410.24164},
  year    = {2024},
}

@article{gemma4_2025,
  title   = {Gemma 4 Technical Report},
  author  = {Google DeepMind},
  year    = {2025},
}
```

---

## License

Apache 2.0.  The Gemma 4 model weights are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
