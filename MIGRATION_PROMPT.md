# Prompt: Add mini-vla features to gemma4_vla

You are a senior robotics ML engineer. Your task is to add ALL features from the
mini-vla project into the gemma4_vla project so that gemma4_vla becomes a complete
drop-in replacement with the same simulation, observability, and tooling ecosystem.

**CONSTRAINT: DO NOT modify any file in /Users/andresjc/beitlab/mini-vla. All changes
go into /Users/andresjc/beitlab/gemma4_vla only.**

---

## CODEBASES

- **mini-vla** (reference only — DO NOT EDIT): `/Users/andresjc/beitlab/mini-vla`
- **gemma4_vla** (target — ALL changes here): `/Users/andresjc/beitlab/gemma4_vla`

---

## WHAT GEMMA4_VLA ALREADY HAS (DO NOT BREAK)

These components are complete and working. Do not rewrite them — integrate with them:

| Component | File | What it does |
|-----------|------|-------------|
| Model | `src/gemma4_vla/model.py` | Gemma4VLA class with compute_loss(), predict_action(), save/load_pretrained() |
| Backbone | `src/gemma4_vla/model.py:130-247` | Gemma4Backbone wrapping HuggingFace Gemma 4 + LoRA |
| ActionExpert | `src/gemma4_vla/action_expert.py` | 8-layer transformer (~300M params) with cross-attention |
| Flow matching | `src/gemma4_vla/flow_matching.py` | OT interpolation, Euler + RK4 integration, flow_matching_loss |
| Inference | `src/gemma4_vla/inference.py` | PolicyRunner with predict(), stream(), from_pretrained() |
| Config | `src/gemma4_vla/config.py` | Gemma4VLAConfig dataclass hierarchy, YAML loading |
| Dataset | `src/gemma4_vla/dataset.py` | LeRobotDataset (HDF5), RandomDemoDataset, transforms |
| Training | `src/gemma4_vla/train.py` | train() loop with mixed precision, optimizer, scheduler |
| Tests | `tests/test_*.py` | Unit tests for config, inference, dataset, flow matching, action expert |
| Configs | `configs/*.yaml` | base_config |
| Examples | `examples/*.py` | quick_start, finetune_lerobot, custom_robot, evaluate |

---

## WHAT TO ADD (from mini-vla)

### 1. MetaWorld / MuJoCo / Gymnasium Environment Wrapper

**Source:** `mini-vla/envs/metaworld_env.py` (74 lines)
**Target:** Create `src/gemma4_vla/envs/metaworld_env.py`

Port the `MetaWorldMT1Wrapper` class exactly, preserving:
- Constructor: `MetaWorldMT1Wrapper(env_name, seed, render_mode, camera_name)`
- Supported tasks: `push-v3, reach-v3, pick-place-v3`
- Camera views: `corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV`
- `reset()` -> `(image, state, info)` where image is `[H,W,3] uint8`, state is `[state_dim] float32`
- `step(action)` -> `(image, state, reward, done, info)`
- Properties: `state_dim`, `action_dim`, `obs_shape`, `action_low`, `action_high`
- State extraction: flattening dict-based Meta-World obs (robot_state + object_state)

Also port `mini-vla/envs/metaworld_mt1.py` (35 lines) — the standalone visualization script.

Create `src/gemma4_vla/envs/__init__.py` exporting `MetaWorldMT1Wrapper`.

### 2. Observability Module (Rerun + MLflow)

**Source:** `mini-vla/utils/observability.py` (188 lines)
**Target:** Create `src/gemma4_vla/observability.py`

Port both classes exactly:

#### RerunLogger
- `__init__(mode, app_id, save_path, connect_url)` — modes: "off", "spawn", "save", "connect"
- `start()` — initializes rerun SDK per mode (rr.spawn / rr.save / rr.connect_grpc)
- `set_step(step)` — sets time context
- `log_image(path, image)` — logs RGB image
- `log_vector(path, vector, labels)` — logs labeled vector
- `log_tensor(path, tensor)` — logs tensor
- `log_scalar(path, value)` — logs scalar metric
- `log_text(path, text)` — logs text string

#### MlflowRun
- `__init__(enabled, tracking_uri, experiment_name, run_name)` — default URI from env var `MLFLOW_TRACKING_URI` or `http://127.0.0.1:5001`
- `start()` — initializes mlflow experiment, starts run, checks artifact URI
- `log_params(params)` — logs sanitized params dict
- `log_metric(key, value, step)` — logs single metric
- `log_metrics(metrics, step)` — logs dict of metrics, filters None
- `log_artifact(path)` — uploads file to artifact store
- `log_dict(data, filename)` — logs dict as artifact file
- `end()` — closes run

#### Helpers
- `ensure_parent(path)` — creates parent directories
- `sanitize_params(params)` — converts complex types to JSON strings for MLflow

### 3. Data Collection Script

**Source:** `mini-vla/scripts/collect_data.py` (197 lines)
**Target:** Create `scripts/collect_data.py`

Port the full expert demonstration collection pipeline:
- Uses `MetaWorldMT1Wrapper` for environment
- Uses `metaworld.policies.ENV_POLICY_MAP[env_name]()` for expert policy
- Collects: images `[H,W,3] uint8`, states `[state_dim] float32`, actions `[action_dim] float32`
- **IMPORTANT:** Save data in HDF5 format (gemma4_vla's native format), NOT .npz:

```
data/
  episode_000000.hdf5
    observation/images/<camera_name>/  [T, H, W, 3] uint8
    observation/state                  [T, state_dim] float32
    action                             [T, action_dim] float32
    attribute: language_instruction     str
  episode_000001.hdf5
  ...
```

- `extract_state(obs)` — flattens dict obs via `.ravel()`
- Tracks per-episode success via `info.get("success", False)`

**ALL CLI args must match mini-vla:**
```
--env-name (str, default: "push-v3")
--camera-name (str, default: "topview")
--seed (int, default: 42)
--episodes (int, default: 50)
--max-steps (int, default: 150)
--output-dir (str, default: "data/metaworld_demos")
--sleep (float, default: 0.0)
--instruction (str, default: "push the object to the goal")
--rerun-mode (choices: off/spawn/save/connect, default: "off")
--rerun-path (str, default: "rerun/collect_data.rrd")
--rerun-connect-url (str, default: None)
--rerun-log-every (int, default: 1)
--mlflow (store_true)
--mlflow-tracking-uri (str, default: None)
--mlflow-experiment (str, default: "gemma4-vla-data")
--mlflow-run-name (str, default: None)
--mlflow-log-dataset-artifact (store_true)
```

**Rerun logging per step** (when enabled):
- Camera image
- State vector (labeled by dimension)
- Action vector
- Reward scalar
- Success scalar

**MLflow logging:**
- Params: env_name, camera_name, episodes, max_steps, instruction
- Per-episode metrics: success, steps
- Summary: total_transitions, success_rate, avg_steps_per_episode
- Optional: upload dataset directory as artifact

### 4. Training Script with Full Observability

**Source:** `mini-vla/scripts/train.py` (251 lines)
**Target:** Modify existing `src/gemma4_vla/train.py` to add observability

Add these features to the existing training loop (DO NOT rewrite the training loop,
ADD to it):

#### JSONL Metrics Logging
- `append_metric(path, record)` helper — appends JSON records to JSONL file
- `train_start` record: dataset_path, state_dim, action_dim, batch_size, max_steps, lr, config dump
- Per-step record: type="step", step#, loss, lr, duration_s
- `train_end` record: checkpoint_path, total_time

#### MLflow Integration
- CLI args: `--mlflow`, `--mlflow-tracking-uri`, `--mlflow-experiment` (default: "gemma4-vla-train"), `--mlflow-run-name`, `--mlflow-log-artifacts`
- Log params at start: all config fields, dataset path, device, mixed precision
- Log metrics per log step: loss, lr
- Log summary: final_loss, total_steps, training_time
- Upload checkpoint + metrics JSONL as artifacts when `--mlflow-log-artifacts`

#### Additional CLI args to add:
```
--metrics-path (str, default: None) — JSONL output path
--mlflow (store_true)
--mlflow-tracking-uri (str, default: None)
--mlflow-experiment (str, default: "gemma4-vla-train")
--mlflow-run-name (str, default: None)
--mlflow-log-artifacts (store_true)
```

#### Action/State Normalization Stats
- Compute per-DOF mean/std from dataset at training start
- Save normalization stats in checkpoint alongside model weights:
  ```python
  "state_mean", "state_std", "action_mean", "action_std", "normalize": True
  ```
- Support `--no-normalize` flag (skip normalization, save identity stats)
- Apply normalization to states/actions before model forward pass
- These stats are needed by the test script for denormalization

### 5. Evaluation / Test Script

**Source:** `mini-vla/scripts/test.py` (448 lines)
**Target:** Create `scripts/test.py`

Build a full evaluation script that uses `PolicyRunner` with `MetaWorldMT1Wrapper`:

#### Core Loop
```python
runner = PolicyRunner.from_pretrained(args.checkpoint, device=args.device)
env = MetaWorldMT1Wrapper(env_name, seed, camera_name=camera_name)

for ep in range(episodes):
    image, state, info = env.reset()
    for step in range(max_steps):
        obs = {"images": [image], "state": state, "instruction": instruction}
        actions = runner.predict(obs)     # [50, action_dim]
        action = actions[0]               # take first action
        # Denormalize action if normalization stats exist in checkpoint
        action_env = action * norm["action_std"] + norm["action_mean"]
        action_env = np.clip(action_env, env.action_low, env.action_high)
        image, state, reward, done, info = env.step(action_env)
```

#### Video Recording
- CLI args: `--save-video`, `--video-dir` (default: "videos")
- Accumulate frames per episode
- Save as MP4 via `imageio.get_writer()`, FPS=20
- Filename: `{env_name}_ep{ep+1:03d}.mp4`

#### JSON Metrics Output
- CLI arg: `--metrics-path` (default: None)
- Top-level: checkpoint, env_name, camera_name, instruction, episodes, normalize
- Per-step record: step, reward, done, success, action_model (normalized), action_env (denormalized), action_env_clipped
- Per-episode summary: total_reward, steps, success
- Overall summary: avg_reward, success_rate, per-DOF action statistics
- `json.dump(metrics, f, indent=2)` at end

#### Flow Matching Trace (equivalent to diffusion trace)
- CLI arg: `--save-flow-trace` (replaces `--save-diffusion-trace`)
- During `model.predict_action()`, capture all ODE integration steps
- Store trajectory `[num_steps, action_horizon, action_dim]` per env step in metrics
- Both model-space and denormalized versions

#### Rerun Logging
- CLI args: `--rerun-mode`, `--rerun-path` (default: "rerun/test.rrd"), `--rerun-connect-url`, `--rerun-log-every`
- Per step log: camera image, state vector, action vectors (model + env + clipped), reward, success, instruction text
- Optional: flow trace tensor

#### MLflow Logging
- CLI args: `--mlflow`, `--mlflow-tracking-uri`, `--mlflow-experiment` (default: "gemma4-vla-eval"), `--mlflow-run-name`, `--mlflow-log-artifacts`
- Params: checkpoint, env_name, camera_name, instruction, episodes, device, normalize
- Per-episode metrics: reward, steps, success
- Summary: avg_reward, success_rate
- Artifacts: metrics JSON, video files

**ALL CLI args:**
```
--checkpoint (str, default: "checkpoints/best")
--env-name (str, default: "push-v3")
--seed (int, default: 42)
--episodes (int, default: 5)
--max-steps (int, default: 150)
--camera-name (str, default: "topview")
--instruction (str, default: "push the object to the goal")
--device (str, default: "cpu")
--save-video (store_true)
--video-dir (str, default: "videos")
--metrics-path (str, default: None)
--save-flow-trace (store_true)
--rerun-mode (choices: off/spawn/save/connect, default: "off")
--rerun-path (str, default: "rerun/test.rrd")
--rerun-connect-url (str, default: None)
--rerun-log-every (int, default: 1)
--mlflow (store_true)
--mlflow-tracking-uri (str, default: None)
--mlflow-experiment (str, default: "gemma4-vla-eval")
--mlflow-run-name (str, default: None)
--mlflow-log-artifacts (store_true)
```

### 6. Dataset Inspection Script

**Source:** `mini-vla/scripts/inspect_dataset.py` (125 lines)
**Target:** Create `scripts/inspect_dataset.py`

Port the full dataset inspection tool, adapted for HDF5:
- `print_dataset_summary()` — lists all episodes, keys, shapes, dtypes, value ranges, language instructions
- `save_frame(episode_path, frame_index, output_path)` — exports single frame as PNG
- `save_video(episode_path, output_path, max_frames, fps)` — exports frame sequence as MP4

**CLI args:**
```
--dataset-dir (str, required) — path to HDF5 episodes directory
--episode-index (int, default: 0) — which episode to inspect
--frame-output (str, default: None) — PNG output path
--frame-index (int, default: 0)
--video-output (str, default: None) — MP4 output path
--max-frames (int, default: 300)
--fps (int, default: 20)
```

### 7. Cleanup Script

**Source:** `mini-vla/scripts/clean_outputs.py` (111 lines)
**Target:** Create `scripts/clean_outputs.py`

Port the full cleanup utility:
- Dry-run by default, `--yes` to actually delete
- `--all` selects everything
- Categories: videos, metrics, rerun, mlruns, previews, probes, cache, data, checkpoints
- Glob patterns for each category
- Skips `.git` and `.venv`
- Handles both files (unlink) and directories (shutil.rmtree)

**CLI args:**
```
--yes (store_true)
--all (store_true)
--videos (store_true)
--metrics (store_true)
--rerun (store_true)
--mlruns (store_true)
--previews (store_true)
--probes (store_true)
--cache (store_true)
--data (store_true)
--checkpoints (store_true)
```

### 8. Docker MLflow Stack

**Source:** `mini-vla/docker-compose.mlflow.yml`, `mini-vla/docker/mlflow/Dockerfile`
**Target:** Create same structure in gemma4_vla

Port exactly:
- `docker-compose.mlflow.yml`: PostgreSQL 16 + MLflow 3.12.0, port 5001
- `docker/mlflow/Dockerfile`: Python 3.11, MLflow, psycopg2-binary
- Backend store: `postgresql://mlflow:mlflow@postgres:5432/mlflow`
- Artifact root: `/mlflow/artifacts` (named volume)

### 9. Dependencies Update

**Target:** Update `requirements.txt` to add:
```
# simulation (from mini-vla)
gymnasium
mujoco
metaworld

# observability (from mini-vla)
rerun-sdk
mlflow

# video / image utils (from mini-vla)
imageio
imageio[ffmpeg]
opencv-python
```

Keep all existing gemma4_vla dependencies.

### 10. .gitignore Update

**Target:** Update `.gitignore` to add:
```
data/
videos*/
checkpoints/
metrics/
rerun/
mlruns/
MUJOCO_LOG.txt
__pycache__/
```

---

## KEY INTERFACE BRIDGING

### Action Horizon Handling
- Gemma4VLA predicts `[50, action_dim]` actions per call.
- MetaWorld expects one `[action_dim]` action per `env.step()`.
- In test.py: use `actions[0]` from `runner.predict()`, OR use `runner.stream(obs, replan_every=1)` which yields single actions.

### Normalization Bridging
- mini-vla computes per-DOF mean/std from dataset, stores in checkpoint, denormalizes at test time.
- gemma4_vla currently has only `action_scale` (a single scalar).
- Add dataset-computed normalization stats to the training pipeline:
  1. At dataset load, compute `state_mean, state_std, action_mean, action_std` across all episodes
  2. Apply `(x - mean) / std` normalization before model forward
  3. Save stats in checkpoint alongside model weights
  4. At test time, load stats and denormalize: `action_env = action_model * std + mean`
  5. Clip to env bounds: `np.clip(action_env, env.action_low, env.action_high)`

### Text Handling
- mini-vla uses `SimpleTokenizer` (custom whitespace tokenizer with fixed vocab).
- gemma4_vla uses Gemma 4's `AutoProcessor` internally.
- No tokenizer porting needed — `PolicyRunner._preprocess()` handles tokenization from raw strings.
- Pass `instruction` as a plain string to `obs["instruction"]`.

### Image Handling
- mini-vla resizes images to 64x64, normalizes to [0,1], permutes to [3,H,W].
- gemma4_vla resizes to 224x224, normalizes to [-1,1] via `preprocess_image()`.
- `PolicyRunner._preprocess()` handles this internally — pass raw `[H,W,3] uint8` from env.

### Config Mapping (mini-vla args -> gemma4_vla config)
```python
cfg = so100_config()  # or build from Gemma4VLAConfig
cfg.robot.state_dim = env.state_dim       # 39 for MetaWorld
cfg.robot.action_dim = env.action_dim     # 4 for MetaWorld
cfg.robot.max_state_dim = max(env.state_dim, 18)
cfg.vision.num_cameras = 1                # single camera in MetaWorld
cfg.flow_matching.action_horizon = 50     # default, adjust if needed
```

---

## FILE STRUCTURE AFTER MIGRATION

```
gemma4_vla/
├── src/gemma4_vla/
│   ├── __init__.py              # add MetaWorldMT1Wrapper, RerunLogger, MlflowRun exports
│   ├── model.py                 # UNCHANGED
│   ├── action_expert.py         # UNCHANGED
│   ├── flow_matching.py         # UNCHANGED
│   ├── inference.py             # UNCHANGED
│   ├── config.py                # UNCHANGED
│   ├── dataset.py               # UNCHANGED
│   ├── train.py                 # MODIFIED: add MLflow, JSONL metrics, normalization stats
│   ├── observability.py         # NEW: RerunLogger + MlflowRun from mini-vla
│   └── envs/
│       ├── __init__.py          # NEW: exports MetaWorldMT1Wrapper
│       ├── metaworld_env.py     # NEW: MetaWorldMT1Wrapper from mini-vla
│       └── metaworld_mt1.py     # NEW: visualization helper from mini-vla
├── scripts/
│   ├── collect_data.py          # NEW: expert demo collection with HDF5 output
│   ├── test.py                  # NEW: full eval with video/rerun/mlflow/metrics
│   ├── inspect_dataset.py       # NEW: HDF5 dataset inspection
│   └── clean_outputs.py         # NEW: artifact cleanup
├── configs/
│   ├── base_config.yaml         # UNCHANGED
│   ├── so100_config.yaml        # UNCHANGED
│   ├── bimanual_config.yaml     # UNCHANGED
│   └── metaworld_push.yaml      # NEW: MetaWorld push-v3 config preset
├── docker/
│   └── mlflow/
│       └── Dockerfile           # NEW: MLflow container
├── docker-compose.mlflow.yml    # NEW: Postgres + MLflow stack
├── examples/                    # UNCHANGED
├── tests/                       # UNCHANGED (add new tests for new modules)
├── docs/                        # UNCHANGED
├── requirements.txt             # MODIFIED: add gymnasium, mujoco, metaworld, rerun-sdk, mlflow, imageio
├── .gitignore                   # MODIFIED: add data/, videos/, metrics/, rerun/, mlruns/
├── pyproject.toml               # UNCHANGED
├── setup.py                     # UNCHANGED
└── README.md                    # UNCHANGED
```

---

## ACCEPTANCE CRITERIA

### Data Collection
```bash
python -m scripts.collect_data \
  --env-name push-v3 --camera-name corner --episodes 20 \
  --output-dir data/push_demos --instruction "push the object to the goal"
```
Produces `data/push_demos/episode_000000.hdf5` through `episode_000019.hdf5`.
Each HDF5 has `observation/images/corner`, `observation/state`, `action`, and `language_instruction`.

### Data Inspection
```bash
python -m scripts.inspect_dataset --dataset-dir data/push_demos --episode-index 0 \
  --video-output videos/preview.mp4
```
Prints summary. Saves MP4 preview.

### Training
```bash
python -m gemma4_vla.train \
  --config configs/metaworld_push.yaml \
  --data_root data/push_demos \
  --metrics-path metrics/train.jsonl \
  --mlflow --mlflow-experiment gemma4-vla-push
```
Trains model. JSONL file has train_start, per-step, and train_end records.
MLflow tracks params/metrics. Checkpoint contains normalization stats.

### Evaluation
```bash
python -m scripts.test \
  --checkpoint checkpoints/best --env-name push-v3 \
  --instruction "push the object to the goal" --episodes 5 \
  --save-video --video-dir videos \
  --metrics-path metrics/eval.json \
  --rerun-mode spawn \
  --mlflow --mlflow-experiment gemma4-vla-eval
```
Runs 5 episodes in MetaWorld. Saves MP4 videos. Writes JSON metrics with per-step
action data. Streams to rerun viewer. Logs to MLflow.

### Observability with Rerun
```bash
python -m scripts.test --checkpoint checkpoints/best --env-name push-v3 \
  --rerun-mode save --rerun-path rerun/eval.rrd --episodes 2
```
Produces `.rrd` file loadable in `rerun rerun/eval.rrd`.

### Observability with MLflow
```bash
docker compose -f docker-compose.mlflow.yml up -d
python -m scripts.test --checkpoint checkpoints/best --env-name push-v3 \
  --mlflow --mlflow-tracking-uri http://127.0.0.1:5001 --mlflow-log-artifacts --episodes 3
```
Experiment visible in MLflow UI at localhost:5001 with metrics + video artifacts.

### Cleanup
```bash
python -m scripts.clean_outputs --videos --metrics --rerun --yes
```
Removes generated artifacts.

### Zero Diff on mini-vla
```bash
cd /Users/andresjc/beitlab/mini-vla && git diff
```
Shows no changes.

### Zero Diff on gemma4_vla Core Model
```bash
# These files must be UNCHANGED:
# src/gemma4_vla/model.py
# src/gemma4_vla/action_expert.py
# src/gemma4_vla/flow_matching.py
# src/gemma4_vla/inference.py
# src/gemma4_vla/config.py
# src/gemma4_vla/dataset.py
```

---

## IMPLEMENTATION ORDER

1. `src/gemma4_vla/observability.py` — port RerunLogger + MlflowRun
2. `src/gemma4_vla/envs/` — port MetaWorldMT1Wrapper
3. `scripts/collect_data.py` — data collection with HDF5 output
4. `scripts/inspect_dataset.py` — dataset inspection
5. Modify `src/gemma4_vla/train.py` — add MLflow, JSONL metrics, normalization
6. `scripts/test.py` — full evaluation script
7. `scripts/clean_outputs.py` — cleanup utility
8. `configs/metaworld_push.yaml` — MetaWorld-specific config preset
9. Docker files — MLflow stack
10. Update `requirements.txt` and `.gitignore`
