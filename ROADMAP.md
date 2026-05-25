# Gemma4VLA — Roadmap & Deployment Proposal

A forward-looking companion to [README.md](README.md). The existing README
describes the *intended* architecture; this document lists the **concrete
changes** needed to move Gemma4VLA from "research scaffold" to "something you
can actually train, evaluate on LeRobot, and run at 50 Hz on a Jetson Thor
dev kit."

The proposal is organised in three tracks:

1. **Training** — what's missing in the loop today and how to fix it.
2. **LeRobot** — migrate the dataset/policy path onto the official `lerobot`
   package and its real-robot runtime.
3. **Jetson Thor** — a reproducible setup + launch recipe for NVIDIA's new
   Blackwell-based robotics dev kit.

---

## 0. Snapshot of the current state

Confirmed working today (see [CODEX_README.md](CODEX_README.md) and a full
pass over `src/`):

- End-to-end architecture: `Gemma4Backbone` → `ActionExpert` → conditional
  flow-matching loss ([model.py:254](src/gemma4_vla/model.py)).
- LoRA injection via `peft` when available
  ([model.py:189](src/gemma4_vla/model.py)).
- Euler + RK4 integrators for inference
  ([flow_matching.py:145](src/gemma4_vla/flow_matching.py)).
- HDF5 LeRobot-style dataset adapter + flat sampler
  ([dataset.py:68](src/gemma4_vla/dataset.py)).
- AMP training loop with cosine + warmup
  ([train.py:49](src/gemma4_vla/train.py)).
- Checkpoint save/load is JSON-first and `torch.load(weights_only=True)`-safe
  ([model.py:65](src/gemma4_vla/model.py)).

Recently landed (see also the bottom of §4):

- Vision-tower assertion in `Gemma4Backbone.__init__` fails loud when the
  loaded backbone has no vision tower — no more silent text-only fallback.
- Dataset-fit action / state normalisation
  ([stats.py](src/gemma4_vla/stats.py), `--normalize-stats` CLI,
  `normalization.pt` next to the checkpoint, applied symmetrically at train
  and inference time).
- DDP via `torchrun` ([train.py](src/gemma4_vla/train.py)) — DistributedSampler,
  DDP-wrapped forward, rank-0-only logs/saves, NCCL teardown.

Partial / risky today:

- `bitsandbytes` is in `[quant]` extras but not actually wired into the
  forward pass.
- `LeRobotDataset` is HDF5-only — it does **not** call the official
  `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.
- No ONNX/TensorRT export, no on-robot server.

---

## 1. Training — proposed changes

### 1.1 Make the two-stage recipe a first-class entry point

Today the recipe lives in prose and in `examples/02_finetune_lerobot.py`.
Promote it to a CLI flag so it's reproducible:

```bash
gemma4vla-train --config configs/so100_config.yaml --recipe two_stage
```

Implementation: add a `training.recipe` enum (`single`, `two_stage`,
`progressive_unfreeze`) in `TrainingConfig` and branch in `train.main`.

### 1.2 Dataset-fit action / state normalisation — **DONE**

Implemented in [stats.py](src/gemma4_vla/stats.py):

- `DatasetStats.compute_from_loader` streams per-dim mean/std over the train
  loader (capped at `cfg.training.normalize_stats_batches`).
- Stats are saved as `normalization.pt` next to the checkpoint with the keys
  `state_mean`, `state_std`, `action_mean`, `action_std`, `normalize` —
  compatible with the loader the eval script already uses.
- The metaworld HDF5 dataset normalises in `__getitem__` when stats are
  attached; `PolicyRunner.predict` denormalises symmetrically.
- Opt in via `cfg.training.normalize_stats = True` or the
  `--normalize-stats` CLI flag.

Open: `RobotConfig.action_scale` still applies on top; consider deprecating
it once the new path has burn-in on a real robot.

### 1.3 DDP + gradient accumulation — **DONE**

`gemma4_vla.train.train()` now detects `RANK` / `WORLD_SIZE` / `LOCAL_RANK`
(set by torchrun), initialises NCCL, wraps the train/val loaders with
`DistributedSampler`, wraps the model in `DistributedDataParallel`, and
gates logging/MLflow/JSONL metrics/checkpoint saves to rank 0. `Gemma4VLA`
gained a `forward(batch)` entry point so DDP's grad-sync hooks fire.
`Gemma4Backbone` disables `device_map="auto"` under torchrun. Gradient
accumulation was already in place and continues to work.

```bash
torchrun --nproc_per_node=4 -m gemma4_vla.train \
    --config configs/so100_config.yaml \
    --output_dir checkpoints/so100_ddp
```

Still open: end-to-end soak test on real multi-GPU hardware, and teaching
`machine_config.py` about `WORLD_SIZE` so it doesn't recommend a per-rank
batch size meant for single-process training.

### 1.4 Validation, early stopping, eval metrics

Add an evaluation hook that reports more than a single loss:

- Action MSE and per-DOF MAE on held-out episodes.
- Open-loop rollout MSE (predict full horizon, compare against ground truth).
- Dynamic-time-warping distance for trajectory similarity.

Early stopping on a configurable metric (default: open-loop MSE), with
`patience` and `min_delta` — this keeps two-stage runs short on small
datasets.

### 1.5 Data augmentation beyond colour jitter

For robustness — especially on small LeRobot datasets:

- Time-series augmentation: random subsampling / temporal jitter within the
  action horizon.
- Action noise injection (σ = 1–2 % of per-DOF std) during stage 2.
- Camera-level domain randomisation (background swap, brightness, mild
  geometric warp). Keep it behind a `training.augment.domain_random` flag
  so simulation runs can disable it.

### 1.6 Cross-embodiment curriculum

`max_state_dim = 18` padding is already in place. Add:

- A `MixedEmbodimentDataset` that wraps several per-robot datasets with
  configurable sampling weights.
- Per-embodiment prompt prefixes (`"[SO100] <instruction>"`) so the
  language conditioning stays separable.
- A small warm-start on Open X-Embodiment subsets before the task-specific
  stage 2 fine-tune (tracked under the existing Roadmap item "RLDS / Open
  X-Embodiment dataset adapter").

---

## 2. Running on LeRobot

### 2.1 Adopt `lerobot.LeRobotDataset` as the primary data path

Today [dataset.py:68](src/gemma4_vla/dataset.py) only reads a custom HDF5
layout. Replace/augment it with a thin adapter around the official class:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

lerobot_ds = LeRobotDataset(
    repo_id="lerobot/pusht",
    delta_timestamps={
        "action": [i / fps for i in range(cfg.flow_matching.action_horizon)],
    },
)
```

Wrap it in a `LeRobotDatasetAdapter` that:

- renames LeRobot keys (`observation.images.top`, `observation.state`,
  `action`, `task`) to our internal dict layout;
- reuses the same image transforms and normaliser as the HDF5 path;
- exposes `.meta.features` so the config builder can auto-populate
  `state_dim`, `action_dim`, and `num_cameras`.

Keep the HDF5 path as a secondary backend for offline/raw captures.

### 2.2 Task / language metadata

LeRobot v2 stores `task` as first-class metadata with per-episode indexes.
Wire that into the sample dict so the instruction can vary per frame,
not just per episode.

### 2.3 Real-robot policy server

To actually drive an SO-100, SO-101, Koch, or ALOHA arm via LeRobot's
`control_robot.py`, we need a thin inference server. Proposal: add
`gemma4_vla.server`:

```bash
# on the training workstation (or on Thor itself)
python -m gemma4_vla.server \
    --checkpoint checkpoints/best \
    --port 8765 \
    --device cuda \
    --precision fp8   # fp8 on Thor, bf16 elsewhere
```

API surface (WebSocket + msgpack, or gRPC if we want strict typing):

| Message | Direction | Payload |
|---------|-----------|---------|
| `reset` | client → server | `{instruction: str}` |
| `observe` | client → server | images (JPEG), state (float32[]), ts |
| `action` | server → client | `action_chunk[H, action_dim]`, ts, confidence |

Client side: a `LeRobotPolicyClient` implementing LeRobot's `Policy`
protocol (`select_action(batch) -> Tensor`) so it plugs straight into
`lerobot/scripts/control_robot.py`. That gives us teleop handoff,
emergency-stop handling, and recording for free.

Replan cadence: the server keeps the most recent chunk cached and returns
the next unused action in-flight; it only re-runs flow matching every
`replan_every` steps (default 25). This matches the streaming path already
in [inference.py:138](src/gemma4_vla/inference.py) and hides backbone
latency.

### 2.4 Eval harness on LeRobot benchmarks

Add a `gemma4_vla.eval.lerobot_rollout` that runs `N` LeRobot simulated
episodes and reports success rate + reward. Targets: `pusht`, `aloha_sim_*`,
`xarm_lift_medium`. These are what reviewers will expect in the first
public release.

---

## 3. Jetson Thor — setup and launch

Target: NVIDIA **Jetson AGX Thor Developer Kit** — Blackwell GPU (~2000
TFLOPS FP4 / ~1000 TFLOPS FP8), 14-core Neoverse V3AE (ARM64), 128 GB
LPDDR5X unified memory, JetPack 7, CUDA 13, TensorRT 10, Ubuntu 24.04.

Unified memory changes the game: the 128 GB pool is shared between CPU and
GPU, so Gemma 4 E2B / E4B fit comfortably with room for the inference stack,
image IO, and ROS.

### 3.1 System setup (one-time, on the board)

```bash
# Flash JetPack 7 from SDK Manager on an x86 host first.
# Then on Thor:

sudo apt update && sudo apt install -y \
    python3.12 python3.12-venv python3-pip \
    libopenblas-dev libjpeg-dev libpng-dev \
    libhdf5-dev pkg-config git-lfs cmake ninja-build

# Enable max clocks + fan profile for the policy server
sudo nvpmodel -m 0           # MAXN_SUPER
sudo jetson_clocks

# Hugging Face login (you must have accepted the Gemma license)
pip install --user huggingface_hub
huggingface-cli login
```

### 3.2 Project install on Thor

Do **not** use the generic `pip install torch` — we need the JetPack
ARM64/CUDA 13 wheels NVIDIA publishes on `pypi.jetson-ai-lab.dev`:

```bash
git clone https://github.com/beitlab-inc/beitlab-gemma4-vla
cd beitlab-gemma4-vla
python3.12 -m venv .venv
source .venv/bin/activate

# Jetson-specific wheels (pinned to JetPack 7 / CUDA 13)
pip install --extra-index-url https://pypi.jetson-ai-lab.dev/jp7/cu130 \
    torch torchvision

# Everything else: use the regular dependency set but skip torch
pip install -e ".[lerobot]" \
    --no-deps-torch  # custom flag — see proposal 3.6
```

> **Proposed change:** add a `[jetson]` extras group in `pyproject.toml`
> that lists `onnx>=1.17`, `onnxruntime-gpu`, `tensorrt>=10.5`,
> `torch2trt`, and `pyzmq` (for the policy server transport), and
> documents the Jetson wheel index in [README.md](README.md).

### 3.3 Precision strategy

Blackwell on Thor supports BF16, FP8 (E4M3 / E5M2), and FP4 natively.
Recommended default for inference:

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Gemma 4 backbone | FP8 weights, BF16 activations | Biggest memory + latency win |
| Vision tower | BF16 | Small, stability matters |
| Action expert | BF16 | 300 M params, latency-critical |
| Flow-matching integrator | FP32 | Numerical stability over 10 Euler steps |

Implementation: extend `BackboneConfig` with a `precision` field
(`bf16` | `fp16` | `fp8_e4m3`) and plumb it through `from_pretrained`
using `torchao.float8` or `transformer_engine` (both ship with JetPack 7).

Remove the current unconditional `torch.bfloat16` cast in
[model.py:162](src/gemma4_vla/model.py); respect the config.

### 3.4 TensorRT export path

The runtime-critical bottleneck is the *backbone*; the action expert is
cheap enough to leave in PyTorch eager. Propose:

```bash
python -m gemma4_vla.export.trt \
    --checkpoint checkpoints/best \
    --output trt/best.plan \
    --precision fp8 \
    --max_cameras 2 \
    --max_batch 1
```

Steps the exporter performs:

1. Merge LoRA adapters into the base weights (`peft.merge_and_unload`).
2. Trace the backbone + `obs_proj` as a single `nn.Module` with static
   image input `[1, num_cameras, 3, 224, 224]` and a fixed token length.
3. `torch.onnx.export(..., opset=20, dynamo=True)`.
4. `trtexec --onnx=... --fp8 --saveEngine=best.plan`.
5. Keep the action expert in PyTorch eager with `torch.compile(mode=
   "reduce-overhead")` — graph break penalty is lower than the TRT
   re-export cost, and the 10-step Euler loop is dynamic anyway.

Ship a `Gemma4VLATRT` wrapper that swaps the backbone forward pass for a
TRT engine call but keeps the rest of the Python inference path identical.

### 3.5 Launching the policy on Thor

Once the engine is built, running it is a one-liner:

```bash
# Terminal 1 — policy server (on Thor)
GEMMA4VLA_BACKEND=trt python -m gemma4_vla.server \
    --engine trt/best.plan \
    --action_expert checkpoints/best \
    --port 8765 \
    --precision fp8

# Terminal 2 — robot driver (on Thor or the host PC talking to the arm)
python -m lerobot.scripts.control_robot teleop \
    --robot-path lerobot/configs/robot/so100.yaml \
    --policy ws://thor.local:8765 \
    --fps 50
```

Expected end-to-end latency budget on Thor (Gemma 4 E2B, 2 cameras, 10
Euler steps, FP8 backbone + BF16 expert):

| Stage | Budget |
|-------|--------|
| Camera capture + resize | 3 ms |
| Backbone (TRT, FP8) | 8–12 ms |
| Action expert × 10 steps | 6–8 ms |
| Serialisation + transport | 1–2 ms |
| **Total per replan** | **~20–25 ms** |

With `replan_every=25` and temporal chunking, that gives **~50 Hz
closed-loop control** with headroom for a second camera and ROS2
integration.

### 3.6 Optional: ROS2 Jazzy node

JetPack 7 ships ROS2 Jazzy. Add `gemma4_vla_ros/` with:

- `policy_node.py`: subscribes to `/camera/*/image_raw` and
  `/joint_states`, publishes `/policy/action` (`JointTrajectory`).
- A launch file that boots the policy server + node together.

This is the cleanest path to integrate with MoveIt, Isaac ROS, or a
Foxglove dashboard without inventing yet another transport.

---

## 4. Proposed work order

Priority-ordered, rough sizing. Each item below is one focused PR.

1. ~~**[M]** Dataset-fit normaliser (§1.2) — unblocks everything downstream.~~ **DONE**
2. **[M]** `LeRobotDatasetAdapter` on top of `lerobot.LeRobotDataset`
   (§2.1).
3. **[S]** `training.recipe` flag and curriculum wiring (§1.1).
4. ~~**[M]** DDP + grad accumulation (§1.3).~~ **DONE** (needs real multi-GPU soak test)
5. **[M]** `gemma4_vla.server` + LeRobot `Policy` client (§2.3).
6. **[S]** Eval harness on LeRobot sim benchmarks (§2.4).
7. **[L]** TensorRT export + `Gemma4VLATRT` wrapper (§3.4).
8. **[S]** `[jetson]` extras + Thor install doc section (§3.1 / 3.2).
9. **[S]** Precision config plumbing, replace hard-coded bfloat16 (§3.3).
10. **[M]** ROS2 Jazzy node (§3.6).
11. **[M]** Validation metrics + early stopping (§1.4).
12. **[S]** Augmentation flags + cross-embodiment sampler (§1.5–1.6).

Sizes: S ≈ ½–1 day, M ≈ 2–4 days, L ≈ 1 week.

---

## 5. Known risks / open questions

- **Gemma 4 availability.** The code assumes `google/gemma-4-E2B-it` etc.
  exist on HF Hub. If Google publishes under a different ID or gated
  release, the fallback to `AutoModelForCausalLM` in
  [model.py](src/gemma4_vla/model.py) would otherwise load a text-only
  model. As of the latest change `Gemma4Backbone.__init__` now raises a
  `RuntimeError` when the loaded model exposes no `vision_tower` /
  `embed_vision` / `vision_model` attribute, so this fails loud rather than
  training silently on text only. Override by setting
  `cfg.vision.use_external_encoder = True` if you intend to supply your own
  vision encoder.
- **LoRA + FP8 interaction.** `merge_and_unload` into FP8 weights needs
  validation; we may have to merge into BF16, then quantise post-merge.
- **LeRobot API stability.** `lerobot` is pre-1.0; pin to a specific
  commit in the `[lerobot]` extras and upgrade deliberately.
- **Thor thermal headroom.** 50 Hz at FP8 is comfortable on paper but
  needs a 30-minute soak test with the real camera pipeline before
  shipping on a robot.
- **Action horizon vs. replan cadence.** `action_horizon=50` and
  `replan_every=25` are defaults carried over from pi0; re-tune them on
  the target task — shorter horizons usually win on contact-rich tasks.

---

## 6. What this document is not

This is a **proposal**, not a spec. Before implementing any item, open
an issue or a design note in [docs/](docs/) and confirm the approach —
especially anything touching checkpoint I/O, tensor shapes, or the public
API surface (see the guardrails in [CODEX_README.md](CODEX_README.md)).
