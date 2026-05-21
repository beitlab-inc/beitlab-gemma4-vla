# Examples

Step-by-step examples for Gemma4VLA, ordered by complexity.

## Prerequisites

```bash
uv sync --all-extras
uv run huggingface-cli login     # required for Gemma 4 weights
```

---

## 01 — Quick start (`01_quick_start.py`)

The simplest possible inference demo.  No data or checkpoint required.

```bash
# Run with random (untrained) weights to test the forward pass
uv run python 01_quick_start.py

# Run with a trained checkpoint
uv run python 01_quick_start.py --checkpoint ../checkpoints/best

# Choose device and denoising steps
uv run python 01_quick_start.py --device cpu --steps 5
```

**What it shows:**
- Building a `PolicyRunner` from a config or checkpoint
- Predicting an action chunk from dummy camera images + language
- Temporal action chunking with `runner.stream()`

**Expected output:**
```
No checkpoint specified — using freshly initialised weights.

Observation:
  Instruction : 'Pick up the red cube and place it in the bowl.'
  Images      : 1 × (224, 224, 3)
  State       : (6,)

Running inference (10 denoising steps)…

Predicted actions: (50, 6)
  horizon    = 50 steps
  action_dim = 6 DOF
  First action : [-0.012  0.034  0.008 -0.001  0.021 -0.005]
  Action range : [-0.312, 0.287]

Simulating temporal action chunking (first 5 steps):
  step  0: [-0.012  0.034  0.008 ...]
  step  1: [-0.009  0.031  0.011 ...]
  ...
```

---

## 02 — Train on MetaWorld data

Uses the MetaWorld robot adapter under `robots/metaworld/`.  See
`robots/metaworld/README.md` for full details.

```bash
# 1. Collect expert demonstrations
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 --episodes 50

# 2. Train on collected data
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push.yaml \
    --data-dir data/metaworld_demos

# 3. Evaluate the checkpoint
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push/best \
    --env-name push-v3 --episodes 10 --save-video
```

**Two-stage training strategy:**

```
Stage 1 (2 000 steps, backbone frozen):
  -> Action expert learns action structure quickly at high LR
  -> Val loss drops from ~1.0 to ~0.15

Stage 2 (50 000 steps, LoRA + action expert):
  -> Language-action alignment refined
  -> Val loss drops from ~0.15 to ~0.04
```

**Supported MetaWorld tasks:**

| Task | Action dim | Difficulty |
|------|-----------|-----------|
| `push-v3` | 4 | Easy |
| `reach-v3` | 4 | Easy |
| `pick-place-v3` | 4 | Medium |

---

## 03 — Custom robot configuration (`03_custom_robot.py`)

Shows how to configure Gemma4VLA for any robot with arbitrary DOF and camera setup.

```bash
# Default: 14-DOF bimanual arm with 3 cameras
uv run python 03_custom_robot.py

# 7-DOF single arm with 2 cameras
uv run python 03_custom_robot.py \
    --robot my_franka \
    --state_dim 7 \
    --action_dim 7 \
    --num_cameras 2

# Use the larger 4B backbone
uv run python 03_custom_robot.py --large
```

**Key concepts demonstrated:**
- `RobotConfig` for any state/action dimensionality
- Zero-padding for cross-embodiment (up to 18 DOF)
- Memory-efficient training with gradient checkpointing
- Exporting just the action expert for edge deployment

**Cross-embodiment compatibility matrix:**

| Robot | DOF | Cameras | Config preset |
|-------|-----|---------|--------------|
| MetaWorld push-v3 | 4 | 1 | `metaworld_push_config()` |
| Franka Panda | 7 | 2 | Custom |
| UR5 | 6 | 2 | Custom |
| ALOHA | 14 | 3 | Custom |
| Stretch RE3 (mobile) | 10 | 2 | Custom |

All fit within `max_state_dim=18` with zero-padding.

---

## 04 — Evaluate a checkpoint (`04_evaluate.py`)

Runs comprehensive evaluation: flow matching loss, action prediction error,
and inference latency.

```bash
# Evaluate a checkpoint
uv run python 04_evaluate.py --checkpoint ../checkpoints/best

# Specify dataset and number of batches
uv run python 04_evaluate.py \
    --checkpoint ../checkpoints/best \
    --data_root ../data/my_robot \
    --num_batches 100 \
    --inference_steps 10

# Skip latency benchmarking (faster)
uv run python 04_evaluate.py --checkpoint ../checkpoints/best --no_latency
```

**Sample output:**

```
Evaluating on validation set (50 batches)…
  batch   0/50  loss=0.0421  l2=0.0183
  batch  10/50  loss=0.0398  l2=0.0171
  ...

──────────────────────────────────────────────────
  Validation Metrics
──────────────────────────────────────────────────
  cosine_similarity             0.9823
  flow_matching_loss            0.0413
  l1_error                      0.0127
  l2_error                      0.0179
  l2_joint_0                    0.0142
  l2_joint_1                    0.0198
  l2_joint_2                    0.0165
  ...
──────────────────────────────────────────────────

Benchmarking inference latency…
──────────────────────────────────────────────────
  Inference Latency (10 steps)
──────────────────────────────────────────────────
  hz                            43.2000
  mean_ms                       23.1500
  p95_ms                        26.4200
  std_ms                        1.2300
──────────────────────────────────────────────────

  Control frequency: 43.2 Hz
  ✓ Fast enough for 50 Hz real-time control
```

---

## Tips

### Choosing `num_inference_steps`

| Steps | Latency (A100, 2B) | Use case |
|-------|--------------------|----------|
| 5 | ~12 ms (83 Hz) | Fast manipulation, table clearing |
| 10 | ~22 ms (45 Hz) | Most tasks (recommended) |
| 25 | ~55 ms (18 Hz) | Slow/dexterous tasks, quality matters |

### Choosing backbone size

| Backbone | VRAM | Quality | Recommended for |
|----------|------|---------|----------------|
| Gemma 4 E2B | 12 GB | Good | Single-task fine-tuning |
| Gemma 4 E4B | 20 GB | Better | Multi-task / complex manipulation |
| Gemma 4 31B | 55 GB | Best | Research, hard dexterous tasks |

### Debugging flow matching loss

A healthy training run looks like:

```
step    100  loss=0.842   ← initial noise
step    500  loss=0.421
step   1000  loss=0.183   ← action structure learned
step   5000  loss=0.071
step  10000  loss=0.041   ← converged
step  50000  loss=0.028   ← fine-grained refinement
```

If loss is stuck above 0.5 after 1 000 steps:
- Increase batch size
- Check that images are normalised to `[-1, 1]`
- Check that actions are in a reasonable range (`[-1, 1]` or similar)
- Increase `lora_rank` for more expressive backbone adaptation
