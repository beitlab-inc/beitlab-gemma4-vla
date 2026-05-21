# Gemma4VLA — Technical Documentation

Deep-dive documentation on the math, engineering, and design choices behind
every module in Gemma4VLA.

If you just want to **run** the code, see the top-level [README](../README.md)
and the [examples/](../examples/) directory.  This folder is for people who
want to **understand** the code — whether to modify it, port it, or extend it
for research.

---

## Reading order

### For first-time readers
Read these in order for a complete mental model:

1. [**Architecture overview**](architecture.md) — how all the pieces fit together
2. [**Flow matching**](02_flow_matching.md) — the math behind training and inference
3. [**Action expert**](03_action_expert.md) — the transformer that generates actions
4. [**Model assembly**](04_model.md) — how Gemma 4 + action expert are stitched together

### For implementers
Additional modules you need to touch when training on new data:

5. [**Config system**](01_config.md)
6. [**Datasets & preprocessing**](05_dataset.md)
7. [**Training loop**](06_training.md)
8. [**Inference pipeline**](07_inference.md)

### For simulation users
How the model connects to robotic environments:

9. [**Simulation pipeline**](08_simulation.md) — MetaWorld + MuJoCo + Gymnasium integration, data collection, evaluation, observability

---

## Module index

| # | Module | What it does | Key math |
|---|--------|--------------|----------|
| 1 | [`config.py`](01_config.md) | Hierarchical dataclass config | – |
| 2 | [`flow_matching.py`](02_flow_matching.md) | CFM training + ODE integration | OT probability paths, Euler/RK4 |
| 3 | [`action_expert.py`](03_action_expert.md) | Transformer that predicts velocities | RoPE, RMSNorm, cross-attention |
| 4 | [`model.py`](04_model.md) | Full Gemma4VLA model | Prefix-LM + action expert composition |
| 5 | [`dataset.py`](05_dataset.md) | Robot data loading | Temporal action chunking, image normalisation |
| 6 | [`train.py`](06_training.md) | Training loop | AdamW + cosine schedule, LoRA, feature caching |
| 7 | [`inference.py`](07_inference.md) | Real-time policy execution | Action chunking, replanning |
| 8 | [Simulation pipeline](08_simulation.md) | MetaWorld + MuJoCo + Gymnasium | Env wrapper, data collection, eval |

---

## Conceptual summary

At the highest level, Gemma4VLA is a function:

```
π(a_{t..t+H} | o_t, ℓ)
```

- `o_t` — the current observation (images + robot state)
- `ℓ` — a natural-language instruction
- `a_{t..t+H}` — a chunk of H future actions

We model this as a **conditional probability distribution** and draw samples
from it at inference time.  The distribution is parameterised as a **flow**
(in the flow-matching sense) conditioned on `(o_t, ℓ)`.  Training minimises
a **regression loss on the flow velocity field**, not a likelihood — which
turns out to be both more stable and faster to train than diffusion.

The conditioning is split into two transformers with different roles:

- **Gemma 4 backbone** reads the observation and language, producing a
  dense feature sequence.  It's huge (2–27 B parameters) and most of it
  stays frozen / LoRA-adapted during robot training.
- **Action expert** is a small (~300 M) transformer that reads the
  proprioceptive state + a noisy action chunk, cross-attends to the
  backbone features, and predicts the flow velocity.  This is where most
  of the robot-specific learning happens.

Every module in the codebase serves one of three purposes:
1. **Produce observation features** (backbone, vision, dataset)
2. **Predict the flow velocity** (action expert, flow matching)
3. **Orchestrate training or inference** (train, inference, config)

The following docs explain each one in detail.
