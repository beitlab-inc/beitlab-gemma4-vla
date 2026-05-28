# 10 — Multi-camera two-stage training

**Modules:**
- [`robots/metaworld/configs/metaworld_push_multicam_stage1.yaml`](../robots/metaworld/configs/metaworld_push_multicam_stage1.yaml)
- [`robots/metaworld/configs/metaworld_push_multicam_stage2.yaml`](../robots/metaworld/configs/metaworld_push_multicam_stage2.yaml)
- [`robots/metaworld/scripts/cache_features.py`](../robots/metaworld/scripts/cache_features.py)
- [`robots/metaworld/scripts/train_cached.py`](../robots/metaworld/scripts/train_cached.py)
- [`robots/metaworld/scripts/train.py`](../robots/metaworld/scripts/train.py) — `--init-from`
- [`src/gemma4_vla/model.py`](../src/gemma4_vla/model.py#L687) — `Gemma4VLA.from_pretrained`

This document is the concrete recipe for combining the two-stage training
schedule (see [06_training.md §8](06_training.md#8-two-stage-training-recommended-recipe))
with multiple camera views (see [09_multi_camera.md](09_multi_camera.md)).

It assumes you already have HDF5 episodes collected with `--cameras a,b,c`
under `data/<dataset>/`, and want a deployable policy at the end.

---

## 1. Why two-stage with multi-cam in particular

Multi-camera amplifies the failure mode that single-stage training already
risks:

- **Three views push the prompt to ~768 image tokens.** The action expert
  has to learn to cross-attend selectively across `3 × num_patches` features
  it has never seen. While the expert is still random, its gradients on
  those features are nearly worthless and — without a frozen backbone —
  would leak into the SigLIP2 + Gemma representations that already encode
  "topview vs gripperPOV" structure for free.
- **Storage cost is linear in cameras.** With expert-only Stage 1 you can
  cache backbone features once and reuse them for thousands of training
  steps. Skipping caching here means re-running a 5B-param forward over
  ~3× the image tokens every step — multi-cam pays the per-step compute
  penalty the most.

So the two-stage recipe is not optional dressing; it's the structural fix
that makes multi-cam training cheap *and* safe.

---

## 2. Architecture of the two stages

```
data/<dataset>/                ← raw HDF5, N eps × C cams
        │
        ▼
[1] cache_features.py          ← one expensive Gemma forward per sample
        │                         (uses Stage 1 config — backbone frozen)
        ▼
data/<dataset>_features/        ← .npz files: obs_features, state, actions
        │
        ▼
[2] train_cached.py            ← fast loop, only expert + obs_proj train
        │                         backbone never executes
        ▼
checkpoints/<run>/stage1/final/   ← config.json + weights.pt
        │
        │ --init-from
        ▼
[3] train.py                   ← real images every step, LoRA + expert
        │                         backbone forward + backward (slow)
        ▼
checkpoints/<run>/stage2/best/    ← the deployable policy
```

The 2-vs-3 script split is purely an optimisation. Conceptually,
**scripts 1 + 2 together = Stage 1 (backbone frozen).** Script 3 = Stage 2
(LoRA + expert). They can't be fused because cached features are only
valid while the backbone is frozen.

---

## 3. Stage 1 — Action expert warm-up

### Config

[`metaworld_push_multicam_stage1.yaml`](../robots/metaworld/configs/metaworld_push_multicam_stage1.yaml):

```yaml
vision:
  num_cameras: 3
  camera_names: ["topview", "corner", "gripperPOV"]   # match HDF5 attr order

backbone:
  freeze_backbone: true        # whole Gemma 4 is requires_grad=False
  use_lora: false              # no adapters — pure feature extractor
  max_sequence_length: 1024    # ~256 tokens/cam × 3 cams + text headroom

training:
  learning_rate: 0.0005        # high LR — small fresh expert needs to move fast
  max_steps: 3000              # 2k–5k typical; early stopping usually fires sooner
  warmup_steps: 200
  early_stopping_patience: 5
  eval_every_n_steps: 100
```

### What's trainable

Only `obs_proj` (the linear that maps Gemma's hidden_size → action expert's
hidden_size) and the action expert transformer (~100 M params). The
backbone (~5 B) is held fixed.

### Why caching is correct

Because Gemma's weights never change in Stage 1, *and* the inputs to it
(your images + the fixed instruction) are identical every epoch, **its
output is bit-exact deterministic across epochs**. Running a 5B-param
forward every step to produce the same tensor over and over is pure
waste.

[`cache_features.py`](../robots/metaworld/scripts/cache_features.py)
iterates the dataset once, runs Gemma forward, and saves one
`sample_NNNNNN.npz` per `(episode, timestep)` containing:

```
obs_features:  [seq_len, hidden_size]    ← Gemma's last-layer output (cached)
state:         [state_dim]                ← proprioceptive vector
actions:       [horizon, action_dim]      ← future 50-step chunk
```

Speedup vs. running the backbone every step: ~100× (≈ 20 s → ≈ 0.1 s per
batch).

### Run

```bash
# 1) Cache backbone features (one-time, ~25–45 min on a 4090; longer on Thor)
uv run python -m robots.metaworld.scripts.cache_features \
    --config robots/metaworld/configs/metaworld_push_multicam_stage1.yaml \
    --data-dir data/push_demos_multicam \
    --output-dir data/push_demos_multicam_features

# 2) Train action expert on cached features (~minutes)
uv run python -m robots.metaworld.scripts.train_cached \
    --config robots/metaworld/configs/metaworld_push_multicam_stage1.yaml \
    --cache-dir data/push_demos_multicam_features \
    --mlflow --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-multicam-stage1 \
    --mlflow-log-artifacts
```

### Convergence target

Val loss drops fast from ~0.6 → ~0.15 in the first few hundred steps,
then plateaus. The plateau is structural — the expert can't extract more
signal from frozen features. Early stopping (`patience: 5`) is expected
to fire well before `max_steps: 3000` on small datasets; that is the
desired behaviour, not a problem to fix.

### When to **recache**

The cache encodes a specific prompt and image preprocessing. Regenerate
it from scratch any time you change:

- `vision.num_cameras` or `vision.camera_names`
- `backbone.max_sequence_length`
- The instruction string passed to `cache_features.py --instruction`
- The image augmentation pipeline (currently disabled in Stage 1; if you
  enable it, caching becomes invalid)

The cache files have no schema versioning. Delete the output directory
and rebuild when in doubt.

---

## 4. Stage 2 — Joint LoRA fine-tune

### Config

[`metaworld_push_multicam_stage2.yaml`](../robots/metaworld/configs/metaworld_push_multicam_stage2.yaml):

```yaml
vision:
  num_cameras: 3
  camera_names: ["topview", "corner", "gripperPOV"]   # must match Stage 1
  freeze_vision: true            # SigLIP2 always frozen in LoRA mode

backbone:
  freeze_backbone: false         # backbone unfrozen — LoRA controls what trains
  use_lora: true
  lora_rank: 16
  lora_alpha: 32.0
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  max_sequence_length: 1024      # match Stage 1

training:
  learning_rate: 0.0002          # 2.5× lower than Stage 1
  backbone_lr_multiplier: 0.05   # LoRA learns 20× slower than expert — the safety knob
  batch_size: 8                  # 3 cams ~doubles VRAM vs 1 cam
  grad_accum_steps: 2            # effective batch = 16
  max_steps: 30000               # 10k–100k per the recipe
  gradient_checkpointing: true   # backbone backward pass needs activation savings
```

### What's trainable

LoRA A/B matrices on `{q,k,v,o}_proj` (~few-M params) **plus** everything
that was trainable in Stage 1 (`obs_proj` + action expert). The base
Gemma weights stay frozen even though `freeze_backbone: false`; PEFT
freezes them inside the LoRA wrapping.

### Why no caching in Stage 2

LoRA updates the effective backbone weights every optimizer step, so its
outputs change every step. Cached features would be stale after a single
step. Stage 2 therefore pays the full backbone forward+backward cost,
which is why `gradient_checkpointing: true` is on and `batch_size`
drops vs. Stage 1.

### How `--init-from` chains the stages

[`train.py`](../robots/metaworld/scripts/train.py) exposes an
`--init-from <checkpoint_dir>` flag. When provided, before training it
calls:

```python
init_model = Gemma4VLA.from_pretrained(args.init_from, cfg=cfg)
```

— which builds the *Stage 2* model from `cfg` (so LoRA layers exist), then
loads Stage 1's state dict with `strict=False`
([model.py:705](../src/gemma4_vla/model.py#L705)):

| Parameter group | Source |
|---|---|
| Backbone base weights | Stage 1 checkpoint (pristine Gemma) |
| `obs_proj` | Stage 1 checkpoint (trained) |
| Action expert + velocity head | Stage 1 checkpoint (trained) |
| LoRA A / B matrices | Default init (B = 0 by convention) |

Because LoRA's B matrices are zero-initialised, **Stage 2 step 0
produces exactly the same output as Stage 1's final checkpoint**. There
is no discontinuity in the loss curve from switching configs — Stage 2
picks up exactly where Stage 1 left off, then gradually carves
task-specific structure into Gemma's features.

### Run

```bash
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push_multicam_stage2.yaml \
    --data-dir data/push_demos_multicam \
    --init-from checkpoints/metaworld_push_multicam_stage1/final \
    --mlflow --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-multicam-stage2 \
    --mlflow-log-artifacts
```

If Stage 1 stopped early, `--init-from` can point at any `step_NNNN/` or
`best/` directory under the Stage 1 output — the `.../final/` directory
is just the last step written when training reached `max_steps`.

### Convergence target

Stage 2 val loss continues descending from Stage 1's plateau (~0.15) down
to ~0.04. That ≈3× drop is the model learning task-specific grounding —
LoRA aligns Gemma's features to the action distribution, and multi-cam
disambiguation (e.g. `gripperPOV` resolving occlusions the `topview`
alone can't) becomes useful.

If Stage 2 doesn't drop below Stage 1's plateau within a few thousand
steps, the bottleneck is your dataset, not the model. Adding more LoRA
capacity won't help — collect more demonstrations.

---

## 5. Composite loss-curve picture

```
Stage 1 (cached, expert only)        Stage 2 (LoRA + expert, --init-from)
val_loss ──────────────────────────────────────────────────────────────────
  ~0.8 ┐\
       │ \  rapid drop — expert learns
       │  \ action shape from fixed features
  ~0.2 │   \________________  plateau: backbone is fixed,
       │                       expert can't extract more       ←── --init-from seam:
       │                                                            loss continues
       │                                                            (LoRA-B = 0, no jump)
  ~0.1 │                                              \___
       │                                                  \_
  ~0.04│                                                    \___ slow descent as LoRA
       │                                                        specialises features
       └─────────────────────────────────────────────────────────
        step 0           ~3,000          step 0 (stage 2)        ~30,000
```

The seam between the two MLflow runs is exactly continuous in expected
loss. If you see a *jump* upward at the start of Stage 2, the most likely
causes are:

1. `--init-from` was omitted (Stage 2 reinitialised expert from scratch).
2. `vision.camera_names` or `max_sequence_length` differs between the
   two stage configs (model architecture mismatch — `strict=False` then
   silently drops weights that no longer fit).
3. The Stage 1 checkpoint passed to `--init-from` was a much earlier
   step than the actual best.

---

## 6. Pitfalls

### Camera order mismatch between stages

Both stage configs must list `camera_names` in the **same** order as the
collected data's `camera_names` HDF5 attribute. Stage 1 caches features
keyed to that order; Stage 2 reads images in the same order. A reorder
between stages won't crash — `strict=False` will load weights and Stage 2
will train on a permuted view, producing a quietly broken policy.

### Stale cache after config change

If you bump `max_sequence_length`, `num_cameras`, `camera_names`, or the
instruction text after caching, **delete and regenerate the cache
directory.** The cache has no checksum or schema check; stale features
will silently produce wrong gradients.

### LoRA didn't actually freeze the backbone

`backbone.freeze_backbone: false` with `use_lora: true` relies on PEFT to
freeze the base Gemma weights. Confirm this in the Stage 2 logs — the
training script prints `Model params: N M total, M M trainable`. The
trainable count for the Stage 2 multicam config should be ~100 M
(expert + obs_proj) + a few M (LoRA), not several B.

### Stage 2 batch size too high

Three cameras roughly doubles the activation memory of the backbone
forward vs. one camera. The shipped Stage 2 config uses `batch_size: 8`
+ `grad_accum_steps: 2` (effective batch 16) and gradient checkpointing
on. On smaller GPUs, drop to `batch_size: 4, grad_accum_steps: 4` rather
than disabling checkpointing.

### Evaluating the resulting policy

The shipped MetaWorld evaluation path
([`MetaWorldMT1Wrapper`](../robots/metaworld/env.py)) renders a single
camera per step. To evaluate a 3-cam policy, extend the wrapper to
render `camera_names` per step and pass `images=[...]` to
`runner.predict(...)`. See
[09_multi_camera.md §5 Option A](09_multi_camera.md#option-a--extend-metaworldmt1wrapper-recommended)
for the ~30-line patch sketch. This caveat is independent of two-stage
training.

---

## 7. Summary

| Step | Script | Backbone | What trains | Speed |
|------|--------|----------|-------------|-------|
| 1 | `cache_features.py` | runs once, frozen | nothing — extracts features | one-shot, ~tens of minutes |
| 2 | `train_cached.py` (Stage 1) | not loaded | `obs_proj` + action expert | fast, ~minutes |
| 3 | `train.py --init-from` (Stage 2) | forward + backward every step (LoRA) | LoRA + `obs_proj` + action expert | slow, ~hours |

Single config knob that switches a one-stage run into the right Stage 1:
`backbone.freeze_backbone: true` + `use_lora: false`. Single CLI knob
that chains Stage 1 → Stage 2: `--init-from <stage1>/final`. Everything
else is hyper-parameter tuning around those two switches.
