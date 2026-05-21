# 06 — Training Loop

**Module:** [`src/gemma4_vla/train.py`](../src/gemma4_vla/train.py)

The training script: optimiser, learning-rate schedule, mixed precision,
checkpointing, validation, and feature caching.

---

## 0. Training strategies at a glance

Three strategies are supported, each a YAML config swap:

| Strategy | Config | What trains | VRAM | Data needed | Speed |
|----------|--------|-------------|------|-------------|-------|
| **Action expert only** | `metaworld_push_expert_only.yaml` | Action expert (~100M) | ~12 GB | 100-1K demos | Fast |
| **LoRA + action expert** | `metaworld_push.yaml` | LoRA adapters + expert (~115M) | ~24 GB | 1K-10K demos | Moderate |
| **Full fine-tune** | `metaworld_push_full.yaml` | Everything (~3.5B) | ~40 GB | 50K+ demos | Slow |

An Apple M3 variant is also provided (`metaworld_push_m3.yaml`) — action
expert only with batch=1 + gradient accumulation.

All configs live in `robots/metaworld/configs/`.  Switching strategies is
just changing `--config`:

```bash
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push_expert_only.yaml \
    --data-dir data/metaworld_demos
```

### How the config flags map to strategies

```
freeze_backbone=True,  use_lora=False  → Action expert only
freeze_backbone=False, use_lora=True   → LoRA + expert
freeze_backbone=False, use_lora=False  → Full fine-tune
```

`freeze_vision` controls the vision tower independently (always `True`
for LoRA, set to `False` only for full fine-tune).

### Decision guide

- **Visually standard tasks** (tabletop, household objects): action expert
  only is usually enough. The pre-trained backbone already understands
  everyday scenes.
- **Language-grounded multi-task** (varied instructions): LoRA helps the
  backbone map new instructions to actions.
- **Novel visual domains** (underwater, microscopy, industrial): full
  fine-tune or at minimum LoRA.
- **Limited data** (< 1K demos): never full fine-tune — you'll overfit.
  Start with action expert only.

---

## 1. Training loop anatomy

The `train()` function is a standard PyTorch loop with gradient
accumulation support:

```python
micro_step = 0
running_loss = 0.0

while step < cfg.training.max_steps:
    for batch in train_loader:
        batch = move_to_device(batch)

        with torch.autocast(dtype=bf16):
            out = model.compute_loss(batch)
            loss = out["loss"] / grad_accum_steps  # scale for accumulation

        scaler.scale(loss).backward()              # accumulate gradients
        micro_step += 1

        if micro_step % grad_accum_steps != 0:
            continue                               # keep accumulating

        scaler.unscale_(optimizer)
        clip_grad_norm(model.parameters(), 1.0)
        scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        step += 1
        if step % log_every: log()
        if step % eval_every: evaluate()
        if step % save_every: save_checkpoint()
```

### Gradient accumulation

When `grad_accum_steps > 1`, the loop accumulates gradients over multiple
micro-batches before stepping the optimiser.  The effective batch size is
`batch_size * grad_accum_steps`.

This is essential for memory-constrained hardware:

```yaml
# M3 config: batch_size=1 fits in memory, but effective batch=8
training:
  batch_size: 1
  grad_accum_steps: 8   # effective batch = 8
```

The loss is divided by `grad_accum_steps` before `.backward()` so that
the accumulated gradient is equivalent to a single large-batch gradient.

The meaningful decisions are in the details below.

---

## 2. Optimiser: AdamW with parameter groups

We use AdamW (Adam with decoupled weight decay) with **two parameter groups**:

```python
param_groups = [
    {"params": backbone_params,
     "lr": lr * backbone_lr_multiplier,
     "name": "backbone"},
    {"params": other_params,
     "lr": lr,
     "name": "action_expert"},
]
optimizer = AdamW(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
```

### Why two groups?

The backbone contains pretrained knowledge we want to preserve.  The
action expert is trained from scratch.  Using the same learning rate for
both means either:
- The action expert learns too slowly (if LR is low enough for the
  backbone), or
- The backbone forgets everything (if LR is high enough for the action
  expert)

The fix: give them different learning rates.  Default
`backbone_lr_multiplier = 0.1` means backbone LR is 10× lower than
action expert LR.  This matches standard multi-LR fine-tuning practice
for pretrained vision-language models.

### Why β₂ = 0.95 instead of 0.999?

Adam's second-moment EMA coefficient ($\beta_2$) determines how
quickly the running variance estimate adapts.  The standard default
$\beta_2 = 0.999$ assumes **many updates with similar gradient statistics**.
For large-scale LLM training, $\beta_2 = 0.95$ has become the norm because:
- Gradient statistics shift more often (rare tokens, curriculum effects)
- Shorter EMA adapts faster to those shifts
- Empirically improves stability for long runs

Flow matching training is similar in spirit — the distribution of `t`
values and action scales creates noisy gradient statistics.  $\beta_2 = 0.95$
gives more stable training than the default.

### Why AdamW and not Adam?

Adam applies weight decay by adding it to the gradient, which gets
rescaled by the adaptive learning rate.  This means the effective decay
depends on the gradient magnitude, which is not what you want — you want
decay proportional to the weight, period.

AdamW decouples weight decay from the gradient update:

$$
\theta_t = \theta_{t-1} - \eta \cdot (\text{Adam update}) - \eta \cdot \lambda \theta_{t-1}
$$

This is the pragmatic default for all modern transformer training and
the theoretical right thing to do.

---

## 3. Learning rate schedule

We use **cosine annealing with linear warmup**:

```
      LR
       ▲
       │
  peak ├─────╱───────────╮
       │    ╱             ╲╮
       │   ╱                ╲_____
       │  ╱                       ╲_
  min  ├─╯________________________╲___
       │                                ▶ step
       └──── warmup ──────── cosine ────
```

### Linear warmup (steps 0 → `warmup_steps`)

Starts at near-zero LR and ramps up linearly to the peak value.
Why: at step 0, Adam's moment estimates are completely unreliable
(initialised to zero), so full-LR updates are effectively random.
Warmup gives Adam time to calibrate its moment estimates before we
trust them with big steps.

Typical warmup: 500–2 000 steps for fine-tuning, 5 000–10 000 steps
for from-scratch training.

### Cosine annealing (steps `warmup_steps` → `max_steps`)

After warmup, LR follows a cosine curve down to `eta_min = 1e-7`:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
\left(1 + \cos\!\left(\pi \cdot \frac{t - t_{\text{warmup}}}{t_{\max} - t_{\text{warmup}}}\right)\right)
$$

Why cosine: you want to end training with a small LR to fine-tune weight
placement without introducing new noise.  Cosine gives a smooth, monotonic
decrease that's empirically better than step decay or linear decay for
transformer training.

### Putting it together

```python
warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                  total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer,
                           T_max=total_steps - warmup_steps,
                           eta_min=1e-7)
return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
```

`SequentialLR` chains them so step `warmup_steps` transitions from one
to the other automatically.

---

## 4. Mixed precision (bf16)

We use **bfloat16 autocast** during forward pass and loss computation,
with full-precision (`float32`) weights, gradients, and optimiser state:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model.compute_loss(batch)["loss"]
```

### Why bf16 over fp16

| | fp16 | bf16 |
|--|------|------|
| Dynamic range | $\pm 65{,}504$ | $\pm 3.4 \times 10^{38}$ |
| Precision | 10 mantissa bits | 7 mantissa bits |
| Loss scaling required | Yes | No |
| Supported on | Most GPUs | A100, H100, RTX 4090+ |

fp16 has a tiny dynamic range — gradients below $\sim 6 \times 10^{-5}$
underflow to zero.  That's why fp16 training needs loss scaling (multiply
the loss by a large constant before backward, unscale before optimiser
step) to push gradients into a representable range.

bf16 has the same dynamic range as fp32, so no loss scaling is needed.
The trade-off is lower precision (7 mantissa bits vs 10), but for
gradient-descent updates this is harmless.

We use bf16 on supported GPUs and fall back to fp32 elsewhere.

### The `GradScaler`

Despite using bf16, we still create a `GradScaler` — with
`enabled=(mixed_precision == "fp16")`.  When disabled, it's a no-op
wrapper that just calls `.backward()` and `.step()` normally.  This
lets the code path for fp16 and bf16 be identical.

---

## 5. Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips the **global gradient norm** across all parameters to 1.0.  If the
norm exceeds 1.0, all gradients are scaled down uniformly.

Why it matters: flow matching loss occasionally produces outlier
gradients, especially early in training when the model's velocity
predictions are far from target.  Without clipping, one bad minibatch
can blow up the optimiser state and wreck training.  Clipping is a cheap
insurance policy.

The order matters — you have to:
1. `scaler.scale(loss).backward()` (compute scaled grads)
2. `scaler.unscale_(optimizer)` (undo scaling so clipping sees true grads)
3. `clip_grad_norm_(params, 1.0)` (clip)
4. `scaler.step(optimizer)` (apply)

Getting the order wrong clips the scaled gradient, which clips at a
dramatically different effective norm.

---

## 6. Validation loop

```python
if step % eval_every_n_steps == 0:
    val_loss = evaluate(model, val_loader, device, ...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(output_dir / "best")
    model.train()
```

Runs `compute_loss` on a fixed number of validation batches and saves a
new "best" checkpoint if the loss improved.  Note we use the same
**teacher-forced** loss as training, not free-running inference, for
speed — running the full ODE integrator on every validation batch would
be too slow.

If you want to track the free-running inference error, do it in a
separate, less-frequent evaluation phase using `model.predict_action()`.
See `examples/04_evaluate.py` for a full implementation.

### Why track best val loss separately

The "best" checkpoint is usually what you deploy.  The "final" checkpoint
captures the end of training.  They diverge when the model starts
overfitting — which happens on small robot datasets surprisingly often.

We save both.  If you're tight on disk space, you can disable the
periodic checkpoints (`save_every_n_steps`) and keep only best + final.

---

## 7. Feature caching for fast training

**Scripts:**
- [`robots/metaworld/scripts/cache_features.py`](../robots/metaworld/scripts/cache_features.py)
- [`robots/metaworld/scripts/train_cached.py`](../robots/metaworld/scripts/train_cached.py)
- [`robots/metaworld/cached_dataset.py`](../robots/metaworld/cached_dataset.py)

When the backbone is frozen (action-expert-only training), it produces
the same features every time for the same input.  Running a 5B-param
forward pass on every training step is wasteful.  **Feature caching**
eliminates this bottleneck:

```
Standard training (slow):
  Image → [Backbone 5B, ~20s] → features → [Expert 100M, ~0.1s] → loss
                ↑ runs every step, output never changes

Cached training (fast):
  Step 1 (once):  Image → [Backbone] → save to disk    (~15 min total)
  Step 2 (train): Load .npz → [Expert] → loss           (~0.1s per step)
```

### Step 1: Cache backbone features

```bash
uv run python -m robots.metaworld.scripts.cache_features \
    --config robots/metaworld/configs/metaworld_push_m3.yaml \
    --data-dir data/metaworld_demos \
    --output-dir data/metaworld_features
```

This runs the backbone on every training sample once and saves per-sample
`.npz` files containing:
- `obs_features`: `[S, hidden_size]` — backbone output
- `state`: `[state_dim]`
- `actions`: `[horizon, action_dim]`

For 550 samples on an M3, this takes about 15 minutes.

### Step 2: Train on cached features

```bash
uv run python -m robots.metaworld.scripts.train_cached \
    --config robots/metaworld/configs/metaworld_push_m3.yaml \
    --cache-dir data/metaworld_features \
    --mlflow --mlflow-tracking-uri http://127.0.0.1:5000
```

The cached training script calls `model.compute_loss_cached(batch)`,
which skips the backbone forward pass and uses the pre-loaded features
directly.

### Performance comparison

| | Standard training | Cached training |
|---|---|---|
| Per-step compute | Backbone (5B) + Expert (100M) | Expert only (100M) |
| Time per step (M3 MPS) | ~20-30s | ~0.1s |
| 15K steps total | ~12 hours | ~5 minutes |
| Recache needed? | — | Only if images or instruction change |

### When NOT to cache

Feature caching only works when the backbone is frozen.  If you're using
LoRA or full fine-tuning, the backbone weights change during training,
so its outputs change too — you must use standard training.

---

## 8. Two-stage training (recommended recipe)

The suggested schedule for fine-tuning on a small dataset:

### Stage 1: action expert warm-up

- `cfg.freeze_backbone = True`
- `learning_rate = 5e-4`
- `max_steps = 2 000` – `5 000`
- `warmup_steps = 200`

Use feature caching for this stage (see section 7) — it's ~100x faster.

Goal: let the action expert learn the action distribution and the
coarse structure of flow matching.  With the backbone frozen, gradients
are small and fast.  This stage typically converges to val loss around
0.1–0.2 very quickly.

### Stage 2: joint fine-tuning (optional)

- `cfg.freeze_backbone = False` (LoRA still applied)
- `learning_rate = 2e-4`
- `backbone_lr_multiplier = 0.05`
- `max_steps = 10 000` – `100 000`
- `warmup_steps = 500`

This stage requires standard training (no caching), since the backbone
weights change.  Use `robots/metaworld/scripts/train.py` for this.

Goal: refine the language-action alignment using LoRA updates to the
backbone.  This is where the model learns task-specific grounding.
Expect val loss to drop from ~0.15 to ~0.04.

The MetaWorld training script (`robots/metaworld/scripts/train.py`)
implements exactly this workflow.

---

## 9. Memory breakdown (Gemma 4 E2B at bf16)

Understanding where memory goes helps you choose the right strategy:

| Component | Expert Only | LoRA + Expert | Full Fine-tune |
|-----------|-------------|---------------|----------------|
| Model weights | ~7 GB | ~7.5 GB | ~7 GB |
| Optimiser states | ~0.8 GB | ~4.9 GB | ~14 GB |
| Gradients | ~0.4 GB | ~2.5 GB | ~7 GB |
| Activations (bs=1) | ~3 GB | ~8 GB | ~12 GB |
| **Total** | **~11 GB** | **~23 GB** | **~40 GB** |
| Hardware | M3 / RTX 3060 | RTX 4090 / A10 | A100 40GB+ |

With feature caching + expert-only, memory drops further since the
backbone doesn't even need to be loaded during training (only during
the one-time caching step).

---

## 10. Distributed training

The current script uses single-process training.  To run on multiple GPUs
with DistributedDataParallel (DDP), the minimal changes are:

```python
# 1. Initialise process group
torch.distributed.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2. Wrap model
model = DDP(model, device_ids=[local_rank])

# 3. Distributed sampler for data
sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=sampler, ...)

# 4. Launch
# torchrun --nproc_per_node=4 -m gemma4_vla.train ...
```

The `train()` function is written to accept a pre-built model and
DataLoaders, so you can do the DDP setup outside and pass them in
without modifying the training loop itself.

---

## 11. Resuming from a checkpoint

There's no explicit "resume" logic in the current training script —
you'd have to:

1. `model = Gemma4VLA.from_pretrained(checkpoint_path)`
2. Optionally restore optimiser state (not saved by default; add
   `torch.save(optimizer.state_dict(), ...)` to `save_pretrained` if
   you need bit-exact resumption)
3. Pass the model to `train()` via the `model=` kwarg

For most use cases, resuming from just the model weights is fine — Adam's
moment estimates re-warm-up within a few dozen steps.  For strict
reproducibility you want the full optimiser + scheduler state.

---

## 12. Common training issues

### Loss stuck above 0.5

Most likely causes, in order of frequency:
- Actions not normalised (see [dataset doc](05_dataset.md) §8)
- Learning rate too low (try 10× higher for action expert)
- `num_heads` too high relative to `hidden_size` (head_dim becomes tiny,
  attention can't learn)

### Loss oscillates wildly

- Gradient clipping not applied (check the order of operations in §5)
- Batch size too small (< 4)
- Learning rate too high — drop by 3× and retry

### Backbone loss climbing while action expert loss drops

Your LoRA is too aggressive, or you forgot `freeze_backbone` in stage 1.
Reduce `backbone_lr_multiplier` by 3–10×.

### CUDA OOM

- Enable gradient checkpointing:
  `model.backbone.model.gradient_checkpointing_enable()`
- Reduce `batch_size` and increase `grad_accum_steps` to compensate
- Enable 4-bit quantisation via `bitsandbytes` (see `03_custom_robot.py`
  for an example)
- Lower `lora_rank` from 16 → 8

### MPS (Apple Silicon) OOM or too slow

- Use the M3 config: `metaworld_push_m3.yaml` (frozen backbone, bs=1,
  grad accumulation=8)
- Use feature caching (section 7) — avoids loading the 5B backbone
  during training entirely. **This is the recommended approach for M3.**
- Gradient checkpointing doesn't help when backbone is frozen (no
  gradients flow through it)

### Training too slow on CPU/MPS

The 5B backbone forward pass dominates training time.  If the backbone
is frozen, use feature caching (section 7) for a ~100x speedup.  This
is almost always the right choice for action-expert-only training on
consumer hardware.
