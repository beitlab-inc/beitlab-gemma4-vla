# 01 — Configuration System

**Module:** [`src/gemma4_vla/config.py`](../src/gemma4_vla/config.py)

---

## Purpose

Centralise every tunable hyperparameter in a single hierarchical dataclass
tree.  Separating config from code means:

1. **Reproducibility** — a checkpoint stores its config, so you can rebuild
   the exact model that produced a given set of weights.
2. **Experiment sweeping** — you can run 10 variants without touching
   model code, just by mutating fields.
3. **Type safety** — dataclasses give you IDE autocomplete, mypy checks,
   and failure at construction time instead of at runtime in `model.py`.

---

## Structure

Config is a tree with six leaves, composed under one root:

```
Gemma4VLAConfig
 ├── vision          : VisionConfig         (cameras, image size)
 ├── backbone        : BackboneConfig       (Gemma 4 model ID, LoRA)
 ├── action_expert   : ActionExpertConfig   (depth, heads, hidden size)
 ├── flow_matching   : FlowMatchingConfig   (horizon, σ_min, steps)
 ├── robot           : RobotConfig          (DOF, max_dim)
 └── training        : TrainingConfig       (LR, batch, schedules)
```

Each leaf is a plain `@dataclass` with default values.  No inheritance,
no Hydra magic, no YAML parsing at import time.  When you do want YAML,
the supported path is:

- load it with [`PyYAML`](https://pyyaml.org/)
- pass the nested dict through `Gemma4VLAConfig.from_dict(...)`
- or let `train.main()` do that for you via `--config config.yaml`

That keeps the runtime config API dataclass-first while still allowing
file-based experiment configs from the CLI.

---

## Why dataclasses and not a dict

A dict has zero guarantees — you can misspell `learning_rate` as
`learning-rate` and Python won't notice until your first epoch of training
uses the default LR instead of yours.

Dataclasses fail loudly on typos:

```python
cfg = TrainingConfig(learing_rate=1e-4)   # TypeError: unexpected keyword
```

They also make it trivial to write `cfg.training.learning_rate` and have
type checkers verify you wrote it correctly.

---

## Cross-field validation

The `__post_init__` hook on `Gemma4VLAConfig` runs after construction and
checks invariants that can't be expressed via default values alone, e.g.:

```python
assert ae.hidden_size % ae.num_heads == 0
```

If you add new config fields with inter-dependencies, add the check here —
finding a misconfigured run at import time is much cheaper than debugging
it 3 000 training steps later.

`Gemma4VLAConfig.from_dict(...)` re-runs `__post_init__` after applying
overrides, so YAML- or dict-driven config mutations still get the same
validation as direct construction.

---

## Preset factory functions

Certain configurations come up often enough to deserve named constructors:

| Factory | Robot | Cameras | Backbone |
|---------|-------|---------|----------|
| `metaworld_push_config()` | MetaWorld push-v3 | 1 | Gemma 4 E2B |

These are helpers, not a type hierarchy — they just return a mutated
`Gemma4VLAConfig`.  You can further tweak the returned object:

```python
cfg = metaworld_push_config()
cfg.training.max_steps = 200_000
cfg.training.batch_size = 64
```

---

## Why the `RobotConfig.max_state_dim` field exists

Cross-embodiment training requires every robot's state/action vector to be
the same shape.  `max_state_dim` is the padding target — set it large
enough to cover the highest-dimensional robot you plan to train on (e.g.
39 for MetaWorld's observation space).

Raising `max_state_dim` is safe but wastes a small amount of compute on
the padding dimensions.  Lowering it breaks compatibility with any
pre-trained checkpoints that used the larger value, because the action
expert's input/output projection weights become shape-incompatible.

---

## Why LR is split into two groups

`TrainingConfig.backbone_lr_multiplier = 0.1` means the Gemma 4 backbone
trains at 10× lower learning rate than the action expert.  This matches
standard practice for fine-tuning pretrained transformers:

- The backbone starts near a good solution (internet pretraining).  Large
  updates would destroy that knowledge.
- The action expert starts from scratch and needs to move a lot.

You will want **larger** multipliers (closer to 1.0) when:
- You have millions of robot episodes (fine-tune more aggressively)
- You're training from random Gemma 4 weights (never recommended)

And **smaller** multipliers (0.01 or even 0.0) when:
- You have very little data and want LoRA-only updates
- You see backbone loss increasing while action expert loss decreases
  (a sign the backbone is being damaged)

---

## Adding your own config field

Say you want to add a reward-to-go auxiliary loss.  The steps are:

```python
# 1. In action_expert or training config:
@dataclass
class TrainingConfig:
    ...
    rtg_loss_weight: float = 0.0

# 2. Access it in train.py:
if cfg.training.rtg_loss_weight > 0:
    loss = loss + cfg.training.rtg_loss_weight * rtg_loss
```

That's it.  No registration step, no global state, no decorators.
