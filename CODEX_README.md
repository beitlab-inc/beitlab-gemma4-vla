# Codex README

This file is a working handoff for future agents operating on `gemma4_vla`.
It is intentionally practical: where the code lives, what is implemented,
which recent integration traps were patched, and how to make changes without
guessing.

## Project Purpose

`gemma4_vla` is a PyTorch implementation of a pi0-style
Vision-Language-Action model:

- Gemma 4 backbone for image + text observation encoding
- Separate action expert transformer for action-sequence denoising
- Conditional flow matching for training
- ODE integration for inference

The codebase is closer to a research scaffold than a production-ready policy
stack. The docs are ambitious; the implementation is partial and some paths
still need careful end-to-end validation against real Gemma 4 checkpoints.

## Repository Map

### Core model
- `src/gemma4_vla/config.py`
  Dataclass configuration and preset builders. `TrainingConfig` carries
  the `normalize_stats` flag.
- `src/gemma4_vla/model.py`
  Main model, backbone wrapper, checkpoint save/load helpers.
  `Gemma4Backbone.__init__` raises if the loaded model exposes no vision
  tower (no silent text-only fallback). `device_map="auto"` is disabled
  under `torchrun`. `Gemma4VLA.forward(batch)` delegates to `compute_loss`
  so DDP's grad-sync hooks fire. `save_pretrained` / `from_pretrained`
  round-trip `normalization.pt` when present.
- `src/gemma4_vla/action_expert.py`
  Standalone action expert transformer.
- `src/gemma4_vla/flow_matching.py`
  OT interpolation, loss, Euler and RK4 integrators.
- `src/gemma4_vla/dataset.py`
  Image transforms, RandomDemoDataset for smoke-testing, collate_fn.
- `src/gemma4_vla/inference.py`
  `PolicyRunner` inference wrapper and preprocessing path. Uses the same
  `apply_chat_template` path as the training dataset and auto-denormalises
  predicted actions when the checkpoint carries `normalization.pt`.
- `src/gemma4_vla/train.py`
  Training loop, CLI entry point, JSONL metrics, MLflow integration, and
  the DDP scaffold (`_is_distributed`, `_init_distributed`,
  `_wrap_loader_for_distributed`). Rank-0-only logging/saves.
- `src/gemma4_vla/stats.py`
  `DatasetStats` — per-dim state/action mean/std. `compute_from_loader`
  streams a single pass; `save`/`load` round-trip `normalization.pt`;
  `normalize_state` / `normalize_actions` / `denormalize_actions` apply.
- `src/gemma4_vla/machine_config.py`
  Auto-detect Jetson Thor / cloud GPU class. `apply_machine_defaults`
  reads `TrainingConfig` defaults at runtime instead of hard-coding
  sentinel values.

### Robot adapters (robots/)
- `robots/metaworld/env.py`
  `MetaWorldMT1Wrapper` — wraps MetaWorld MT1 tasks (MuJoCo + Gymnasium) into
  a simple `(image, state) → action → (image, state, reward, done)` loop.
  Supports tasks `push-v3`, `reach-v3`, `pick-place-v3` and 7 camera views.
- `robots/metaworld/dataset.py`
  `MetaWorldHDF5Dataset` — loads HDF5 episodes for training.
  `build_metaworld_dataloaders()` — splits episodes into train/val DataLoaders.
- `robots/metaworld/visualize.py`
  Standalone visualization script for MetaWorld environments with expert policies.
- `robots/metaworld/scripts/collect_data.py`
  Collects expert demonstrations using scripted policies. Saves per-episode HDF5.
- `robots/metaworld/scripts/train.py`
  MetaWorld-specific training entry point. Wires HDF5 dataset into core train loop.
- `robots/metaworld/scripts/test.py`
  Evaluates a checkpoint in MetaWorld. PolicyRunner + env loop. Videos/metrics/Rerun/MLflow.
- `robots/metaworld/scripts/inspect_dataset.py`
  Inspects HDF5 episode files. Prints shapes/dtypes/ranges, exports previews.
- `robots/metaworld/configs/metaworld_push.yaml`
  MetaWorld push-v3 preset (state_dim=39, action_dim=4, 1 camera).

### Observability
- `src/gemma4_vla/observability.py`
  `RerunLogger` — wraps rerun-sdk for real-time visualization (modes: spawn,
  save to .rrd, connect to remote viewer). Logs images, vectors, tensors,
  scalars, text.
  `MlflowRun` — wraps MLflow for experiment tracking (params, metrics,
  artifacts). No-op when disabled.

### Utilities
- `clean_outputs.py`
  Removes generated artifacts (videos, metrics, rerun, checkpoints, etc.).
  Dry-run by default; `--yes` to actually delete.

### Config presets
- `configs/base_config.yaml`
  Base configuration template.

### Docker
- `docker-compose.mlflow.yml`
  Postgres 16 + MLflow 3.12.0 server. Exposes UI on port 5001.
- `docker/mlflow/Dockerfile`
  MLflow container image.

### Other
- `tests/`
  Unit tests for action expert, flow matching, config, dataset, inference.
- `docs/`
  Technical deep-dives on each module.
- `examples/`
  Example Python scripts for quick start, fine-tuning, custom robots, eval.

## Current Architecture

High-level data flow:

1. Images + task text go through `Gemma4Backbone`.
2. Backbone hidden states are projected with `obs_proj`.
3. `ActionExpert` consumes:
   - projected observation features
   - padded proprioceptive state
   - noisy action chunk
   - scalar noise level embedding
4. Training uses `ot_flow_interpolate(...)` and `flow_matching_loss(...)`.
5. Inference integrates the learned velocity field with Euler or RK4.

Core public entry points:

- `Gemma4VLA(cfg)`
- `Gemma4VLA.compute_loss(batch)`
- `Gemma4VLA.predict_action(obs)`
- `PolicyRunner(model).predict(obs)`
- `PolicyRunner.from_pretrained(checkpoint_path)`
- `train(cfg)`
- `robots.metaworld.MetaWorldMT1Wrapper(env_name, camera_name=...)`
- `RerunLogger(mode=...).start()`
- `MlflowRun(enabled=True, ...).start(params)`

## MetaWorld Simulation Pipeline

The repo includes a full collect → train → eval pipeline for MetaWorld
robotic manipulation tasks.

### Data flow

```
Expert Policy (MetaWorld)
  ├── robots/metaworld/scripts/collect_data.py ──► data/push_demos/episode_*.hdf5
  │                                                  └── observation/images/<cam>
  │                                                  └── observation/state
  │                                                  └── action
  │                                                  └── attr: language_instruction
  │
  ├── robots/metaworld/scripts/train.py ──► checkpoints/metaworld_push/{best,final}/
  │     └── reads HDF5 via MetaWorldHDF5Dataset
  │     └── writes config.json + weights.pt
  │
  └── robots/metaworld/scripts/test.py ──► PolicyRunner + MetaWorldMT1Wrapper
        └── videos/{env}_ep001.mp4
        └── metrics/eval.json
        └── rerun/test.rrd
```

### Key dimensions for MetaWorld

| Task | state_dim | action_dim | Image size |
|------|-----------|------------|------------|
| push-v3 | 39 | 4 | 480x480 |
| reach-v3 | 39 | 4 | 480x480 |
| pick-place-v3 | 39 | 4 | 480x480 |

The model internally resizes images to 224x224 and zero-pads state/action
to `max_state_dim` for cross-embodiment compatibility.

### Action horizon bridging

Gemma4VLA predicts 50-step action horizons. MetaWorld consumes one action per
step. The test script takes `actions[0]` from each prediction. For temporal
action chunking, use `PolicyRunner.stream(obs, replan_every=N)`.

### Observability flags

All pipeline scripts accept the same observability flags:

```
--rerun-mode {off,spawn,save,connect}
--rerun-path <path.rrd>
--rerun-connect-url <grpc-url>
--rerun-log-every N

--mlflow
--mlflow-tracking-uri <uri>
--mlflow-experiment <name>
--mlflow-run-name <name>
--mlflow-log-artifacts
```

## Patched Integration Notes

These are recent fixes that future agents should preserve.

### 1. Checkpoint config artifacts are now JSON-first

Checkpoint save/load now writes `config.json` and keeps a compatibility
`config.pt` artifact that stores a plain dict rather than a pickled dataclass.

Why this matters:

- Modern PyTorch defaults `torch.load(...)` to `weights_only=True`.
- Pickled dataclass configs are brittle under that default.
- `load_config_artifact(...)` now supports:
  - `config.json`
  - dict-based `config.pt`
  - legacy pickled-dataclass `config.pt` via a compatibility fallback

Relevant code:

- `src/gemma4_vla/model.py`

### 2. `pixel_values` now has one canonical shape inside the repo

The internal contract is now:

- sample-level: `[num_cameras, 3, H, W]`
- batch-level: `[B, num_cameras, 3, H, W]`

`Gemma4Backbone` flattens the batch and camera axes only at the final boundary
before the backbone call. Avoid reintroducing mixed 4-D / 5-D behavior in
dataset or inference paths.

Relevant code:

- `src/gemma4_vla/dataset.py`
- `src/gemma4_vla/inference.py`
- `src/gemma4_vla/model.py`

### 3. YAML config loading now applies nested overrides

`train.py` now loads YAML through `Gemma4VLAConfig.from_dict(...)`, so nested
sections such as `vision`, `robot`, `backbone`, and `flow_matching` are all
applied instead of only `training`.

Relevant code:

- `src/gemma4_vla/config.py`
- `src/gemma4_vla/train.py`

## Lower-Risk Areas

These modules are relatively self-contained and are good places to work first:

- `flow_matching.py`
  Math utilities and integrators are isolated and easy to test.
- `action_expert.py`
  Self-contained transformer with decent unit coverage.
- `config.py`
  Safe place to improve validation and config loading behavior.
- `observability.py`
  RerunLogger and MlflowRun are thin wrappers with no-op defaults.
- `robots/metaworld/env.py`
  Environment wrapper is self-contained; only depends on gymnasium + metaworld.
- `clean_outputs.py` and `robots/metaworld/scripts/inspect_dataset.py`
  Utility scripts with no model dependencies.

## Higher-Risk Areas

Touch these carefully because the integration surface is wide:

- `model.py`
  Backbone loading, checkpointing, and inference all meet here.
- `inference.py`
  Raw user observations are converted here; shape mistakes propagate quickly.
- `dataset.py`
  Dataset schema, prompt formatting, and image tensor conventions all converge
  here.
- `robots/metaworld/scripts/test.py`
  Combines PolicyRunner, MetaWorldMT1Wrapper, normalization stats,
  observability, and video recording — many integration points.

## Recommended Work Order

If you are extending or fixing the repo, the safest sequence is:

1. Keep config serialization backward-compatible when touching checkpoint I/O.
2. Preserve the canonical `pixel_values` contract when changing image paths.
3. Extend config handling through `Gemma4VLAConfig.from_dict(...)` rather than
   reintroducing ad-hoc YAML merging.
4. Add smoke tests for any integration boundary you change.

## Local Development Notes

Useful commands from the repo root:

```bash
uv run python -m compileall src
uv run python -c "from gemma4_vla.config import metaworld_push_config; print(metaworld_push_config())"
uv run python -m pytest tests/ -x -q
```

## Editing Guidance For Future Agents

- Preserve the dataclass config API unless you are explicitly migrating it.
- Avoid changing architecture docs until the code path is verified; the docs
  are already ahead of the implementation in a few places.
- When changing tensor shapes, update:
  - `dataset.py`
  - `inference.py`
  - `model.py`
  - tests
  - any shape comments or docstrings
- If you touch checkpointing, validate both:
  - save then reload in the same environment
  - load from a fresh Python process

## First Files To Read

If you are starting cold, read in this order:

**Core model:**
1. `src/gemma4_vla/config.py` — dataclass config, preset builders
2. `src/gemma4_vla/model.py` — Gemma4VLA class, backbone, checkpoint I/O
3. `src/gemma4_vla/action_expert.py` — action denoising transformer
4. `src/gemma4_vla/inference.py` — PolicyRunner, preprocessing, streaming

**MetaWorld adapter:**
5. `robots/metaworld/env.py` — MetaWorld wrapper
6. `robots/metaworld/dataset.py` — HDF5 dataset adapter
7. `robots/metaworld/scripts/train.py` — MetaWorld training entry point
8. `robots/metaworld/scripts/test.py` — full evaluation loop

**Training and data:**
9. `src/gemma4_vla/dataset.py` — image transforms, collate_fn
10. `src/gemma4_vla/train.py` — training loop + observability
11. `src/gemma4_vla/observability.py` — Rerun + MLflow wrappers

That sequence gives the quickest path from architecture to the full
simulation pipeline.
