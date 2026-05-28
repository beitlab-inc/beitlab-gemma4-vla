# 09 — Training and evaluating with multiple cameras

**Modules:**
- [`robots/metaworld/scripts/collect_data.py`](../robots/metaworld/scripts/collect_data.py)
- [`robots/metaworld/dataset.py`](../robots/metaworld/dataset.py)
- [`robots/metaworld/env.py`](../robots/metaworld/env.py)
- [`robots/metaworld/scripts/test.py`](../robots/metaworld/scripts/test.py)
- [`src/gemma4_vla/config.py`](../src/gemma4_vla/config.py) — `VisionConfig`

How to take Gemma4VLA from "single camera" to "N cameras" without changing the
model code. The architecture is multi-camera from day one — the work is in
**how you record data, how you tell the config to use those views, and how you
feed them at inference time**.

> All MetaWorld configs shipped with the repo (`metaworld_push*.yaml`) default
> to `num_cameras: 1`. Multi-camera is a config + collection change, not a
> code change.

---

## 1. What "multi-camera" means in this codebase

The canonical batch-level image tensor is:

```python
batch["pixel_values"]   # FloatTensor [B, C, 3, H, W]
```

`C` is **the number of camera views per timestep**, set by
`cfg.vision.num_cameras`. The dataset stacks one image per configured camera
into the `C` axis at the **same** timestep, and the prompt contains one
`<image>` token per camera (see [05_dataset.md §5](05_dataset.md#5-the-prompt-template)).

This means:

- The backbone (Gemma 4 SigLIP2 vision tower) processes every camera as an
  independent image and concatenates the resulting feature tokens.
- The model does **not** assume any spatial relationship between cameras —
  views can be arbitrarily different (top, wrist, third-person, …).
- The order of cameras is **load-bearing**: image #0 in the prompt is image
  #0 at inference. Mismatching order between training and deployment will
  silently degrade the policy. See "Pitfalls" below.

---

## 2. Step 1 — Collect data with multiple cameras

`collect_data.py` accepts a comma-separated `--cameras` flag. When provided,
it overrides `--camera-name` and renders every configured camera at every
step.

```bash
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 \
    --cameras topview,corner,gripperPOV \
    --episodes 100 \
    --instruction "push the object to the goal" \
    --output-dir data/push_demos_multicam \
    --rerun-mode save \
    --rerun-path rerun/collect_multicam.rrd
```

`--rerun-mode save` is recommended for multi-camera collection — every view
is logged side-by-side under `collect/camera/<name>` in the resulting `.rrd`,
so you can sanity-check alignment, lighting, and gripper occlusion across
views before you commit to training on hours of data. Open it later with
`uv run rerun rerun/collect_multicam.rrd` (or stream live with
`--rerun-mode spawn`).

Each episode HDF5 file gets one image group per camera:

```
data/push_demos_multicam/episode_000000.hdf5
├── observation/
│   ├── images/
│   │   ├── topview/data       [T, H, W, 3]  uint8
│   │   ├── corner/data        [T, H, W, 3]  uint8
│   │   └── gripperPOV/data    [T, H, W, 3]  uint8
│   └── state                  [T, state_dim] float32
├── action                     [T, action_dim] float32
└── attrs:
    ├── language_instruction = "push the object to the goal"
    └── camera_names = '["topview","corner","gripperPOV"]'
```

The `camera_names` HDF5 attribute records the order used at collection time —
keep this in mind when you write the training config.

### A note on storage cost

Multi-camera data scales linearly: 3 cameras × 480×480×3 uint8 × 50 Hz ×
30 s ≈ **100 MB per episode**. Plan disk capacity accordingly; HDF5 chunking
+ gzip can cut this by 4–8× if storage is tight.

---

## 3. Step 2 — Configure training for multiple cameras

Two fields in `VisionConfig` control multi-camera behaviour:

```yaml
# robots/metaworld/configs/metaworld_push_multicam.yaml (new)
vision:
  num_cameras: 3
  camera_names: ["topview", "corner", "gripperPOV"]
  image_size: 224
```

- `num_cameras` — number of `<image>` tokens in the prompt and the size of
  the `C` axis in `pixel_values`.
- `camera_names` — **explicit** list of HDF5 keys to read, in the order the
  model will see them. **Always set this when collecting more than one
  camera.** If you leave it as `null`, the dataset falls back to
  *alphabetical order of HDF5 keys*, which is fragile (`corner` < `topview`
  but you may not want that).

The dataset validates the contract: every requested camera must exist in
the episode, and `len(camera_names) == num_cameras`. See
[`MetaWorldHDF5Dataset._resolve_cameras`](../robots/metaworld/dataset.py#L113-L129)
and [`config.py` validation](../src/gemma4_vla/config.py#L306-L309).

### Increasing `max_sequence_length`

Each additional camera adds image tokens to the prompt. Gemma 4's vision
tower produces ~256 tokens per image, so a 3-camera prompt is ~768 image
tokens plus your text. Make sure `backbone.max_sequence_length` has
headroom:

```yaml
backbone:
  max_sequence_length: 1024   # was 512 for 1 camera, bump for 3
```

If you don't, the language part of the prompt gets truncated and the model
loses task conditioning.

### Memory cost

Each extra camera adds one forward pass through the vision tower per
sample. The dominant cost is activation memory for those tokens during the
backbone forward — expect roughly:

| Cameras | Approx VRAM at bs=8, bf16 (Gemma 4 E2B + LoRA) |
|---------|-----------------------------------------------|
| 1       | ~12 GB |
| 2       | ~16 GB |
| 3       | ~20 GB |

If you blow the VRAM budget, lower `training.batch_size` and raise
`training.grad_accum_steps` to keep the effective batch the same (see
[06_training.md §1](06_training.md#gradient-accumulation)).

### Feature caching is still valid

[06_training.md §7](06_training.md#7-feature-caching-for-fast-training) works
identically with multi-camera data — the cache file just contains a larger
`obs_features` tensor (more camera tokens per sample). Recache from scratch
if you change `camera_names` or `num_cameras`.

---

## 4. Step 3 — Train

Once the YAML is set, training is identical to the single-camera flow:

```bash
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push_multicam.yaml \
    --data-dir data/push_demos_multicam \
    --metrics-path metrics/train_multicam.jsonl \
    --mlflow \
    --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-push-multicam \
    --mlflow-log-artifacts
```

Each batch element now contains an `[C, 3, H, W]` image tensor; the
collate function preserves the camera axis and the backbone handles the
rest (see [05_dataset.md §7](05_dataset.md#7-collate-function)).

The checkpoint saves `config.json` containing the camera setup, so
`PolicyRunner.from_pretrained(...)` automatically knows it needs 3 images
per observation at inference.

---

## 5. Step 4 — Evaluate with multiple cameras

> **Current limitation:** `MetaWorldMT1Wrapper`
> ([env.py](../robots/metaworld/env.py)) renders a **single** camera per
> `env.render()` call, and `scripts/test.py` passes a single image to
> `runner.predict(...)`. The shipped MetaWorld evaluation path is
> single-camera even when the trained model expects multiple views.

There are two ways to evaluate a multi-camera policy in MetaWorld:

### Option A — Extend `MetaWorldMT1Wrapper` (recommended)

Mirror what `collect_data.render_cameras` already does
([collect_data.py:111-120](../robots/metaworld/scripts/collect_data.py#L111-L120)):
mutate the renderer's `camera_id` and capture one frame per camera per step.

A minimal patch:

```python
# robots/metaworld/env.py (sketch)
class MetaWorldMT1Wrapper:
    def __init__(self, env_name, camera_names=("topview",), ...):
        self.env = gym.make("Meta-World/MT1", env_name=env_name,
                            render_mode="rgb_array",
                            camera_name=camera_names[0])
        self.camera_names = list(camera_names)
        mj_model = self.env.unwrapped.mujoco_renderer.model
        self._camera_ids = {n: mj_model.camera(n).id for n in self.camera_names}

    def _get_images(self):
        renderer = self.env.unwrapped.mujoco_renderer
        imgs = []
        for n in self.camera_names:
            renderer.camera_id = self._camera_ids[n]
            imgs.append(renderer.render(self.render_mode).astype(np.uint8))
        return imgs   # list of [H, W, 3] in the order of camera_names

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._get_images(), self._extract_state(obs), info

    def step(self, action):
        obs, reward, trunc, term, info = self.env.step(action)
        done = bool(trunc or term) or int(info.get("success", 0)) == 1
        return self._get_images(), self._extract_state(obs), reward, done, info
```

Then `test.py` builds the observation as:

```python
imgs, state, info = env.reset()    # imgs is a list
obs = {
    "images": imgs,                 # list of [H, W, 3] uint8 in camera_names order
    "state": state,
    "instruction": args.instruction,
}
actions = runner.predict(obs)
```

The order of `imgs` **must** match the `camera_names` used at training
time, or the policy will see a permuted observation it never trained on.

### Option B — Real-robot / custom env

If you're deploying on a real robot or a custom simulator, you already
control how cameras are captured. Just pass a list of `[H, W, 3] uint8`
arrays to `runner.predict({"images": [...], ...})` in the trained
`camera_names` order. No simulator changes needed.

---

## 6. End-to-end minimal recipe (3 cameras, MetaWorld push)

```bash
# 1. Collect 100 episodes with 3 views
uv run python -m robots.metaworld.scripts.collect_data \
    --env-name push-v3 \
    --cameras topview,corner,gripperPOV \
    --episodes 100 \
    --instruction "push the object to the goal" \
    --output-dir data/push_demos_multicam \
    --rerun-mode save \
    --rerun-path rerun/collect_multicam.rrd

# 2. Train with the multi-cam config
uv run python -m robots.metaworld.scripts.train \
    --config robots/metaworld/configs/metaworld_push_multicam.yaml \
    --data-dir data/push_demos_multicam \
    --mlflow --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-push-multicam \
    --mlflow-log-artifacts

# 3. Evaluate (after extending MetaWorldMT1Wrapper — see §5 Option A)
uv run python -m robots.metaworld.scripts.test \
    --checkpoint checkpoints/metaworld_push_multicam/best \
    --env-name push-v3 \
    --cameras topview,corner,gripperPOV \
    --episodes 10 \
    --device cuda \
    --rerun-mode save --rerun-path rerun/eval_multicam.rrd \
    --mlflow --mlflow-tracking-uri http://127.0.0.1:5001 \
    --mlflow-experiment gemma4-vla-eval --mlflow-log-artifacts
```

Step 3 currently requires the env / test.py extension from §5; it's a
~30-line change and a good candidate for a contributing PR.

---

## 7. Pitfalls

### Camera order mismatch (silent failure)

The dataset reads cameras alphabetically if `camera_names` is `null`. If
collection wrote `["topview", "corner", "gripperPOV"]` but the training
config leaves `camera_names: null`, the loader will read them as
`[corner, gripperPOV, topview]`. The model still trains — on a permuted
view it never saw during deployment.

**Always set `camera_names` explicitly in both the config and the
inference call.**

### Missing cameras in some episodes

If you mix datasets collected with different `--cameras` settings, episodes
that don't contain every name in `camera_names` will crash at
`_resolve_cameras`. Either re-collect to the common camera set or split
into separate training runs.

### Inference image list is the wrong length

`PolicyRunner.predict` expects `obs["images"]` to be a list of length
`vision.num_cameras`. Passing one image (or three when the model was
trained on one) raises an error at the backbone's image-token splice.

### Prompt truncation when adding cameras

Doubling cameras roughly doubles image tokens in the prompt. If
`backbone.max_sequence_length` is left at 512, the language tokens get
truncated and the policy degrades to "image-only" conditioning. See §3.

### Resource budget

Multi-camera scales **storage** (linear in cameras), **VRAM** (sub-linear
because the LoRA / expert stays the same size), and **per-step compute**
(linear in the vision-tower forward). Caching backbone features
([06_training.md §7](06_training.md#7-feature-caching-for-fast-training))
neutralises the per-step compute hit during training, but not the cache
build time.

---

## 8. Summary

| Stage | What changes |
|-------|--------------|
| Collect (`collect_data.py`) | Pass `--cameras a,b,c`. Each camera lands in `observation/images/<name>/data`. |
| Config (`vision`) | Set `num_cameras: N` and `camera_names: [...]` in the explicit order. Bump `backbone.max_sequence_length`. |
| Train (`train.py`) | Unchanged — dataset and model handle the `[B, C, 3, H, W]` axis automatically. |
| Evaluate in MetaWorld (`test.py`) | Currently single-camera in the shipped wrapper. Extend `MetaWorldMT1Wrapper` to render `camera_names` per step (§5 Option A) and pass `images=[...]` to the runner. |
| Real robot | Just pass `images: [...]` in the trained `camera_names` order. No code changes. |
