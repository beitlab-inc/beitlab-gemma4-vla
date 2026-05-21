# 07 — Inference Pipeline

**Module:** [`src/gemma4_vla/inference.py`](../src/gemma4_vla/inference.py)

How a trained Gemma4VLA model is used for real-time robot control.

---

## 1. The inference API

The `PolicyRunner` class wraps a `Gemma4VLA` model and adds:

- Observation preprocessing (numpy images → normalised tensors)
- Prompt templating
- Temporal action chunking buffer
- Latency-friendly inference defaults

Usage:

```python
runner = PolicyRunner.from_pretrained("checkpoints/best", device="cuda")

# One-shot: get an action chunk
actions = runner.predict(obs)                 # [horizon, action_dim]

# Streaming: yield actions one at a time, replanning periodically
for action in runner.stream(obs, replan_every=25):
    robot.apply(action)
```

`predict` is for offline use or when you control replanning externally.
`stream` is the recommended online path.

---

## 2. The control-loop problem

A real robot runs a control loop at some fixed rate (often 50 Hz for
dexterous tasks, 10–20 Hz for mobile manipulation).  At each tick, the
loop needs a fresh action to send to the motors.

Naïve approach: run the full model at every tick.

```
tick 0: encode obs → run backbone → run ODE → execute action[0]
tick 1: encode obs → run backbone → run ODE → execute action[0]   (next obs)
tick 2: ...
```

Problem: the backbone forward pass takes ~15–30 ms on an A100 with
Gemma 4 E2B.  That leaves no budget for anything else, and it barely
fits the 20 ms budget at 50 Hz.  On a smaller GPU, it doesn't fit at all.

---

## 3. Temporal action chunking (the solution)

**Key idea**: predict a chunk of $H$ actions every $k$ ticks, not every tick.

```
tick 0:  encode obs → run backbone → run ODE → buffer = [a₀..a₄₉]
tick 1:  execute a₀          (buffer pop)
tick 2:  execute a₁
...
tick 24: execute a₂₃
tick 25: encode obs → run backbone → run ODE → buffer = [a₂₅..a₇₄]
tick 26: execute a₂₅
...
```

With `replan_every = 25`, we run the backbone once every 500 ms instead
of once every 20 ms.  Effective inference cost per tick drops by 25×.

### Choosing the replan rate

| `replan_every` | Reactivity | Throughput |
|----------------|-----------|-----------|
| 1 | Max | Min (backbone every tick) |
| $H/4$ | Good | 4× amortisation |
| $H/2$ | OK | 2× amortisation |
| $H$ | Poor | Max (backbone once per chunk) |

We default `replan_every = H` for maximum throughput, but recommend
$H/2$ for most tasks — it halves the "stale plan" window without much
extra cost.

### What if the robot needs to react to something unexpected?

Out-of-chunk events (e.g. the human moves an object) aren't caught until
the next replan.  Two options:

- **Shorten `replan_every`** (costs compute)
- **External interrupt**: keep a separate "surprise detector" and call
  `runner.stream(new_obs, replan_every=...)` from scratch when the
  detector fires.  This forces an immediate replan at the cost of
  discarding the current buffer.

For most tasks the default is fine because the robot re-plans fast enough
that "stale" chunks rarely cause problems.

---

## 4. Observation preprocessing

`_preprocess` converts raw sensor data into model-ready tensors:

```python
def _preprocess(self, obs):
    # 1. Pad / trim camera count
    while len(images) < num_cameras: images.append(images[-1])
    images = images[:num_cameras]

    # 2. Resize + normalise each image to [-1, 1]
    pixel_values = torch.stack([preprocess_image(im, size) for im in images])
    pixel_values = pixel_values.unsqueeze(0)   # [1, C, 3, H, W]

    # 3. Build prompt template
    prompt = "<image>\n" * num_cameras + f"Task: {instruction}"

    # 4. Tokenise
    encoding = processor(text=prompt, return_tensors="pt",
                         padding="max_length", max_length=max_seq_len)

    # 5. State to tensor
    state = torch.tensor(obs["state"]).unsqueeze(0)

    return { ... }
```

The internal image-tensor contract during inference is therefore:

- single observation: `[1, num_cameras, 3, H, W]`
- batched training data: `[B, num_cameras, 3, H, W]`

The backbone flattens the batch and camera axes internally right before the
Gemma 4 forward call.

### Why pad camera count?

Your robot might have 1 camera during debugging and 3 in production.
Training was done with a specific number (`cfg.vision.num_cameras`).
We pad by replicating the last image so the tensor shape matches what
the model expects.

This is a simplistic handling — a smarter approach would mask the
duplicated images so the model ignores them.  For now, the duplication
approach works because Gemma 4 is robust to repeated inputs.

### Why `[-1, 1]` normalisation again?

Same reason as the dataset doc: Gemma 4's vision tower was trained with
this normalisation.  Using the wrong one degrades features by 5–10%.
`PolicyRunner` uses its own transform to guarantee consistency even if
somebody modifies the dataset's transform.

---

## 5. Action post-processing

After the model returns a raw action chunk:

```python
actions = model.predict_action(...)[0].cpu().numpy()
actions = actions * self._action_scale
```

### `action_scale`

A scalar from `RobotConfig.action_scale` that maps model output (in
roughly $[-1, 1]$) to actuator commands (in whatever units your robot
uses).

Simple example: if your robot uses joint positions in radians on
$[-\pi, \pi]$, set `action_scale = π` to rescale.  For more complex
per-joint ranges, store a vector in `action_scale` and multiply
element-wise in the runner.

### Missing: action clipping

We don't currently clip actions to actuator limits.  If you're plugging
this into a real robot, add:

```python
actions = np.clip(actions, self.action_min, self.action_max)
```

before executing.  Adjust `action_min` / `action_max` per joint.  A model
trained on clean data is unlikely to predict wildly out-of-range actions
but it's not impossible — better to enforce at the robot side than trust
the model.

---

## 6. Streaming implementation

```python
def stream(self, obs, replan_every=None):
    if replan_every is None:
        replan_every = self._horizon

    step = 0
    while True:
        if step % replan_every == 0:
            actions = self.predict(obs)
            self._action_buffer = list(actions)

        if not self._action_buffer:
            break

        yield self._action_buffer.pop(0)
        step += 1
```

A generator with a simple buffer:

1. When `step % replan_every == 0`, refill the buffer with a fresh chunk.
2. Otherwise, pop the next action from the buffer and yield it.
3. Terminate when the buffer is empty (shouldn't happen in normal use).

**Important**: the caller is expected to **update `obs`** between steps.
This module doesn't know what your camera or robot looks like, so it
can't re-read sensors automatically.  Typical usage:

```python
for step, action in enumerate(runner.stream(obs, replan_every=25)):
    robot.apply(action)
    obs = {
        "images":      [camera.capture() for camera in cameras],
        "state":       robot.get_state(),
        "instruction": obs["instruction"],       # usually constant
    }
```

On replan ticks, the next iteration will use the updated `obs`.

---

## 7. Inference speed breakdown

For Gemma 4 E2B on an A100 with `num_inference_steps = 10`:

| Phase | Time |
|-------|------|
| Image preprocessing (CPU) | ~3 ms |
| Tokenisation | ~1 ms |
| Backbone forward | ~15 ms |
| Action expert × 10 | ~5 ms |
| Post-processing | ~1 ms |
| **Total** | **~25 ms (40 Hz)** |

Per-phase optimisation targets:

- **Backbone** is the bottleneck at ~60% of total time.  Options:
  - 4-bit quantisation via bitsandbytes (~40% speedup)
  - Smaller model (1B or 2B-it instead of 4B+)
  - KV cache (not implemented — the backbone runs on a fresh prompt
    each replan, so there's no cache reuse by default)
- **Action expert** is already fast.  Reducing `num_inference_steps`
  from 10 to 5 gives an extra 2.5 ms for minimal quality loss.
- **Image preprocessing** is CPU-bound and can become significant on
  ARM/embedded systems.  Moving to GPU-side preprocessing
  (`torchvision.transforms.v2` + `to("cuda")` early) helps.

---

## 8. Benchmarking utility

`benchmark_inference()` runs a warm-up, then times repeated calls to
`PolicyRunner.predict`:

```python
times = []
for _ in range(num_runs):
    t0 = time.perf_counter()
    runner.predict(dummy_obs)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

return {"mean_ms": np.mean(times), "hz": 1000 / np.mean(times), ...}
```

The `torch.cuda.synchronize()` call is essential — CUDA kernels are
asynchronous, so `time.perf_counter()` alone would measure only the
dispatch time, not the actual execution time.

See [`examples/04_evaluate.py`](../examples/04_evaluate.py) for the full
benchmarking workflow and reporting format.

---

## 9. Batch inference

`PolicyRunner.predict` takes a single observation by default but the
underlying model accepts batches.  For offline trajectory generation
(e.g. evaluating many initial states in parallel), you can construct
batched tensors manually and call `model.predict_action` directly:

```python
batched_obs = {
    "input_ids":      torch.stack([...]),  # [B, T]
    "attention_mask": torch.stack([...]),
    "pixel_values":   torch.stack([...]),
    "state":          torch.stack([...]),
}
batched_actions = model.predict_action(batched_obs, num_steps=10)
```

Benchmark first — on small models, the GPU is often under-utilised
with batch size 1, and increasing batch size doesn't cost much.

---

## 10. Production-quality extras (not in the current code)

Things you'll want to add before deploying to a real robot:

- **Action clipping** to actuator limits (see §5)
- **Action smoothing** (low-pass filter or exponential moving average)
  to kill high-frequency jitter from the flow matching output
- **Watchdog** that kills the policy if inference takes >2× the expected
  latency (avoid runaway robots on GPU hangs)
- **Observation staleness check** to refuse to act on old sensor data
- **Emergency-stop handling** that gracefully unwinds the action buffer
- **Per-joint velocity/acceleration limits** enforced at the controller
  (this should live in the robot's low-level driver, not in the policy)

None of these are about the model — they're engineering requirements for
real-world deployment.  Keep them out of `PolicyRunner` and in your
robot integration layer so the policy stays testable in isolation.
