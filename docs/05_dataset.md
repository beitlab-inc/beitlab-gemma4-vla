# 05 — Datasets and Preprocessing

**Module:** [`src/gemma4_vla/dataset.py`](../src/gemma4_vla/dataset.py)

How raw robot data becomes training batches for Gemma4VLA.

---

## 1. What a single sample looks like

Every `__getitem__` call returns a dict:

```python
{
    "input_ids":      LongTensor  [T]           # text tokens
    "attention_mask": LongTensor  [T]           # text mask
    "pixel_values":   FloatTensor [C, 3, H, W]  # C cameras
    "state":          FloatTensor [state_dim]   # current proprioception
    "actions":        FloatTensor [horizon, action_dim]  # future chunk
}
```

After `collate_fn(...)`, the batch-level image tensor is:

```python
batch["pixel_values"]   # FloatTensor [B, C, 3, H, W]
```

That explicit camera axis is the canonical internal contract.  The backbone
flattens `[B, C, 3, H, W]` to `[B*C, 3, H, W]` only at the final boundary
before calling Gemma 4.

Each sample is a **(start_step, episode)** pair.  From that pair we extract:
- The current observation (images + state at `start_step`)
- The next `horizon` actions (`start_step` to `start_step + horizon`)
- The episode's language instruction (constant for a whole episode)

Note that one episode of length `T_ep` produces `T_ep - horizon` training
samples — one for each valid starting step.  This is critical for data
efficiency: a 30-second episode at 50 Hz (1500 frames) yields ~1450
samples, not 1.

---

## 2. Temporal action chunking

A "chunk" is a window of $H$ consecutive actions.  The model is trained
to predict all $H$ actions from a single observation.  This design is
different from classical imitation learning (which predicts one action
at a time) and is core to pi0's approach.

### Why predict a chunk?

Three reasons:

1. **Smoothness**: predicting a single action independently at each
   timestep causes jittery motion.  Chunk prediction lets the network
   enforce temporal coherence directly.

2. **Compute amortisation**: the backbone (Gemma 4) is expensive.  If we
   ran it at 50 Hz, inference would dominate the robot's control loop.
   Running it once per chunk and executing 25–50 actions from that one
   forward pass cuts inference cost by 25–50×.

3. **Modelling capacity**: predicting 50 correlated actions jointly is a
   much richer target than 50 independent actions.  Flow matching with
   a single joint distribution can capture the multi-modal structure
   (e.g. "I could pick with the left or right hand"), which independent
   prediction cannot.

### The horizon-length trade-off

| Horizon | Latency | Coherence | Reactivity |
|---------|---------|-----------|-----------|
| 1 | Very high | Poor | Best |
| 10 | High | OK | Good |
| **50** | **Low** | **Good** | **OK** |
| 100 | Very low | Excellent | Poor |

- Too short → backbone runs too often → latency spikes, poor smoothness
- Too long → robot can't react to new events → outdated plans

$H = 50$ at 50 Hz = 1 second of action, which matches pi0's choice.  Most
dexterous tasks happen on sub-second timescales, so this is short enough
to react but long enough to plan a coherent motion.

---

## 3. Image preprocessing

Every image goes through a torchvision pipeline:

**Training** (with augmentation):
```python
[
    Resize(224, 224),
    RandomResizedCrop(224, scale=(0.9, 1.0)),    # slight zoom augmentation
    ColorJitter(brightness=0.3, contrast=0.3,     # robust to lighting
                saturation=0.2, hue=0.05),
    ToTensor(),                                    # → [0, 1]
    Normalize(mean=[0.5]*3, std=[0.5]*3),         # → [-1, 1]
]
```

**Validation** (no augmentation):
```python
[
    Resize(224, 224),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3),
]
```

### Why `[-1, 1]` normalisation?

Gemma 4's vision tower (SigLIP2) was trained on images normalised to
`[-1, 1]` rather than ImageNet statistics.  Using the wrong normalisation
systematically shifts the distribution of input activations and degrades
performance noticeably (~5–10% on vision-language tasks).

If you ever see a "vision encoder's features look degenerate" warning,
double-check that `Normalize(mean=[0.5]*3, std=[0.5]*3)` is still what
the upstream vision encoder expects — sometimes new model releases change
this.

### Why these specific augmentations?

- **RandomResizedCrop with small scale range** (`0.9 – 1.0`) adds enough
  jitter to prevent overfitting to specific camera positions without
  destroying alignment between action and perception.
- **ColorJitter** simulates lighting variation between training and
  deployment.  Brightness is the most impactful axis; hue is set low
  because large hue shifts can change the *meaning* of colour instructions
  ("pick the red cube").
- **No flips** — robotic actions are not symmetric under horizontal flips
  (left/right matters).  A flip augmentation would require also flipping
  the action commands, which is robot-specific.

---

## 4. Robot-specific dataset adapters

Each robot adapter under `robots/<name>/` provides its own dataset class.
For MetaWorld, see `robots/metaworld/dataset.py` (`MetaWorldHDF5Dataset`).

Dataset adapters read per-episode HDF5 files.

### Directory layout

```
data_root/
  episode_000000.hdf5
  episode_000001.hdf5
  ...
```

Each HDF5 file contains:

```
observation/
  images/
    <camera_name>   [T, H, W, 3]   uint8
    ...
  state             [T, state_dim] float32
action              [T, action_dim] float32
language_instruction  str (attribute or dataset)
```

### How it builds the index

```python
for ep_idx, ep_len in enumerate(self.episode_lengths):
    for step in range(ep_len - self.horizon):
        self.index.append((ep_idx, step))
```

The flat `self.index` list has one entry per valid (episode, start_step)
pair.  `__len__` returns its length.  `__getitem__(i)` looks up the
corresponding episode + step and reads only that slice.

### Why we load HDF5 on every `__getitem__`

H5 files support random access, so reading one window is cheap.  We open
and close the file inside the method (no persistent handle) because:
- PyTorch DataLoaders use `multiprocessing`, and HDF5 handles don't
  survive process forks reliably.
- Holding thousands of open file descriptors for thousands of episodes
  is a great way to hit per-process file-descriptor limits.

If you have the RAM, you can trivially cache entire episodes in the
constructor — it'll make `__getitem__` much faster at the cost of memory.

### Validation split

```python
n_val = max(1, int(len(all_episodes) * val_fraction))
if split == "train":
    self.episodes = all_episodes[n_val:]
else:
    self.episodes = all_episodes[:n_val]
```

We split **by episode**, not by frame.  Splitting by frame would allow
the training and validation sets to contain consecutive frames from the
same episode, which leaks information.

A ~10% validation fraction is usually enough for robot datasets, which
tend to be small (thousands of episodes).  Going lower can cause noisy
validation curves; going higher wastes precious training data.

---

## 5. The prompt template

We assemble the language prompt as:

```
<image>
<image>
Task: Pick up the red cube.
Execute the task by predicting robot actions.
```

One `<image>` token per camera, followed by the task description.  The
number of `<image>` tokens tells the Gemma 4 processor how many image
embeddings to splice into the sequence.

The "Execute the task …" suffix is a mild prompt-engineering hack.  Gemma 4
is an *instruction-tuned* model, so it responds better to explicit task
framing than to a bare instruction.  In practice the difference is small,
but it's free and consistent.

---

## 6. RandomDemoDataset

This is a synthetic dataset that produces random tensors of the correct
shape.  It serves three purposes:

1. **Testing**: lets you run unit tests without real robot data.
2. **Benchmarking**: you can measure training throughput on a new machine
   in minutes, without downloading anything.
3. **Debugging**: if a batch through `RandomDemoDataset` crashes your
   model, the bug is in the model, not the data pipeline.

It's deliberately very small and has no dependencies beyond PyTorch.

---

## 7. Collate function

```python
def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        collated[k] = torch.stack(tensors, dim=0)
    return collated
```

The default PyTorch `default_collate` would also work for our case, but
we write our own to make the stacking rule explicit.  Every sample in a
batch must have the exact same shape for this to work — which is true
for us because:
- Images are resized to a fixed size
- State and action dims are determined by `RobotConfig`
- Text is padded to `max_sequence_length`

If any of those fail, `torch.stack` throws a clear shape mismatch error
pointing at the offending key.

For images, this means `collate_fn` preserves the camera axis and produces
`[B, C, 3, H, W]`, which keeps single- and multi-camera paths consistent.

---

## 8. Known pitfalls

### Action scale

The single most common bug.  Your robot probably reports joint angles in
radians (`[-π, π]`) or motor positions in some motor-specific unit.  Flow
matching expects actions roughly on the `[-1, 1]` range because:
- The noise distribution is $\mathcal{N}(0, 1)$
- The interpolation `(1-t)*noise + t*action` only makes sense if they're
  on comparable scales

**Always normalise your actions** before feeding them to the dataset.
A simple min-max scaling per joint, computed over the training set and
saved to the config, is usually enough.  Denormalise in
`inference.PolicyRunner` via `action_scale`.

### Camera order

The dataset reads cameras in alphabetical order of their HDF5 key names.
If your prompt template assumes a specific order (e.g. "first image is
the overhead view"), set `camera_names=["top", "wrist"]` explicitly when
constructing the dataset.

Otherwise, your model might learn that the "first image" is the wrist
camera during training, then fail at deployment when you pass the top
camera first.

### Horizon longer than episode

`self.index` only includes start steps where `step + horizon <= ep_len`.
Episodes shorter than the horizon are skipped entirely, which can silently
drop data if your episodes are very short.  Check `len(self.index)`
against `sum(episode_lengths)` to spot this.

### Language instruction present?

If `language_instruction` is missing from the HDF5, we fall back to the
generic `"Follow the task instruction."`.  A missing-instruction dataset
will train a model that ignores language altogether — check your data
if you notice no language conditioning in the model's behavior.

---

## 9. Adapting to a new data format

To support a new format (e.g. RLDS, raw MP4 files, your own logger),
write a subclass of `torch.utils.data.Dataset` with:

1. `__len__` returning the total number of training samples
2. `__getitem__(i)` returning a dict with the 5 keys above
3. Image tensors in `[-1, 1]` with shape `[C, 3, H, W]`
4. State/actions as `float32` in a reasonable range

That's it.  Everything else in the pipeline (collate, DataLoader, training)
is format-agnostic.
