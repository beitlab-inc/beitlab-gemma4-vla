# Architecture Overview

This document gives a complete picture of how Gemma4VLA works end-to-end,
from raw observations to executed motor commands.

---

## 1. The VLA problem

A **Vision-Language-Action** model learns a policy:

```
π : (images, language, state)  →  action sequence
```

Unlike classical imitation learning, we want *one* policy that works across
many tasks and many robots, conditioned on natural-language instructions.
The research insight that made this tractable is:

> Pretrained vision-language models already know an enormous amount about
> how objects look, how language describes them, and how the physical world
> is organised.  We should reuse that knowledge and add only a thin layer
> of robot-specific machinery on top.

Gemma4VLA implements this idea with two transformers that talk to each other:

```
      observations                          actions
 (images, language, state)                (50-step chunks)
           │                                     ▲
           ▼                                     │
┌──────────────────────┐              ┌──────────────────────┐
│   Gemma 4 backbone   │ ──features──▶│    Action Expert     │
│   (2B – 27B params)  │              │     (~300M params)   │
│   frozen / LoRA      │              │    trained fully     │
└──────────────────────┘              └──────────────────────┘
      "what is here?"                   "what should I do?"
      "what is asked?"                  "where should I move?"
```

---

## 2. Information flow

### Training forward pass

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  1. Observation Encoding                                           │
│     ─────────────────────                                          │
│     images   ──┐                                                   │
│                ├──►  Gemma 4  ──►  obs_features [B, S_obs, D_b]   │
│     tokens   ──┘                                                   │
│                                                                    │
│  2. Action Corruption                                              │
│     ───────────────────                                            │
│     clean_actions  x_1                                             │
│     noise          x_0 ~ N(0, I)                                   │
│     noise level    t   ~ U(0, 1)                                   │
│                                                                    │
│     x_t = (1 − (1−σ)t)·x_0 + t·x_1        ← OT interpolant        │
│     u   = x_1 − (1−σ)·x_0                  ← target velocity      │
│                                                                    │
│  3. Velocity Prediction                                            │
│     ────────────────────                                           │
│     state_token   ─┐                                               │
│     action_tokens ─┤──► Action Expert ──► v̂ = velocity           │
│     noise_emb(t)  ─┘         ▲                                     │
│                              │ cross-attention                     │
│                         obs_features                               │
│                                                                    │
│  4. Loss                                                           │
│     ────                                                           │
│     L = ||v̂ − u||²                                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Inference forward pass

Inference is ODE integration of the learned velocity field:

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  1. Encode observation   (ONCE — reused across all denoising steps)│
│     obs_features = Gemma4(images, language)                        │
│                                                                    │
│  2. Initialise           x ← random noise                          │
│                                                                    │
│  3. Integrate ODE        for i in 0..num_steps:                    │
│                              t = i / num_steps                     │
│                              v = ActionExpert(x, state, obs, t)    │
│                              x = x + v · dt                        │
│                                                                    │
│  4. Return               clean_action_chunk = x                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The key efficiency trick is that **the backbone runs only once** per action
chunk.  The action expert runs `num_inference_steps` times (typically 10),
but it's 10× smaller, so total latency is dominated by a single Gemma 4
forward pass plus 10 cheap action expert passes.

---

## 3. Why two transformers instead of one?

You might reasonably ask: why not just add a few extra tokens to Gemma 4
and have it produce actions directly?

Three reasons:

### 3.1 Protect the VLM's pre-training

If the backbone had to produce velocity predictions directly, gradients from
the robot loss would flow through every parameter every step.  On a small
robot dataset (tens of thousands of episodes), those gradients would
catastrophically overwrite the internet-scale knowledge baked into Gemma 4.

By isolating the action-specific computation in a separate transformer with
separate weights, we keep most of the backbone frozen (or LoRA-adapted) and
let the action expert absorb the high-variance robot-specific gradients.

### 3.2 Match the information structure

The two halves of the problem have very different structure:

| | Observation | Action |
|--|------------|--------|
| Source | Pixels, text | Joint angles |
| Cardinality | ~1000 tokens | 50 tokens |
| Temporal structure | Static snapshot | Sequential trajectory |
| Output type | Dense features | Continuous velocity |
| Pretraining benefit | Huge | None |

A single unified transformer would have to learn both simultaneously.
Splitting them lets each module use an architecture suited to its job.

### 3.3 Decouple inference steps

During inference, the backbone runs once but the action expert runs
`num_inference_steps` times.  If they were fused, we'd pay for the
(expensive) backbone on every denoising step.  The split gives us a
~10× speedup for free.

---

## 4. Observation encoding details

Gemma 4 processes a prompt that interleaves image tokens and language tokens:

```
<image><image>Task: Pick up the red cube.
```

Internally:
- Each image is chunked into 16×16 patches by the SigLIP2 vision tower
- Patches are projected to Gemma 4's hidden size and prepended to the text
- The full sequence goes through 18–46 transformer layers (depending on size)

The output we care about is the **last-layer hidden states** for every token
in the sequence.  We don't do any generation — we treat Gemma 4 as a fixed
feature extractor that happens to understand language and images jointly.

The hidden states are then linearly projected to the action expert's hidden
size (usually smaller) by `model.obs_proj`.  This is the single learnable
bridge between the two transformers.

---

## 5. Action representation: why flow matching?

We need to model `p(action_chunk | obs, language)`, a continuous distribution
over 50×6 ≈ 300 real numbers.  The main options are:

| Approach | Pros | Cons |
|----------|------|------|
| Discrete tokenisation | Simple, reuses LLM | Quantisation error, complex inference |
| Gaussian MLE | Fast, simple | Unimodal, poor for multi-modal tasks |
| Diffusion | Multi-modal, flexible | Slow inference (50–100 steps) |
| **Flow matching (OT)** | **Multi-modal, fast (5–10 steps)** | Newer, less tooling |

Flow matching with OT paths is essentially "diffusion with straight-line
interpolants and an MSE loss".  It trains more stably than diffusion and
samples in far fewer steps.  The original pi0 paper showed it was the key
to real-time dexterous control at 50 Hz.

For the full derivation, see [02_flow_matching.md](02_flow_matching.md).

---

## 6. Cross-embodiment: one model, many robots

Different robots have different DOF.  Naïvely you'd need a separate model
per robot.

The pi0 trick, which we preserve, is **zero-padding to a fixed maximum**:

```
MetaWorld push (4):  [a1, a2, a3, a4,  0,  0,  ..., 0]   ← max_state_dim
7-DOF arm    (7):    [q1, q2, ..., q7,  0,  0,  ..., 0]   ← max_state_dim
```

The action expert always operates on `max_state_dim`-dim vectors.  The
velocity output for padded dimensions is ignored during loss computation
and clipped to zero at inference.

This lets you train a single model on a mixed-robot dataset and fine-tune
it on any subset.  The only per-robot cost is a learned embedding of the
robot name (not currently implemented here — see the roadmap).

---

## 7. Putting it all together

A single training step looks like:

```python
# 1. Get a batch of (images, text, state, clean_actions)
batch = next(dataloader)

# 2. Run the backbone on observations
obs_features = backbone(batch["input_ids"], batch["attention_mask"],
                        batch["pixel_values"])     # [B, S, D_backbone]

# 3. Sample noise and noise level
t     = torch.rand(B)
noise = torch.randn_like(batch["actions"])

# 4. Compute the OT interpolant + target velocity
x_t, u = ot_flow_interpolate(noise, batch["actions"], t)

# 5. Predict velocity with the action expert
v_hat = action_expert(batch["state"], x_t, t, obs_features)

# 6. MSE loss
loss = ((v_hat - u) ** 2).mean()

# 7. Backprop
loss.backward()
optimizer.step()
```

A single inference call:

```python
# 1. Encode observation ONCE
obs_features = backbone(images, text)

# 2. Start from noise
x = torch.randn(B, H, D)

# 3. Integrate the ODE for num_steps iterations
for i in range(num_steps):
    t = i / num_steps
    v = action_expert(state, x, t, obs_features)
    x = x + v * (1.0 / num_steps)

# 4. Execute x step-by-step on the robot
for action in x:
    robot.apply(action)
```

That's the entire algorithm.  The remaining docs explain the engineering
details of each component.
