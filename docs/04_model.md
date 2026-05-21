# 04 — Model Assembly

**Module:** [`src/gemma4_vla/model.py`](../src/gemma4_vla/model.py)

This module wires together the Gemma 4 backbone and the action expert into
a single end-to-end `nn.Module`.  It's the smallest file in the project
that actually *does* something useful — most of the complexity is in the
two components it composes.

---

## 1. Component responsibilities

```
Gemma4VLA
 ├── backbone   : Gemma4Backbone        (Gemma 4 + optional LoRA)
 ├── obs_proj   : Linear                (backbone_dim → action_expert_dim)
 └── action_expert : ActionExpert       (velocity predictor)
```

The `obs_proj` linear layer is the only module *invented* in this file —
its job is to bridge the dimensionality gap between the backbone (e.g.
2048 for Gemma 4 E2B) and the action expert (typically 1024 for
efficiency).

Making them different sizes is intentional: the backbone needs a lot of
capacity to understand images and language, but the action expert only
needs to produce a few dozen action tokens.  Using a smaller hidden size
in the expert saves significant compute during the inner denoising loop
at inference.

---

## 2. Gemma4Backbone wrapper

`Gemma4Backbone` is a thin wrapper around a HuggingFace Gemma 4 model.  It:

1. Loads the model via `AutoProcessor` + `Gemma4ForConditionalGeneration`
   (falling back to `AutoModelForCausalLM` for older transformers versions)
2. Optionally freezes all backbone parameters
3. Optionally injects LoRA adapters using [`peft`](https://github.com/huggingface/peft)
4. Exposes a single `forward()` that returns **last-layer hidden states**
   rather than logits

The important line is:

```python
outputs = self.model(..., output_hidden_states=True, return_dict=True)
return outputs.hidden_states[-1]   # [B, S, hidden_size]
```

We deliberately don't use the language-model head.  Gemma4VLA never
generates text — it uses Gemma 4 purely as a feature extractor.  This
is the same pattern used by LLaVA, PaliGemma, and most "VLM as encoder"
robot papers.

### Why return only the last layer?

You could return a concatenation of multiple layers (à la ELMo) or a
learned weighted average.  We return just the last layer for two reasons:

- **Simplicity**: it keeps the interface between backbone and action expert
  as a single fixed-shape tensor.
- **Information content**: the last layer of a pretrained LLM already
  contains all the information from earlier layers — by design, because
  residual connections preserve lower-layer signals.

If you have a research case for using multiple layers, override
`Gemma4Backbone.forward` to return whatever you need.

---

## 3. LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) freezes a pretrained weight matrix $W$ and adds a
trainable low-rank update:

$$
W' = W + \Delta W, \quad \Delta W = B A
$$

where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$,
with rank $r \ll d$ (typically $r = 8, 16, 32$).

The number of trainable parameters for one attention projection becomes:

$$
\underbrace{r(d + d)}_{\text{LoRA } A + B} \ll \underbrace{d^2}_{\text{full matrix}}
$$

For Gemma 4 E2B with $d = 2048$ and $r = 16$:

$$
\text{LoRA params per projection} = 2 \cdot 16 \cdot 2048 = 65{,}536
$$

vs

$$
\text{Full params per projection} = 2048^2 = 4{,}194{,}304
$$

That's a 64× reduction.  With LoRA on four projections per layer × 18
layers, we end up training about **4.7 M backbone parameters** instead of
**~1.9 B**.  Memory and compute savings are proportional.

### Scaling factor

LoRA introduces two hyperparameters: rank $r$ and a scaling factor
$\alpha$.  The effective update is:

$$
\Delta W = \frac{\alpha}{r} B A
$$

Keeping $\alpha / r$ constant (e.g. $\alpha = 2r$) makes the magnitude of
the update roughly invariant to rank.  Our defaults use $\alpha = 32$,
$r = 16$, giving $\alpha / r = 2$.

### Which modules to adapt?

We apply LoRA to the attention projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`) only.  These are where most cross-modal alignment happens.

Adding LoRA to the FFN layers (`gate_proj`, `up_proj`, `down_proj` in
Gemma 4) can squeeze out a little more performance at the cost of more
parameters.  You can configure this in `BackboneConfig.lora_target_modules`.

---

## 4. Cross-embodiment via padding

### The padding helpers

```python
def _pad_state(self, state):
    B, D = state.shape
    pad = torch.zeros(B, self.max_dim - D, ...)
    return torch.cat([state, pad], dim=-1)

def _pad_actions(self, actions):
    B, H, D = actions.shape
    pad = torch.zeros(B, H, self.max_dim - D, ...)
    return torch.cat([actions, pad], dim=-1)
```

Every state and action tensor is padded to `max_dim = 18` before touching
the action expert.  At the output, `_unpad_actions` slices off the padding:

```python
def _unpad_actions(self, actions):
    return actions[:, :, : self.action_dim]
```

### Why this works

The action expert doesn't know that the last few dimensions are fake.
It happily predicts velocities for them.  During training, **the loss is
computed only on the real dimensions**:

```python
pred_trimmed   = pred_velocity[:, :, : self.action_dim]
target_trimmed = target_velocity[:, :, : self.action_dim]
loss = flow_matching_loss(pred_trimmed, target_trimmed)
```

So gradients only flow through the real-dimension outputs.  The padded-
dimension outputs get no signal and quickly become garbage — but we don't
care because we throw them away.

If you want to be extra-safe, you can also zero out the padded dimensions
explicitly after the model returns, so that downstream code can't
accidentally use them.  We do this in `_unpad_actions`.

---

## 5. Training forward pass in detail

`Gemma4VLA.compute_loss` is a six-step function:

```python
def compute_loss(self, batch):
    # 1. Encode observations with Gemma 4
    obs_features = self.backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
    )                                                 # [B, S_obs, D_b]

    # 2. Pad state / actions for cross-embodiment
    state   = self._pad_state(batch["state"])         # [B, 18]
    actions = self._pad_actions(batch["actions"])     # [B, H, 18]

    # 3. Sample flow matching noise and time
    t     = torch.rand(B, device=device)              # [B]
    noise = torch.randn_like(actions)                 # [B, H, 18]

    # 4. OT interpolant + target velocity
    noisy_actions, target_velocity = ot_flow_interpolate(
        x_0=noise, x_1=actions, t=t, sigma_min=self.sigma_min
    )

    # 5. Predict velocity
    pred_velocity = self._predict_velocity(
        noisy_actions=noisy_actions,
        noise_level=t,
        obs_features=obs_features,
        state=state,
    )                                                 # [B, H, 18]

    # 6. Trim padding and compute MSE loss
    pred   = pred_velocity[:,  :, :self.action_dim]
    target = target_velocity[:, :, :self.action_dim]
    loss = flow_matching_loss(pred, target)

    return {"loss": loss, "metrics": {"flow_matching_loss": loss.item()}}
```

Every step should be clear by now if you've read the flow matching doc.
The only subtle thing is that **the backbone runs inside this function**,
not upstream — because we want its gradients in the same backward pass as
the action expert (when LoRA / full fine-tuning is enabled).

---

## 6. Inference forward pass in detail

`Gemma4VLA.predict_action` is structured around the **single-backbone,
many-expert-calls** pattern:

```python
@torch.no_grad()
def predict_action(self, obs, num_steps=None, use_rk4=False):
    n_steps = num_steps or self.num_inference_steps

    # 1. Encode observation ONCE
    obs_features = self.backbone(
        input_ids=obs["input_ids"],
        attention_mask=obs["attention_mask"],
        pixel_values=obs.get("pixel_values"),
    )
    state = self._pad_state(obs["state"])

    # 2. Define velocity function using the cached obs_features
    def velocity_fn(noisy_actions, t):
        return self._predict_velocity(
            noisy_actions=noisy_actions,
            noise_level=t,
            obs_features=obs_features,  # <-- cached!
            state=state,
        )

    # 3. Integrate the ODE
    shape = (B, self.action_horizon, self.max_dim)
    actions_padded = euler_integration(
        velocity_fn=velocity_fn, shape=shape,
        num_steps=n_steps, ...
    )

    # 4. Trim padding
    return self._unpad_actions(actions_padded)
```

Cost accounting:

| | Training | Inference |
|--|---------|-----------|
| Backbone forward | 1× | 1× |
| Action expert forward | 1× | 10× (num_steps) |

Because the action expert is ~50× smaller than Gemma 4 E2B, the 10 expert
calls add only about 20% to the total latency.  That's the core trick
that makes real-time VLA inference feasible.

---

## 7. Save / load

Checkpoints are now stored as:

```
checkpoint_dir/
  config.json  # canonical config artifact
  config.pt    # compatibility artifact storing a plain dict
  weights.pt   # pickled state_dict
```

`save_pretrained` and `from_pretrained` handle both together.  We use
JSON for the canonical config plus PyTorch serialization for the weights.
`from_pretrained` prefers `config.json`, but still supports older
checkpoints whose `config.pt` contains a pickled `Gemma4VLAConfig`
instance.  If you need safetensors compatibility for production, the
weights artifact is the piece to swap.

Since we use `strict=False` when loading, you can load checkpoints into
a model with slightly different architecture (e.g. different LoRA rank).
Missing and unexpected keys will be printed as warnings but not cause a
crash.

---

## 8. Common bugs and how the code guards against them

### Hidden size mismatch

If you change `backbone.hidden_size` in config but load a checkpoint saved
with a different hidden size, you'll get a shape error in `obs_proj`.
`Gemma4Backbone.__init__` fetches the *actual* hidden size from the loaded
model config, overriding whatever's in `BackboneConfig`, to prevent this.

### Frozen backbone + LoRA conflict

Applying LoRA to a fully-frozen model produces a model with no trainable
parameters.  We guard against this:

```python
if bk.use_lora and not bk.freeze_backbone:
    self._apply_lora(bk)
```

LoRA is only injected when the backbone isn't fully frozen.

### Missing peft

LoRA is an optional feature.  If `peft` isn't installed, we print a
warning and fall back to full fine-tuning rather than crashing.

### Action dim > max_dim

If you accidentally pass `action_dim = 20` while `max_state_dim = 18`,
padding will fail silently (pad size becomes negative).  Catch this in
`Gemma4VLAConfig.__post_init__` if you add many new robots.

---

## 9. Extending the model

### Add language-only instruction conditioning (no images)

Remove the `pixel_values` kwarg from `compute_loss` and `predict_action`.
Make sure your dataset provides `input_ids` with only the text prompt.
The backbone will still work — Gemma 4 is a language model first.

### Add depth / pointcloud inputs

Two common patterns:
- **Extra vision stream**: add a small PointNet / depth encoder, project
  its features to `backbone_dim`, and concatenate onto `obs_features`.
- **Extra cross-attention path**: have the action expert cross-attend to
  two separate feature streams (image, depth) with two cross-attention
  sub-layers per block.

The first is simpler and usually works well.

### Use the model as part of a hierarchical policy

Treat `predict_action` as a primitive "motion-plan-for-one-second" step,
and wrap it in an outer loop that re-encodes the observation every
$k$ steps.  This is the natural fit for longer-horizon tasks.
