# 03 — Action Expert

**Module:** [`src/gemma4_vla/action_expert.py`](../src/gemma4_vla/action_expert.py)

The action expert is the network that actually predicts flow velocities.
It's a lightweight (~300 M parameter) transformer with its own weights —
separate from the Gemma 4 backbone.

Input: a noisy action chunk + proprioceptive state + noise level + observation features
Output: predicted velocity for each noisy action token

---

## 1. Token layout

The action expert operates on a short sequence of tokens:

```
position  0         1         2         ...     H
          ┌───────┐ ┌───────┐ ┌───────┐       ┌───────┐
          │ state │ │ act_0 │ │ act_1 │  ...  │ act_H−1│
          └───────┘ └───────┘ └───────┘       └───────┘
             ▲         ▲         ▲              ▲
             │         │         │              │
       state_proj  action_proj  action_proj  action_proj
        (18→d)       (18→d)       (18→d)        (18→d)

  Each token also receives:
    + sinusoidal_embed(noise_level_t)   (added to every position)
    + action_pos_embed(position_index)   (action tokens only)
```

- `H = action_horizon` (default 50).
- `d = action_expert.hidden_size` (default 1024).
- The state token is always at position 0; action tokens fill the rest.

Why bundle state with actions?  Because the action expert needs to know
the robot's current joint configuration to predict a valid motion, and
putting the state in the same sequence lets the network naturally attend
to it from every action position without a second cross-attention path.

---

## 2. Layer structure

Each of the $L$ transformer blocks (default $L = 8$) has this shape:

```
        x  [B, 1+H, d]
        │
   RMSNorm
        │
   Self-Attention      ──► attends over all 1+H tokens
        │
        ⊕  (residual)
        │
   RMSNorm
        │
   Cross-Attention     ──► queries over 1+H, keys/values from obs_features
        │
        ⊕  (residual)
        │
   RMSNorm
        │
   Feed-Forward (FFN)
        │
        ⊕  (residual)
        │
     output
```

Three sub-layers per block, each with pre-normalisation (RMSNorm before
the operation, residual after).  This is the same structure as LLaMA and
Gemma, and it's chosen because pre-norm transformers are easier to train
deeply.

---

## 3. RMSNorm instead of LayerNorm

**RMSNorm** (Zhang & Sennrich, 2019) is a simplified version of LayerNorm
that drops the mean-subtraction step:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

where $g$ is a learned per-dimension scale.  Compare to LayerNorm:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot g + b
$$

RMSNorm:
- Saves one sum and one subtraction per token (~10% faster)
- Removes the learnable bias $b$ (slightly fewer parameters)
- Works just as well empirically for modern LLM-style architectures

Gemma, LLaMA, and most recent transformers use RMSNorm, so we follow suit.
This also makes it easier to load pretrained attention blocks from those
models if you want to transfer from a language-only checkpoint.

---

## 4. Rotary Position Embeddings (RoPE)

Classical position embeddings add a learned vector to each token.  RoPE
instead **rotates the query and key vectors** by an angle that depends on
absolute position.  The magical property: the **dot product** between a
query at position $m$ and a key at position $n$ ends up depending only on
$(m - n)$, giving us relative position awareness "for free".

### The math

Split each head's query vector into pairs $(q_{2k}, q_{2k+1})$ and apply a
2D rotation matrix depending on position $m$:

$$
R(m, \theta_k) =
\begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\
                \sin(m\theta_k) &  \cos(m\theta_k) \end{pmatrix}
$$

with per-pair frequencies $\theta_k = 10{,}000^{-2k/d_{\text{head}}}$.

The same rotation is applied to keys.  Then attention proceeds normally.

### Why it matters here

Our action tokens are ordered in time — position $0$ is "next action",
position $H-1$ is "furthest future action".  RoPE encodes this temporal
structure without us having to carve out dedicated parameters for it.

It also extrapolates better than learned embeddings if you ever want to
change `action_horizon` at inference time (e.g. for longer-horizon planning).

### Implementation

Our `RotaryEmbedding` class precomputes `cos` and `sin` caches up to
`max_seq_len` and applies them via `rotate_half`:

```python
def rotate_half(self, x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

q = q * cos + self.rotate_half(q) * sin
k = k * cos + self.rotate_half(k) * sin
```

This is the "half-rotation" trick: instead of literally building a 2×2
rotation matrix for each pair, we split the vector in half, apply the
appropriate sign flip, and use element-wise multiplication.  Same result,
much cheaper.

---

## 5. Self-attention vs cross-attention

**Self-attention** is standard multi-head attention where Q, K, V all come
from the same input tensor.  Action tokens attend to each other and to the
state token, learning things like "the third joint should follow the
second joint smoothly over the horizon".

**Cross-attention** reuses the same attention mechanism but sources its
keys and values from a *different* tensor — the observation features from
the Gemma 4 backbone.  Queries still come from the action expert's internal
representation.

The mathematical difference is tiny (a few different linear projections),
but the information flow is completely different:

| | Self-attention | Cross-attention |
|--|---------------|-----------------|
| Query | action expert | action expert |
| Key, Value | action expert | Gemma 4 backbone |
| What it learns | Temporal dependencies between actions | Grounding actions in what the camera sees |

Without cross-attention, the action expert would have no way to see the
image or the language instruction.  It's the bridge that makes this a VLA
rather than a blind state-based policy.

### Why both?

You could imagine skipping self-attention and using cross-attention
only — the action expert would then behave more like a Perceiver.  That
actually works, but empirically is slightly worse at producing smooth
trajectories.  The intuition: neighbouring action tokens need to coordinate
(smoothness, coherence), and doing that through cross-attention requires
an inefficient round-trip through the observation features.

You could also skip cross-attention and just concatenate image tokens to
the input sequence.  This works too, but inflates the sequence length
from $1 + H \approx 51$ tokens to $1 + H + S_{\text{obs}} \approx 1000$
tokens, which is 20× more attention compute per layer.

---

## 6. Noise-level conditioning

The noise level $t$ is broadcast to **every token** before the first layer:

```python
noise_cond = self.noise_proj(self.noise_emb(noise_level))  # [B, d]
state_tok  = state_tok  + noise_cond.unsqueeze(1)           # [B, 1, d]
action_tok = action_tok + noise_cond.unsqueeze(1)           # [B, H, d]
```

Adding is simpler than concatenating, uses no extra parameters, and works
well in practice.  Other popular options include AdaLN (adaptive layer
norm with FiLM-style conditioning) and cross-attention to a $t$ token —
both are slightly more expressive but also more parameters.  For a first
implementation, additive conditioning is plenty.

The noise conditioning gets mixed with the token representation at every
layer (via residual connections), so later layers still have access to it.

---

## 7. Output head: zero-init trick

The velocity head maps from `hidden_size` to `action_dim`:

```python
self.velocity_head = nn.Linear(hidden_size, action_dim)
```

We **initialise its weights and biases to zero**:

```python
nn.init.zeros_(self.velocity_head.weight)
nn.init.zeros_(self.velocity_head.bias)
```

This is borrowed from diffusion model literature and serves one purpose:
**at step 0 of training, the predicted velocity is exactly 0**, so the
initial flow is just the noise-to-noise identity (no drift).  The network
then slowly learns to deviate from zero as training progresses.

The practical benefit: very stable early training.  Without zero-init,
random velocity predictions at step 0 can push the optimisation in chaotic
directions, particularly for large learning rates.

---

## 8. FFN structure

The feed-forward sublayer is a simple two-layer MLP with SiLU activation:

```python
self.ffn = nn.Sequential(
    nn.Linear(hidden_size, hidden_size * ffn_multiplier, bias=False),
    nn.SiLU(),
    nn.Linear(hidden_size * ffn_multiplier, hidden_size, bias=False),
)
```

`ffn_multiplier = 4` is standard.  SiLU (a.k.a. Swish) is smooth and
non-monotonic, which empirically produces better gradients in deep
transformers than plain ReLU.

More exotic designs like SwiGLU (used in LLaMA and Gemma) offer a small
additional improvement but add parameters and complexity.  You can swap
it in if you're chasing the last few points of validation loss.

---

## 9. Parameter count

For the default config (`hidden_size=1024, num_layers=8, num_heads=16`):

| Component | Parameters |
|-----------|-----------|
| Input/output projections (state, action, velocity head) | ~50 K |
| Noise embedding MLP | ~130 K |
| Action positional embedding | ~50 K |
| Self-attention (8 layers) | ~33 M |
| Cross-attention (8 layers) | ~33 M |
| FFN (8 layers) | ~67 M |
| RMSNorm layers | ~50 K |
| **Total** | **~135 M** |

For a larger config (`hidden_size=2048, num_layers=16,
num_heads=32`), it's closer to ~1.1 B parameters — still ~25× smaller than
the Gemma 4 31B backbone.

---

## 10. Modifying the action expert

Common things you might want to change:

**Add block-wise attention masking (pi0-style)**:
The original pi0 uses a specific mask where action tokens attend to each
other fully, state tokens attend only to backbone, and backbone tokens
don't attend to action tokens.  We use full self-attention for simplicity;
you can recover the pi0 behavior by building a boolean mask and passing it
to `SelfAttention.forward`.

**Use a different position encoding**:
Replace `RotaryEmbedding` with `AbsolutePositionalEncoding` for long
horizons, or with ALiBi for even longer extrapolation.

**Swap SiLU for SwiGLU**:
```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, ffn_multiplier):
        super().__init__()
        inner = hidden_size * ffn_multiplier
        self.w_gate = nn.Linear(hidden_size, inner, bias=False)
        self.w_up   = nn.Linear(hidden_size, inner, bias=False)
        self.w_down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

**Grouped-query attention (GQA)**:
Use fewer KV heads than Q heads to save KV cache memory.  Useful when you
push `action_horizon` into the hundreds.
