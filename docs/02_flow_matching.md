# 02 — Flow Matching

**Module:** [`src/gemma4_vla/flow_matching.py`](../src/gemma4_vla/flow_matching.py)

This is the mathematical heart of Gemma4VLA.  It defines **how action
distributions are represented and how we sample from them**.  Everything
else (backbone, action expert, dataset) exists to feed this module.

---

## 1. The generative modelling problem

We want to learn a conditional distribution

$$
p(\mathbf{x}_1 \mid \mathbf{c})
$$

where:
- $\mathbf{x}_1 \in \mathbb{R}^{H \times D}$ is a clean action chunk
  (H time steps of D-DOF actions)
- $\mathbf{c}$ is the conditioning (observation features, state, language)

A common trick is to relate $\mathbf{x}_1$ to a simple **source distribution**
$p_0 = \mathcal{N}(0, I)$ via a smooth interpolation path
$\{p_t\}_{t\in[0,1]}$ with $p_0$ at $t = 0$ and $p_1$ at $t = 1$.

Diffusion models do this with a stochastic differential equation (SDE):
random noise is added along the way, and a neural network learns to reverse
the noise process.

**Flow matching** does it with an ordinary differential equation (ODE):
there's no stochasticity — samples follow smooth deterministic trajectories
from noise to data, and the neural network learns the **velocity field**
governing those trajectories.

This removes a lot of the variance (and the messy SDE solvers) that make
diffusion training and sampling slow.

---

## 2. The Optimal Transport (OT) probability path

There are infinitely many valid interpolations from $p_0$ to $p_1$.  We use
the one that's **straight lines** in sample space:

$$
\mathbf{x}_t = (1 - (1 - \sigma_{\min}) t) \mathbf{x}_0 + t\, \mathbf{x}_1
$$

where $\mathbf{x}_0 \sim \mathcal{N}(0, I)$, $\mathbf{x}_1 \sim p_{\text{data}}$,
and $\sigma_{\min}$ is a small positive constant (we use $10^{-4}$).

This is the **OT path** from Lipman et al. (2023).  It's called that because
in the infinite-data limit, linear interpolation between samples is the
optimal transport map between $p_0$ and $p_1$.

### Why $\sigma_{\min}$?

If $\sigma_{\min} = 0$, at $t = 1$ the interpolant is exactly $\mathbf{x}_1$.
That's what we want!  But it also means that for $t$ slightly less than 1,
the path is almost noiseless — the network has to predict the clean action
with almost no perturbation to learn from.  Empirically this causes
instabilities near $t = 1$.

Setting $\sigma_{\min} = 10^{-4}$ keeps a tiny residual of noise at the
endpoint:

$$
\mathbf{x}_{t=1} = \sigma_{\min}\,\mathbf{x}_0 + \mathbf{x}_1
$$

This introduces imperceptible error at inference but massively stabilises
training.

### Code

```python
def ot_flow_interpolate(x_0, x_1, t, sigma_min=1e-4):
    t_ = t.view(-1, 1, 1)                             # broadcast
    x_t = (1.0 - (1.0 - sigma_min) * t_) * x_0 + t_ * x_1
    u   = x_1 - (1.0 - sigma_min) * x_0               # target velocity
    return x_t, u
```

---

## 3. The conditional velocity field

For a given source-target pair $(\mathbf{x}_0, \mathbf{x}_1)$, the path is
a straight line.  The velocity along that line is **constant** and equal to:

$$
\frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - (1 - \sigma_{\min})\,\mathbf{x}_0
$$

This is `u` in the code.  It's the thing the network tries to predict.

### Why constant?

Because the path is linear in $t$.  Differentiating:

$$
\mathbf{x}_t = (1 - (1-\sigma_{\min})t)\mathbf{x}_0 + t\,\mathbf{x}_1
$$

with respect to $t$ gives

$$
\frac{d\mathbf{x}_t}{dt} = -(1 - \sigma_{\min})\mathbf{x}_0 + \mathbf{x}_1 = \mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0
$$

which doesn't depend on $t$.  So the velocity is constant *along a given
trajectory* — but different trajectories have different constant velocities.
The network has to learn which velocity is correct for each $(x_t, c)$.

---

## 4. The flow matching loss

Conditional flow matching minimises the expected squared error between
the predicted and target velocity:

$$
\mathcal{L}_{\text{CFM}}(\theta) =
\mathbb{E}_{t,\, \mathbf{x}_0,\, \mathbf{x}_1}
\big[\|\, v_\theta(\mathbf{x}_t, \mathbf{c}, t) - u\,\|^2 \big]
$$

where:
- $t \sim \mathcal{U}(0, 1)$
- $\mathbf{x}_0 \sim \mathcal{N}(0, I)$
- $\mathbf{x}_1 \sim p_{\text{data}}$  (a clean action from the dataset)
- $\mathbf{x}_t$ is the interpolant
- $u$ is the target velocity from Section 3
- $\mathbf{c}$ is the conditioning

**This is just MSE.**  That's the whole training loss.  No KL term, no
log-likelihood, no variational bound.  Because the OT path is deterministic
and the velocity field is constant along each trajectory, the regression
target is well-defined.

### Why this works (intuition)

If $v_\theta(\mathbf{x}_t, \mathbf{c}, t) = u$ for every $(\mathbf{x}_t, \mathbf{c}, t)$,
then integrating the ODE

$$
\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}, \mathbf{c}, t),
\qquad \mathbf{x}(0) \sim \mathcal{N}(0, I)
$$

from $t = 0$ to $t = 1$ produces samples from $p_{\text{data}}(\cdot \mid \mathbf{c})$.
So learning the velocity field is equivalent to learning how to sample.

The reason we can *regress on constant velocities* per-trajectory but still
end up with a useful network is that the network is asked to **predict the
average velocity over all trajectories passing through** $(\mathbf{x}_t, \mathbf{c}, t)$.
This marginalised velocity field is exactly the correct one for generating
samples from the target distribution — a non-trivial theoretical result
proved in Lipman et al. (2023).

### Code

```python
def flow_matching_loss(predicted_velocity, target_velocity, mask=None):
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
    if mask is not None:
        loss = loss * mask.unsqueeze(-1).float()
        return loss.sum() / (mask.float().sum() * D + 1e-8)
    return loss.mean()
```

The optional mask lets you ignore padded timesteps (e.g. when an episode
ends mid-chunk) or padded action dimensions (for cross-embodiment training).

---

## 5. Inference: integrating the ODE

At inference time, we solve:

$$
\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}, \mathbf{c}, t),
\qquad \mathbf{x}(0) \sim \mathcal{N}(0, I)
$$

numerically from $t = 0$ to $t = 1$ and return $\mathbf{x}(1)$.

### Forward Euler (default)

The simplest integrator:

$$
\mathbf{x}_{t + \Delta t} = \mathbf{x}_t + \Delta t \cdot v_\theta(\mathbf{x}_t, \mathbf{c}, t)
$$

With `num_steps = N`, we use $\Delta t = 1/N$ and take N steps.

```python
def euler_integration(velocity_fn, shape, num_steps, ...):
    x = torch.randn(shape, ...)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = step * dt
        velocity = velocity_fn(x, torch.full((B,), t))
        x = x + velocity * dt
    return x
```

Global error for Euler is $O(\Delta t)$, so 10 steps gives
$\approx 10\%$ residual error on a generic smooth ODE.  For flow matching
with OT paths, the true trajectories are almost straight lines, so Euler
is **much more accurate** than the generic bound suggests.  Empirically,
5–10 steps are enough for high-quality robot actions.

### Runge-Kutta 4 (optional, higher quality)

RK4 combines four velocity evaluations per step:

$$
\begin{aligned}
k_1 &= v(\mathbf{x}_t, t) \\
k_2 &= v(\mathbf{x}_t + \tfrac{\Delta t}{2} k_1, t + \tfrac{\Delta t}{2}) \\
k_3 &= v(\mathbf{x}_t + \tfrac{\Delta t}{2} k_2, t + \tfrac{\Delta t}{2}) \\
k_4 &= v(\mathbf{x}_t + \Delta t\, k_3, t + \Delta t) \\
\mathbf{x}_{t+\Delta t} &= \mathbf{x}_t + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

Global error is $O(\Delta t^4)$ — dramatically better — but you pay 4×
more model evaluations per step.  Use RK4 when quality matters more than
speed (offline trajectory generation, evaluation, research ablations).

```python
def rk4_integration(velocity_fn, shape, num_steps, ...):
    x = torch.randn(shape, ...)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t    = step * dt
        k1 = velocity_fn(x,              t)
        k2 = velocity_fn(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = velocity_fn(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = velocity_fn(x + dt*k3,     t + dt)
        x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x
```

---

## 6. The noise-level embedding

The action expert needs to know $t$ (where we are along the denoising
trajectory) to predict the right velocity.  We feed it a continuous
**sinusoidal embedding**, analogous to position encodings in the
original Transformer:

$$
\phi(t)_{2k}   = \sin\!\big(t \cdot \omega_k\big),
\quad
\phi(t)_{2k+1} = \cos\!\big(t \cdot \omega_k\big)
$$

with frequencies

$$
\omega_k = \exp\!\left(-\frac{k}{K/2} \ln \omega_{\max}\right),
\qquad k = 0, \ldots, K/2 - 1
$$

Here $K$ is the embedding dimension and $\omega_{\max} = 10{,}000$.

### Why sinusoidal?

1. **Continuous** — any $t \in [0, 1]$ gets a unique embedding, so we can
   use arbitrary inference step counts at test time.
2. **Smooth** — nearby $t$ values produce nearby embeddings, matching our
   assumption that the velocity field is smooth in $t$.
3. **Frequency-spread** — the multiple frequencies let the network
   distinguish fine differences in $t$ while also capturing coarse
   structure.

### After the sinusoid

We pass the raw sinusoidal features through a small MLP to let the network
learn a task-appropriate embedding:

```python
self.proj = nn.Sequential(
    nn.Linear(output_dim, output_dim * 2),
    nn.SiLU(),
    nn.Linear(output_dim * 2, output_dim),
)
```

SiLU (also known as Swish) works well here because it's smooth and
non-monotonic, matching the needs of diffusion-style models.

---

## 7. Why is this better than diffusion?

Both diffusion and flow matching learn a transformation from noise to data
using a regression loss.  The key differences:

| | Diffusion | Flow matching (OT) |
|--|-----------|--------------------|
| Interpolation | Noisy SDE | Deterministic straight line |
| Training target | Noise $\epsilon$ or score $\nabla \log p_t$ | Constant velocity $u$ |
| Loss | Reweighted MSE | Plain MSE |
| Inference | SDE solver or DDIM | Euler ODE (5–10 steps) |
| Multi-modality | Yes | Yes |
| Training stability | Sensitive to noise schedule | Stable |

The big win is **inference speed**.  Flow matching trajectories are
approximately straight, so even Euler with very few steps works well.
Diffusion models typically need 25–100 steps or a clever distillation
procedure to match that quality.

For robotics at 50 Hz we have a 20 ms budget per action chunk.  Flow
matching with 10 steps fits; diffusion with 50 steps does not.

---

## 8. Known failure modes

- **Action scale mismatch**: if your clean actions are in $[-100, 100]$
  but the noise is $\mathcal{N}(0, 1)$, the interpolation is dominated by
  actions at $t > 0.01$ and by noise at $t < 0.01$.  Always normalise your
  actions to roughly unit variance before training.

- **Insufficient num_steps**: Euler with 1–2 steps can produce
  out-of-distribution actions even if the velocity field is perfect.
  Always validate with at least 5 steps.

- **σ_min too large**: setting $\sigma_{\min} > 0.01$ introduces visible
  noise in the final action.  Keep it below $10^{-3}$.

- **Horizon padding leaking into loss**: if your episodes end mid-chunk,
  pass a `mask` to `flow_matching_loss` to zero out the padded timesteps.
  Otherwise the network trains to predict zero velocity for padded steps,
  which is wrong.

---

## 9. Further reading

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative
  Modeling*.  ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Pooladian et al. *Multisample Flow Matching: Straightening Flows with
  Minibatch Couplings*.  ICML 2023. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
- Black et al. *π0: A Vision-Language-Action Flow Model for General
  Robot Control*. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
