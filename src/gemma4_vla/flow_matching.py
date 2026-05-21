"""
Conditional Flow Matching (CFM) utilities.

We implement the optimal-transport (OT) probability path from:
  Lipman et al. "Flow Matching for Generative Modeling" (2023)

This is the same formulation used in pi0. Given:
  - x_0 ~ N(0, I)   (noise / source distribution)
  - x_1              (clean action / target distribution)

The OT path interpolates linearly:

    x_t = (1 - (1 - sigma_min) * t) * x_0  +  t * x_1

The conditional vector field (target velocity) is:

    u(x_t | x_1) = x_1 - (1 - sigma_min) * x_0

The model learns to approximate u.  At inference we integrate the ODE:

    dx/dt = v_theta(x_t, obs, t)   from t=0 (noise) to t=1 (action)

using simple Euler steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for the noise level t ∈ [0, 1].

    Encodes t as a fixed set of sin/cos frequencies, then projects to
    `output_dim` via a small MLP.  This gives the action expert a smooth,
    high-frequency representation of the current denoising step.
    """

    def __init__(self, output_dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.output_dim = output_dim
        self.max_period = max_period

        half = output_dim // 2
        # Fixed frequency schedule
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)  # [half]

        # Small MLP to project concatenated sin+cos to output_dim
        self.proj = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Noise levels, shape [B].  Values in [0, 1].

        Returns:
            Embeddings of shape [B, output_dim].
        """
        assert t.dim() == 1, f"Expected t of shape [B], got {t.shape}"
        # [B, half]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, output_dim]
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Core flow matching functions
# ---------------------------------------------------------------------------

def ot_flow_interpolate(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the OT interpolant x_t and the target velocity u.

    Args:
        x_0:       Source noise samples,  shape [B, H, D].
        x_1:       Target clean actions,  shape [B, H, D].
        t:         Noise levels,           shape [B].  Values in [0, 1].
        sigma_min: Small floor to prevent degenerate paths.

    Returns:
        x_t:    Interpolated noisy actions, shape [B, H, D].
        u:      Target velocity,            shape [B, H, D].
    """
    # Broadcast t to match action dimensions
    t_ = t.view(-1, 1, 1)  # [B, 1, 1]

    # OT interpolant
    x_t = (1.0 - (1.0 - sigma_min) * t_) * x_0 + t_ * x_1

    # Conditional vector field
    u = x_1 - (1.0 - sigma_min) * x_0

    return x_t, u


def flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean-squared-error loss between predicted and target velocities.

    Args:
        predicted_velocity: Shape [B, H, D].
        target_velocity:    Shape [B, H, D].
        mask:               Optional boolean mask [B, H] (True = valid step).

    Returns:
        Scalar loss.
    """
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")  # [B, H, D]

    if mask is not None:
        # mask: [B, H] → [B, H, 1]
        loss = loss * mask.unsqueeze(-1).float()
        return loss.sum() / (mask.float().sum() * predicted_velocity.shape[-1] + 1e-8)

    return loss.mean()


# ---------------------------------------------------------------------------
# Inference: ODE integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_integration(
    velocity_fn,
    shape: Tuple[int, ...],
    num_steps: int = 10,
    sigma_min: float = 1e-4,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    **velocity_fn_kwargs,
) -> torch.Tensor:
    """
    Integrate the learned ODE from t=0 (noise) to t=1 (action) using
    the forward Euler method.

    Args:
        velocity_fn:         Callable (x_t, t, **kwargs) → velocity [B, H, D].
        shape:               Output shape (B, H, D).
        num_steps:           Number of Euler integration steps.
        sigma_min:           Must match the value used during training.
        device:              Torch device.
        dtype:               Tensor dtype.
        **velocity_fn_kwargs: Extra kwargs forwarded to velocity_fn (e.g. obs).

    Returns:
        Clean action tensor of shape `shape`.
    """
    if device is None:
        device = torch.device("cpu")

    B = shape[0]
    dt = 1.0 / num_steps

    # Start from pure noise (t = 0)
    x = torch.randn(shape, device=device, dtype=dtype)

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((B,), t, device=device, dtype=dtype)

        velocity = velocity_fn(x, t_tensor, **velocity_fn_kwargs)
        x = x + velocity * dt

    return x


@torch.no_grad()
def rk4_integration(
    velocity_fn,
    shape: Tuple[int, ...],
    num_steps: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    **velocity_fn_kwargs,
) -> torch.Tensor:
    """
    4th-order Runge-Kutta integration.  Higher quality than Euler but
    4× more model evaluations per step.  Good for offline / evaluation use.

    Args:
        velocity_fn:          Callable (x_t, t, **kwargs) → velocity [B, H, D].
        shape:                Output shape (B, H, D).
        num_steps:            Number of RK4 steps.
        device:               Torch device.
        dtype:                Tensor dtype.
        **velocity_fn_kwargs: Extra kwargs forwarded to velocity_fn.

    Returns:
        Clean action tensor of shape `shape`.
    """
    if device is None:
        device = torch.device("cpu")

    B = shape[0]
    dt = 1.0 / num_steps

    x = torch.randn(shape, device=device, dtype=dtype)

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((B,), t, device=device, dtype=dtype)
        t_mid = torch.full((B,), t + 0.5 * dt, device=device, dtype=dtype)
        t_end = torch.full((B,), t + dt, device=device, dtype=dtype)

        k1 = velocity_fn(x, t_tensor, **velocity_fn_kwargs)
        k2 = velocity_fn(x + 0.5 * dt * k1, t_mid, **velocity_fn_kwargs)
        k3 = velocity_fn(x + 0.5 * dt * k2, t_mid, **velocity_fn_kwargs)
        k4 = velocity_fn(x + dt * k3, t_end, **velocity_fn_kwargs)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x
