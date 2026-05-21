"""
Action Expert Transformer.

The action expert is a dedicated ~300 M parameter transformer with its own
weights.  It operates on two token types:

  1. State tokens   – a single token representing the proprioceptive state
  2. Action tokens  – one token per future action step (horizon H)

Crucially, the action expert cross-attends to observation features produced
by the Gemma 4 backbone (images + language).  This design mirrors pi0 and
allows the VLM weights to remain largely frozen while the action-specific
parameters train quickly on robot data.

Block-wise attention (matching pi0):
  - Action tokens attend to: all observation features (full), each other (full)
  - State  tokens attend to: all observation features (full)
  - Observation tokens do NOT attend to state/action tokens
    (their keys/values are read-only from the action expert's perspective)

This asymmetry preserves the VLM's pre-training and prevents robot noise
from corrupting the language representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .flow_matching import SinusoidalEmbedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (used in Gemma / LLaMA style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # [1,1,S,D]
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, S)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from action expert, keys/values from backbone."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Sq, _ = x.shape
        _, Sc, _ = context.shape

        q = self.q_proj(x).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            # context_mask: [B, Sc] → additive mask
            additive = (1.0 - context_mask.float()).unsqueeze(1).unsqueeze(1) * -1e9
            attn = attn + additive
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Sq, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------

class ActionExpertBlock(nn.Module):
    """
    One transformer block for the action expert.

    Structure:
      x → RMSNorm → SelfAttention  → residual
        → RMSNorm → CrossAttention → residual   (if use_cross_attention)
        → RMSNorm → FFN            → residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_dim: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        ffn_dim = hidden_size * ffn_multiplier

        self.norm1 = RMSNorm(hidden_size)
        self.self_attn = SelfAttention(hidden_size, num_heads, dropout)

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm2 = RMSNorm(hidden_size)
            self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads, dropout)

        self.norm3 = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.drop(self.self_attn(self.norm1(x), attn_mask=self_attn_mask))

        # Cross-attention to backbone observations
        if self.use_cross_attention and context is not None:
            x = x + self.drop(self.cross_attn(self.norm2(x), context, context_mask))

        # Feed-forward
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Action Expert
# ---------------------------------------------------------------------------

class ActionExpert(nn.Module):
    """
    Lightweight transformer that processes proprioceptive state + noisy actions
    and produces velocity predictions for flow matching.

    Input tokens:
      [state_token, action_token_0, action_token_1, ..., action_token_{H-1}]
          ^ 1 token                 ^  H tokens

    The state token attends to the backbone features and to all action tokens.
    Action tokens attend fully to each other and cross-attend to backbone features.

    The noise level embedding is added to every token before the first layer,
    giving each layer awareness of where we are in the denoising trajectory.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        context_dim: int,
        action_dim: int,
        state_dim: int,
        action_horizon: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        noise_embedding_dim: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_horizon = action_horizon

        # Input projections
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)

        # Noise level conditioning
        self.noise_emb = SinusoidalEmbedding(noise_embedding_dim)
        self.noise_proj = nn.Linear(noise_embedding_dim, hidden_size)

        # Learnable positional embeddings for action steps
        self.action_pos_emb = nn.Embedding(action_horizon, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            ActionExpertBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_dim=context_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = RMSNorm(hidden_size)

        # Output projection: predict velocity for each action token
        self.velocity_head = nn.Linear(hidden_size, action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

        # Zero-init the velocity head for stable early training
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        noise_level: torch.Tensor,
        obs_features: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state:          Proprioceptive state,     shape [B, state_dim].
            noisy_actions:  Noisy action sequence,    shape [B, H, action_dim].
            noise_level:    Diffusion time t ∈ [0,1], shape [B].
            obs_features:   Backbone observation feats, shape [B, S_obs, context_dim].
            obs_mask:       Padding mask for obs_features [B, S_obs] (True = valid).

        Returns:
            Predicted velocities, shape [B, H, action_dim].
        """
        B, H, _ = noisy_actions.shape

        # 1. Embed noise level → [B, hidden_size]
        noise_cond = self.noise_proj(self.noise_emb(noise_level))  # [B, hidden]

        # 2. Embed state → single state token [B, 1, hidden]
        state_tok = self.state_proj(state).unsqueeze(1)  # [B, 1, hidden]
        state_tok = state_tok + noise_cond.unsqueeze(1)

        # 3. Embed actions → [B, H, hidden]
        pos_ids = torch.arange(H, device=noisy_actions.device)
        action_toks = self.action_proj(noisy_actions)           # [B, H, hidden]
        action_toks = action_toks + self.action_pos_emb(pos_ids)  # add position
        action_toks = action_toks + noise_cond.unsqueeze(1)       # add noise level

        # 4. Concatenate: [state_token, action_tokens_0..H-1]
        x = torch.cat([state_tok, action_toks], dim=1)  # [B, 1+H, hidden]

        # 5. Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, context=obs_features, context_mask=obs_mask)

        x = self.norm_out(x)

        # 6. Extract action tokens (skip the state token at position 0)
        action_out = x[:, 1:, :]  # [B, H, hidden]

        # 7. Project to velocity
        velocity = self.velocity_head(action_out)  # [B, H, action_dim]
        return velocity
