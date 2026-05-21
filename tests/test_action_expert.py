"""Tests for the ActionExpert module (no Gemma 4 weights needed)."""

import pytest
import torch
from gemma4_vla.action_expert import ActionExpert, SelfAttention, CrossAttention


class TestSelfAttention:
    def test_output_shape(self):
        B, S, D = 2, 20, 64
        attn = SelfAttention(hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        out = attn(x)
        assert out.shape == (B, S, D)

    def test_with_mask(self):
        B, S, D = 2, 10, 64
        attn = SelfAttention(hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        # Causal mask
        mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
        out = attn(x, attn_mask=mask)
        assert out.shape == (B, S, D)


class TestCrossAttention:
    def test_output_shape(self):
        B, Sq, Sc = 2, 51, 256  # query=state+actions, context=obs
        Q_DIM, C_DIM = 64, 128
        xattn = CrossAttention(query_dim=Q_DIM, context_dim=C_DIM, num_heads=4)
        q = torch.randn(B, Sq, Q_DIM)
        ctx = torch.randn(B, Sc, C_DIM)
        out = xattn(q, ctx)
        assert out.shape == (B, Sq, Q_DIM)

    def test_context_mask(self):
        B, Sq, Sc = 2, 5, 32
        xattn = CrossAttention(query_dim=64, context_dim=64, num_heads=4)
        q = torch.randn(B, Sq, 64)
        ctx = torch.randn(B, Sc, 64)
        # Mask out the second half of context
        mask = torch.ones(B, Sc)
        mask[:, Sc // 2:] = 0
        out = xattn(q, ctx, context_mask=mask)
        assert out.shape == (B, Sq, 64)


class TestActionExpert:
    @pytest.fixture
    def expert(self):
        return ActionExpert(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            context_dim=128,
            action_dim=6,
            state_dim=6,
            action_horizon=10,
            ffn_multiplier=2,
            dropout=0.0,
            use_cross_attention=True,
            noise_embedding_dim=32,
        )

    def test_output_shape(self, expert):
        B, H, D = 3, 10, 6
        state = torch.randn(B, 6)
        noisy_actions = torch.randn(B, H, 6)
        noise_level = torch.rand(B)
        obs_features = torch.randn(B, 20, 128)

        velocity = expert(
            state=state,
            noisy_actions=noisy_actions,
            noise_level=noise_level,
            obs_features=obs_features,
        )
        assert velocity.shape == (B, H, D)

    def test_different_t_give_different_velocities(self, expert):
        B, H = 2, 10
        state = torch.randn(B, 6)
        noisy_actions = torch.randn(B, H, 6)
        obs_features = torch.randn(B, 15, 128)

        # The model zero-initializes the velocity head for stable training,
        # so we give it a non-zero projection before testing time conditioning.
        with torch.no_grad():
            torch.nn.init.normal_(expert.velocity_head.weight, std=0.02)

        with torch.no_grad():
            v1 = expert(state, noisy_actions, torch.zeros(B), obs_features)
            v2 = expert(state, noisy_actions, torch.ones(B), obs_features)
        assert not torch.allclose(v1, v2), "Different noise levels should give different velocities"

    def test_gradients_flow(self, expert):
        B, H = 2, 10
        state = torch.randn(B, 6)
        noisy_actions = torch.randn(B, H, 6)
        noise_level = torch.rand(B)
        obs_features = torch.randn(B, 15, 128)

        velocity = expert(state, noisy_actions, noise_level, obs_features)
        velocity.sum().backward()

        has_grad = any(p.grad is not None for p in expert.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed"

    def test_zero_velocity_head_init(self, expert):
        """Output should be near-zero at init (we zero-init the velocity head)."""
        B, H = 2, 10
        state = torch.zeros(B, 6)
        noisy_actions = torch.zeros(B, H, 6)
        noise_level = torch.zeros(B)
        obs_features = torch.zeros(B, 5, 128)

        with torch.no_grad():
            velocity = expert(state, noisy_actions, noise_level, obs_features)

        assert velocity.abs().max().item() < 1e-3, (
            f"Velocity head should be near-zero at init, got max={velocity.abs().max().item()}"
        )

    def test_without_cross_attention(self):
        """ActionExpert should work without cross attention (context=None)."""
        expert = ActionExpert(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            context_dim=128,
            action_dim=6,
            state_dim=6,
            action_horizon=5,
            use_cross_attention=False,
            noise_embedding_dim=32,
        )
        B, H = 2, 5
        state = torch.randn(B, 6)
        noisy_actions = torch.randn(B, H, 6)
        noise_level = torch.rand(B)
        obs = torch.randn(B, 10, 128)

        # context is passed but should be ignored
        velocity = expert(state, noisy_actions, noise_level, obs)
        assert velocity.shape == (B, H, 6)

    def test_parameter_count(self):
        """Approximate parameter count for the default (pi0-style) config."""
        expert = ActionExpert(
            hidden_size=1024,
            num_layers=8,
            num_heads=16,
            context_dim=1024,
            action_dim=18,
            state_dim=18,
            action_horizon=50,
            ffn_multiplier=4,
        )
        n_params = sum(p.numel() for p in expert.parameters())
        # Should be in the ~100–400 M range for this config
        assert 50_000_000 < n_params < 500_000_000, (
            f"Unexpected parameter count: {n_params / 1e6:.1f} M"
        )
