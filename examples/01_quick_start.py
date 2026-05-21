"""
Example 01 — Quick start: run inference with Gemma4VLA.

This script shows the minimal code to load a trained model and predict
robot actions from camera images and a language instruction.

Prerequisites:
    uv sync

    You must accept the Gemma 4 license on Hugging Face and log in:
        uv run huggingface-cli login

Usage:
    uv run python examples/01_quick_start.py
    uv run python examples/01_quick_start.py --checkpoint checkpoints/best
    uv run python examples/01_quick_start.py --device cpu --steps 5
"""

import argparse
import numpy as np
import torch

# ── Optional: avoid tokeniser parallelism warnings ──────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def make_dummy_obs(num_cameras: int = 1, image_size: int = 224, state_dim: int = 6):
    """Create a fake observation for demonstration purposes."""
    return {
        # List of uint8 RGB images, one per camera
        "images": [
            np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            for _ in range(num_cameras)
        ],
        # Robot proprioceptive state (joint angles, gripper opening, …)
        "state": np.zeros(state_dim, dtype=np.float32),
        # Natural-language task description
        "instruction": "Pick up the red cube and place it in the bowl.",
    }


def run_inference(checkpoint: str | None, device: str, num_steps: int):
    """Run a single inference pass and print the predicted actions."""
    from gemma4_vla import Gemma4VLA, Gemma4VLAConfig, PolicyRunner, metaworld_push_config

    # ── 1. Build or load the model ───────────────────────────────────────────
    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        runner = PolicyRunner.from_pretrained(checkpoint, device=device)
        cfg = runner.cfg
    else:
        print("No checkpoint specified — using freshly initialised weights.")
        print("(Actions will be random but demonstrate the forward pass.)")
        cfg = metaworld_push_config()
        # Override model name if you want a smaller/faster model for testing
        # cfg.backbone.model_name = "google/gemma-4-E2B-it"
        model = Gemma4VLA(cfg)
        runner = PolicyRunner(model, device=device)

    # ── 2. Build a dummy observation ─────────────────────────────────────────
    obs = make_dummy_obs(
        num_cameras=cfg.vision.num_cameras,
        image_size=cfg.vision.image_size,
        state_dim=cfg.robot.state_dim,
    )

    print(f"\nObservation:")
    print(f"  Instruction : '{obs['instruction']}'")
    print(f"  Images      : {len(obs['images'])} × {obs['images'][0].shape}")
    print(f"  State       : {obs['state'].shape}")

    # ── 3. Predict an action chunk ───────────────────────────────────────────
    print(f"\nRunning inference ({num_steps} denoising steps)…")
    actions = runner.predict(obs, num_inference_steps=num_steps)

    print(f"\nPredicted actions: {actions.shape}")
    print(f"  horizon   = {actions.shape[0]} steps")
    print(f"  action_dim = {actions.shape[1]} DOF")
    print(f"  First action : {actions[0]}")
    print(f"  Last  action : {actions[-1]}")
    print(f"  Action range : [{actions.min():.3f}, {actions.max():.3f}]")

    # ── 4. Streaming / temporal action chunking ──────────────────────────────
    print("\nSimulating temporal action chunking (first 5 steps):")
    for i, action in enumerate(runner.stream(obs, replan_every=10)):
        print(f"  step {i:2d}: {action}")
        if i >= 4:
            break

    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of flow matching denoising steps")
    args = parser.parse_args()

    actions = run_inference(args.checkpoint, args.device, args.steps)
    print("\nDone!")
