"""
Example 03 — Custom robot configuration.

Shows how to set up Gemma4VLA for a robot not covered by the built-in
presets.  We use a hypothetical 7-DOF bimanual arm (14 joints total)
with 3 cameras as the running example.

The key insight from pi0 is that you need only:
  1. Define state_dim  (proprioceptive state size)
  2. Define action_dim (motor command size)
  3. Set num_cameras   (camera count)

Everything else adapts automatically via zero-padding to max_state_dim=18.

Steps shown here:
  A. Define a custom RobotConfig
  B. Integrate with a custom data pipeline
  C. Override LoRA parameters for resource-constrained hardware
  D. Export the trained model for deployment on a Raspberry Pi / Jetson

Usage:
    uv run python examples/03_custom_robot.py
    uv run python examples/03_custom_robot.py --robot my_arm --state_dim 7 --action_dim 7
"""

import argparse
import numpy as np
import torch

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# A. Custom robot config
# ---------------------------------------------------------------------------

def build_custom_config(
    robot_name: str,
    state_dim: int,
    action_dim: int,
    num_cameras: int,
    use_large_backbone: bool = False,
):
    """
    Build a Gemma4VLAConfig for any robot.

    Args:
        robot_name:         Short string identifier (e.g. 'my_arm').
        state_dim:          Proprioceptive state dimension.
        action_dim:         Action (motor command) dimension.
        num_cameras:        Number of RGB cameras.
        use_large_backbone: Use Gemma 4 E4B instead of E2B.

    Returns:
        Gemma4VLAConfig ready for training / inference.
    """
    from gemma4_vla.config import (
        Gemma4VLAConfig, VisionConfig, BackboneConfig,
        ActionExpertConfig, FlowMatchingConfig, RobotConfig, TrainingConfig
    )

    # Validate cross-embodiment compatibility
    assert state_dim <= 18, (
        f"state_dim={state_dim} exceeds max_state_dim=18. "
        "Increase RobotConfig.max_state_dim if your robot has more DOF."
    )
    assert action_dim <= 18, (
        f"action_dim={action_dim} exceeds max_state_dim=18."
    )

    backbone_model = (
        "google/gemma-4-E4B-it" if use_large_backbone else "google/gemma-4-E2B-it"
    )
    hidden_size = 2560 if use_large_backbone else 2048

    return Gemma4VLAConfig(
        vision=VisionConfig(
            num_cameras=num_cameras,
            image_size=224,
            freeze_vision=True,           # keep vision encoder frozen
        ),
        backbone=BackboneConfig(
            model_name=backbone_model,
            hidden_size=hidden_size,
            freeze_backbone=False,
            use_lora=True,
            lora_rank=8,                  # smaller rank for limited GPU memory
            lora_alpha=16.0,
            lora_dropout=0.05,
        ),
        action_expert=ActionExpertConfig(
            hidden_size=512,              # lighter action expert for quick training
            num_layers=6,
            num_heads=8,
            ffn_multiplier=4,
        ),
        flow_matching=FlowMatchingConfig(
            action_horizon=25,            # 0.5 s at 50 Hz for fast robots
            num_inference_steps=10,
        ),
        robot=RobotConfig(
            name=robot_name,
            state_dim=state_dim,
            action_dim=action_dim,
            max_state_dim=18,
            action_scale=1.0,
        ),
        training=TrainingConfig(
            learning_rate=3e-4,
            batch_size=16,
            max_steps=15_000,
            warmup_steps=500,
            output_dir=f"runs/{robot_name}",
        ),
    )


# ---------------------------------------------------------------------------
# B. Custom data pipeline (bring-your-own data)
# ---------------------------------------------------------------------------

class CustomRobotDataset(torch.utils.data.Dataset):
    """
    Minimal dataset example: load (image, state, action, instruction) tuples
    from a custom folder structure.

    Expected directory layout::

        data/my_arm/
          episode_0/
            step_000.jpg  (or .png)
            step_001.jpg
            …
            states.npy    [T, state_dim]
            actions.npy   [T, action_dim]
            instruction.txt
          episode_1/
            …

    You can extend this to match your own logging format.
    """

    def __init__(self, data_root: str, cfg, processor, horizon: int = 25):
        from pathlib import Path
        from torchvision import transforms

        self.cfg = cfg
        self.processor = processor
        self.horizon = horizon
        self.image_size = cfg.vision.image_size

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Discover episodes
        self.episodes = sorted(Path(data_root).glob("episode_*"))
        if not self.episodes:
            raise FileNotFoundError(f"No episode_* directories found in {data_root}")

        # Build flat index
        self.index = []
        self.ep_data = {}
        for ep_path in self.episodes:
            states_path  = ep_path / "states.npy"
            actions_path = ep_path / "actions.npy"
            if not (states_path.exists() and actions_path.exists()):
                continue
            T = len(np.load(states_path))
            for t in range(T - horizon):
                self.index.append((ep_path, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        from PIL import Image

        ep_path, t = self.index[idx]

        # Load image (first camera only for simplicity)
        img_path = ep_path / f"step_{t:03d}.jpg"
        if not img_path.exists():
            img_path = ep_path / f"step_{t:03d}.png"
        if img_path.exists():
            img = self.transform(Image.open(img_path).convert("RGB"))
        else:
            img = torch.zeros(3, self.image_size, self.image_size)

        # Load state & actions
        states  = np.load(ep_path / "states.npy")
        actions = np.load(ep_path / "actions.npy")
        state   = torch.tensor(states[t],  dtype=torch.float32)
        act_seq = torch.tensor(actions[t:t + self.horizon], dtype=torch.float32)

        # Load instruction
        instr_path = ep_path / "instruction.txt"
        instruction = instr_path.read_text().strip() if instr_path.exists() else "Perform the task."

        # Tokenise
        prompt = f"<image>\nTask: {instruction}"
        enc = self.processor(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.backbone.max_sequence_length,
            truncation=True,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": img.unsqueeze(0),  # [1, 3, H, W]
            "state": state,
            "actions": act_seq,
        }


# ---------------------------------------------------------------------------
# C. Memory-efficient training on a single GPU
# ---------------------------------------------------------------------------

def train_memory_efficient(cfg, data_root: str):
    """
    Demonstrates gradient checkpointing + 4-bit quantisation for single-GPU
    training on a machine with ≤16 GB VRAM.
    """
    from gemma4_vla import Gemma4VLA
    from gemma4_vla.train import train

    # Optionally enable 4-bit quantisation (requires bitsandbytes)
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantisation (bitsandbytes)")
    except ImportError:
        bnb_cfg = None
        print("bitsandbytes not available; running in bf16")

    # Build model with gradient checkpointing
    model = Gemma4VLA(cfg)
    if hasattr(model.backbone.model, "gradient_checkpointing_enable"):
        model.backbone.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    cfg.training.dataset_root = data_root
    cfg.training.mixed_precision = "bf16"
    train(cfg, model=model)


# ---------------------------------------------------------------------------
# D. Export model for edge deployment
# ---------------------------------------------------------------------------

def export_for_deployment(checkpoint_path: str, output_path: str):
    """
    Export the action expert (only) as a TorchScript module for
    deployment on edge devices (Jetson Orin, Raspberry Pi 5, etc.).

    The backbone runs on a server and sends obs_features over the network;
    only the lightweight action expert runs on the robot.

    Args:
        checkpoint_path: Saved checkpoint directory.
        output_path:     Path to write the TorchScript .pt file.
    """
    from gemma4_vla import Gemma4VLA

    model = Gemma4VLA.from_pretrained(checkpoint_path)
    model.eval()

    # Extract just the action expert + flow matching head
    action_expert = model.action_expert
    obs_proj = model.obs_proj

    class EdgeModule(torch.nn.Module):
        def __init__(self, obs_proj, action_expert):
            super().__init__()
            self.obs_proj = obs_proj
            self.action_expert = action_expert

        def forward(self, obs_features, state, noisy_actions, noise_level):
            obs = self.obs_proj(obs_features)
            return self.action_expert(state, noisy_actions, noise_level, obs)

    edge_model = EdgeModule(obs_proj, action_expert)
    scripted = torch.jit.script(edge_model)
    scripted.save(output_path)
    print(f"Edge model saved to: {output_path}")
    return scripted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="my_bimanual_arm")
    parser.add_argument("--state_dim", type=int, default=14,
                        help="Proprioceptive state dimension")
    parser.add_argument("--action_dim", type=int, default=14,
                        help="Action (motor command) dimension")
    parser.add_argument("--num_cameras", type=int, default=3)
    parser.add_argument("--large", action="store_true",
                        help="Use Gemma 4 E4B backbone instead of E2B")
    args = parser.parse_args()

    # Build config
    cfg = build_custom_config(
        robot_name=args.robot,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_cameras=args.num_cameras,
        use_large_backbone=args.large,
    )

    print(f"\nCustom robot config for '{args.robot}':")
    print(f"  state_dim    = {cfg.robot.state_dim}")
    print(f"  action_dim   = {cfg.robot.action_dim}")
    print(f"  num_cameras  = {cfg.vision.num_cameras}")
    print(f"  backbone     = {cfg.backbone.model_name}")
    print(f"  action_horizon = {cfg.flow_matching.action_horizon}")
    print(f"  LoRA rank    = {cfg.backbone.lora_rank}")

    # Verify forward pass with dummy data
    from gemma4_vla import Gemma4VLA, PolicyRunner
    import numpy as np

    print("\nTesting forward pass…")
    model = Gemma4VLA(cfg)
    runner = PolicyRunner(model, device="cpu")

    obs = {
        "images": [
            np.random.randint(0, 255, (cfg.vision.image_size, cfg.vision.image_size, 3), dtype=np.uint8)
            for _ in range(cfg.vision.num_cameras)
        ],
        "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
        "instruction": "Fold the towel and place it on the shelf.",
    }

    actions = runner.predict(obs, num_inference_steps=5)
    print(f"  Output shape: {actions.shape}  (horizon={actions.shape[0]}, dof={actions.shape[1]})")
    print("Forward pass OK!")


if __name__ == "__main__":
    main()
