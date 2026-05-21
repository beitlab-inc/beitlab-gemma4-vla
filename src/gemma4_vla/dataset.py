"""
Core dataset utilities for Gemma4VLA.

This module provides reusable building blocks that any robot-specific dataset
adapter can use:

  - Image transforms (train + eval)
  - RandomDemoDataset for testing without real data
  - collate_fn for batching

Robot-specific dataset adapters live under ``robots/<name>/dataset.py`` and
import these utilities.  For example ``robots/metaworld/dataset.py`` provides
the MetaWorld HDF5 dataset adapter.

The dataset contract — every adapter must produce batches with::

    {
      "input_ids":      LongTensor  [T]     - tokenised instruction
      "attention_mask": LongTensor  [T]
      "pixel_values":   FloatTensor [num_cameras, 3, H, W]  - stacked images
      "state":          FloatTensor [state_dim]
      "actions":        FloatTensor [horizon, action_dim]
    }

Images are normalised to [-1, 1] (Gemma 4 convention).
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import Gemma4VLAConfig


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def build_train_transform(image_size: int, use_color_jitter: bool, use_random_crop: bool, crop_scale: float):
    t = [transforms.Resize((image_size, image_size), antialias=True)]
    if use_random_crop:
        t.append(transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0), antialias=True))
    if use_color_jitter:
        t.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05))
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(t)


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Simple in-memory demo dataset (no real data required)
# ---------------------------------------------------------------------------

class RandomDemoDataset(Dataset):
    """
    Synthetic random dataset for testing model forward/backward passes
    without needing real robot data.
    """

    def __init__(
        self,
        cfg: Gemma4VLAConfig,
        n_samples: int = 256,
        processor=None,
        instruction: str = "Pick up the red cup.",
    ):
        self.cfg = cfg
        self.n_samples = n_samples
        self.processor = processor
        self.instruction = instruction
        self.horizon = cfg.flow_matching.action_horizon
        self.state_dim = cfg.robot.state_dim
        self.action_dim = cfg.robot.action_dim
        self.num_cameras = cfg.vision.num_cameras
        self.image_size = cfg.vision.image_size
        self.max_seq_len = cfg.backbone.max_sequence_length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        state = torch.randn(self.state_dim) * 0.3
        actions = torch.randn(self.horizon, self.action_dim) * 0.1

        if self.processor is not None:
            import numpy as np
            from PIL import Image

            # Generate random PIL images for proper processor pipeline
            images = []
            for _ in range(self.num_cameras):
                arr = np.random.randint(
                    0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8
                )
                images.append(Image.fromarray(arr))

            # Build chat template for correct <|image|> token generation
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": f"Task: {self.instruction}"})
            messages = [{"role": "user", "content": content}]

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            proc_images = images[0] if len(images) == 1 else images
            enc = self.processor(
                text=prompt,
                images=proc_images,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
            )

            result = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "pixel_values": enc["pixel_values"].squeeze(0),
                "state": state,
                "actions": actions,
            }
            if "image_position_ids" in enc:
                result["image_position_ids"] = enc["image_position_ids"].squeeze(0)
            if "mm_token_type_ids" in enc:
                result["mm_token_type_ids"] = enc["mm_token_type_ids"].squeeze(0)
            return result
        else:
            # No processor — text-only fallback (no pixel_values)
            input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "state": state,
                "actions": actions,
            }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack a list of sample dicts into a batch dict."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        collated[k] = torch.stack(tensors, dim=0)
    return collated
