"""Tests for inference preprocessing and camera tensor shapes."""

import numpy as np
import torch

from gemma4_vla.config import Gemma4VLAConfig, metaworld_push_config
from gemma4_vla.inference import PolicyRunner


class DummyEncoding(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}


class DummyProcessor:
    def __call__(self, text, return_tensors, padding, max_length, truncation):
        return DummyEncoding({
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        })


class DummyBackbone:
    def __init__(self):
        self.processor = DummyProcessor()


class DummyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone = DummyBackbone()

    def to(self, device):
        return self

    def eval(self):
        return self


class TestPolicyRunnerPreprocess:
    def test_single_camera_keeps_explicit_camera_axis(self):
        cfg = metaworld_push_config()
        runner = PolicyRunner(DummyModel(cfg), device="cpu")

        obs = {
            "images": [np.zeros((32, 32, 3), dtype=np.uint8)],
            "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
            "instruction": "test",
        }

        batch = runner._preprocess(obs)

        assert batch["pixel_values"].shape == (1, 1, 3, cfg.vision.image_size, cfg.vision.image_size)

    def test_multi_camera_keeps_explicit_camera_axis(self):
        cfg = Gemma4VLAConfig()
        cfg.vision.num_cameras = 3
        runner = PolicyRunner(DummyModel(cfg), device="cpu")

        obs = {
            "images": [
                np.zeros((32, 32, 3), dtype=np.uint8)
                for _ in range(cfg.vision.num_cameras)
            ],
            "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
            "instruction": "test",
        }

        batch = runner._preprocess(obs)

        assert batch["pixel_values"].shape == (
            1,
            cfg.vision.num_cameras,
            3,
            cfg.vision.image_size,
            cfg.vision.image_size,
        )
