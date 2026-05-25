"""Tests for inference preprocessing and camera tensor shapes."""

import numpy as np
import torch
from PIL import Image

from gemma4_vla.config import Gemma4VLAConfig, metaworld_push_config
from gemma4_vla.inference import PolicyRunner


class DummyEncoding(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}


class DummyProcessor:
    """Mimics the new chat-template + multimodal processor path enough for
    PolicyRunner._preprocess to exercise the multi-camera glue."""

    def __init__(self):
        self.last_call = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<chat-template-prompt>"

    def __call__(self, text, images=None, return_tensors=None, padding=None,
                 max_length=None, truncation=None, **kwargs):
        if isinstance(images, list):
            num_images = len(images)
        elif images is None:
            num_images = 0
        else:
            num_images = 1
        self.last_call = {"text": text, "num_images": num_images}
        result = {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }
        if num_images > 0:
            # Mirror the canonical [B, num_cameras, 3, H, W] shape the real
            # processor returns when given multiple images.
            result["pixel_values"] = torch.zeros(1, num_images, 3, 224, 224)
        return DummyEncoding(result)


class DummyBackbone:
    def __init__(self):
        self.processor = DummyProcessor()


class DummyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone = DummyBackbone()
        self.stats = None

    def to(self, device):
        return self

    def eval(self):
        return self


class TestPolicyRunnerPreprocess:
    def test_single_camera_passes_one_image_to_processor(self):
        cfg = metaworld_push_config()
        runner = PolicyRunner(DummyModel(cfg), device="cpu")

        obs = {
            "images": [np.zeros((32, 32, 3), dtype=np.uint8)],
            "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
            "instruction": "test",
        }

        batch = runner._preprocess(obs)

        assert runner._processor.last_call["num_images"] == 1
        assert batch["pixel_values"].shape[1] == 1

    def test_multi_camera_passes_all_images_to_processor(self):
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

        assert runner._processor.last_call["num_images"] == cfg.vision.num_cameras
        assert batch["pixel_values"].shape[1] == cfg.vision.num_cameras

    def test_accepts_pil_images(self):
        cfg = metaworld_push_config()
        runner = PolicyRunner(DummyModel(cfg), device="cpu")

        obs = {
            "images": [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))],
            "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
            "instruction": "test",
        }

        batch = runner._preprocess(obs)

        assert batch["state"].shape == (1, cfg.robot.state_dim)
        assert batch["input_ids"].shape == (1, cfg.backbone.max_sequence_length)
