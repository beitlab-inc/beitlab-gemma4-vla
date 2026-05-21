"""Tests for the dataset utilities (no robot data required)."""

import pytest
import torch
from torch.utils.data import DataLoader

from gemma4_vla.config import metaworld_push_config
from gemma4_vla.dataset import RandomDemoDataset, collate_fn


class TestRandomDemoDataset:
    @pytest.fixture
    def cfg(self):
        cfg = metaworld_push_config()
        cfg.backbone.max_sequence_length = 32  # small for testing
        return cfg

    def test_len(self, cfg):
        ds = RandomDemoDataset(cfg, n_samples=100)
        assert len(ds) == 100

    def test_item_keys(self, cfg):
        ds = RandomDemoDataset(cfg, n_samples=10)
        item = ds[0]
        expected_keys = {"input_ids", "attention_mask", "pixel_values", "state", "actions"}
        assert set(item.keys()) == expected_keys

    def test_item_shapes(self, cfg):
        ds = RandomDemoDataset(cfg, n_samples=10)
        item = ds[0]

        H = cfg.flow_matching.action_horizon
        D = cfg.robot.action_dim
        S = cfg.robot.state_dim
        C = cfg.vision.num_cameras
        I = cfg.vision.image_size
        T = cfg.backbone.max_sequence_length

        assert item["input_ids"].shape == (T,), f"input_ids: {item['input_ids'].shape}"
        assert item["attention_mask"].shape == (T,)
        assert item["pixel_values"].shape == (C, 3, I, I), f"pixels: {item['pixel_values'].shape}"
        assert item["state"].shape == (S,)
        assert item["actions"].shape == (H, D)

    def test_pixel_values_range(self, cfg):
        ds = RandomDemoDataset(cfg, n_samples=10)
        item = ds[0]
        # pixel_values should be random floats (not normalised in RandomDemoDataset)
        pv = item["pixel_values"]
        assert pv.dtype == torch.float32

    def test_dataloader_collation(self, cfg):
        ds = RandomDemoDataset(cfg, n_samples=16)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        H = cfg.flow_matching.action_horizon
        D = cfg.robot.action_dim
        C = cfg.vision.num_cameras
        I = cfg.vision.image_size
        T = cfg.backbone.max_sequence_length

        assert batch["input_ids"].shape == (4, T)
        assert batch["pixel_values"].shape == (4, C, 3, I, I)
        assert batch["state"].shape == (4, cfg.robot.state_dim)
        assert batch["actions"].shape == (4, H, D)


class TestCollateFn:
    def test_stacks_tensors(self):
        samples = [
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor(3.0)}
            for _ in range(4)
        ]
        batch = collate_fn(samples)
        assert batch["a"].shape == (4, 2)
        assert batch["b"].shape == (4,)

    def test_preserves_keys(self):
        samples = [{"x": torch.randn(3), "y": torch.randn(5)} for _ in range(2)]
        batch = collate_fn(samples)
        assert set(batch.keys()) == {"x", "y"}
