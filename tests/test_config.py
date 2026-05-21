"""Tests for config loading and checkpoint config artifacts."""

import json

import pytest
import torch

from gemma4_vla.config import Gemma4VLAConfig, metaworld_push_config
from gemma4_vla.model import load_config_artifact, save_config_artifact


class TestGemma4VLAConfig:
    def test_from_dict_applies_nested_overrides(self):
        cfg = Gemma4VLAConfig.from_dict({
            "vision": {"num_cameras": 3},
            "robot": {"name": "bimanual", "state_dim": 14, "action_dim": 14},
            "training": {"batch_size": 16},
        })

        assert cfg.vision.num_cameras == 3
        assert cfg.robot.name == "bimanual"
        assert cfg.robot.state_dim == 14
        assert cfg.robot.action_dim == 14
        assert cfg.training.batch_size == 16

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(KeyError):
            Gemma4VLAConfig.from_dict({"robot": {"unknown_field": 1}})

    def test_to_dict_round_trip(self):
        cfg = metaworld_push_config()
        cfg.training.batch_size = 8

        restored = Gemma4VLAConfig.from_dict(cfg.to_dict())

        assert restored.vision.num_cameras == 1
        assert restored.robot.name == "metaworld-push"
        assert restored.training.batch_size == 8

    def test_from_dict_revalidates_dependent_fields(self):
        with pytest.raises(AssertionError):
            Gemma4VLAConfig.from_dict({
                "action_expert": {"hidden_size": 1000, "num_heads": 16},
            })


class TestConfigArtifacts:
    def test_save_config_artifact_writes_json_and_pt(self, tmp_path):
        cfg = metaworld_push_config()

        save_config_artifact(cfg, str(tmp_path))

        json_path = tmp_path / "config.json"
        pt_path = tmp_path / "config.pt"

        assert json_path.exists()
        assert pt_path.exists()

        raw_json = json.loads(json_path.read_text())
        assert raw_json["robot"]["name"] == "metaworld-push"

    def test_load_config_artifact_from_json(self, tmp_path):
        cfg = metaworld_push_config()
        save_config_artifact(cfg, str(tmp_path))

        loaded = load_config_artifact(str(tmp_path))

        assert loaded.robot.name == "metaworld-push"
        assert loaded.vision.num_cameras == 1

    def test_load_config_artifact_from_legacy_pickled_dataclass(self, tmp_path):
        cfg = metaworld_push_config()
        torch.save(cfg, tmp_path / "config.pt")

        loaded = load_config_artifact(str(tmp_path))

        assert loaded.robot.name == "metaworld-push"
        assert loaded.robot.action_dim == 4
