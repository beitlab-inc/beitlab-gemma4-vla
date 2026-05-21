"""
gemma4_vla — Vision-Language-Action model built on Gemma 4.

A PyTorch re-implementation of the pi0 architecture that replaces the
PaliGemma backbone with Google's Gemma 4 multimodal model.

The core model is robot-agnostic.  Robot-specific adapters (environments,
datasets, training scripts) live under ``robots/<name>/``.

Quick start::

    from gemma4_vla import Gemma4VLA, Gemma4VLAConfig, PolicyRunner

    cfg    = Gemma4VLAConfig()
    model  = Gemma4VLA(cfg)
    runner = PolicyRunner(model, device="cuda")

    actions = runner.predict({
        "images":      [camera_image],
        "state":       robot_state,
        "instruction": "Pick up the red cube",
    })
"""

from .config import (
    Gemma4VLAConfig,
    VisionConfig,
    BackboneConfig,
    ActionExpertConfig,
    FlowMatchingConfig,
    RobotConfig,
    TrainingConfig,
    metaworld_push_config,
)
from .model import Gemma4VLA
from .inference import PolicyRunner
from .flow_matching import (
    SinusoidalEmbedding,
    ot_flow_interpolate,
    flow_matching_loss,
    euler_integration,
    rk4_integration,
)
from .observability import RerunLogger, MlflowRun

__version__ = "0.1.0"
__author__ = "Gemma4VLA contributors"

__all__ = [
    # Config
    "Gemma4VLAConfig",
    "VisionConfig",
    "BackboneConfig",
    "ActionExpertConfig",
    "FlowMatchingConfig",
    "RobotConfig",
    "TrainingConfig",
    "metaworld_push_config",
    # Model
    "Gemma4VLA",
    # Inference
    "PolicyRunner",
    # Flow matching utilities
    "SinusoidalEmbedding",
    "ot_flow_interpolate",
    "flow_matching_loss",
    "euler_integration",
    "rk4_integration",
    # Observability
    "RerunLogger",
    "MlflowRun",
]
