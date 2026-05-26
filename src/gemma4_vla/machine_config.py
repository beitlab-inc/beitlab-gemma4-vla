"""
Machine-specific configuration detection and defaults.

Allows training to run on Jetson Thor, desktop GPUs, or cloud instances
with automatic environment-specific adjustments.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MachineProfile:
    """Detect machine type and set environment defaults."""

    def __init__(self):
        self.device_name = self._detect_device()
        self.machine_type = self._detect_machine_type()
        self.available_vram = self._get_available_vram()

    def _detect_device(self) -> str:
        """Detect GPU device type."""
        if not torch.cuda.is_available():
            return "cpu"
        return torch.cuda.get_device_name(0)

    def _detect_machine_type(self) -> str:
        """Detect machine type from device name."""
        device = self.device_name.lower()
        if "thor" in device or "jetson" in device:
            return "jetson_thor"
        elif "a100" in device:
            return "cloud_a100"
        elif "h100" in device:
            return "cloud_h100"
        elif "l40" in device or "l40s" in device:
            return "cloud_l40s"
        elif "v100" in device:
            return "cloud_v100"
        elif "gpu" in device:
            return "desktop_gpu"
        else:
            return "unknown"

    def _get_available_vram(self) -> int:
        """Get available GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    @property
    def is_jetson(self) -> bool:
        return self.machine_type == "jetson_thor"

    @property
    def is_cloud(self) -> bool:
        return self.machine_type.startswith("cloud_")

    @property
    def recommended_batch_size(self) -> int:
        """Recommend batch size based on VRAM and machine type."""
        if self.is_jetson:
            return 2  # Jetson Thor: shared system RAM, conservative
        elif self.available_vram < 16:
            return 4
        elif self.available_vram < 40:
            return 8
        else:
            return 16

    @property
    def recommended_grad_accum_steps(self) -> int:
        """Recommend gradient accumulation steps."""
        if self.is_jetson:
            return 4  # Simulate larger batch on constrained hardware
        elif self.available_vram < 40:
            return 2
        else:
            return 1

    @property
    def use_gradient_checkpointing(self) -> bool:
        """Whether to enable gradient checkpointing."""
        return self.is_jetson or self.available_vram < 20

    @property
    def recommended_precision(self) -> str:
        """Recommend mixed precision based on hardware."""
        # Check if GPU supports bfloat16
        if torch.cuda.is_available():
            # Most modern GPUs support bf16
            if hasattr(torch.cuda, "is_bf16_supported"):
                if torch.cuda.is_bf16_supported():
                    return "bf16"
        return "fp16"

    def log_profile(self):
        """Log detected machine profile."""
        logger.info(f"Machine Type: {self.machine_type}")
        logger.info(f"Device: {self.device_name}")
        logger.info(f"Available VRAM: {self.available_vram:.1f} GB")
        logger.info(f"Recommended batch size: {self.recommended_batch_size}")
        logger.info(f"Recommended grad accum steps: {self.recommended_grad_accum_steps}")
        logger.info(f"Recommended precision: {self.recommended_precision}")


class EnvironmentConfig:
    """Load environment-specific paths and MLflow settings."""

    def __init__(self):
        self.data_dir = self._resolve_data_dir()
        self.output_dir = self._resolve_output_dir()
        self.mlflow_tracking_uri = self._resolve_mlflow_uri()
        self.mlflow_enabled = self._mlflow_enabled()

    def _resolve_data_dir(self) -> str:
        """Resolve data directory from env var or default."""
        return os.environ.get(
            "GEMMA4VLA_DATA_DIR",
            os.path.join(os.getcwd(), "data", "metaworld_demos")
        )

    def _resolve_output_dir(self) -> str:
        """Resolve output directory from env var or default."""
        return os.environ.get(
            "GEMMA4VLA_OUTPUT_DIR",
            os.path.join(os.getcwd(), "outputs")
        )

    def _resolve_mlflow_uri(self) -> str:
        """Resolve MLflow tracking URI from env var or default."""
        return os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5001"
        )

    def _mlflow_enabled(self) -> bool:
        """Check if MLflow should be enabled."""
        return os.environ.get("GEMMA4VLA_MLFLOW", "").lower() in ("true", "1", "yes")

    def log_paths(self):
        """Log resolved paths."""
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")
        logger.info(f"MLflow enabled: {self.mlflow_enabled}")


def apply_machine_defaults(
    cfg,
    machine: Optional[MachineProfile] = None,
    force: bool = False,
) -> None:
    """Apply machine-specific defaults to training config.

    Args:
        cfg: Gemma4VLAConfig object to modify in-place.
        machine: MachineProfile (auto-detected if None).
        force: If True, overwrite even non-default training fields. If False
            (default), only overwrite fields that still hold their dataclass
            default — explicit overrides from YAML/CLI are preserved.
    """
    if machine is None:
        machine = MachineProfile()

    from .config import TrainingConfig

    defaults = TrainingConfig()
    tr = cfg.training

    if force or tr.batch_size == defaults.batch_size:
        tr.batch_size = machine.recommended_batch_size

    if force or tr.grad_accum_steps == defaults.grad_accum_steps:
        tr.grad_accum_steps = machine.recommended_grad_accum_steps

    if force or tr.gradient_checkpointing == defaults.gradient_checkpointing:
        # Gradient checkpointing only saves memory when the backbone actually
        # runs a backward pass — pointless overhead when it's frozen.
        backbone_trains = (
            not cfg.backbone.freeze_backbone
        )
        tr.gradient_checkpointing = (
            machine.use_gradient_checkpointing and backbone_trains
        )

    if force or tr.mixed_precision == defaults.mixed_precision:
        tr.mixed_precision = machine.recommended_precision

    machine.log_profile()
