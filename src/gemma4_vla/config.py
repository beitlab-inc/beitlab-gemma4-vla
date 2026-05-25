"""
Configuration classes for Gemma4VLA.

Gemma4VLA follows the pi0 architecture but replaces PaliGemma with Gemma 4
as the vision-language backbone. Gemma 4 natively supports multimodal inputs
(images + text) with up to 256K context via its hybrid local/global attention.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, List, Mapping, Optional


def _apply_dataclass_overrides(instance: Any, overrides: Mapping[str, Any], path: str = "config") -> None:
    """Recursively apply nested overrides onto a dataclass instance."""
    field_map = {f.name: f for f in fields(instance)}
    unknown_keys = sorted(set(overrides) - set(field_map))
    if unknown_keys:
        raise KeyError(f"Unknown config keys under '{path}': {', '.join(unknown_keys)}")

    for name, value in overrides.items():
        current = getattr(instance, name)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_dataclass_overrides(current, value, path=f"{path}.{name}")
        else:
            setattr(instance, name, value)


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """
    Configuration for the visual processing.

    Gemma 4 has a built-in vision tower, so we use it directly through the
    Gemma4ForConditionalGeneration backbone rather than a separate SigLIP
    encoder. If you prefer a standalone vision encoder (e.g. for ablations),
    set `use_external_encoder=True` and supply `external_encoder_name`.
    """
    num_cameras: int = 2
    """Number of RGB camera inputs per observation."""

    camera_names: Optional[List[str]] = None
    """
    Names of the cameras to load from each episode, in the order they should
    be fed to the vision encoder. Dataset adapters that store frames keyed by
    camera name (e.g. the MetaWorld HDF5 layout) use this to pick exactly
    which cameras train. When None, adapters fall back to picking the first
    `num_cameras` keys in whatever order the storage backend reports them
    (alphabetical for HDF5) — this is unsafe when an episode contains more
    cameras than the model uses, so set this list explicitly in production.
    When set, `len(camera_names)` must equal `num_cameras`.
    """

    image_size: int = 224
    """Spatial resolution fed to the vision encoder (height == width)."""

    use_external_encoder: bool = False
    """
    When False, Gemma 4's built-in SigLIP2 vision tower is used.
    When True, a standalone SigLIP2 encoder is loaded separately.
    """

    external_encoder_name: str = "google/siglip2-so400m-patch16-224"
    """HuggingFace model ID for the standalone vision encoder."""

    freeze_vision: bool = True
    """Freeze vision encoder weights during training."""


@dataclass
class BackboneConfig:
    """Configuration for the Gemma 4 language/multimodal backbone."""

    model_name: str = "google/gemma-4-E2B-it"
    """
    HuggingFace model ID.  Choose based on your compute budget:
      - google/gemma-4-E2B-it   (~2B,   ~4 GB VRAM at bf16)
      - google/gemma-4-E4B-it   (~4B,   ~8 GB VRAM at bf16)
      - google/gemma-4-26B-A4B-it (26B MoE, ~4B active, ~24 GB VRAM at bf16)
      - google/gemma-4-31B-it   (31B,  ~55 GB VRAM at bf16)
    """

    hidden_size: int = 2048
    """Hidden dimension (set automatically from model_name when loading)."""

    freeze_backbone: bool = False
    """
    Freeze the entire backbone. Useful when fine-tuning only the action
    expert on a small robot dataset.
    """

    use_lora: bool = True
    """Apply LoRA to the backbone for parameter-efficient fine-tuning."""

    lora_rank: int = 16
    """LoRA rank.  Higher = more capacity, more parameters."""

    lora_alpha: float = 32.0
    """LoRA scaling factor (alpha / rank scales the updates)."""

    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    """Which attention projections to apply LoRA to."""

    lora_dropout: float = 0.05
    """Dropout rate inside LoRA adapters."""

    max_sequence_length: int = 1024
    """Maximum number of tokens (images + text) passed to the backbone.

    The HF processor pads every sample to this length, so a high value costs
    compute even when instructions are short. Pick the smallest value that
    holds your longest prompt + the image tokens for your `vision.num_cameras`.
    The figure in the README about 256K context is the Gemma 4 *capability*;
    in practice we run with the much smaller value here.
    """


@dataclass
class ActionExpertConfig:
    """
    Configuration for the action expert transformer.

    The action expert is a separate lightweight transformer (~300 M params)
    with its own weights.  It processes [state_tokens, noisy_action_tokens]
    and cross-attends to observation features from the Gemma 4 backbone.
    """

    hidden_size: int = 1024
    """Hidden dimension of the action expert."""

    num_layers: int = 8
    """Number of transformer layers."""

    num_heads: int = 16
    """Number of attention heads."""

    head_dim: int = 64
    """Dimension per head (hidden_size / num_heads = 64)."""

    ffn_multiplier: int = 4
    """Feed-forward expansion factor."""

    dropout: float = 0.0
    """Attention / FFN dropout."""

    use_cross_attention: bool = True
    """
    Whether action expert uses cross-attention to backbone features.
    Set False for a simpler concatenation-based architecture.
    """

    noise_embedding_dim: int = 256
    """Sinusoidal noise-level embedding dimension."""


@dataclass
class FlowMatchingConfig:
    """
    Configuration for the conditional flow matching action head.

    We implement the optimal-transport (OT) probability path from
    Lipman et al. (2023), matching the pi0 formulation exactly.
    """

    action_horizon: int = 50
    """
    Number of future action steps to predict at once.
    pi0 predicts 50 steps at 50 Hz → 1 second of motion.
    """

    num_inference_steps: int = 10
    """
    Number of Euler integration steps during inference.
    10 steps gives a good quality / speed trade-off.
    Increase to 50 for smoother trajectories.
    """

    sigma_min: float = 1e-4
    """Minimum noise std for the OT flow path (avoids degenerate samples)."""

    time_schedule: str = "linear"
    """
    Noise schedule for the OT path.
    Options: 'linear', 'cosine'.
    """


@dataclass
class RobotConfig:
    """
    Per-embodiment robot configuration.

    Following pi0, we pad state / action vectors to `max_dim` so that
    a single model can control diverse robots without architecture changes.
    """

    name: str = "metaworld-push"
    """Human-readable robot name."""

    state_dim: int = 39
    """Proprioceptive state dimension (e.g. 39 for MetaWorld)."""

    action_dim: int = 4
    """Action space dimension."""

    max_state_dim: int = 39
    """
    Padding target for cross-embodiment training.
    All state / action vectors are zero-padded to this size.

    NOTE: this must be >= the max of `state_dim` and `action_dim` across every
    embodiment you intend to mix.  Setting it equal to `state_dim` (as
    `metaworld_push_config()` does) is the right floor for single-robot
    training but leaves no headroom — bump it when wiring up cross-embodiment
    training (roadmap §1.6). Changing it invalidates existing checkpoints,
    because the action expert's state_proj / action_proj / velocity_head are
    sized to this value.
    """

    action_scale: float = 1.0
    """
    Scalar applied to actions before / after the model.
    Set to the range of your actuator commands to keep actions in [-1, 1].
    """


@dataclass
class TrainingConfig:
    """Optimizer and schedule hyper-parameters."""

    learning_rate: float = 2e-4
    backbone_lr_multiplier: float = 0.1
    """Backbone LR = learning_rate * backbone_lr_multiplier."""

    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 32
    grad_accum_steps: int = 1
    """Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps."""
    grad_clip_norm: float = 1.0
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing on the backbone to save memory."""
    mixed_precision: str = "bf16"
    """Options: 'no', 'fp16', 'bf16'."""

    save_every_n_steps: int = 5_000
    eval_every_n_steps: int = 1_000
    log_every_n_steps: int = 50
    output_dir: str = "./checkpoints"

    # Data
    dataset_root: str = "./data"
    num_workers: int = 4
    prefetch_factor: int = 2

    # Augmentation
    use_color_jitter: bool = True
    use_random_crop: bool = True
    crop_scale: float = 0.9

    # Normalisation
    normalize_stats: bool = False
    """Compute per-dim state / action stats once before training, normalise
    inputs during training, and save them next to the checkpoint so
    ``PolicyRunner`` can denormalise at inference time."""
    normalize_stats_batches: int = 200
    """Maximum number of batches scanned when computing stats. Set None to
    walk the whole loader once."""


@dataclass
class Gemma4VLAConfig:
    """Top-level configuration for Gemma4VLA."""

    vision: VisionConfig = field(default_factory=VisionConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate and sync dependent fields."""
        ae = self.action_expert
        assert ae.hidden_size % ae.num_heads == 0, (
            f"action_expert hidden_size ({ae.hidden_size}) must be divisible "
            f"by num_heads ({ae.num_heads})"
        )
        v = self.vision
        if v.camera_names is not None and len(v.camera_names) != v.num_cameras:
            raise ValueError(
                f"vision.camera_names has {len(v.camera_names)} entries "
                f"({v.camera_names}) but vision.num_cameras is {v.num_cameras}; "
                f"they must match."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the config tree to a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]] = None) -> "Gemma4VLAConfig":
        """Build a config from nested dict overrides."""
        cfg = cls()
        if raw:
            _apply_dataclass_overrides(cfg, raw)
            cfg.__post_init__()
        return cfg


# ---------------------------------------------------------------------------
# Pre-built configs for common robots
# ---------------------------------------------------------------------------

def metaworld_push_config() -> Gemma4VLAConfig:
    """Config for MetaWorld push-v3 task (state_dim=39, action_dim=4)."""
    cfg = Gemma4VLAConfig()
    cfg.robot = RobotConfig(name="metaworld-push", state_dim=39, action_dim=4, max_state_dim=39)
    cfg.vision.num_cameras = 1
    return cfg


