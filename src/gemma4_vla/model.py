"""
Gemma4VLA — Vision-Language-Action model.

Architecture (mirrors pi0 but replaces PaliGemma with Gemma 4):

  ┌──────────────────────────────────────────────────────────────┐
  │                       Observation                            │
  │  Camera images ──► Gemma 4 Vision Tower                      │
  │  Language instr ──► Gemma 4 Tokenizer + Embeddings           │
  │                          │                                   │
  │                    Gemma 4 Backbone                          │
  │               (hybrid local + global attn)                   │
  │                          │                                   │
  │                   obs_features [B, S, D]                     │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ cross-attention
  ┌──────────────────────────────▼───────────────────────────────┐
  │                    Action Expert                              │
  │  state ──► state_token                                        │
  │  noisy_actions ──► action_tokens [B, H, D]                   │
  │  noise_level   ──► sinusoidal embedding                      │
  │                          │                                   │
  │              Action Expert Transformer                        │
  │                 (8 layers, cross-attn)                       │
  │                          │                                   │
  │                  velocity [B, H, action_dim]                 │
  └──────────────────────────────────────────────────────────────┘

Training:  Conditional Flow Matching (OT path)
Inference: Euler ODE integration, t: 0 → 1 in `num_inference_steps` steps
"""

import json
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import Gemma4VLAConfig
from .action_expert import ActionExpert
from .flow_matching import (
    ot_flow_interpolate,
    flow_matching_loss,
    euler_integration,
    rk4_integration,
)
from .stats import DatasetStats


CONFIG_JSON_NAME = "config.json"
LEGACY_CONFIG_NAME = "config.pt"
WEIGHTS_NAME = "weights.pt"


def _coerce_loaded_config(raw_cfg) -> Gemma4VLAConfig:
    """Normalise config artifacts from JSON, dict, or legacy pickled dataclasses."""
    if isinstance(raw_cfg, Gemma4VLAConfig):
        return raw_cfg
    if isinstance(raw_cfg, dict):
        return Gemma4VLAConfig.from_dict(raw_cfg)
    raise TypeError(f"Unsupported config artifact type: {type(raw_cfg)!r}")


def load_config_artifact(checkpoint_path: str) -> Gemma4VLAConfig:
    """Load a checkpoint config from JSON or the legacy torch artifact."""
    json_path = os.path.join(checkpoint_path, CONFIG_JSON_NAME)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return Gemma4VLAConfig.from_dict(json.load(f))

    legacy_path = os.path.join(checkpoint_path, LEGACY_CONFIG_NAME)
    if not os.path.exists(legacy_path):
        raise FileNotFoundError(
            f"No {CONFIG_JSON_NAME} or {LEGACY_CONFIG_NAME} found in {checkpoint_path}"
        )

    try:
        raw_cfg = torch.load(legacy_path, map_location="cpu")
    except pickle.UnpicklingError:
        # Backward compatibility for checkpoints created before PyTorch flipped
        # the default to weights_only=True.
        raw_cfg = torch.load(legacy_path, map_location="cpu", weights_only=False)

    return _coerce_loaded_config(raw_cfg)


def save_config_artifact(cfg: Gemma4VLAConfig, checkpoint_path: str) -> None:
    """Persist the config in JSON and torch formats."""
    config_dict = cfg.to_dict()
    json_path = os.path.join(checkpoint_path, CONFIG_JSON_NAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)

    # Keep the legacy filename for compatibility, but store a plain dict so it
    # remains loadable with weights_only=True.
    torch.save(config_dict, os.path.join(checkpoint_path, LEGACY_CONFIG_NAME))


def prepare_pixel_values(
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_list: Optional[List[torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """Normalise image tensors to [B*num_cameras, 3, H, W] for the backbone."""
    if pixel_values is None and pixel_values_list is None:
        return None

    if pixel_values is None:
        if len(pixel_values_list) == 0:
            return None
        pixel_values = torch.stack(pixel_values_list, dim=1)

    if pixel_values.dim() == 5:
        batch_size, num_cameras, channels, height, width = pixel_values.shape
        return pixel_values.reshape(batch_size * num_cameras, channels, height, width)

    if pixel_values.dim() == 4:
        return pixel_values

    raise ValueError(
        "pixel_values must have shape [B, num_cameras, 3, H, W] or [B*num_cameras, 3, H, W], "
        f"got {tuple(pixel_values.shape)}"
    )


# ---------------------------------------------------------------------------
# Gemma 4 backbone wrapper
# ---------------------------------------------------------------------------

class Gemma4Backbone(nn.Module):
    """
    Thin wrapper around a HuggingFace Gemma 4 multimodal model.

    We use the model as a feature extractor:
      - Feed image patches + text tokens
      - Return the last hidden states (not logits)

    The wrapper also handles optional LoRA injection for parameter-efficient
    fine-tuning.
    """

    def __init__(self, cfg: "Gemma4VLAConfig"):
        super().__init__()
        self.cfg = cfg
        bk = cfg.backbone

        # Lazy import so the module can be imported without transformers installed
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            # Try multimodal first, fall back to causal
            try:
                from transformers import Gemma4ForConditionalGeneration
                model_cls = Gemma4ForConditionalGeneration
            except ImportError:
                model_cls = AutoModelForCausalLM

            self.processor = AutoProcessor.from_pretrained(
                bk.model_name, trust_remote_code=True
            )
            # device_map="auto" shards the backbone across visible GPUs, which
            # conflicts with DDP (each rank wants its model on one device).
            # Disable it when we detect a torchrun launch.
            in_distributed = (
                "WORLD_SIZE" in os.environ
                and int(os.environ.get("WORLD_SIZE", "1")) > 1
            )
            use_device_map = (
                torch.cuda.is_available()
                and not bk.freeze_backbone
                and not in_distributed
            )
            self.model = model_cls.from_pretrained(
                bk.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if use_device_map else None,
                trust_remote_code=True,
            )

            # Disable cache if gradient checkpointing will be used (incompatible)
            if cfg.training.gradient_checkpointing:
                self.model.config.use_cache = False
        except Exception as e:
            raise ImportError(
                f"Could not load Gemma 4 model '{bk.model_name}'. "
                "Install transformers>=4.51.0 and ensure you have accepted "
                "the Gemma 4 license on Hugging Face.\n"
                f"Original error: {e}"
            )

        # Infer hidden_size from loaded model — Gemma 4 multimodal stores
        # it under text_config, not at the top level.
        try:
            self.hidden_size = self.model.config.text_config.hidden_size
        except AttributeError:
            try:
                self.hidden_size = self.model.config.hidden_size
            except AttributeError:
                self.hidden_size = bk.hidden_size

        # Fail loud if the resolved model has no vision tower — otherwise we'd
        # silently train a text-only fallback while pretending to ingest images.
        # Gemma4ForConditionalGeneration nests these on the inner Gemma4Model
        # (self.model.model.vision_tower), so probe both levels.
        vision_attrs = ("vision_tower", "embed_vision", "vision_model")
        inner = getattr(self.model, "model", None)
        has_vision = any(hasattr(self.model, a) for a in vision_attrs) or any(
            hasattr(inner, a) for a in vision_attrs
        )
        if not has_vision and not getattr(cfg.vision, "use_external_encoder", False):
            raise RuntimeError(
                f"Loaded backbone '{bk.model_name}' exposes no vision tower "
                "(no vision_tower / embed_vision / vision_model attribute). "
                "This usually means the Gemma 4 multimodal weights are not "
                "yet published under that ID and AutoModelForCausalLM loaded "
                "a text-only fallback. Use a multimodal model ID, or set "
                "cfg.vision.use_external_encoder=True if you intend to "
                "supply a separate vision encoder."
            )

        # --- Freeze / LoRA strategy ---
        #
        # Strategy 1 – Action expert only:
        #   freeze_backbone=True, use_lora=False  → all backbone frozen
        #
        # Strategy 2 – LoRA + action expert:
        #   freeze_backbone=False, use_lora=True  → LoRA adapters train,
        #   base weights frozen (handled by PEFT), vision tower frozen
        #
        # Strategy 3 – Full fine-tune:
        #   freeze_backbone=False, use_lora=False → everything trains
        #   (freeze_vision still respected)

        if bk.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # Optionally apply LoRA (only when backbone is not fully frozen)
        if bk.use_lora and not bk.freeze_backbone:
            self._apply_lora(bk)

        # Freeze vision tower independently — important for LoRA mode
        # where PEFT only freezes the language model, leaving vision
        # tower accidentally unfrozen.
        vis = cfg.vision
        if vis.freeze_vision and not bk.freeze_backbone:
            self._freeze_vision_tower()

    def _apply_lora(self, bk):
        """Inject LoRA adapters into the language model only.

        Gemma 4's vision/audio towers use Gemma4ClippableLinear which PEFT
        cannot wrap.  Applying LoRA to model.model.language_model targets
        only the standard nn.Linear projections in the text transformer.
        """
        try:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=bk.lora_rank,
                lora_alpha=bk.lora_alpha,
                lora_dropout=bk.lora_dropout,
                target_modules=bk.lora_target_modules,
                bias="none",
            )
            # Apply to language model only — vision/audio towers use
            # Gemma4ClippableLinear which is unsupported by PEFT.
            self.model.model.language_model = get_peft_model(
                self.model.model.language_model, lora_cfg
            )
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"[Gemma4Backbone] LoRA applied to language model. "
                f"Trainable: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.1f}%)"
            )
        except ImportError:
            print(
                "[Gemma4Backbone] peft not installed — skipping LoRA. "
                "Install with: uv add peft"
            )

    def _freeze_vision_tower(self):
        """Freeze the vision tower parameters.

        In LoRA mode, PEFT only freezes the language model.  The vision
        tower (SigLIP2) would remain unfrozen unless we freeze it
        explicitly here.  Full fine-tuning with freeze_vision=False
        skips this call entirely.
        """
        frozen = 0
        if hasattr(self.model, "vision_tower"):
            for p in self.model.vision_tower.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        # Also freeze the vision embedder / projector if present
        if hasattr(self.model, "embed_vision"):
            for p in self.model.embed_vision.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        if frozen > 0:
            print(f"[Gemma4Backbone] Vision tower frozen: {frozen:,} params")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the Gemma 4 backbone and return last-layer hidden states.

        Args:
            input_ids:          Text token IDs [B, T].
            attention_mask:     Padding mask    [B, T].
            pixel_values:       Processor output [B, num_patches, patch_dim].
            image_position_ids: Patch position coordinates [B, num_patches, 2].
            mm_token_type_ids:  Multimodal token type IDs [B, T].

        Returns:
            Hidden states [B, S, hidden_size] where S = image_tokens + T.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        # Return last-layer hidden states
        return outputs.hidden_states[-1]  # [B, S, hidden_size]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Gemma4VLA(nn.Module):
    """
    Gemma4VLA: Vision-Language-Action model based on Gemma 4.

    Usage:
        config = Gemma4VLAConfig()
        model  = Gemma4VLA(config)

        # Training forward pass
        loss = model.compute_loss(batch)

        # Inference
        actions = model.predict_action(obs)
    """

    def __init__(self, cfg: Gemma4VLAConfig):
        super().__init__()
        self.cfg = cfg
        fm  = cfg.flow_matching
        ae  = cfg.action_expert
        rob = cfg.robot

        # --- Backbone ---
        self.backbone = Gemma4Backbone(cfg)
        backbone_dim = self.backbone.hidden_size

        # --- Projection: backbone_dim → action_expert_dim ---
        # Needed when they differ (e.g. Gemma 4 E2B has hidden_size=2048,
        # but we may want action expert hidden_size=1024 for efficiency)
        self.obs_proj = nn.Linear(backbone_dim, ae.hidden_size, bias=False)

        # --- Action Expert ---
        # State / action dims are padded to rob.max_state_dim for
        # cross-embodiment compatibility
        self.action_expert = ActionExpert(
            hidden_size=ae.hidden_size,
            num_layers=ae.num_layers,
            num_heads=ae.num_heads,
            context_dim=ae.hidden_size,  # after obs_proj
            action_dim=rob.max_state_dim,
            state_dim=rob.max_state_dim,
            action_horizon=fm.action_horizon,
            ffn_multiplier=ae.ffn_multiplier,
            dropout=ae.dropout,
            use_cross_attention=ae.use_cross_attention,
            noise_embedding_dim=ae.noise_embedding_dim,
        )

        # --- Config ---
        self.sigma_min = fm.sigma_min
        self.action_horizon = fm.action_horizon
        self.num_inference_steps = fm.num_inference_steps
        self.max_dim = rob.max_state_dim
        self.action_dim = rob.action_dim
        self.state_dim = rob.state_dim

        # Dataset-fit normalisation stats — populated by the trainer (or by
        # `from_pretrained` if a `normalization.pt` is present).
        self.stats: Optional[DatasetStats] = None

    def set_stats(self, stats: Optional[DatasetStats]) -> None:
        """Attach dataset-fit normalisation stats to this model."""
        self.stats = stats

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Standard `nn.Module` entry point. Delegates to `compute_loss` so
        the model can be wrapped in `DistributedDataParallel` and called via
        `ddp_model(batch)` (which is what triggers DDP's gradient sync)."""
        return self.compute_loss(batch)

    # ------------------------------------------------------------------
    # Padding helpers for cross-embodiment
    # ------------------------------------------------------------------

    def _pad_state(self, state: torch.Tensor) -> torch.Tensor:
        """Zero-pad state to max_state_dim."""
        B, D = state.shape
        if D == self.max_dim:
            return state
        pad = torch.zeros(B, self.max_dim - D, device=state.device, dtype=state.dtype)
        return torch.cat([state, pad], dim=-1)

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Zero-pad actions to max_state_dim."""
        B, H, D = actions.shape
        if D == self.max_dim:
            return actions
        pad = torch.zeros(B, H, self.max_dim - D, device=actions.device, dtype=actions.dtype)
        return torch.cat([actions, pad], dim=-1)

    def _unpad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Remove padding from predicted actions."""
        return actions[:, :, : self.action_dim]

    # ------------------------------------------------------------------
    # Core velocity function (used both in training and inference)
    # ------------------------------------------------------------------

    def _predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        noise_level: torch.Tensor,
        obs_features: torch.Tensor,
        state: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given noisy actions and observations, predict the flow velocity.

        Args:
            noisy_actions: [B, H, max_dim]
            noise_level:   [B]
            obs_features:  [B, S_obs, backbone_dim]
            state:         [B, max_dim]
            obs_mask:      [B, S_obs] optional

        Returns:
            velocity: [B, H, max_dim]
        """
        # Project backbone features to action expert dim
        # Backbone may output bf16 while obs_proj is fp32
        obs = self.obs_proj(obs_features.to(self.obs_proj.weight.dtype))  # [B, S, ae_hidden]

        velocity = self.action_expert(
            state=state,
            noisy_actions=noisy_actions,
            noise_level=noise_level,
            obs_features=obs,
            obs_mask=obs_mask,
        )
        return velocity

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the flow matching loss for a training batch.

        Expected batch keys:
            input_ids:      [B, T]          — text token IDs
            attention_mask: [B, T]          — text attention mask
            pixel_values:   [B, num_cameras, 3, H, W] — stacked camera images (optional)
            state:          [B, state_dim]  — proprioceptive state
            actions:        [B, H, action_dim] — clean ground-truth actions

        Returns:
            Dict with 'loss' (scalar) and 'metrics' (dict of floats).
        """
        device = batch["state"].device
        B = batch["state"].shape[0]

        # 1. Run backbone to extract observation features
        obs_features = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_position_ids=batch.get("image_position_ids"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )  # [B, S, backbone_dim]

        obs_mask = batch.get("obs_mask")

        # 2. Pad state & actions for cross-embodiment
        state = self._pad_state(batch["state"])                # [B, max_dim]
        actions = self._pad_actions(batch["actions"])          # [B, H, max_dim]

        # 3. Sample noise and noise level
        t = torch.rand(B, device=device)
        noise = torch.randn_like(actions)

        # 4. Compute OT interpolant
        noisy_actions, target_velocity = ot_flow_interpolate(
            x_0=noise, x_1=actions, t=t, sigma_min=self.sigma_min
        )

        # 5. Predict velocity
        pred_velocity = self._predict_velocity(
            noisy_actions=noisy_actions,
            noise_level=t,
            obs_features=obs_features,
            state=state,
            obs_mask=obs_mask,
        )  # [B, H, max_dim]

        # 6. Compute loss (only over real action dims, not padding)
        pred_trimmed   = pred_velocity[:, :, : self.action_dim]
        target_trimmed = target_velocity[:, :, : self.action_dim]

        loss = flow_matching_loss(pred_trimmed, target_trimmed)

        return {
            "loss": loss,
            "metrics": {
                "flow_matching_loss": loss.item(),
            },
        }

    def compute_loss_cached(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss using pre-computed backbone features.

        Skips the backbone forward pass entirely — ~100x faster per step.
        Use with CachedFeatureDataset from robots/metaworld/cached_dataset.py.

        Expected batch keys:
            obs_features: [B, S, hidden_size] — pre-computed backbone output
            state:        [B, state_dim]
            actions:      [B, H, action_dim]
        """
        device = batch["state"].device
        B = batch["state"].shape[0]

        obs_features = batch["obs_features"]  # already computed
        state = self._pad_state(batch["state"])
        actions = self._pad_actions(batch["actions"])

        t = torch.rand(B, device=device)
        noise = torch.randn_like(actions)

        noisy_actions, target_velocity = ot_flow_interpolate(
            x_0=noise, x_1=actions, t=t, sigma_min=self.sigma_min
        )

        pred_velocity = self._predict_velocity(
            noisy_actions=noisy_actions,
            noise_level=t,
            obs_features=obs_features,
            state=state,
        )

        pred_trimmed = pred_velocity[:, :, : self.action_dim]
        target_trimmed = target_velocity[:, :, : self.action_dim]

        loss = flow_matching_loss(pred_trimmed, target_trimmed)

        return {
            "loss": loss,
            "metrics": {"flow_matching_loss": loss.item()},
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        use_rk4: bool = False,
    ) -> torch.Tensor:
        """
        Predict an action sequence from observations.

        Args:
            obs: Dict containing:
                - input_ids:      [B, T]
                - attention_mask: [B, T]
                - pixel_values:   [B, num_cameras, 3, H, W]  (optional)
                - state:          [B, state_dim]
                - obs_mask:       [B, S]           (optional)
            num_steps: Override default inference steps.
            use_rk4:   Use 4th-order Runge-Kutta instead of Euler.

        Returns:
            Predicted actions of shape [B, H, action_dim].
        """
        self.eval()
        device = obs["state"].device
        B = obs["state"].shape[0]
        n_steps = num_steps or self.num_inference_steps

        # 1. Extract observation features (computed once, reused every denoising step)
        obs_features = self.backbone(
            input_ids=obs["input_ids"],
            attention_mask=obs["attention_mask"],
            pixel_values=obs.get("pixel_values"),
            image_position_ids=obs.get("image_position_ids"),
            mm_token_type_ids=obs.get("mm_token_type_ids"),
        )

        state = self._pad_state(obs["state"])
        obs_mask = obs.get("obs_mask")

        # 2. Define the velocity function for the integrator
        def velocity_fn(noisy_actions, t):
            return self._predict_velocity(
                noisy_actions=noisy_actions,
                noise_level=t,
                obs_features=obs_features,
                state=state,
                obs_mask=obs_mask,
            )

        # 3. Integrate ODE from t=0 (noise) to t=1 (clean action)
        shape = (B, self.action_horizon, self.max_dim)
        integrator = rk4_integration if use_rk4 else euler_integration

        actions_padded = integrator(
            velocity_fn=velocity_fn,
            shape=shape,
            num_steps=n_steps,
            sigma_min=self.sigma_min,
            device=device,
            dtype=obs_features.dtype,
        )

        # 4. Remove padding and return
        return self._unpad_actions(actions_padded)  # [B, H, action_dim]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable-only) parameter count."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def freeze_backbone(self):
        """Freeze backbone parameters (useful when switching to action-expert-only training)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, cfg: Optional[Gemma4VLAConfig] = None):
        """
        Load a Gemma4VLA model from a checkpoint directory.

        Args:
            checkpoint_path: Path to directory containing `config.json` (or the
                             legacy `config.pt`) and `weights.pt`.
            cfg: Optional config override. If None, loaded from checkpoint.

        Returns:
            Initialised Gemma4VLA model.
        """
        if cfg is None:
            cfg = load_config_artifact(checkpoint_path)

        model = cls(cfg)
        weights_path = os.path.join(checkpoint_path, WEIGHTS_NAME)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        stats = DatasetStats.load(checkpoint_path)
        if stats is not None:
            model.set_stats(stats)
        return model

    def save_pretrained(self, checkpoint_path: str):
        """
        Save model config and weights to a checkpoint directory.

        Args:
            checkpoint_path: Directory to save to (created if it does not exist).
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        save_config_artifact(self.cfg, checkpoint_path)
        torch.save(self.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        if self.stats is not None:
            self.stats.save(checkpoint_path)
        print(f"[Gemma4VLA] Saved to {checkpoint_path}")
