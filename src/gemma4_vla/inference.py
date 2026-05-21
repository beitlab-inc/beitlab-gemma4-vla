"""
Inference pipeline for Gemma4VLA.

Provides a high-level `PolicyRunner` class that wraps the model and handles:
  - Observation preprocessing (image normalisation, state scaling)
  - Action prediction with temporal action chunking
  - Action post-processing (de-scaling, clipping)
  - Optional action smoothing

Usage::

    runner = PolicyRunner.from_pretrained("checkpoints/best", device="cuda")

    # Single step
    obs = {
        "images": [np.ndarray],          # list of [H, W, 3] uint8 arrays
        "state": np.ndarray,             # [state_dim] float32
        "instruction": "Pick up the cup",
    }
    actions = runner.predict(obs)        # [H, action_dim] float32

    # Streaming execution at 50 Hz
    for action in runner.stream(obs):
        robot.apply(action)
"""

import numpy as np
import torch
from typing import Dict, Iterator, List, Optional, Union
from PIL import Image
from torchvision import transforms

from .config import Gemma4VLAConfig
from .model import Gemma4VLA


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    image_size: int,
) -> torch.Tensor:
    """
    Convert a raw image (uint8 numpy array or PIL Image) to a normalised tensor.

    Args:
        image:      [H, W, 3] uint8 numpy array or PIL Image.
        image_size: Target square resolution.

    Returns:
        FloatTensor [3, image_size, image_size] in [-1, 1].
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return _EVAL_TRANSFORM(image)


# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------

class PolicyRunner:
    """
    High-level inference wrapper for Gemma4VLA.

    Attributes:
        model:     The underlying Gemma4VLA model.
        cfg:       Gemma4VLAConfig.
        device:    Torch device the model runs on.
    """

    def __init__(
        self,
        model: Gemma4VLA,
        device: Union[str, torch.device] = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.cfg = model.cfg
        self.device = torch.device(device)

        self._processor = model.backbone.processor
        self._image_size = self.cfg.vision.image_size
        self._num_cameras = self.cfg.vision.num_cameras
        self._max_seq_len = self.cfg.backbone.max_sequence_length
        self._horizon = self.cfg.flow_matching.action_horizon
        self._action_dim = self.cfg.robot.action_dim
        self._action_scale = self.cfg.robot.action_scale

        # Action chunking buffer: we store predicted actions and execute them
        # step by step.  When the buffer is empty we re-run inference.
        self._action_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: Dict,
        num_inference_steps: Optional[int] = None,
        use_rk4: bool = False,
    ) -> np.ndarray:
        """
        Predict an action chunk from raw observations.

        Args:
            obs: Dict with keys:
                - "images":      list of [H, W, 3] uint8 arrays, one per camera.
                - "state":       [state_dim] float array.
                - "instruction": str task description.
            num_inference_steps: Override denoising steps (default from config).
            use_rk4: Use RK4 integration for higher-quality actions.

        Returns:
            Action array of shape [horizon, action_dim], float32.
        """
        batch = self._preprocess(obs)
        with torch.no_grad():
            actions = self.model.predict_action(
                batch,
                num_steps=num_inference_steps,
                use_rk4=use_rk4,
            )  # [1, H, action_dim]
        actions = actions[0].float().cpu().numpy()  # [H, action_dim]
        actions = actions * self._action_scale
        return actions

    def stream(
        self,
        obs: Dict,
        replan_every: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        Generator that yields individual action steps, replanning when
        the action buffer runs out.

        This implements temporal action chunking: predict H steps at once
        but execute them one by one.  Replanning frequency is controlled
        by `replan_every` (default: every horizon steps).

        Args:
            obs:               Initial observation dict (update between steps!).
            replan_every:      How often to re-run the policy (in environment steps).
                               Default is `action_horizon`.
            num_inference_steps: Override denoising steps.

        Yields:
            Individual action arrays of shape [action_dim].
        """
        if replan_every is None:
            replan_every = self._horizon

        step = 0
        while True:
            if step % replan_every == 0:
                actions = self.predict(obs, num_inference_steps=num_inference_steps)
                self._action_buffer = list(actions)

            if not self._action_buffer:
                break

            yield self._action_buffer.pop(0)
            step += 1

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Convert a raw observation dict to tensors ready for the model."""
        images = obs.get("images", [])
        if not images:
            raise ValueError("obs['images'] must be a non-empty list of images.")

        # Pad / trim to expected number of cameras
        while len(images) < self._num_cameras:
            images = images + [images[-1]]  # replicate last camera
        images = images[: self._num_cameras]

        # Process each image
        img_tensors = [
            preprocess_image(img, self._image_size) for img in images
        ]  # list of [3, H, W]
        pixel_values = torch.stack(img_tensors, dim=0).unsqueeze(0).to(self.device)
        # [1, num_cameras, 3, H, W]

        # Build text prompt
        instruction = obs.get("instruction", "Perform the task.")
        prompt = "<image>\n" * self._num_cameras + f"Task: {instruction}"
        encoding = self._processor(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self._max_seq_len,
            truncation=True,
        ).to(self.device)

        # State
        state = obs.get("state", np.zeros(self._action_dim, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.device)  # [1, state_dim]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "pixel_values": pixel_values,
            "state": state,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cuda",
    ) -> "PolicyRunner":
        """
        Load a PolicyRunner from a saved checkpoint.

        Args:
            checkpoint_path: Directory containing 'config.pt' and 'weights.pt'.
            device:          Torch device to run inference on.

        Returns:
            Initialised PolicyRunner.
        """
        model = Gemma4VLA.from_pretrained(checkpoint_path)
        return cls(model, device=device)

    @classmethod
    def from_model(
        cls,
        model: Gemma4VLA,
        device: Union[str, torch.device] = "cuda",
    ) -> "PolicyRunner":
        """Wrap an existing Gemma4VLA model."""
        return cls(model, device=device)


# ---------------------------------------------------------------------------
# Benchmark / timing utility
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: Gemma4VLA,
    num_runs: int = 20,
    device: str = "cuda",
    num_inference_steps: int = 10,
) -> Dict[str, float]:
    """
    Measure inference latency and throughput.

    Args:
        model:               Model to benchmark.
        num_runs:            Number of forward passes to average over.
        device:              Device string.
        num_inference_steps: Number of denoising steps.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'hz' (inferences per second).
    """
    import time

    model.eval()
    runner = PolicyRunner(model, device=device)
    cfg = model.cfg

    # Dummy observation
    obs = {
        "images": [
            np.random.randint(0, 255, (cfg.vision.image_size, cfg.vision.image_size, 3), dtype=np.uint8)
            for _ in range(cfg.vision.num_cameras)
        ],
        "state": np.zeros(cfg.robot.state_dim, dtype=np.float32),
        "instruction": "Pick up the cube.",
    }

    # Warmup
    for _ in range(3):
        runner.predict(obs, num_inference_steps=num_inference_steps)

    # Timing
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        runner.predict(obs, num_inference_steps=num_inference_steps)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    hz = 1000.0 / mean_ms

    return {"mean_ms": mean_ms, "std_ms": std_ms, "hz": hz}
