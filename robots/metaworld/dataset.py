"""
MetaWorld HDF5 dataset adapter for Gemma4VLA.

Loads demonstration episodes collected by ``robots/metaworld/scripts/collect_data.py``
in the standard HDF5 format::

    episode_000000.hdf5
      observation/images/<camera>/data  [T, H, W, 3] uint8
      observation/state                 [T, state_dim] float32
      action                            [T, action_dim] float32
      attrs: language_instruction (str)

Each __getitem__ call samples a random temporal window of length
``action_horizon`` from a random episode, producing the batch dict expected
by ``Gemma4VLA.compute_loss()``.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gemma4_vla.config import Gemma4VLAConfig
from gemma4_vla.dataset import build_eval_transform, build_train_transform, collate_fn
from gemma4_vla.stats import DatasetStats


def _build_pil_augmentation(
    train: bool,
    use_color_jitter: bool,
    use_random_crop: bool,
    crop_scale: float,
    image_size: int,
):
    """PIL-in / PIL-out augmentation used before the HF processor resizes/normalises."""
    if not train:
        return None
    ops = []
    if use_random_crop:
        ops.append(
            transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0), antialias=True)
        )
    if use_color_jitter:
        ops.append(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        )
    return transforms.Compose(ops) if ops else None


class MetaWorldHDF5Dataset(Dataset):
    """
    Dataset that reads MetaWorld HDF5 episodes and produces training samples
    for Gemma4VLA.
    """

    def __init__(
        self,
        data_dir: str,
        cfg: Gemma4VLAConfig,
        processor=None,
        train: bool = True,
        instruction: Optional[str] = None,
        stats: Optional[DatasetStats] = None,
    ):
        self.cfg = cfg
        self.processor = processor
        self.train = train
        self.horizon = cfg.flow_matching.action_horizon
        self.state_dim = cfg.robot.state_dim
        self.action_dim = cfg.robot.action_dim
        self.max_state_dim = cfg.robot.max_state_dim
        self.max_seq_len = cfg.backbone.max_sequence_length
        self.num_cameras = cfg.vision.num_cameras
        self.configured_camera_names = cfg.vision.camera_names
        self.default_instruction = instruction or "push the object to the goal"
        self.stats = stats

        self.episode_paths = sorted(Path(data_dir).glob("episode_*.hdf5"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

        self._selected_cameras = self._resolve_cameras(self.episode_paths[0])

        tr = cfg.training
        if train:
            self.transform = build_train_transform(
                cfg.vision.image_size, tr.use_color_jitter, tr.use_random_crop, tr.crop_scale
            )
        else:
            self.transform = build_eval_transform(cfg.vision.image_size)

        # PIL-stage augmentation runs before the HF processor when one is used,
        # so color jitter / random crop are actually applied in the processor path.
        self.pil_aug = _build_pil_augmentation(
            train=train,
            use_color_jitter=tr.use_color_jitter,
            use_random_crop=tr.use_random_crop,
            crop_scale=tr.crop_scale,
            image_size=cfg.vision.image_size,
        )

        self._index = self._build_index()

    def _resolve_cameras(self, sample_episode: Path) -> List[str]:
        with h5py.File(sample_episode, "r") as f:
            available = sorted(f["observation/images"].keys())
        if self.configured_camera_names is not None:
            missing = [c for c in self.configured_camera_names if c not in available]
            if missing:
                raise ValueError(
                    f"vision.camera_names requests {missing} but episode "
                    f"{sample_episode} only contains {available}."
                )
            return list(self.configured_camera_names)
        if len(available) < self.num_cameras:
            raise ValueError(
                f"vision.num_cameras={self.num_cameras} but episode "
                f"{sample_episode} only contains cameras {available}."
            )
        return available[: self.num_cameras]

    def _build_index(self):
        index = []
        for ep_idx, ep_path in enumerate(self.episode_paths):
            with h5py.File(ep_path, "r") as f:
                n_steps = f["action"].shape[0]
            max_start = max(0, n_steps - self.horizon)
            for t in range(max_start + 1):
                index.append((ep_idx, t, n_steps))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t_start, n_steps = self._index[idx]
        ep_path = self.episode_paths[ep_idx]

        with h5py.File(ep_path, "r") as f:
            instruction = f.attrs.get("language_instruction", self.default_instruction)

            img_group = f["observation/images"]
            images_raw = [img_group[cam]["data"][t_start] for cam in self._selected_cameras]

            state_raw = f["observation/state"][t_start]

            t_end = min(t_start + self.horizon, n_steps)
            actions_raw = f["action"][t_start:t_end]

        pil_imgs: List[Image.Image] = [Image.fromarray(img) for img in images_raw]
        if self.pil_aug is not None:
            pil_imgs = [self.pil_aug(img) for img in pil_imgs]

        state = torch.zeros(self.max_state_dim, dtype=torch.float32)
        s = np.asarray(state_raw, dtype=np.float32)
        state[:len(s)] = torch.from_numpy(s)

        actions = torch.zeros(self.horizon, self.action_dim, dtype=torch.float32)
        a = np.asarray(actions_raw, dtype=np.float32)
        actions[:a.shape[0], :a.shape[1]] = torch.from_numpy(a)

        if self.stats is not None and self.stats.enabled:
            state = self.stats.normalize_state(state)
            actions = self.stats.normalize_actions(actions)

        if self.processor is not None:
            content = [{"type": "image", "image": img} for img in pil_imgs]
            content.append({"type": "text", "text": f"Task: {instruction}"})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            proc_images = pil_imgs[0] if len(pil_imgs) == 1 else pil_imgs
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
                # Keep the image axis: processor returns [N_images, P, D]; the
                # collate fn concatenates on dim 0 → HF expects [B*N, P, D].
                "pixel_values": enc["pixel_values"],
                "state": state,
                "actions": actions,
            }
            if "image_position_ids" in enc:
                result["image_position_ids"] = enc["image_position_ids"]
            if "mm_token_type_ids" in enc:
                result["mm_token_type_ids"] = enc["mm_token_type_ids"].squeeze(0)
            return result
        else:
            pixel = torch.stack([self.transform(img) for img in pil_imgs], dim=0)
            return {
                "input_ids": torch.zeros(self.max_seq_len, dtype=torch.long),
                "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
                "pixel_values": pixel,
                "state": state,
                "actions": actions,
            }


def build_metaworld_dataloaders(
    cfg: Gemma4VLAConfig,
    data_dir: str,
    processor=None,
    train_split: float = 0.9,
    instruction: Optional[str] = None,
    stats: Optional[DatasetStats] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a MetaWorld HDF5 dataset.

    Splits episodes (not samples) into train/val by the given ratio.
    """
    all_episodes = sorted(Path(data_dir).glob("episode_*.hdf5"))
    if not all_episodes:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

    n_train = max(1, int(len(all_episodes) * train_split))
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:] if n_train < len(all_episodes) else all_episodes[-1:]

    train_dir = tempfile.mkdtemp(prefix="mw_train_")
    val_dir = tempfile.mkdtemp(prefix="mw_val_")
    atexit.register(shutil.rmtree, train_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, val_dir, ignore_errors=True)
    for ep in train_episodes:
        os.symlink(ep.resolve(), os.path.join(train_dir, ep.name))
    for ep in val_episodes:
        os.symlink(ep.resolve(), os.path.join(val_dir, ep.name))

    train_ds = MetaWorldHDF5Dataset(
        train_dir, cfg, processor=processor, train=True, instruction=instruction, stats=stats,
    )
    val_ds = MetaWorldHDF5Dataset(
        val_dir, cfg, processor=processor, train=False, instruction=instruction, stats=stats,
    )

    tr = cfg.training
    pin = torch.cuda.is_available()  # pin_memory is CUDA-only (not MPS)
    # persistent_workers=True keeps train workers alive across epoch boundaries
    # so they don't refork from a now-huge parent process (model + optimiser
    # state). Required on unified-memory hardware (Jetson Thor) where each
    # fork materialises several GB via copy-on-write + Python refcount churn.
    train_loader = DataLoader(
        train_ds, batch_size=tr.batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=pin, num_workers=tr.num_workers,
        persistent_workers=tr.num_workers > 0,
        prefetch_factor=tr.prefetch_factor if tr.num_workers > 0 else None,
    )
    # Val loop runs <= 50 batches every eval_every_n_steps. Forking workers
    # for it spikes RAM at the eval boundary (worker count doubles since the
    # train workers are still alive). Serial loading adds ~seconds per eval,
    # negligible vs. the Stage-2 step cost, so num_workers=0 wins on safety.
    val_loader = DataLoader(
        val_ds, batch_size=tr.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=False, num_workers=0,
    )
    return train_loader, val_loader
