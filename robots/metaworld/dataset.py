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

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from gemma4_vla.config import Gemma4VLAConfig
from gemma4_vla.dataset import build_eval_transform, build_train_transform, collate_fn


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
        self.default_instruction = instruction or "push the object to the goal"

        self.episode_paths = sorted(Path(data_dir).glob("episode_*.hdf5"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

        tr = cfg.training
        if train:
            self.transform = build_train_transform(
                cfg.vision.image_size, tr.use_color_jitter, tr.use_random_crop, tr.crop_scale
            )
        else:
            self.transform = build_eval_transform(cfg.vision.image_size)

        self._index = self._build_index()

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
            camera_name = list(img_group.keys())[0]
            image_raw = img_group[camera_name]["data"][t_start]

            state_raw = f["observation/state"][t_start]

            t_end = min(t_start + self.horizon, n_steps)
            actions_raw = f["action"][t_start:t_end]

        pil_img = Image.fromarray(image_raw)

        state = torch.zeros(self.max_state_dim, dtype=torch.float32)
        s = np.asarray(state_raw, dtype=np.float32)
        state[:len(s)] = torch.from_numpy(s)

        actions = torch.zeros(self.horizon, self.action_dim, dtype=torch.float32)
        a = np.asarray(actions_raw, dtype=np.float32)
        actions[:a.shape[0], :a.shape[1]] = torch.from_numpy(a)

        if self.processor is not None:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": f"Task: {instruction}"},
            ]}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            enc = self.processor(
                text=prompt,
                images=pil_img,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
            )
            result = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "pixel_values": enc["pixel_values"].squeeze(0),
                "state": state,
                "actions": actions,
            }
            if "image_position_ids" in enc:
                result["image_position_ids"] = enc["image_position_ids"].squeeze(0)
            if "mm_token_type_ids" in enc:
                result["mm_token_type_ids"] = enc["mm_token_type_ids"].squeeze(0)
            return result
        else:
            pixel = self.transform(pil_img)
            return {
                "input_ids": torch.zeros(self.max_seq_len, dtype=torch.long),
                "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
                "pixel_values": pixel.unsqueeze(0),
                "state": state,
                "actions": actions,
            }


def build_metaworld_dataloaders(
    cfg: Gemma4VLAConfig,
    data_dir: str,
    processor=None,
    train_split: float = 0.9,
    instruction: Optional[str] = None,
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

    import tempfile, shutil
    train_dir = tempfile.mkdtemp(prefix="mw_train_")
    val_dir = tempfile.mkdtemp(prefix="mw_val_")
    for ep in train_episodes:
        os.symlink(ep.resolve(), os.path.join(train_dir, ep.name))
    for ep in val_episodes:
        os.symlink(ep.resolve(), os.path.join(val_dir, ep.name))

    train_ds = MetaWorldHDF5Dataset(train_dir, cfg, processor=processor, train=True, instruction=instruction)
    val_ds = MetaWorldHDF5Dataset(val_dir, cfg, processor=processor, train=False, instruction=instruction)

    tr = cfg.training
    pin = torch.cuda.is_available()  # pin_memory is CUDA-only (not MPS)
    train_loader = DataLoader(
        train_ds, batch_size=tr.batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=pin, num_workers=tr.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin, num_workers=tr.num_workers,
    )
    return train_loader, val_loader
