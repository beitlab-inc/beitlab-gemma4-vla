"""Inspect Gemma4VLA HDF5 datasets and optionally export visual previews."""

import argparse
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a Gemma4VLA HDF5 dataset saved by scripts.collect_data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the HDF5 episodes directory.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode to inspect in detail.",
    )
    parser.add_argument(
        "--frame-output",
        type=str,
        default=None,
        help="Optional path for saving one image frame, for example data/preview_frame.png.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to save when --frame-output is set.",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default=None,
        help="Optional path for saving an MP4 preview, for example videos/data_preview.mp4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to include in the preview video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Preview video frame rate.",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def list_episodes(dataset_dir):
    episodes = sorted(
        p for p in Path(dataset_dir).glob("episode_*.hdf5")
    )
    return episodes


def print_hdf5_tree(group, prefix=""):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {prefix}{key}: shape={item.shape} dtype={item.dtype}")
            if item.size > 0 and np.issubdtype(item.dtype, np.number):
                data = item[:]
                print(f"    min={data.min():.6g} max={data.max():.6g}")
        elif isinstance(item, h5py.Group):
            print(f"  {prefix}{key}/")
            print_hdf5_tree(item, prefix=prefix + "  ")


def print_dataset_summary(dataset_dir):
    episodes = list_episodes(dataset_dir)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Episodes found: {len(episodes)}")

    if not episodes:
        print("  (no episodes)")
        return

    for i, ep_path in enumerate(episodes[:5]):
        print(f"\n--- {ep_path.name} ---")
        with h5py.File(ep_path, "r") as f:
            print_hdf5_tree(f)
            if "language_instruction" in f.attrs:
                print(f"  language_instruction: {f.attrs['language_instruction']}")

    if len(episodes) > 5:
        print(f"\n  ... and {len(episodes) - 5} more episodes")


def get_images_from_episode(ep_path):
    with h5py.File(ep_path, "r") as f:
        img_group = f.get("observation/images")
        if img_group is None:
            raise KeyError(f"No observation/images group in {ep_path}")
        camera_name = list(img_group.keys())[0]
        images = img_group[camera_name]["data"][:]
    return images


def save_frame(images, output_path, frame_index):
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(
            f"--frame-index {frame_index} is out of range for {len(images)} images"
        )

    output_path = Path(output_path)
    ensure_parent(output_path)
    imageio.imwrite(output_path, images[frame_index])
    print("saved frame:", output_path)


def save_video(images, output_path, max_frames, fps):
    output_path = Path(output_path)
    ensure_parent(output_path)

    num_frames = min(len(images), max_frames)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in images[:num_frames]:
            writer.append_data(frame)

    print(f"saved video: {output_path} ({num_frames} frames at {fps} fps)")


def main():
    args = parse_args()

    print_dataset_summary(args.dataset_dir)

    episodes = list_episodes(args.dataset_dir)
    if not episodes:
        return

    if args.episode_index >= len(episodes):
        print(f"Episode index {args.episode_index} out of range (have {len(episodes)} episodes)")
        return

    if args.frame_output or args.video_output:
        images = get_images_from_episode(episodes[args.episode_index])

        if args.frame_output:
            save_frame(images, args.frame_output, args.frame_index)

        if args.video_output:
            save_video(images, args.video_output, args.max_frames, args.fps)


if __name__ == "__main__":
    main()
