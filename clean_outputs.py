"""Dry-run cleanup helper for generated Gemma4VLA artifacts."""

import argparse
import shutil
from pathlib import Path


TARGETS = {
    "videos": ["videos", "videos*"],
    "metrics": ["metrics"],
    "rerun": ["rerun", "*.rrd"],
    "mlruns": ["mlruns"],
    "previews": ["data/preview_frame.png", "videos/data_preview.mp4"],
    "probes": ["data/codex_probe_*.hdf5", "checkpoints/codex*"],
    "cache": ["**/__pycache__", "MUJOCO_LOG.txt"],
    "data": ["data"],
    "checkpoints": ["checkpoints"],
}

DEFAULT_CATEGORIES = ["videos", "metrics", "rerun", "previews", "probes", "cache"]
SKIP_PARTS = {".git", ".venv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove generated datasets, checkpoints, videos, metrics, and cache files."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag, the script only prints a dry run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select every cleanup category, including full data/ and checkpoints/ directories.",
    )
    for category in TARGETS:
        parser.add_argument(
            f"--{category}",
            action="store_true",
            help=f"Clean {category} artifacts.",
        )
    return parser.parse_args()


def selected_categories(args):
    if args.all:
        return list(TARGETS)

    selected = [
        category
        for category in TARGETS
        if getattr(args, category)
    ]
    return selected or DEFAULT_CATEGORIES


def resolve_targets(categories):
    found = []
    seen = set()
    cwd = Path.cwd()

    for category in categories:
        for pattern in TARGETS[category]:
            for path in cwd.glob(pattern):
                if not path.exists():
                    continue
                if any(part in SKIP_PARTS for part in path.relative_to(cwd).parts):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(path)

    return sorted(found, key=lambda p: (len(p.parts), str(p)), reverse=True)


def remove_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main():
    args = parse_args()
    categories = selected_categories(args)
    targets = resolve_targets(categories)

    mode = "DELETE" if args.yes else "DRY RUN"
    print(f"[clean] mode={mode}")
    print(f"[clean] categories={', '.join(categories)}")

    if not targets:
        print("[clean] no matching generated artifacts found")
        return

    for path in targets:
        kind = "dir" if path.is_dir() else "file"
        print(f"[clean] {kind}: {path}")
        if args.yes:
            remove_path(path)

    if not args.yes:
        print("[clean] pass --yes to delete these paths")


if __name__ == "__main__":
    main()
