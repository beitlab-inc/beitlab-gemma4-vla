"""
MetaWorld robot adapter for Gemma4VLA.

Provides environment wrappers, dataset loading, and training/evaluation
scripts for Meta-World MT1 tasks (push-v3, reach-v3, pick-place-v3, etc.)
running on MuJoCo via Gymnasium.

Usage::

    from robots.metaworld import MetaWorldMT1Wrapper, MetaWorldHDF5Dataset

    env = MetaWorldMT1Wrapper(env_name="push-v3")
    dataset = MetaWorldHDF5Dataset("data/metaworld_demos", cfg)
"""

from .dataset import MetaWorldHDF5Dataset, build_metaworld_dataloaders

__all__ = [
    "MetaWorldMT1Wrapper",
    "MetaWorldHDF5Dataset",
    "build_metaworld_dataloaders",
]


def __getattr__(name):
    if name == "MetaWorldMT1Wrapper":
        from .env import MetaWorldMT1Wrapper
        return MetaWorldMT1Wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
