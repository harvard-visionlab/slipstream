"""Pipeline presets for common training workflows.

This package provides factory functions returning pipeline dicts for
``SlipstreamLoader(pipelines=...)``:

- :func:`supervised_train` / :func:`supervised_val` — standard ImageNet
- :func:`simclr` — SimCLR two-view SSL
- :func:`byol` — BYOL two-view SSL (asymmetric augmentation)
- :func:`multicrop` — multi-crop SSL (DINO, iBOT)

For decode stages, import from :mod:`slipstream.decoders`::

    from slipstream.decoders import RandomResizedCrop, CenterCrop

For GPU augmentations, import from :mod:`slipstream.transforms`::

    from slipstream.transforms import Normalize, ToTorchImage
"""

# Pipeline presets
from slipstream.pipelines.supervised import (
    make_train_pipeline,
    make_val_pipeline,
    supervised_train,
    supervised_val,
)
from slipstream.pipelines.simclr import simclr
from slipstream.pipelines.byol import byol
from slipstream.pipelines.multicrop_preset import multicrop

# Seed offset constants
from slipstream.pipelines._common import (
    CROP_OFFSET,
    FLIP_OFFSET,
    COLOR_OFFSET,
    GRAY_OFFSET,
    SOLAR_OFFSET,
    BLUR_OFFSET,
)

__all__ = [
    # Pipeline presets
    "supervised_train",
    "supervised_val",
    "simclr",
    "byol",
    "multicrop",
    # Legacy convenience (deprecated in favor of presets)
    "make_train_pipeline",
    "make_val_pipeline",
    # Seed offset constants
    "CROP_OFFSET",
    "FLIP_OFFSET",
    "COLOR_OFFSET",
    "GRAY_OFFSET",
    "SOLAR_OFFSET",
    "BLUR_OFFSET",
]
