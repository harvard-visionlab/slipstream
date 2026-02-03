"""Pipeline presets for common training workflows.

This package provides factory functions returning pipeline dicts for
``SlipstreamLoader(pipelines=...)``:

- :func:`supervised_train` / :func:`supervised_val` — standard ImageNet
- :func:`simclr` / :func:`simclr_symmetric` — SimCLR two-view SSL (symmetric)
- :func:`simclr_standard` — SimCLR two-view SSL (asymmetric, matches lrm-ssl yaml)
- :func:`ipcl` — IPCL N-crop SSL (default 5 crops)
- :func:`lejepa` — L-JEPA multi-crop SSL (6 global crops)
- :func:`multicrop` — multi-crop SSL (DINO, iBOT style: global + local crops)

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
from slipstream.pipelines.simclr import simclr, simclr_symmetric, simclr_standard
from slipstream.pipelines.ipcl import ipcl
from slipstream.pipelines.lejepa import lejepa
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
    "simclr_symmetric",
    "simclr_standard",
    "ipcl",
    "lejepa",
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
