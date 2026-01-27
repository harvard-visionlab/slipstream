"""Slipstream: High-performance data loading for PyTorch vision workloads.

This package provides FFCV-like performance without the FFCV dependency hassle,
using modern dependencies and a more versatile architecture.

Example:
    from slipstream import SlipstreamDataset

    # Simple usage with remote dataset
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/",
        decode_images=True,
    )

    for sample in dataset:
        image = sample['image']  # PIL Image or tensor
        label = sample['label']  # int
"""

__version__ = "0.1.0"

from slipstream.dataset import (
    SlipstreamDataset,
    decode_image,
    ensure_lightning_symlink_on_cluster,
    get_default_cache_dir,
    is_image_bytes,
)

__all__ = [
    "__version__",
    "SlipstreamDataset",
    "decode_image",
    "is_image_bytes",
    "ensure_lightning_symlink_on_cluster",
    "get_default_cache_dir",
    # Future exports:
    # "SlipstreamLoader",
    # "PrefetchingDataLoader",
    # "GPUDecoder",
    # "CPUDecoder",
]
