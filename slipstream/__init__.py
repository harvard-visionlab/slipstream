"""Slipstream: High-performance data loading for PyTorch vision workloads.

Namespace: visionlab.slipstream

This package provides FFCV-like performance without the FFCV dependency hassle,
using modern dependencies and a more versatile architecture.

Example:
    from slipstream import SlipstreamDataset, SlipstreamLoader

    dataset = SlipstreamDataset(
        input_dir="s3://bucket/dataset/",
        cache_dir="/local/cache",
    )

    loader = SlipstreamLoader(
        dataset,
        batch_size=256,
        num_workers=8,
    )

    for batch in loader:
        images = batch['image']  # [B, C, H, W] tensor
        labels = batch['label']  # [B] tensor
"""

__version__ = "0.1.0"

# Imports will be added as modules are implemented
# from slipstream.dataset import SlipstreamDataset
# from slipstream.loader import SlipstreamLoader, PrefetchingDataLoader
# from slipstream.decoders import GPUDecoder, CPUDecoder

__all__ = [
    "__version__",
    # "SlipstreamDataset",
    # "SlipstreamLoader",
    # "PrefetchingDataLoader",
    # "GPUDecoder",
    # "CPUDecoder",
]
