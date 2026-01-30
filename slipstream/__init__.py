"""Slipstream: High-performance data loading for PyTorch vision workloads.

This package provides FFCV-like performance without the FFCV dependency hassle,
using modern dependencies and a more versatile architecture.

Example:
    from slipstream import SlipstreamDataset, SlipstreamLoader

    # Simple usage with remote dataset
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/",
        decode_images=True,
    )

    for sample in dataset:
        image = sample['image']  # PIL Image or tensor
        label = sample['label']  # int

High-Performance Training:
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream import RandomResizedCrop, Normalize

    # Create dataset
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/train/",
        decode_images=False,  # Let loader handle decoding
    )

    # Create high-performance loader with pipelines
    loader = SlipstreamLoader(
        dataset,
        batch_size=256,
        pipelines={
            'image': [
                RandomResizedCrop(224, device='cuda'),
                Normalize(),
            ],
        },
    )

    for batch in loader:
        images = batch['image']  # [B, 3, 224, 224] normalized tensor
        labels = batch['label']  # [B] tensor
        # Training...

    # Raw I/O (no pipelines, for benchmarking)
    raw_loader = SlipstreamLoader(dataset, batch_size=256)
"""

__version__ = "0.1.0"

# Core dataset
# Native FFCV file support (this IS the FFCV .beton format)
from slipstream.backends.ffcv_file import (
    FFCVFileDataset,
    FFCVFilePrefetchingDataLoader,
)

# Optimized cache (internal, but exposed for advanced users)
from slipstream.cache import (
    OptimizedCache,
    write_index,
)
from slipstream.dataset import (
    SlipstreamDataset,
    decode_image,
    ensure_lightning_symlink_on_cluster,
    get_default_cache_dir,
    is_image_bytes,
    list_collate_fn,
)

# Decoders
from slipstream.decoders import (
    CPUDecoder,
    GPUDecoder,
    GPUDecoderFallback,
    check_gpu_decoder_available,
    check_turbojpeg_available,
    get_decoder,
)

# High-level loader
from slipstream.loader import SlipstreamLoader

# Pipelines
from slipstream.pipelines import (
    BatchTransform,
    CenterCrop,
    Compose,
    DecodeOnly,
    MultiCropRandomResizedCrop,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Normalize,
    RandomResizedCrop,
    ResizeCrop,
    ToDevice,
    make_train_pipeline,
    make_val_pipeline,
)

# Readers (dataset format adapters)
from slipstream.readers import FFCVFileReader

# Crop utilities
from slipstream.utils.crop import (
    CropParams,
    align_to_mcu,
    generate_center_crop_params,
    generate_random_crop_params,
)

__all__ = [
    "__version__",
    # Core dataset
    "SlipstreamDataset",
    "decode_image",
    "is_image_bytes",
    "ensure_lightning_symlink_on_cluster",
    "get_default_cache_dir",
    "list_collate_fn",
    # High-level loader
    "SlipstreamLoader",
    # Pipelines
    "BatchTransform",
    "Compose",
    "DecodeOnly",
    "RandomResizedCrop",
    "MultiCropRandomResizedCrop",
    "CenterCrop",
    "ResizeCrop",
    "Normalize",
    "ToDevice",
    "make_train_pipeline",
    "make_val_pipeline",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Optimized cache (advanced)
    "OptimizedCache",
    "write_index",
    # Crop utilities
    "CropParams",
    "align_to_mcu",
    "generate_random_crop_params",
    "generate_center_crop_params",
    # Decoders
    "CPUDecoder",
    "GPUDecoder",
    "GPUDecoderFallback",
    "check_turbojpeg_available",
    "check_gpu_decoder_available",
    "get_decoder",
    # Native FFCV file support
    "FFCVFileDataset",
    "FFCVFilePrefetchingDataLoader",
    # Readers
    "FFCVFileReader",
]
