"""Slipstream: High-performance data loading for PyTorch vision workloads.

This package provides FFCV-like performance without the FFCV dependency hassle,
using modern dependencies and a more versatile architecture.

Example:
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.pipelines import supervised_train

    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/train/",
        decode_images=False,
    )

    loader = SlipstreamLoader(
        dataset,
        batch_size=256,
        pipelines=supervised_train(224, device='cuda'),
    )

    for batch in loader:
        images = batch['image']  # [B, 3, 224, 224] normalized float tensor
        labels = batch['label']  # [B] tensor
"""

# Every public name loads on first access (PEP 562): importing torch, numba, litdata and
# torchvision up front made `import slipstream` cost seconds on cluster filesystems, even for
# `slipstream status` or a notebook that only touches a few names. `from slipstream import X`
# and `slipstream.X` work exactly as before; submodules (`slipstream.decoders`, ...) too.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from slipstream.version import __version__

if TYPE_CHECKING:  # eager imports for type checkers / IDEs
    from slipstream.backends.ffcv_file import (
        FFCVFileDataset,
        FFCVFilePrefetchingDataLoader,
    )
    from slipstream.cache import (
        CacheIntegrityError,
        OptimizedCache,
        write_index,
    )
    from slipstream.dataset import (
        SlipstreamDataset,
        decode_image,
        ensure_lightning_symlink_on_cluster,
        get_default_cache_dir,
        is_hf_image_dict,
        is_image_bytes,
        list_collate_fn,
    )

    # Decoders (low-level + fused decode+crop stages)
    from slipstream.decoders import (
        BatchTransform,
        CPUDecoder,
        GPUDecoder,
        GPUDecoderFallback,
        check_gpu_decoder_available,
        check_turbojpeg_available,
        get_decoder,
        # Fused decode+crop stages (new names)
        DecodeOnly,
        DecodeYUVFullRes,
        DecodeYUVPlanes,
        DecodeCenterCrop,
        DecodeRandomResizedCrop,
        DecodeDirectRandomResizedCrop,
        DecodeResizeCrop,
        DecodeRandomResizeShortCropLong,
        DecodeMultiRandomResizedCrop,
        DecodeMultiRandomResizeShortCropLong,
        DecodeMultiResizeCropEmbed,
        DecodeUniformMultiRandomResizedCrop,
        MultiCropPipeline,
        NamedCopies,
        estimate_rejection_fallback_rate,
        # Backward-compatible aliases (deprecated)
        CenterCrop,
        RandomResizedCrop,
        DirectRandomResizedCrop,
        ResizeCrop,
        RandomResizeShortCropLong,
        MultiCropRandomResizedCrop,
        MultiRandomResizedCrop,
        MultiRandomResizeShortCropLong,
    )

    # Transforms (GPU batch augmentations + pipeline-level transforms)
    from slipstream.transforms import (
        Compose,
        IMAGENET_MEAN,
        IMAGENET_STD,
        Normalize,
        RandomBackgroundBlend,
        RandomEmbed,
        ToDevice,
        ToTorchImage,
    )

    # High-level loader
    from slipstream.loader import SlipstreamLoader

    # Pipeline presets
    from slipstream.pipelines import (
        make_train_pipeline,
        make_val_pipeline,
        supervised_train,
        supervised_val,
        simclr,
        ipcl,
        lejepa,
        multicrop,
    )

    # Readers (dataset format adapters)
    from slipstream.readers import FFCVFileReader, StreamingReader

    # Visualization
    from slipstream.vis import show_batch, show_rgba

    # Utilities
    from slipstream.s3_sync import sync_s3_dataset
    from slipstream.stats import compute_normalization_stats

    # Seed derivation
    from slipstream.resolution import ResolutionSchedule
    from slipstream.seeds import derive_seed

    # Crop utilities
    from slipstream.utils.crop import (
        CropParams,
        align_to_mcu,
        generate_center_crop_params,
        generate_random_crop_params,
    )

    # Cache directory utilities
    from slipstream.utils.cache_dir import (
        CACHE_DIR_ENV_VAR,
        get_cache_base,
        get_cache_path,
    )
    from slipstream.readers.imagefolder import SlipstreamImageFolder, open_imagefolder

_LAZY: dict[str, str] = {
    'FFCVFileDataset': 'slipstream.backends.ffcv_file',
    'FFCVFilePrefetchingDataLoader': 'slipstream.backends.ffcv_file',
    'OptimizedCache': 'slipstream.cache',
    'CacheIntegrityError': 'slipstream.cache',
    'write_index': 'slipstream.cache',
    'SlipstreamDataset': 'slipstream.dataset',
    'decode_image': 'slipstream.dataset',
    'ensure_lightning_symlink_on_cluster': 'slipstream.dataset',
    'get_default_cache_dir': 'slipstream.dataset',
    'is_hf_image_dict': 'slipstream.dataset',
    'is_image_bytes': 'slipstream.dataset',
    'list_collate_fn': 'slipstream.dataset',
    'BatchTransform': 'slipstream.decoders',
    'CPUDecoder': 'slipstream.decoders',
    'GPUDecoder': 'slipstream.decoders',
    'GPUDecoderFallback': 'slipstream.decoders',
    'check_gpu_decoder_available': 'slipstream.decoders',
    'check_turbojpeg_available': 'slipstream.decoders',
    'get_decoder': 'slipstream.decoders',
    'DecodeOnly': 'slipstream.decoders',
    'DecodeYUVFullRes': 'slipstream.decoders',
    'DecodeYUVPlanes': 'slipstream.decoders',
    'DecodeCenterCrop': 'slipstream.decoders',
    'DecodeRandomResizedCrop': 'slipstream.decoders',
    'DecodeDirectRandomResizedCrop': 'slipstream.decoders',
    'DecodeResizeCrop': 'slipstream.decoders',
    'DecodeRandomResizeShortCropLong': 'slipstream.decoders',
    'DecodeMultiRandomResizedCrop': 'slipstream.decoders',
    'DecodeMultiRandomResizeShortCropLong': 'slipstream.decoders',
    'DecodeMultiResizeCropEmbed': 'slipstream.decoders',
    'DecodeUniformMultiRandomResizedCrop': 'slipstream.decoders',
    'MultiCropPipeline': 'slipstream.decoders',
    'NamedCopies': 'slipstream.decoders',
    'estimate_rejection_fallback_rate': 'slipstream.decoders',
    'CenterCrop': 'slipstream.decoders',
    'RandomResizedCrop': 'slipstream.decoders',
    'DirectRandomResizedCrop': 'slipstream.decoders',
    'ResizeCrop': 'slipstream.decoders',
    'RandomResizeShortCropLong': 'slipstream.decoders',
    'MultiCropRandomResizedCrop': 'slipstream.decoders',
    'MultiRandomResizedCrop': 'slipstream.decoders',
    'MultiRandomResizeShortCropLong': 'slipstream.decoders',
    'Compose': 'slipstream.transforms',
    'IMAGENET_MEAN': 'slipstream.transforms',
    'IMAGENET_STD': 'slipstream.transforms',
    'Normalize': 'slipstream.transforms',
    'RandomBackgroundBlend': 'slipstream.transforms',
    'RandomEmbed': 'slipstream.transforms',
    'ToDevice': 'slipstream.transforms',
    'ToTorchImage': 'slipstream.transforms',
    'SlipstreamLoader': 'slipstream.loader',
    'make_train_pipeline': 'slipstream.pipelines',
    'make_val_pipeline': 'slipstream.pipelines',
    'supervised_train': 'slipstream.pipelines',
    'supervised_val': 'slipstream.pipelines',
    'simclr': 'slipstream.pipelines',
    'ipcl': 'slipstream.pipelines',
    'lejepa': 'slipstream.pipelines',
    'multicrop': 'slipstream.pipelines',
    'FFCVFileReader': 'slipstream.readers',
    'StreamingReader': 'slipstream.readers',
    'show_batch': 'slipstream.vis',
    'show_rgba': 'slipstream.vis',
    'sync_s3_dataset': 'slipstream.s3_sync',
    'compute_normalization_stats': 'slipstream.stats',
    'derive_seed': 'slipstream.seeds',
    'ResolutionSchedule': 'slipstream.resolution',
    'CropParams': 'slipstream.utils.crop',
    'align_to_mcu': 'slipstream.utils.crop',
    'generate_center_crop_params': 'slipstream.utils.crop',
    'generate_random_crop_params': 'slipstream.utils.crop',
    'CACHE_DIR_ENV_VAR': 'slipstream.utils.cache_dir',
    'get_cache_base': 'slipstream.utils.cache_dir',
    'get_cache_path': 'slipstream.utils.cache_dir',
    'SlipstreamImageFolder': 'slipstream.readers.imagefolder',
    'open_imagefolder': 'slipstream.readers.imagefolder',
}

__all__ = [
    "__version__",
    # Core dataset
    "SlipstreamDataset",
    "decode_image",
    "is_hf_image_dict",
    "is_image_bytes",
    "ensure_lightning_symlink_on_cluster",
    "get_default_cache_dir",
    "list_collate_fn",
    # High-level loader
    "SlipstreamLoader",
    # Decode stages (new names)
    "BatchTransform",
    "DecodeOnly",
    "DecodeYUVFullRes",
    "DecodeYUVPlanes",
    "DecodeCenterCrop",
    "DecodeRandomResizedCrop",
    "DecodeDirectRandomResizedCrop",
    "DecodeResizeCrop",
    "DecodeRandomResizeShortCropLong",
    "DecodeMultiRandomResizedCrop",
    "DecodeMultiRandomResizeShortCropLong",
    "DecodeMultiResizeCropEmbed",
    "DecodeUniformMultiRandomResizedCrop",
    "MultiCropPipeline",
    "NamedCopies",
    "estimate_rejection_fallback_rate",
    # Backward-compatible aliases (deprecated)
    "CenterCrop",
    "RandomResizedCrop",
    "DirectRandomResizedCrop",
    "ResizeCrop",
    "RandomResizeShortCropLong",
    "MultiCropRandomResizedCrop",
    "MultiRandomResizedCrop",
    "MultiRandomResizeShortCropLong",
    # Transforms
    "Compose",
    "Normalize",
    "RandomBackgroundBlend",
    "RandomEmbed",
    "ToDevice",
    "ToTorchImage",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Pipeline presets
    "make_train_pipeline",
    "make_val_pipeline",
    "supervised_train",
    "supervised_val",
    "simclr",
    "ipcl",
    "lejepa",
    "multicrop",
    # Optimized cache (advanced)
    "OptimizedCache",
    "CacheIntegrityError",
    "write_index",
    # Seeds
    "derive_seed",
    # Progressive resolution
    "ResolutionSchedule",
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
    "SlipstreamImageFolder",
    "StreamingReader",
    "open_imagefolder",
    # Visualization
    "show_batch",
    "show_rgba",
    # Utilities
    "sync_s3_dataset",
    "compute_normalization_stats",
    # Cache directory utilities
    "CACHE_DIR_ENV_VAR",
    "get_cache_base",
    "get_cache_path",
    # Dataset preparation
    "prep",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is not None:
        value = getattr(importlib.import_module(module), name)
        globals()[name] = value                   # cache: later lookups skip __getattr__
        return value
    try:                                          # submodules: slipstream.decoders, slipstream.prep, ...
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
