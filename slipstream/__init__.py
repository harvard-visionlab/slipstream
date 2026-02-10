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

__version__ = "0.1.0"

# Core dataset
from slipstream.backends.ffcv_file import (
    FFCVFileDataset,
    FFCVFilePrefetchingDataLoader,
)
from slipstream.cache import (
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
    DecodeMultiRandomResizedCrop,
    DecodeUniformMultiRandomResizedCrop,
    MultiCropPipeline,
    estimate_rejection_fallback_rate,
    # Backward-compatible aliases (deprecated)
    CenterCrop,
    RandomResizedCrop,
    DirectRandomResizedCrop,
    ResizeCrop,
    MultiCropRandomResizedCrop,
    MultiRandomResizedCrop,
)

# Transforms (GPU batch augmentations + pipeline-level transforms)
from slipstream.transforms import (
    Compose,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Normalize,
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
from slipstream.readers import FFCVFileReader, SlipstreamImageFolder, StreamingReader, open_imagefolder

# Utilities
from slipstream.s3_sync import sync_s3_dataset
from slipstream.stats import compute_normalization_stats

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
    "DecodeMultiRandomResizedCrop",
    "DecodeUniformMultiRandomResizedCrop",
    "MultiCropPipeline",
    "estimate_rejection_fallback_rate",
    # Backward-compatible aliases (deprecated)
    "CenterCrop",
    "RandomResizedCrop",
    "DirectRandomResizedCrop",
    "ResizeCrop",
    "MultiCropRandomResizedCrop",
    "MultiRandomResizedCrop",
    # Transforms
    "Compose",
    "Normalize",
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
    "SlipstreamImageFolder",
    "StreamingReader",
    "open_imagefolder",
    # Utilities
    "sync_s3_dataset",
    "compute_normalization_stats",
    # Cache directory utilities
    "CACHE_DIR_ENV_VAR",
    "get_cache_base",
    "get_cache_path",
]
