"""Test that all public imports work.

Run after any refactor to verify nothing broke.
"""

import pytest


class TestTopLevelImports:
    """All symbols exported from slipstream.__init__."""

    def test_core_dataset(self):
        from slipstream import SlipstreamDataset, decode_image, is_image_bytes
        from slipstream import ensure_lightning_symlink_on_cluster, get_default_cache_dir
        from slipstream import list_collate_fn

    def test_loader(self):
        from slipstream import SlipstreamLoader

    def test_cache(self):
        from slipstream import OptimizedCache, write_index

    def test_decoders(self):
        from slipstream import CPUDecoder, GPUDecoder, GPUDecoderFallback
        from slipstream import check_turbojpeg_available, check_gpu_decoder_available
        from slipstream import get_decoder

    def test_backends(self):
        from slipstream import FFCVFileDataset, FFCVFilePrefetchingDataLoader

    def test_readers(self):
        from slipstream import FFCVFileReader

    def test_transforms(self):
        from slipstream import Compose, Normalize, ToDevice, ToTorchImage
        from slipstream import IMAGENET_MEAN, IMAGENET_STD

    def test_decode_stages(self):
        from slipstream import BatchTransform
        from slipstream import DecodeOnly, DecodeYUVFullRes, DecodeYUVPlanes
        from slipstream import CenterCrop, RandomResizedCrop, DirectRandomResizedCrop
        from slipstream import ResizeCrop
        from slipstream import MultiCropRandomResizedCrop, MultiRandomResizedCrop
        from slipstream import MultiCropPipeline
        from slipstream import estimate_rejection_fallback_rate

    def test_pipeline_presets(self):
        from slipstream import make_train_pipeline, make_val_pipeline
        from slipstream import supervised_train, supervised_val
        from slipstream import simclr, byol, multicrop

    def test_crop_utils(self):
        from slipstream import CropParams, align_to_mcu
        from slipstream import generate_random_crop_params, generate_center_crop_params

    def test_utilities(self):
        from slipstream import sync_s3_dataset, compute_normalization_stats


class TestCanonicalImports:
    """Import from canonical module locations."""

    def test_decoders_base(self):
        from slipstream.decoders.base import BatchTransform

    def test_decoders_crop(self):
        from slipstream.decoders.crop import (
            CenterCrop, RandomResizedCrop, DirectRandomResizedCrop, ResizeCrop,
        )

    def test_decoders_multicrop(self):
        from slipstream.decoders.multicrop import (
            MultiCropRandomResizedCrop, MultiRandomResizedCrop, MultiCropPipeline,
        )

    def test_decoders_decode(self):
        from slipstream.decoders.decode import DecodeOnly, DecodeYUVFullRes, DecodeYUVPlanes

    def test_decoders_utils(self):
        from slipstream.decoders.utils import estimate_rejection_fallback_rate

    def test_transforms_constants(self):
        from slipstream.transforms import IMAGENET_MEAN, IMAGENET_STD
        from slipstream.transforms.normalization import IMAGENET_MEAN, IMAGENET_STD

    def test_transforms_module(self):
        from slipstream.transforms import BatchAugment, Compose, RandomApply, MultiSample
        from slipstream.transforms import RandomHorizontalFlip
        from slipstream.transforms import RandomColorJitterHSV
        from slipstream.transforms import RandomGrayscale
        from slipstream.transforms import RandomGaussianBlur, RandomSolarization
        from slipstream.transforms import Normalize, ToDevice, ToTorchImage


class TestPresetStructure:
    """Verify presets return valid pipeline dicts."""

    def test_supervised_train_structure(self):
        from slipstream.pipelines import supervised_train
        p = supervised_train(224, seed=42)
        assert isinstance(p, dict)
        assert 'image' in p
        assert isinstance(p['image'], list)
        assert len(p['image']) >= 1

    def test_supervised_val_structure(self):
        from slipstream.pipelines import supervised_val
        p = supervised_val(224)
        assert isinstance(p, dict)
        assert 'image' in p

    def test_simclr_structure(self):
        from slipstream.pipelines import simclr
        p = simclr(224, seed=42)
        assert 'image' in p
        assert len(p['image']) == 2  # two views

    def test_byol_structure(self):
        from slipstream.pipelines import byol
        p = byol(224, seed=42)
        assert 'image' in p
        assert len(p['image']) == 2

    def test_multicrop_structure(self):
        from slipstream.pipelines import multicrop
        p = multicrop(global_crops=2, local_crops=4, seed=42)
        assert 'image' in p
        assert len(p['image']) == 2  # MultiRandomResizedCrop + MultiCropPipeline
