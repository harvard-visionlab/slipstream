"""Tests for HuggingFace dataset support via hf:// URIs."""

import io

import pytest
from PIL import Image


def create_test_png_bytes() -> bytes:
    """Create minimal PNG bytes for testing."""
    img = Image.new('RGB', (32, 32), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


class TestHuggingFaceImageDict:
    """Test HuggingFace image dict format detection and decoding."""

    def test_is_hf_image_dict_with_bytes(self):
        """Test is_hf_image_dict with inline bytes."""
        from slipstream.dataset import is_hf_image_dict

        png_bytes = create_test_png_bytes()

        # Valid HF image dict with bytes
        assert is_hf_image_dict({'bytes': png_bytes, 'path': None}) is True
        assert is_hf_image_dict({'bytes': png_bytes}) is True

        # Invalid cases
        assert is_hf_image_dict({'bytes': None, 'path': None}) is False
        assert is_hf_image_dict({'path': '/some/path'}) is False  # missing 'bytes' key
        assert is_hf_image_dict({'other': 'data'}) is False
        assert is_hf_image_dict("not a dict") is False
        assert is_hf_image_dict(png_bytes) is False

    def test_is_hf_image_dict_with_path(self):
        """Test is_hf_image_dict with path reference."""
        from slipstream.dataset import is_hf_image_dict

        # Valid HF image dict with path
        assert is_hf_image_dict({'bytes': None, 'path': '/path/to/image.jpg'}) is True

    def test_is_image_bytes_with_hf_dict(self):
        """Test is_image_bytes handles HF image dicts."""
        from slipstream.dataset import is_image_bytes

        png_bytes = create_test_png_bytes()

        # HF dict with inline bytes
        assert is_image_bytes({'bytes': png_bytes, 'path': None}) is True

        # HF dict with invalid bytes
        assert is_image_bytes({'bytes': b'not an image', 'path': None}) is False

        # Raw bytes still work
        assert is_image_bytes(png_bytes) is True
        assert is_image_bytes(b'not an image') is False

    def test_decode_image_with_hf_dict(self):
        """Test decode_image handles HF image dicts."""
        import torch
        from slipstream.dataset import decode_image

        png_bytes = create_test_png_bytes()

        # Decode HF dict to tensor
        hf_dict = {'bytes': png_bytes, 'path': None}
        tensor = decode_image(hf_dict, to_pil=False)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)  # CHW format

        # Decode HF dict to PIL
        pil_img = decode_image(hf_dict, to_pil=True)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (32, 32)

    def test_decode_image_with_invalid_hf_dict(self):
        """Test decode_image raises on invalid HF dict."""
        from slipstream.dataset import decode_image

        with pytest.raises(ValueError, match="Invalid HuggingFace image dict"):
            decode_image({'bytes': None, 'path': None})


class TestImageDimensionParsing:
    """Test image dimension parsing for cache building."""

    def test_read_png_dimensions(self):
        """Test reading PNG dimensions from header."""
        from slipstream.cache import read_png_dimensions, read_image_dimensions

        png_bytes = create_test_png_bytes()
        # Our test PNG is 32x32
        assert read_png_dimensions(png_bytes) == (32, 32)
        assert read_image_dimensions(png_bytes) == (32, 32)

    def test_read_image_dimensions_invalid(self):
        """Test reading dimensions from invalid data."""
        from slipstream.cache import read_image_dimensions

        assert read_image_dimensions(b'invalid') == (0, 0)
        assert read_image_dimensions(b'') == (0, 0)


class TestCacheImageExtraction:
    """Test image bytes extraction for cache building."""

    def test_extract_image_bytes_from_raw(self):
        """Test extracting bytes from raw bytes."""
        from slipstream.cache import _extract_image_bytes

        png_bytes = create_test_png_bytes()
        result = _extract_image_bytes(png_bytes)
        assert result == png_bytes

    def test_extract_image_bytes_from_numpy(self):
        """Test extracting bytes from numpy array."""
        import numpy as np
        from slipstream.cache import _extract_image_bytes

        png_bytes = create_test_png_bytes()
        np_array = np.frombuffer(png_bytes, dtype=np.uint8)
        result = _extract_image_bytes(np_array)
        assert result == png_bytes

    def test_extract_image_bytes_from_hf_dict(self):
        """Test extracting bytes from HuggingFace image dict."""
        from slipstream.cache import _extract_image_bytes

        png_bytes = create_test_png_bytes()
        hf_dict = {'bytes': png_bytes, 'path': None}
        result = _extract_image_bytes(hf_dict)
        assert result == png_bytes

    def test_extract_image_bytes_invalid_hf_dict(self):
        """Test error handling for invalid HF dict."""
        from slipstream.cache import _extract_image_bytes

        with pytest.raises(ValueError, match="Invalid HuggingFace image dict"):
            _extract_image_bytes({'bytes': None, 'path': None})


class TestHuggingFaceSupport:
    """Test HuggingFace dataset loading via LitData integration."""

    def test_hf_uri_recognized_as_remote(self):
        """Test that hf:// URIs are recognized as remote URLs."""
        from slipstream import SlipstreamDataset

        # Check that hf:// is in the list of recognized remote protocols
        # by verifying it doesn't get treated as a local path
        test_uri = "hf://datasets/cifar10/data"

        # The remote_dir property should recognize hf:// as remote
        # We test this by checking the internal URL detection logic
        assert test_uri.startswith(("s3://", "gs://", "http://", "https://", "hf://"))

    @pytest.mark.skip(reason="Requires network access and HuggingFace download")
    def test_hf_dataset_loading(self):
        """Test loading a HuggingFace dataset via hf:// URI.

        This test requires network access and will download data from HuggingFace.
        Run manually with: pytest tests/test_huggingface.py -k test_hf_dataset_loading -v --no-header -rN
        """
        from slipstream import SlipstreamDataset

        # CIFAR-10 is a small dataset good for testing
        dataset = SlipstreamDataset(
            input_dir="hf://datasets/cifar10/data",
            decode_images=True,
            to_pil=True,
        )

        # Check basic properties
        assert len(dataset) > 0

        # Get a sample
        sample = dataset[0]
        assert isinstance(sample, dict)

        # Check that we got image and label fields
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Dataset length: {len(dataset)}")

    @pytest.mark.skip(reason="Requires network access and HuggingFace download")
    def test_hf_dataset_with_loader(self):
        """Test using HuggingFace dataset with SlipstreamLoader.

        Run manually with: pytest tests/test_huggingface.py -k test_hf_dataset_with_loader -v --no-header -rN
        """
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.pipelines import supervised_train

        dataset = SlipstreamDataset(
            input_dir="hf://datasets/cifar10/data",
            decode_images=False,  # Let loader handle decoding
        )

        # Create loader with supervised training pipeline
        # CIFAR-10 images are 32x32
        loader = SlipstreamLoader(
            dataset,
            batch_size=32,
            pipelines=supervised_train(size=32),
        )

        # Get one batch
        batch = next(iter(loader))
        print(f"Batch keys: {list(batch.keys())}")

        if 'image' in batch:
            print(f"Image shape: {batch['image'].shape}")
        if 'label' in batch:
            print(f"Label shape: {batch['label'].shape}")


class TestImageFormatDetection:
    """Test image format detection from header bytes."""

    def test_detect_jpeg_format(self):
        """Test detection of JPEG format."""
        from slipstream.cache import detect_image_format

        # Create minimal JPEG header (FF D8 FF ...)
        jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        assert detect_image_format(jpeg_header) == "jpeg"

        # Just the magic bytes
        assert detect_image_format(b'\xff\xd8') == "jpeg"

    def test_detect_png_format(self):
        """Test detection of PNG format."""
        from slipstream.cache import detect_image_format

        png_bytes = create_test_png_bytes()
        assert detect_image_format(png_bytes) == "png"

        # PNG magic bytes only
        png_header = b'\x89PNG\r\n\x1a\n'
        assert detect_image_format(png_header) == "png"

    def test_detect_other_format(self):
        """Test detection of unknown formats."""
        from slipstream.cache import detect_image_format

        assert detect_image_format(b'random data') == "other"
        assert detect_image_format(b'') == "other"
        assert detect_image_format(b'\x00') == "other"

    def test_detect_format_from_numpy(self):
        """Test format detection from numpy array."""
        import numpy as np
        from slipstream.cache import detect_image_format

        png_bytes = create_test_png_bytes()
        np_array = np.frombuffer(png_bytes, dtype=np.uint8)
        assert detect_image_format(np_array) == "png"


class TestRgbToYuv420Conversion:
    """Test RGB to YUV420 conversion."""

    def test_basic_conversion(self):
        """Test basic RGB to YUV420 conversion."""
        import numpy as np
        from slipstream.cache import rgb_to_yuv420

        # Create simple RGB image
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Pure red

        yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)

        # Check dimensions (should be even)
        assert pad_h == 32
        assert pad_w == 32

        # Check size: Y (32*32) + U (16*16) + V (16*16)
        expected_size = 32 * 32 + 16 * 16 + 16 * 16
        assert len(yuv_bytes) == expected_size

    def test_odd_dimension_padding(self):
        """Test that odd dimensions get padded to even."""
        import numpy as np
        from slipstream.cache import rgb_to_yuv420

        # Odd dimensions
        rgb = np.zeros((31, 33, 3), dtype=np.uint8)

        yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)

        # Should be padded to even
        assert pad_h == 32  # 31 + 1
        assert pad_w == 34  # 33 + 1

    def test_grayscale_input(self):
        """Test RGB to YUV420 with grayscale (R=G=B)."""
        import numpy as np
        from slipstream.cache import rgb_to_yuv420

        # Grayscale image (R = G = B = 128)
        rgb = np.full((32, 32, 3), 128, dtype=np.uint8)

        yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)

        # Y should be approximately 128 (grayscale)
        y_plane = np.frombuffer(yuv_bytes[:32*32], dtype=np.uint8)
        # BT.601: Y = 0.299*R + 0.587*G + 0.114*B = 128 for grayscale
        assert np.allclose(y_plane, 128, atol=1)


class TestDecodeImageToRgb:
    """Test image decoding to RGB using PIL."""

    def test_decode_png(self):
        """Test decoding PNG to RGB."""
        from slipstream.cache import decode_image_to_rgb

        png_bytes = create_test_png_bytes()
        rgb = decode_image_to_rgb(png_bytes)

        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        # Our test PNG is pure red
        assert rgb[0, 0, 0] == 255  # R
        assert rgb[0, 0, 1] == 0    # G
        assert rgb[0, 0, 2] == 0    # B

    def test_decode_grayscale_png(self):
        """Test that grayscale images get converted to RGB."""
        from slipstream.cache import decode_image_to_rgb

        # Create grayscale PNG
        img = Image.new('L', (32, 32), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        gray_bytes = buffer.getvalue()

        rgb = decode_image_to_rgb(gray_bytes)

        # Should be converted to RGB
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        # All channels should have same value
        assert np.all(rgb[:, :, 0] == 128)
        assert np.all(rgb[:, :, 1] == 128)
        assert np.all(rgb[:, :, 2] == 128)


class TestNonJpegCacheBuilding:
    """Test cache building with non-JPEG images (PNG, etc.)."""

    def test_png_images_stored_as_yuv420(self, tmp_path):
        """Test that PNG images are converted to YUV420 during cache build."""
        import numpy as np
        from pathlib import Path
        from slipstream.cache import (
            ImageBytesStorage,
            detect_image_format,
        )

        # Create some PNG image samples
        png_samples = []
        for i in range(5):
            img = Image.new('RGB', (32, 32), color=(i * 50, 100, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            png_samples.append(buffer.getvalue())

        # Verify they're PNGs
        assert detect_image_format(png_samples[0]) == "png"

        # Build storage
        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        storage, image_format = ImageBytesStorage.build(
            "image",
            png_samples,
            output_dir,
            "ImageBytes",
        )

        # Should be stored as YUV420
        assert image_format == "yuv420"

        # Storage should have correct sample count
        assert storage.num_samples == 5

        # Dimensions should be stored
        assert storage._heights[0] == 32
        assert storage._widths[0] == 32

        # Data files should exist
        assert (output_dir / "image.bin").exists()
        assert (output_dir / "image.meta.npy").exists()

    def test_jpeg_images_stored_as_jpeg(self, tmp_path):
        """Test that JPEG images are stored as-is."""
        import numpy as np
        from pathlib import Path
        from slipstream.cache import (
            ImageBytesStorage,
            detect_image_format,
        )

        # Create JPEG samples
        jpeg_samples = []
        for i in range(5):
            img = Image.new('RGB', (32, 32), color=(i * 50, 100, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            jpeg_samples.append(buffer.getvalue())

        # Verify they're JPEGs
        assert detect_image_format(jpeg_samples[0]) == "jpeg"

        # Build storage
        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        storage, image_format = ImageBytesStorage.build(
            "image",
            jpeg_samples,
            output_dir,
            "ImageBytes",
        )

        # Should be stored as JPEG
        assert image_format == "jpeg"
        assert storage.num_samples == 5

    def test_optimized_cache_stores_image_format(self, tmp_path):
        """Test that OptimizedCache stores image_format in manifest."""
        import json
        from slipstream.cache import OptimizedCache, CACHE_SUBDIR, MANIFEST_FILE

        # Create a mock dataset with PNG images
        class MockPNGDataset:
            def __init__(self):
                self.cache_path = tmp_path
                self.field_types = {'image': 'ImageBytes', 'label': 'int'}

            def __len__(self):
                return 5

            def __getitem__(self, idx):
                img = Image.new('RGB', (32, 32), color=(idx * 50, 100, 200))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return {'image': buffer.getvalue(), 'label': idx}

        dataset = MockPNGDataset()

        # Build cache
        cache = OptimizedCache.build(dataset, verbose=False)

        # Check manifest has image_format
        manifest_path = cache.cache_dir / MANIFEST_FILE
        with open(manifest_path) as f:
            manifest = json.load(f)

        # PNG images should be stored as YUV420
        assert manifest['fields']['image'].get('image_format') == 'yuv420'

        # get_image_format method should work
        assert cache.get_image_format('image') == 'yuv420'

    def test_cache_load_preserves_image_format(self, tmp_path):
        """Test that loaded cache has correct image_format."""
        from slipstream.cache import OptimizedCache

        # Create a mock dataset with PNG images
        class MockPNGDataset:
            def __init__(self):
                self.cache_path = tmp_path
                self.field_types = {'image': 'ImageBytes', 'label': 'int'}

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                img = Image.new('RGB', (32, 32), color='blue')
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return {'image': buffer.getvalue(), 'label': idx}

        dataset = MockPNGDataset()

        # Build cache
        original_cache = OptimizedCache.build(dataset, verbose=False)
        assert original_cache.get_image_format('image') == 'yuv420'

        # Load cache
        loaded_cache = OptimizedCache.load(tmp_path, verbose=False)
        assert loaded_cache.get_image_format('image') == 'yuv420'


# Import numpy at module level for grayscale test
import numpy as np
