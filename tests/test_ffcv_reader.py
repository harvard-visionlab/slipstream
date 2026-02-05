"""Tests for FFCV reader integration.

Tests FFCVFileReader, SlipstreamDataset auto-detection, and SlipstreamLoader
iteration using synthetic .ffcv files (no external data needed).
"""

import io
import shutil
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from slipstream.backends.ffcv_file import (
    FFCV_ALLOC_ENTRY_DTYPE,
    FFCV_FIELD_DESC_DTYPE,
    FFCV_HEADER_DTYPE,
    FFCV_TYPE_BYTES,
    FFCV_TYPE_INT,
    FFCV_TYPE_RGB_IMAGE,
)
from slipstream.readers.ffcv import FFCVFileReader, _find_image_end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int = 32, height: int = 32, color: tuple = (255, 0, 0)) -> bytes:
    """Create a JPEG image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_synthetic_ffcv(
    path: Path,
    images: list[tuple[bytes, int, int]],
    labels: list[int],
) -> None:
    """Build a minimal synthetic FFCV v2 file.

    Args:
        path: Output file path.
        images: List of (jpeg_bytes, width, height) tuples.
        labels: List of integer labels.
    """
    num_samples = len(images)
    assert len(labels) == num_samples
    num_fields = 2  # image + label
    page_size = 1 << 21  # 2 MiB (minimum valid FFCV page size)

    # ---- Field descriptors ----
    field_descs = np.zeros(num_fields, dtype=FFCV_FIELD_DESC_DTYPE)

    # Field 0: image (RGBImageField)
    field_descs[0]['type_id'] = FFCV_TYPE_RGB_IMAGE
    name0 = b'image'
    field_descs[0]['name'][:len(name0)] = np.frombuffer(name0, dtype='<u1')

    # Field 1: label (IntField)
    field_descs[1]['type_id'] = FFCV_TYPE_INT
    name1 = b'label'
    field_descs[1]['name'][:len(name1)] = np.frombuffer(name1, dtype='<u1')

    # ---- Per-sample metadata (compound dtype, must match FFCV layout) ----
    image_meta_dt = np.dtype([
        ('mode', '<u1'),
        ('width', '<u2'),
        ('height', '<u2'),
        ('data_ptr', '<u8'),
    ])
    int_meta_dt = np.dtype('<i8')

    compound_dt = np.dtype([
        ('f0', image_meta_dt),
        ('f1', int_meta_dt),
    ], align=True)

    metadata = np.zeros(num_samples, dtype=compound_dt)

    # Fill label metadata now; image data_ptr filled below after layout
    for i in range(num_samples):
        jpeg_bytes, w, h = images[i]
        metadata['f0'][i]['mode'] = 0  # JPEG mode
        metadata['f0'][i]['width'] = w
        metadata['f0'][i]['height'] = h
        metadata['f1'][i] = labels[i]

    # ---- Compute data region offset (aligned to page_size) ----
    header_size = FFCV_HEADER_DTYPE.itemsize
    descs_size = field_descs.nbytes
    meta_size = metadata.nbytes
    pre_data = header_size + descs_size + meta_size

    if pre_data % page_size != 0:
        data_region_start = pre_data + page_size - (pre_data % page_size)
    else:
        data_region_start = pre_data

    # ---- Place image data sequentially in data region ----
    alloc_entries = np.zeros(num_samples, dtype=FFCV_ALLOC_ENTRY_DTYPE)
    current_offset = data_region_start

    for i, (jpeg_bytes, w, h) in enumerate(images):
        metadata['f0'][i]['data_ptr'] = current_offset
        alloc_entries[i]['sample_id'] = i
        alloc_entries[i]['ptr'] = current_offset
        alloc_entries[i]['size'] = len(jpeg_bytes)
        current_offset += len(jpeg_bytes)

    alloc_table_offset = current_offset

    # ---- Header ----
    header = np.zeros(1, dtype=FFCV_HEADER_DTYPE)
    header[0]['version'] = 2
    header[0]['num_fields'] = num_fields
    header[0]['page_size'] = page_size
    header[0]['num_samples'] = num_samples
    header[0]['alloc_table_ptr'] = alloc_table_offset

    # ---- Write everything ----
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(field_descs.tobytes())
        f.write(metadata.tobytes())

        # Pad to data region
        current_pos = header_size + descs_size + meta_size
        if current_pos < data_region_start:
            f.write(b'\x00' * (data_region_start - current_pos))

        # Write image data
        for jpeg_bytes, w, h in images:
            f.write(jpeg_bytes)

        # Write alloc table
        f.write(alloc_entries.tobytes())


def _build_synthetic_ffcv_4field(
    path: Path,
    images: list[tuple[bytes, int, int]],
    labels: list[int],
    indices: list[int],
    paths: list[bytes],
) -> None:
    """Build a synthetic FFCV v2 file with 4 fields: image, label, index, path.

    Matches the structure of real ImageNet FFCV files which include a 'path'
    bytes field alongside image, label, and index.
    """
    num_samples = len(images)
    assert len(labels) == len(indices) == len(paths) == num_samples
    num_fields = 4
    page_size = 1 << 21

    # ---- Field descriptors ----
    field_descs = np.zeros(num_fields, dtype=FFCV_FIELD_DESC_DTYPE)
    for i, (name, type_id) in enumerate([
        (b'image', FFCV_TYPE_RGB_IMAGE),
        (b'label', FFCV_TYPE_INT),
        (b'index', FFCV_TYPE_INT),
        (b'path', FFCV_TYPE_BYTES),
    ]):
        field_descs[i]['type_id'] = type_id
        field_descs[i]['name'][:len(name)] = np.frombuffer(name, dtype='<u1')

    # ---- Per-sample metadata (compound dtype) ----
    image_meta_dt = np.dtype([
        ('mode', '<u1'), ('width', '<u2'),
        ('height', '<u2'), ('data_ptr', '<u8'),
    ])
    bytes_meta_dt = np.dtype([('ptr', '<u8'), ('size', '<u8')])
    compound_dt = np.dtype([
        ('f0', image_meta_dt),
        ('f1', np.dtype('<i8')),
        ('f2', np.dtype('<i8')),
        ('f3', bytes_meta_dt),
    ], align=True)

    metadata = np.zeros(num_samples, dtype=compound_dt)
    for i in range(num_samples):
        _, w, h = images[i]
        metadata['f0'][i]['mode'] = 0
        metadata['f0'][i]['width'] = w
        metadata['f0'][i]['height'] = h
        metadata['f1'][i] = labels[i]
        metadata['f2'][i] = indices[i]

    # ---- Data region ----
    header_size = FFCV_HEADER_DTYPE.itemsize
    descs_size = field_descs.nbytes
    meta_size = metadata.nbytes
    pre_data = header_size + descs_size + meta_size
    if pre_data % page_size != 0:
        data_region_start = pre_data + page_size - (pre_data % page_size)
    else:
        data_region_start = pre_data

    # 2 variable-length fields: image and path → interleaved alloc entries
    num_var_fields = 2
    alloc_entries = np.zeros(num_samples * num_var_fields, dtype=FFCV_ALLOC_ENTRY_DTYPE)
    current_offset = data_region_start

    # All data written sequentially: image0, path0, image1, path1, ...
    data_chunks: list[bytes] = []
    for i in range(num_samples):
        jpeg_bytes = images[i][0]
        path_bytes = paths[i]

        # Image alloc entry (interleaved position: i * 2)
        metadata['f0'][i]['data_ptr'] = current_offset
        alloc_entries[i * num_var_fields]['sample_id'] = i
        alloc_entries[i * num_var_fields]['ptr'] = current_offset
        alloc_entries[i * num_var_fields]['size'] = len(jpeg_bytes)
        data_chunks.append(jpeg_bytes)
        current_offset += len(jpeg_bytes)

        # Path alloc entry (interleaved position: i * 2 + 1)
        metadata['f3'][i]['ptr'] = current_offset
        metadata['f3'][i]['size'] = len(path_bytes)
        alloc_entries[i * num_var_fields + 1]['sample_id'] = i
        alloc_entries[i * num_var_fields + 1]['ptr'] = current_offset
        alloc_entries[i * num_var_fields + 1]['size'] = len(path_bytes)
        data_chunks.append(path_bytes)
        current_offset += len(path_bytes)

    alloc_table_offset = current_offset

    # ---- Header ----
    header = np.zeros(1, dtype=FFCV_HEADER_DTYPE)
    header[0]['version'] = 2
    header[0]['num_fields'] = num_fields
    header[0]['page_size'] = page_size
    header[0]['num_samples'] = num_samples
    header[0]['alloc_table_ptr'] = alloc_table_offset

    # ---- Write ----
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(field_descs.tobytes())
        f.write(metadata.tobytes())
        current_pos = header_size + descs_size + meta_size
        if current_pos < data_region_start:
            f.write(b'\x00' * (data_region_start - current_pos))
        for chunk in data_chunks:
            f.write(chunk)
        f.write(alloc_entries.tobytes())


def _build_synthetic_ffcv_with_padding(
    path: Path,
    images: list[tuple[bytes, int, int]],
    labels: list[int],
) -> None:
    """Build a synthetic FFCV v2 file where alloc_ptr != data_ptr.

    This simulates FFCV's page allocator behavior where the alloc table
    entry points to a page-aligned allocation, but metadata data_ptr
    points to the actual data within that page (possibly offset).
    """
    num_samples = len(images)
    assert len(labels) == num_samples
    num_fields = 2
    page_size = 1 << 21  # 2 MiB

    # ---- Field descriptors ----
    field_descs = np.zeros(num_fields, dtype=FFCV_FIELD_DESC_DTYPE)
    field_descs[0]['type_id'] = FFCV_TYPE_RGB_IMAGE
    name0 = b'image'
    field_descs[0]['name'][:len(name0)] = np.frombuffer(name0, dtype='<u1')
    field_descs[1]['type_id'] = FFCV_TYPE_INT
    name1 = b'label'
    field_descs[1]['name'][:len(name1)] = np.frombuffer(name1, dtype='<u1')

    # ---- Metadata ----
    image_meta_dt = np.dtype([
        ('mode', '<u1'), ('width', '<u2'),
        ('height', '<u2'), ('data_ptr', '<u8'),
    ])
    compound_dt = np.dtype([
        ('f0', image_meta_dt), ('f1', np.dtype('<i8')),
    ], align=True)
    metadata = np.zeros(num_samples, dtype=compound_dt)

    for i in range(num_samples):
        jpeg_bytes, w, h = images[i]
        metadata['f0'][i]['mode'] = 0
        metadata['f0'][i]['width'] = w
        metadata['f0'][i]['height'] = h
        metadata['f1'][i] = labels[i]

    # ---- Data region ----
    header_size = FFCV_HEADER_DTYPE.itemsize
    descs_size = field_descs.nbytes
    meta_size = metadata.nbytes
    pre_data = header_size + descs_size + meta_size
    if pre_data % page_size != 0:
        data_region_start = pre_data + page_size - (pre_data % page_size)
    else:
        data_region_start = pre_data

    # Pack images with 64-byte padding before each JPEG (simulates
    # alloc_ptr pointing to allocation start, data_ptr offset within)
    PADDING = 64
    alloc_entries = np.zeros(num_samples, dtype=FFCV_ALLOC_ENTRY_DTYPE)
    current_offset = data_region_start

    for i, (jpeg_bytes, w, h) in enumerate(images):
        alloc_start = current_offset
        data_start = current_offset + PADDING
        total_alloc_size = PADDING + len(jpeg_bytes)

        metadata['f0'][i]['data_ptr'] = data_start
        alloc_entries[i]['sample_id'] = i
        alloc_entries[i]['ptr'] = alloc_start
        alloc_entries[i]['size'] = total_alloc_size
        current_offset += total_alloc_size

    alloc_table_offset = current_offset

    # ---- Header ----
    header = np.zeros(1, dtype=FFCV_HEADER_DTYPE)
    header[0]['version'] = 2
    header[0]['num_fields'] = num_fields
    header[0]['page_size'] = page_size
    header[0]['num_samples'] = num_samples
    header[0]['alloc_table_ptr'] = alloc_table_offset

    # ---- Write ----
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(field_descs.tobytes())
        f.write(metadata.tobytes())

        current_pos = header_size + descs_size + meta_size
        if current_pos < data_region_start:
            f.write(b'\x00' * (data_region_start - current_pos))

        for jpeg_bytes, w, h in images:
            f.write(b'\x00' * PADDING)  # padding (garbage before JPEG)
            f.write(jpeg_bytes)

        f.write(alloc_entries.tobytes())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ffcv(tmp_path):
    """Create a synthetic .ffcv file with 8 small JPEG images."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 128), (64, 192, 32),
    ]
    images = []
    for color in colors:
        jpeg = _create_test_jpeg(width=32, height=24, color=color)
        images.append((jpeg, 32, 24))

    labels = list(range(8))
    ffcv_path = tmp_path / "test.ffcv"
    _build_synthetic_ffcv(ffcv_path, images, labels)
    return ffcv_path, images, labels


@pytest.fixture
def synthetic_ffcv_padded(tmp_path):
    """Create a synthetic .ffcv file where alloc_ptr != data_ptr."""
    images = []
    for i in range(4):
        jpeg = _create_test_jpeg(width=16, height=16, color=(i * 60, 100, 200))
        images.append((jpeg, 16, 16))

    labels = [10, 20, 30, 40]
    ffcv_path = tmp_path / "padded.ffcv"
    _build_synthetic_ffcv_with_padding(ffcv_path, images, labels)
    return ffcv_path, images, labels


# ---------------------------------------------------------------------------
# Tests: _find_image_end helper
# ---------------------------------------------------------------------------

class TestFindImageEnd:
    def test_jpeg_with_eoi(self):
        jpeg = _create_test_jpeg(16, 16)
        # Append garbage after the JPEG
        data = jpeg + b'\x00' * 100
        result = _find_image_end(data, len(data))
        assert result == len(jpeg)

    def test_non_jpeg_returns_max_len(self):
        data = b'\x89PNG' + b'\x00' * 100
        result = _find_image_end(data, 50)
        assert result == 50

    def test_numpy_input(self):
        jpeg = _create_test_jpeg(8, 8)
        arr = np.frombuffer(jpeg + b'\x00' * 50, dtype=np.uint8)
        result = _find_image_end(arr, len(arr))
        assert result == len(jpeg)


# ---------------------------------------------------------------------------
# Tests: FFCVFileReader
# ---------------------------------------------------------------------------

class TestFFCVFileReader:
    def test_basic_read(self, synthetic_ffcv):
        ffcv_path, images, labels = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        assert len(reader) == 8
        assert reader.field_types == {'image': 'ImageBytes', 'label': 'int'}

    def test_image_fields_property(self, synthetic_ffcv):
        ffcv_path, images, labels = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)
        assert reader.image_fields == ['image']

    def test_getitem_returns_correct_data(self, synthetic_ffcv):
        ffcv_path, images, labels = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        for i in range(len(images)):
            sample = reader[i]
            assert 'image' in sample
            assert 'label' in sample
            assert sample['label'] == labels[i]
            # Image bytes should be valid JPEG
            assert sample['image'][:2] == b'\xff\xd8'
            assert sample['image'][-2:] == b'\xff\xd9'
            # Should match the original JPEG exactly
            assert sample['image'] == images[i][0]

    def test_getitem_image_decodable(self, synthetic_ffcv):
        ffcv_path, images, labels = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        sample = reader[0]
        img = PIL.Image.open(io.BytesIO(sample['image']))
        assert img.size == (32, 24)
        assert img.mode == 'RGB'

    def test_read_all_fields(self, synthetic_ffcv):
        ffcv_path, images, labels = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        all_fields = reader.read_all_fields()
        assert all_fields is not None
        assert len(all_fields['image']) == 8
        assert len(all_fields['label']) == 8
        assert all_fields['label'] == labels

        # Image metadata
        assert len(all_fields['__image_sizes']) == 8
        assert len(all_fields['__image_heights']) == 8
        assert len(all_fields['__image_widths']) == 8

        # All images should be valid JPEG
        for i, img_bytes in enumerate(all_fields['image']):
            assert img_bytes[:2] == b'\xff\xd8'
            assert img_bytes[-2:] == b'\xff\xd9'
            assert img_bytes == images[i][0]

    def test_cache_path(self, synthetic_ffcv, tmp_path):
        ffcv_path, _, _ = synthetic_ffcv
        cache_dir = tmp_path / "my_cache"
        reader = FFCVFileReader(str(ffcv_path), cache_dir=str(cache_dir), verbose=False)
        assert reader.cache_path == cache_dir

    def test_repr(self, synthetic_ffcv):
        ffcv_path, _, _ = synthetic_ffcv
        reader = FFCVFileReader(str(ffcv_path), verbose=False)
        r = repr(reader)
        assert 'FFCVFileReader' in r
        assert 'num_samples=8' in r


class TestFFCVFileReaderPadded:
    """Test that the reader correctly handles alloc_ptr != data_ptr."""

    def test_padded_read(self, synthetic_ffcv_padded):
        ffcv_path, images, labels = synthetic_ffcv_padded
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        assert len(reader) == 4

        for i in range(4):
            sample = reader[i]
            assert sample['label'] == labels[i]
            # Must start with JPEG SOI, not with padding garbage
            assert sample['image'][:2] == b'\xff\xd8'
            assert sample['image'][-2:] == b'\xff\xd9'
            assert sample['image'] == images[i][0]

    def test_padded_read_all_fields(self, synthetic_ffcv_padded):
        ffcv_path, images, labels = synthetic_ffcv_padded
        reader = FFCVFileReader(str(ffcv_path), verbose=False)

        all_fields = reader.read_all_fields()
        for i in range(4):
            assert all_fields['image'][i] == images[i][0]
            assert all_fields['label'][i] == labels[i]


# ---------------------------------------------------------------------------
# Tests: SlipstreamDataset auto-detection
# ---------------------------------------------------------------------------

class TestFFCVAutoDetect:
    def test_auto_detect_ffcv_extension(self, synthetic_ffcv, tmp_path):
        from slipstream import SlipstreamDataset

        ffcv_path, images, labels = synthetic_ffcv
        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(tmp_path / "cache"),
            decode_images=False,
        )

        assert len(dataset) == 8
        assert dataset.field_types == {'image': 'ImageBytes', 'label': 'int'}

    def test_auto_detect_beton_extension(self, synthetic_ffcv, tmp_path):
        """Ensure .beton extension is also detected."""
        from slipstream import SlipstreamDataset
        from slipstream.dataset import _is_ffcv_source

        ffcv_path, _, _ = synthetic_ffcv
        beton_path = ffcv_path.with_suffix('.beton')
        shutil.copy2(ffcv_path, beton_path)

        assert _is_ffcv_source(str(beton_path))

        dataset = SlipstreamDataset(
            local_dir=str(beton_path),
            cache_dir=str(tmp_path / "cache"),
            decode_images=False,
        )
        assert len(dataset) == 8

    def test_getitem_through_dataset(self, synthetic_ffcv, tmp_path):
        from slipstream import SlipstreamDataset

        ffcv_path, images, labels = synthetic_ffcv
        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(tmp_path / "cache"),
            decode_images=False,
        )

        sample = dataset[0]
        assert sample['label'] == 0
        assert sample['image'][:2] == b'\xff\xd8'

    def test_getitem_with_decode(self, synthetic_ffcv, tmp_path):
        from slipstream import SlipstreamDataset

        ffcv_path, images, labels = synthetic_ffcv
        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(tmp_path / "cache"),
            decode_images=True,
        )

        sample = dataset[0]
        assert isinstance(sample['image'], PIL.Image.Image)
        assert sample['image'].size == (32, 24)


# ---------------------------------------------------------------------------
# Tests: SlipstreamLoader with FFCV dataset
# ---------------------------------------------------------------------------

class TestFFCVLoader:
    def test_loader_builds_cache_and_iterates(self, synthetic_ffcv, tmp_path):
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.decoders.crop import DecodeCenterCrop

        ffcv_path, images, labels = synthetic_ffcv
        cache_dir = tmp_path / "cache"

        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(cache_dir),
            decode_images=False,
        )

        loader = SlipstreamLoader(
            dataset,
            batch_size=4,
            pipelines={
                "image": [DecodeCenterCrop(size=16)],
            },
        )

        batches = list(loader)
        assert len(batches) == 2  # 8 samples / batch_size 4

        batch = batches[0]
        assert 'image' in batch
        img = batch['image']
        # DecodeCenterCrop outputs HWC numpy by default
        assert img.shape[0] == 4   # batch
        assert img.shape[1] == 16  # height
        assert img.shape[2] == 16  # width
        assert img.shape[3] == 3   # channels

        assert 'label' in batch
        assert len(batch['label']) == 4

    def test_loader_force_rebuild(self, synthetic_ffcv, tmp_path):
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.decoders.crop import DecodeCenterCrop

        ffcv_path, images, labels = synthetic_ffcv
        cache_dir = tmp_path / "cache"

        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(cache_dir),
            decode_images=False,
        )

        # Build once
        loader1 = SlipstreamLoader(
            dataset, batch_size=4,
            pipelines={"image": [DecodeCenterCrop(size=16)]},
        )
        _ = list(loader1)

        # Rebuild
        loader2 = SlipstreamLoader(
            dataset, batch_size=4,
            pipelines={"image": [DecodeCenterCrop(size=16)]},
            force_rebuild=True,
        )
        batches = list(loader2)
        assert len(batches) == 2

    def test_loader_with_padded_ffcv(self, synthetic_ffcv_padded, tmp_path):
        """Ensure loader works when alloc_ptr != data_ptr."""
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.decoders.crop import DecodeCenterCrop

        ffcv_path, images, labels = synthetic_ffcv_padded
        cache_dir = tmp_path / "cache"

        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(cache_dir),
            decode_images=False,
        )

        loader = SlipstreamLoader(
            dataset, batch_size=2,
            pipelines={"image": [DecodeCenterCrop(size=8)]},
        )

        batches = list(loader)
        assert len(batches) == 2  # 4 samples / 2

        # Verify labels are correct
        all_labels = []
        for batch in batches:
            all_labels.extend(batch['label'].tolist() if hasattr(batch['label'], 'tolist') else batch['label'])
        assert sorted(all_labels) == sorted(labels)

    def test_loader_with_bytes_field(self, tmp_path):
        """Regression test: FFCV 'bytes' fields must not be treated as images.

        Real FFCV files (e.g. ImageNet) include 'path' as a bytes field.
        The cache builder must store these as raw bytes, not route them
        through image detection / YUV420 conversion.
        """
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.decoders.crop import DecodeCenterCrop

        # Create 4-field FFCV file matching real ImageNet structure
        images = []
        for i in range(6):
            jpeg = _create_test_jpeg(width=32, height=24, color=(i * 40, 100, 200))
            images.append((jpeg, 32, 24))

        labels = [0, 0, 1, 1, 2, 2]
        indices = list(range(6))
        paths = [
            b'val/n01440764/ILSVRC2012_val_00000001.JPEG',
            b'val/n01440764/ILSVRC2012_val_00000002.JPEG',
            b'val/n01443537/ILSVRC2012_val_00000003.JPEG',
            b'val/n01443537/ILSVRC2012_val_00000004.JPEG',
            b'val/n01484850/ILSVRC2012_val_00000005.JPEG',
            b'val/n01484850/ILSVRC2012_val_00000006.JPEG',
        ]

        ffcv_path = tmp_path / "4field.ffcv"
        _build_synthetic_ffcv_4field(ffcv_path, images, labels, indices, paths)
        cache_dir = tmp_path / "cache"

        # Read via SlipstreamDataset
        dataset = SlipstreamDataset(
            local_dir=str(ffcv_path),
            cache_dir=str(cache_dir),
            decode_images=False,
        )

        assert dataset.field_types == {
            'image': 'ImageBytes', 'label': 'int',
            'index': 'int', 'path': 'str',
        }

        # Build cache and iterate — this was the crash:
        # 'path' bytes field was routed through ImageBytesStorage → YUV420 → crash
        loader = SlipstreamLoader(
            dataset, batch_size=3,
            pipelines={"image": [DecodeCenterCrop(size=16)]},
            use_threading=False,
        )

        batches = list(loader)
        assert len(batches) == 2  # 6 / 3

        batch = batches[0]
        assert 'image' in batch
        assert batch['image'].shape == (3, 16, 16, 3)
        assert 'label' in batch
        assert 'index' in batch
        assert 'path' in batch
