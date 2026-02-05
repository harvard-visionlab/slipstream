"""SlipstreamDataset: High-performance streaming dataset for PyTorch vision workloads.

This module provides SlipstreamDataset, a wrapper around LitData's StreamingDataset
with an intuitive API, automatic field type detection, and pipeline support.

Example:
    from slipstream import SlipstreamDataset

    # Intuitive API
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/",
        cache_dir="/local/cache",
        decode_images=True,
    )

    # Or with pipelines for training
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/dataset/",
        decode_images=False,  # Let the loader handle decoding
        pipelines={"image": my_decoder},
    )
"""

from __future__ import annotations

import io
import os
import pathlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode
from torchvision.io import decode_image as tv_decode_image
from torchvision.transforms import functional as tvf

if TYPE_CHECKING:
    from litdata.streaming.resolver import Dir

__all__ = [
    "SlipstreamDataset",
    "decode_image",
    "is_image_bytes",
    "is_hf_image_dict",
    "extract_hf_image_bytes",
    "ensure_lightning_symlink_on_cluster",
    "get_default_cache_dir",
    "list_collate_fn",
    "detect_local_dataset_type",
    "is_imagefolder_structure",
]


def list_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that keeps variable-sized fields as lists.

    Use this with PyTorch DataLoader when you have variable-sized images
    or other fields that can't be stacked into tensors.

    Example:
        from slipstream import SlipstreamDataset, list_collate_fn
        from torch.utils.data import DataLoader

        dataset = SlipstreamDataset(remote_dir="s3://...", decode_images=False)
        loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            collate_fn=list_collate_fn,
        )

    Args:
        batch: List of sample dicts from the dataset

    Returns:
        Dict with field names as keys. Numeric fields are stacked into tensors,
        variable-sized fields (bytes, images, strings) are kept as lists.
    """
    if not batch:
        return {}

    result: dict[str, Any] = {}
    first = batch[0]

    for key in first.keys():
        values = [sample[key] for sample in batch]
        first_val = values[0]

        # Try to stack numeric types into tensors
        if isinstance(first_val, (int, float)):
            result[key] = torch.tensor(values)
        elif isinstance(first_val, np.ndarray) and first_val.dtype.kind in 'iufb':
            # Numeric numpy arrays with same shape
            try:
                stacked = np.stack(values)
                result[key] = torch.from_numpy(stacked)
            except ValueError:
                # Different shapes, keep as list
                result[key] = values
        elif isinstance(first_val, torch.Tensor):
            # Try to stack tensors
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                # Different shapes, keep as list
                result[key] = values
        else:
            # Bytes, strings, PIL Images, variable-sized arrays -> keep as list
            result[key] = values

    return result


def get_default_cache_dir() -> pathlib.Path:
    """Get the default cache directory for slipstream/LitData.

    Resolution order:
    1. SLIPSTREAM_CACHE_DIR environment variable (if set)
    2. LitData's default cache directory (via litdata.streaming.resolver.get_default_cache_dir)

    On visionlab clusters, ~/.lightning is symlinked to shared storage,
    so the LitData default points to the shared lab cache.

    Returns:
        Path to the cache directory.
    """
    # Check for slipstream-specific override
    slipstream_cache = os.environ.get("SLIPSTREAM_CACHE_DIR")
    if slipstream_cache:
        return pathlib.Path(slipstream_cache)

    # Fall back to LitData's default
    try:
        from litdata.streaming.resolver import get_default_cache_dir as litdata_get_cache_dir
        return pathlib.Path(litdata_get_cache_dir())
    except ImportError:
        # Fallback if litdata internals change
        return pathlib.Path.home() / ".lightning"


def is_slurm_available() -> bool:
    """Check if running on a SLURM cluster."""
    return any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME"])


def ensure_lightning_symlink_on_cluster() -> None:
    """Ensure ~/.lightning symlink exists on SLURM clusters.

    On visionlab clusters, LitData uses ~/.lightning for credentials and cache.
    This function creates a symlink to the shared lab directory.

    Raises:
        RuntimeError: If symlink exists but points to wrong location,
            or if symlink cannot be created.
    """
    if not is_slurm_available():
        return  # Not on cluster, nothing to do

    home = pathlib.Path.home()
    symlink_path = home / ".lightning"
    target_path = pathlib.Path("/n/netscratch/alvarez_lab/Everyone/.lightning")

    # Case 1: Symlink already exists and points correctly
    if symlink_path.is_symlink():
        if symlink_path.resolve() == target_path:
            return  # All good
        raise RuntimeError(
            f"~/.lightning already exists but points to {symlink_path.resolve()}, "
            f"expected {target_path}. Please fix manually."
        )

    # Case 2: ~/.lightning exists but is not a symlink
    if symlink_path.exists():
        raise RuntimeError(
            f"~/.lightning exists but is not a symlink. "
            f"Please remove it and create a symlink to {target_path}."
        )

    # Case 3: Doesn't exist → create it
    try:
        symlink_path.symlink_to(target_path)
        print(f"Created symlink: {symlink_path} -> {target_path}")
    except Exception as e:
        raise RuntimeError(
            f"Could not create symlink {symlink_path} -> {target_path}. "
            f"Please create it manually."
        ) from e


def is_hf_image_dict(value: Any) -> bool:
    """Check if value is a HuggingFace image dict format.

    HuggingFace stores images in Parquet as dicts with 'bytes' and/or 'path' keys:
        {'bytes': b'\\x89PNG...', 'path': None}  # inline bytes
        {'bytes': None, 'path': '/path/to/image.jpg'}  # path reference

    Args:
        value: Value to check.

    Returns:
        True if value matches the HuggingFace image dict format.
    """
    if not isinstance(value, dict):
        return False
    if 'bytes' not in value:
        return False
    # Either has bytes data, or has a path to read from
    return isinstance(value.get('bytes'), bytes) or isinstance(value.get('path'), str)


def extract_hf_image_bytes(value: dict) -> bytes:
    """Extract raw image bytes from a HuggingFace image dict.

    Args:
        value: HuggingFace image dict {'bytes': ..., 'path': ...}

    Returns:
        Raw image bytes.

    Raises:
        ValueError: If dict doesn't contain valid image data.
    """
    if 'bytes' in value and value['bytes']:
        return value['bytes'] if isinstance(value['bytes'], bytes) else bytes(value['bytes'])
    elif 'path' in value and value['path']:
        with open(value['path'], 'rb') as f:
            return f.read()
    else:
        raise ValueError(f"Invalid HuggingFace image dict: {value}")


def is_image_bytes(data: bytes | dict) -> bool:
    """Check if data represents a valid image.

    Handles:
    - Raw bytes (JPEG, PNG, etc.)
    - HuggingFace image dicts: {'bytes': ..., 'path': ...}

    Uses PIL's verify() which checks file structure without decoding pixels.
    This is robust and handles edge cases well.

    Note: Only called once per field during _set_field_types(), so performance
    impact is negligible.

    Args:
        data: Raw bytes or HuggingFace image dict to check.

    Returns:
        True if data can be opened as an image, False otherwise.
    """
    # Handle HuggingFace image dict format
    if isinstance(data, dict):
        if 'bytes' in data and isinstance(data['bytes'], bytes):
            data = data['bytes']
        elif 'path' in data and data['path']:
            # Path-based image, assume valid if path exists
            return True
        else:
            return False

    if not isinstance(data, bytes):
        return False

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False


def decode_image(image_data: bytes | dict | np.ndarray | torch.Tensor | Image.Image,
                 to_pil: bool = False) -> torch.Tensor | Image.Image:
    """Decode image data to tensor (CHW format) or PIL Image.

    Handles various input types gracefully:
    - torch.Tensor: returned as-is (or converted to PIL)
    - PIL.Image: converted to tensor CHW (or returned as-is)
    - np.ndarray (HWC): converted to tensor CHW (or PIL)
    - bytes or np.ndarray (1D): decoded using torchvision
    - HuggingFace image dict: {'bytes': ..., 'path': ...}

    Args:
        image_data: Image data in various formats.
        to_pil: If True, return PIL Image instead of tensor.

    Returns:
        RGB image as torch.Tensor in CHW format (uint8 [0-255]) or PIL Image.
    """
    # Handle HuggingFace image dict format
    if isinstance(image_data, dict):
        if 'bytes' in image_data and image_data['bytes']:
            image_data = image_data['bytes']
        elif 'path' in image_data and image_data['path']:
            # Read from path
            with open(image_data['path'], 'rb') as f:
                image_data = f.read()
        else:
            raise ValueError(f"Invalid HuggingFace image dict: {image_data}")

    # Already a tensor - return as-is or convert to PIL
    if isinstance(image_data, torch.Tensor):
        return tvf.to_pil_image(image_data) if to_pil else image_data

    # Already a PIL Image - convert to tensor or return as-is
    if isinstance(image_data, Image.Image):
        return image_data if to_pil else tvf.pil_to_tensor(image_data)

    # Already a decoded numpy array (HWC) - convert to tensor (CHW) or PIL
    if isinstance(image_data, np.ndarray) and image_data.ndim > 1:
        if to_pil:
            return Image.fromarray(image_data)
        return torch.from_numpy(image_data).permute(2, 0, 1)

    # Numpy array of bytes - convert to Python bytes
    if isinstance(image_data, np.ndarray):
        image_data = image_data.tobytes()

    # Decode bytes to tensor using torchvision (uses libjpeg-turbo backend)
    img_buffer = torch.frombuffer(image_data, dtype=torch.uint8)
    img = tv_decode_image(img_buffer, mode=ImageReadMode.RGB)

    if to_pil:
        img = tvf.to_pil_image(img)
    return img


# Supported image extensions for ImageFolder detection
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def detect_local_dataset_type(path: pathlib.Path | str) -> str:
    """Detect dataset type from local directory structure.

    Resolution order:
    1. Streaming dataset: Has index.json with LitData structure
    2. HuggingFace dataset: Has index.json with item_loader="ParquetLoader"
    3. ImageFolder dataset: Has subdirectories with image files
    4. Unknown: None of the above

    Args:
        path: Local directory path to check

    Returns:
        Dataset type string: "streaming", "huggingface", "imagefolder", or "unknown"

    Example:
        >>> detect_local_dataset_type("/path/to/imagenet/val")
        'imagefolder'
        >>> detect_local_dataset_type("/path/to/litdata/dataset")
        'streaming'
    """
    import json

    path = pathlib.Path(path)
    if not path.exists() or not path.is_dir():
        return "unknown"

    index_path = path / "index.json"

    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)

            config = index.get("config", {})
            if config.get("item_loader") == "ParquetLoader":
                return "huggingface"
            if "data_format" in config:
                return "streaming"
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for ImageFolder structure (subdirs with images)
    if is_imagefolder_structure(path):
        return "imagefolder"

    return "unknown"


def is_imagefolder_structure(path: pathlib.Path | str) -> bool:
    """Check if path has ImageFolder structure (class subdirs with images).

    An ImageFolder has the structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...

    Args:
        path: Directory path to check

    Returns:
        True if path contains subdirectories with image files.
    """
    path = pathlib.Path(path)
    if not path.is_dir():
        return False

    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return False

    # Check if subdirs contain image files (sample first 3 subdirs)
    for subdir in subdirs[:3]:
        files = list(subdir.iterdir())[:10]  # Sample first 10 files
        if any(f.suffix.lower() in _IMG_EXTENSIONS for f in files):
            return True

    return False


# Tar archive extensions for S3 ImageFolder detection
_TAR_EXTENSIONS = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}


def _is_tar_archive(path: str) -> bool:
    """Check if path looks like a tar archive."""
    path_lower = path.lower()
    for ext in _TAR_EXTENSIONS:
        if path_lower.endswith(ext):
            return True
    return False


def _is_ffcv_source(source: str | None) -> bool:
    """Check if source is an FFCV .beton/.ffcv file."""
    if source is None:
        return False
    path_part = str(source).split("?")[0]  # strip query params
    return path_part.endswith((".ffcv", ".beton"))


def _is_imagefolder_source(source: str | None) -> bool:
    """Check if source should be handled as an ImageFolder dataset.

    Returns True for:
    - S3 tar archives (s3://bucket/path/data.tar.gz)
    - Local tar archives (/path/to/data.tar.gz)
    - Local ImageFolder directories

    Returns False for:
    - Streaming datasets (s3://bucket/path/ with index.json)
    - HuggingFace datasets (hf://...)
    - None
    """
    if source is None:
        return False

    source_str = str(source)

    # HuggingFace datasets are never ImageFolders
    if source_str.startswith("hf://"):
        return False

    # Check for tar archive (S3 or local)
    if _is_tar_archive(source_str):
        return True

    # For local paths, check if it's an ImageFolder structure
    if not source_str.startswith(("s3://", "gs://", "http://", "https://")):
        path = pathlib.Path(source_str)
        if path.exists() and path.is_dir():
            # Check if it's a streaming dataset (has index.json)
            if (path / "index.json").exists():
                return False
            # Check if it's an ImageFolder
            return is_imagefolder_structure(path)

    return False


class SlipstreamDataset(torch.utils.data.Dataset):
    """High-performance streaming dataset for PyTorch vision workloads.

    SlipstreamDataset automatically detects and handles multiple dataset formats:
    - **Streaming datasets**: LitData format with index.json (s3://, local paths)
    - **HuggingFace datasets**: Via hf:// URIs
    - **ImageFolder datasets**: Local directories with class subdirectories
    - **S3 tar archives**: Auto-download, hash, and extract (.tar, .tar.gz, etc.)

    Uses composition: always returns a SlipstreamDataset wrapping an internal
    ``_reader`` that handles source-specific I/O. Processing logic (decode,
    transform, pipelines) lives here and is shared across all source types.

    Args:
        remote_dir: Remote URL (S3, GCS, HuggingFace, etc.) for the dataset.
            Examples:
                - "s3://bucket/dataset/train/" (streaming)
                - "s3://bucket/imagenet/val.tar.gz" (ImageFolder tar)
                - "hf://datasets/cifar10/data" (HuggingFace)
        cache_dir: Local directory to cache downloaded data.
            Defaults to lab shared cache on clusters, ~/.cache/slipstream locally.
        local_dir: Path to local dataset (no remote). Mutually exclusive with remote_dir.
            Auto-detects streaming vs ImageFolder format.
        input_dir: Direct LitData Dir object or URI string for power users.
            If provided, remote_dir/cache_dir/local_dir are ignored.
            Supports: s3://, gs://, http://, https://, hf://
        decode_images: If True, automatically decode image bytes to tensors/PIL.
        to_pil: If True and decode_images=True, return PIL Images instead of tensors.
        transform: Global transform applied to all image fields.
        pipelines: Dict mapping field names to transform functions.
            Cannot be used with `transform`.
        expected_version: Expected dataset version string. Raises if mismatch.
        profile: Storage profile ('wasabi' for Wasabi S3-compatible storage).
        storage_options: Custom storage options dict for S3/cloud access.
        max_cache_size: Maximum cache size (e.g., '350GB'). Default: '350GB'.
        **kwargs: Additional arguments passed to the underlying reader.

    Example:
        # Streaming dataset (LitData format)
        dataset = SlipstreamDataset(
            remote_dir="s3://visionlab-datasets/imagenet1k/.../val/",
            decode_images=True,
        )

        # S3 tar archive (auto-download and extract)
        dataset = SlipstreamDataset(
            remote_dir="s3://visionlab-datasets/imagenet1k-raw/val.tar.gz",
        )

        # Local ImageFolder (auto-detected)
        dataset = SlipstreamDataset(local_dir="/path/to/imagenet/val")

        # HuggingFace dataset
        dataset = SlipstreamDataset(
            input_dir="hf://datasets/cifar10/data",
            decode_images=True,
        )
    """

    def __init__(
        self,
        # Intuitive API
        remote_dir: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        # Power user API (pass-through to LitData)
        input_dir: Dir | str | None = None,
        # Dataset behavior
        decode_images: bool = False,
        to_pil: bool = True,
        transform: Callable | None = None,
        pipelines: Mapping[str, Callable] | None = None,
        # Validation and storage
        expected_version: str | None = None,
        profile: str | None = None,
        storage_options: dict[str, Any] | None = None,
        max_cache_size: str = "350GB",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Validate mutually exclusive options
        if pipelines is not None and transform is not None:
            raise ValueError(
                "Cannot specify both 'pipelines' and 'transform'. "
                "Use 'transform' for simple image transforms, or "
                "'pipelines' for per-field transforms."
            )

        # Store processing options
        self.pipelines = dict(pipelines) if pipelines is not None else None
        self.decode_images = decode_images
        self.to_pil = to_pil
        self._transform = transform

        # Create the source-specific reader
        self._reader = self._create_reader(
            remote_dir=remote_dir,
            cache_dir=cache_dir,
            local_dir=local_dir,
            input_dir=input_dir,
            expected_version=expected_version,
            profile=profile,
            storage_options=storage_options,
            max_cache_size=max_cache_size,
            **kwargs,
        )

    def _create_reader(
        self,
        remote_dir: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        input_dir: Dir | str | None = None,
        expected_version: str | None = None,
        profile: str | None = None,
        storage_options: dict[str, Any] | None = None,
        max_cache_size: str = "350GB",
        **kwargs: Any,
    ) -> Any:
        """Create the appropriate reader based on data source.

        Returns:
            A reader instance (StreamingReader or SlipstreamImageFolder).
        """
        # Determine the source to check
        source = input_dir or remote_dir or local_dir

        # Check for FFCV .beton/.ffcv files
        if source is not None and _is_ffcv_source(str(source)):
            from slipstream.readers.ffcv import FFCVFileReader

            verbose = kwargs.pop("verbose", True)
            return FFCVFileReader(
                ffcv_path=str(source),
                cache_dir=cache_dir,
                verbose=verbose,
            )

        # Check if this should be an ImageFolder dataset
        if source is not None and _is_imagefolder_source(str(source)):
            from slipstream.readers.imagefolder import open_imagefolder

            # Build s3_config from storage_options if present
            s3_config = None
            if storage_options:
                s3_config = {
                    "endpoint_url": storage_options.get("S3_ENDPOINT_URL"),
                }

            # Resolve cache_dir for imagefolder
            if cache_dir is None:
                cache_dir = get_default_cache_dir() / "slipstream" / "imagefolder"

            verbose = kwargs.pop("verbose", True)

            return open_imagefolder(
                source=str(source),
                cache_dir=cache_dir,
                s3_config=s3_config,
                verbose=verbose,
            )

        # Default: streaming reader (LitData)
        from slipstream.readers.streaming import StreamingReader

        return StreamingReader(
            remote_dir=remote_dir,
            cache_dir=cache_dir,
            local_dir=local_dir,
            input_dir=input_dir,
            profile=profile,
            storage_options=storage_options,
            max_cache_size=max_cache_size,
            expected_version=expected_version,
            **kwargs,
        )

    # ---- Delegated properties ----

    @property
    def field_types(self) -> dict[str | int, str | type]:
        """Field type mapping from the underlying reader."""
        return self._reader.field_types

    @property
    def image_fields(self) -> list[str | int]:
        """List of image field names from the underlying reader."""
        return self._reader.image_fields

    @property
    def cache_path(self) -> pathlib.Path | None:
        """Get the local cache directory path."""
        return self._reader.cache_path

    def __len__(self) -> int:
        return len(self._reader)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying reader for reader-specific attrs.

        This handles attributes like: classes, samples, input_dir, version,
        remote_dir, on_demand_bytes, cache, _dataset, etc.
        """
        # Avoid infinite recursion during init (before _reader is set)
        if name == "_reader":
            raise AttributeError(name)
        try:
            return getattr(self._reader, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    # ---- Processing logic (source-agnostic) ----

    def __getitem__(self, idx: int) -> dict[str, Any] | tuple[Any, ...]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Sample dict or tuple with optional decoding and transforms applied.
        """
        # Reader handles source-specific normalization (e.g. HF image dicts)
        sample = self._reader[idx]

        # Decode images if requested
        if self.decode_images:
            sample = self._decode_images(sample)

        # Apply global transform to image fields
        if self._transform is not None:
            sample = self._apply_transform(sample)

        # Apply per-field pipelines
        if self.pipelines is not None:
            sample = self._apply_pipelines(sample)

        return sample

    def _decode_images(self, sample: dict | tuple) -> dict | tuple:
        """Decode image bytes in sample."""
        if hasattr(sample, "items"):
            for key in self.image_fields:
                if key in sample:
                    sample[key] = decode_image(sample[key], to_pil=self.to_pil)
        else:
            sample = list(sample)
            for idx in self.image_fields:
                sample[idx] = decode_image(sample[idx], to_pil=self.to_pil)
            sample = tuple(sample)
        return sample

    def _apply_transform(self, sample: dict | tuple) -> dict | tuple:
        """Apply global transform to image fields."""
        if hasattr(sample, "items"):
            for key in self.image_fields:
                if key in sample:
                    sample[key] = self._transform(sample[key])
        else:
            sample = list(sample)
            for idx in self.image_fields:
                sample[idx] = self._transform(sample[idx])
            sample = tuple(sample)
        return sample

    def _apply_pipelines(self, sample: dict | tuple) -> dict | tuple:
        """Apply per-field pipeline transforms."""
        if hasattr(sample, "items"):
            for key, transform in self.pipelines.items():
                if key in sample:
                    sample[key] = transform(sample[key])
        else:
            sample = list(sample)
            for key, transform in self.pipelines.items():
                if isinstance(key, int) and key < len(sample):
                    sample[key] = transform(sample[key])
            sample = tuple(sample)
        return sample

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        indent = "    "
        lines = ["SlipstreamDataset("]

        # Basic info
        lines.append(f"{indent}num_samples={len(self)},")

        # Paths - delegate to reader for source-specific info
        input_dir = getattr(self._reader, 'input_dir', None)
        if input_dir is not None:
            if hasattr(input_dir, "path") and hasattr(input_dir, "url"):
                lines.append(f"{indent}cache_dir='{input_dir.path}',")
                if input_dir.url:
                    lines.append(f"{indent}remote_dir='{input_dir.url}',")
            elif isinstance(input_dir, str):
                if input_dir.startswith(("s3://", "gs://", "http://", "https://", "hf://")):
                    lines.append(f"{indent}remote_dir='{input_dir}',")
                    lines.append(f"{indent}cache_dir='~/.lightning/ (LitData default)',")
                else:
                    lines.append(f"{indent}local_dir='{input_dir}',")
        elif hasattr(self._reader, '_remote_path'):
            # FFCV reader
            lines.append(f"{indent}ffcv_path='{self._reader._remote_path}',")
        else:
            # ImageFolder reader
            root = getattr(self._reader, '_root_path', None)
            if root is not None:
                lines.append(f"{indent}local_dir='{root}',")

        version = getattr(self._reader, 'version', None)
        if version:
            lines.append(f"{indent}version='{version}',")

        # Field info
        lines.append(f"{indent}fields={{")
        for key, field_type in self.field_types.items():
            type_str = field_type if isinstance(field_type, str) else field_type.__name__
            lines.append(f"{indent}{indent}'{key}': {type_str},")
        lines.append(f"{indent}}},")

        # Reader type
        reader_type = type(self._reader).__name__
        lines.append(f"{indent}reader={reader_type},")

        # Options
        lines.append(f"{indent}decode_images={self.decode_images},")
        if self.decode_images:
            lines.append(f"{indent}to_pil={self.to_pil},")

        # Pipelines
        if self.pipelines:
            lines.append(f"{indent}pipelines={{")
            for key, pipeline in self.pipelines.items():
                pipeline_repr = repr(pipeline)
                if "\n" in pipeline_repr:
                    pipeline_lines = pipeline_repr.split("\n")
                    pipeline_repr = pipeline_lines[0] + "\n" + "\n".join(
                        f"{indent}{indent}    {line}" for line in pipeline_lines[1:]
                    )
                lines.append(f"{indent}{indent}'{key}': {pipeline_repr},")
            lines.append(f"{indent}}},")

        # Transform
        if self._transform is not None:
            transform_repr = repr(self._transform).split("\n")[0]
            lines.append(f"{indent}transform={transform_repr},")

        lines.append(")")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        return f"<pre>{self.__repr__()}</pre>"
