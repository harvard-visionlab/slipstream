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
from litdata import StreamingDataset as LitDataStreamingDataset
from litdata.streaming.resolver import Dir
from litdata.utilities.dataset_utilities import _read_updated_at
from PIL import Image
from torchvision.io import ImageReadMode
from torchvision.io import decode_image as tv_decode_image
from torchvision.transforms import functional as tvf

if TYPE_CHECKING:
    pass

__all__ = [
    "SlipstreamDataset",
    "decode_image",
    "is_image_bytes",
    "is_hf_image_dict",
    "extract_hf_image_bytes",
    "ensure_lightning_symlink_on_cluster",
    "get_default_cache_dir",
    "list_collate_fn",
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


class SlipstreamDataset(LitDataStreamingDataset):
    """High-performance streaming dataset for PyTorch vision workloads.

    SlipstreamDataset wraps LitData's StreamingDataset with:
    - Intuitive API: Use `remote_dir` and `cache_dir` instead of `Dir(...)`
    - Automatic field type detection: Identifies image fields automatically
    - Pipeline support: Per-field transforms for flexible data processing
    - Cluster integration: Auto-setup for visionlab SLURM clusters
    - HuggingFace support: Stream datasets directly via hf:// URIs

    Args:
        remote_dir: Remote URL (S3, GCS, HuggingFace, etc.) for the dataset.
            Examples:
                - "s3://bucket/dataset/train/"
                - "hf://datasets/cifar10/data"
        cache_dir: Local directory to cache downloaded data.
            Defaults to lab shared cache on clusters, ~/.cache/slipstream locally.
        local_dir: Path to local dataset (no remote). Mutually exclusive with remote_dir.
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
        **kwargs: Additional arguments passed to LitData StreamingDataset.

    Example:
        # Simple usage with automatic decoding
        dataset = SlipstreamDataset(
            remote_dir="s3://visionlab-datasets/imagenet1k/.../val/",
            decode_images=True,
            to_pil=True,
        )
        sample = dataset[0]
        pil_image = sample['image']  # PIL Image

        # HuggingFace dataset (via input_dir)
        dataset = SlipstreamDataset(
            input_dir="hf://datasets/cifar10/data",
            decode_images=True,
        )

        # Training usage with custom pipeline
        dataset = SlipstreamDataset(
            remote_dir="s3://visionlab-datasets/imagenet1k/.../train/",
            decode_images=False,
            pipelines={"image": my_decoder_pipeline},
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
        # Ensure cluster symlinks are set up
        ensure_lightning_symlink_on_cluster()

        # Validate mutually exclusive options
        if pipelines is not None and transform is not None:
            raise ValueError(
                "Cannot specify both 'pipelines' and 'transform'. "
                "Use 'transform' for simple image transforms, or "
                "'pipelines' for per-field transforms."
            )

        # Store our options
        self.pipelines = dict(pipelines) if pipelines is not None else None
        self.decode_images = decode_images
        self.to_pil = to_pil

        # Resolve input_dir from intuitive API if not provided directly
        if input_dir is None:
            input_dir = self._resolve_input_dir(remote_dir, cache_dir, local_dir)

        # Set up storage options
        storage_options = self._resolve_storage_options(storage_options, profile)

        # Initialize parent class
        super().__init__(
            input_dir=input_dir,
            transform=transform,
            storage_options=storage_options,
            max_cache_size=max_cache_size,
            **kwargs,
        )

        # Store transform for later use (parent may modify it)
        self._transform = transform

        # Get dataset version
        self.version = (
            _read_updated_at(self.input_dir) if self.input_dir is not None else None
        )

        # Validate version if expected
        if expected_version is not None and self.version != str(expected_version):
            raise ValueError(
                f"Dataset version mismatch: expected '{expected_version}', "
                f"got '{self.version}'"
            )

        # Detect field types from first sample
        self._set_field_types()

    @property
    def cache_path(self) -> pathlib.Path | None:
        """Get the local cache directory path.

        Note: This is named cache_path (not cache_dir) to avoid conflicting
        with the parent class's cache_dir attribute.

        Returns:
            Path to the local cache directory, or None if using remote-only mode.
        """
        # First check if parent class set cache_dir (from LitData's internal resolution)
        parent_cache = getattr(super(), 'cache_dir', None)
        if parent_cache is not None:
            if hasattr(parent_cache, 'path'):
                return pathlib.Path(parent_cache.path)
            if isinstance(parent_cache, str):
                return pathlib.Path(parent_cache)

        if self.input_dir is None:
            return None

        # Dir object with explicit cache
        if hasattr(self.input_dir, "path") and self.input_dir.path:
            return pathlib.Path(self.input_dir.path)

        # String path (local directory)
        if isinstance(self.input_dir, str):
            # Check if it's a local path vs remote URL
            if not self.input_dir.startswith(("s3://", "gs://", "http://", "https://", "hf://")):
                return pathlib.Path(self.input_dir)

        # Remote URL without explicit cache - LitData uses default cache location
        # We can try to find it, but it's complex (hash-based subdirectories)
        return None

    @property
    def remote_dir(self) -> str | None:
        """Get the remote URL if using remote storage.

        Returns:
            Remote URL string, or None if using local-only mode.
        """
        if self.input_dir is None:
            return None

        # Dir object with URL
        if hasattr(self.input_dir, "url") and self.input_dir.url:
            return self.input_dir.url

        # String that looks like a URL (including HuggingFace hf://)
        if isinstance(self.input_dir, str):
            if self.input_dir.startswith(("s3://", "gs://", "http://", "https://", "hf://")):
                return self.input_dir

        return None

    def _resolve_input_dir(
        self,
        remote_dir: str | None,
        cache_dir: str | pathlib.Path | None,
        local_dir: str | pathlib.Path | None,
    ) -> Dir | str:
        """Resolve input_dir from intuitive API parameters.

        Args:
            remote_dir: Remote URL for the dataset (s3://, gs://, hf://, etc.).
            cache_dir: Local cache directory.
            local_dir: Local-only dataset path.

        Returns:
            Dir object or string path for LitData.

        Raises:
            ValueError: If parameters are invalid or conflicting.
        """
        # Case 1: Local-only dataset
        if local_dir is not None:
            if remote_dir is not None:
                raise ValueError(
                    "Cannot specify both 'local_dir' and 'remote_dir'. "
                    "Use 'local_dir' for local datasets, or "
                    "'remote_dir' (with optional 'cache_dir') for remote datasets."
                )
            return str(local_dir)

        # Case 2: Remote dataset with optional cache
        if remote_dir is not None:
            # If cache_dir specified, use Dir(path=cache_dir, url=remote_dir)
            # Otherwise, just pass the URL and let LitData use its default caching
            # (under ~/.lightning/litdata/, organized by URL hash)
            if cache_dir is not None:
                return Dir(path=str(cache_dir), url=remote_dir)
            return remote_dir  # LitData handles caching automatically

        # Case 3: Neither specified - error
        raise ValueError(
            "Must specify one of: 'remote_dir', 'local_dir', or 'input_dir'. "
            "Example: SlipstreamDataset(remote_dir='s3://bucket/dataset/')\n"
            "         SlipstreamDataset(input_dir='hf://datasets/user/dataset/data')"
        )

    def _resolve_storage_options(
        self,
        storage_options: dict[str, Any] | None,
        profile: str | None,
    ) -> dict[str, Any]:
        """Resolve storage options based on profile.

        Args:
            storage_options: User-provided storage options.
            profile: Storage profile name ('wasabi', etc.).

        Returns:
            Resolved storage options dict.
        """
        if storage_options is None:
            storage_options = {}

        # Apply profile defaults if no custom options provided
        if profile == "wasabi" and not storage_options:
            storage_options = {
                "AWS_NO_SIGN_REQUEST": "yes",
                "S3_ENDPOINT_URL": "https://s3.wasabisys.com",
            }

        return storage_options

    def _set_field_types(self) -> None:
        """Detect field types by inspecting the first sample.

        Sets:
            self.field_types: Dict mapping field names to type info.
            self.image_fields: List of field names containing image bytes.

        Handles:
            - Raw bytes (JPEG, PNG, etc.)
            - HuggingFace image dicts: {'bytes': ..., 'path': ...}
        """
        # Get raw sample without transforms
        raw_sample = super().__getitem__(0)

        self.image_fields: list[str | int] = []
        self.field_types: dict[str | int, str | type] = {}

        if hasattr(raw_sample, "items"):
            # Dict-like sample
            for key, value in raw_sample.items():
                if isinstance(value, bytes):
                    if is_image_bytes(value):
                        self.field_types[key] = "ImageBytes"
                        self.image_fields.append(key)
                    else:
                        self.field_types[key] = bytes
                elif isinstance(value, dict) and is_hf_image_dict(value):
                    # HuggingFace image dict format
                    if is_image_bytes(value):
                        self.field_types[key] = "HFImageDict"
                        self.image_fields.append(key)
                    else:
                        self.field_types[key] = dict
                else:
                    self.field_types[key] = type(value)
        else:
            # Tuple-like sample
            for idx, value in enumerate(raw_sample):
                if isinstance(value, bytes):
                    if is_image_bytes(value):
                        self.field_types[idx] = "ImageBytes"
                        self.image_fields.append(idx)
                    else:
                        self.field_types[idx] = bytes
                elif isinstance(value, dict) and is_hf_image_dict(value):
                    # HuggingFace image dict format
                    if is_image_bytes(value):
                        self.field_types[idx] = "HFImageDict"
                        self.image_fields.append(idx)
                    else:
                        self.field_types[idx] = dict
                else:
                    self.field_types[idx] = type(value)

    def __getitem__(self, idx: int) -> dict[str, Any] | tuple[Any, ...]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Sample dict or tuple with optional decoding and transforms applied.
        """
        sample = super().__getitem__(idx)

        # Normalize HF image dicts to raw bytes (for downstream compatibility)
        sample = self._normalize_hf_images(sample)

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

    def _normalize_hf_images(self, sample: dict | tuple) -> dict | tuple:
        """Convert HuggingFace image dicts to raw bytes.

        This ensures downstream code (loaders, decoders) only sees raw bytes,
        not HF-specific dict format.
        """
        if hasattr(sample, "items"):
            for key in self.image_fields:
                if key in sample:
                    value = sample[key]
                    if isinstance(value, dict) and is_hf_image_dict(value):
                        sample[key] = extract_hf_image_bytes(value)
        else:
            sample = list(sample)
            for idx in self.image_fields:
                value = sample[idx]
                if isinstance(value, dict) and is_hf_image_dict(value):
                    sample[idx] = extract_hf_image_bytes(value)
            sample = tuple(sample)
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

        # Paths
        if self.input_dir is not None:
            if hasattr(self.input_dir, "path") and hasattr(self.input_dir, "url"):
                # Dir object with explicit cache
                lines.append(f"{indent}cache_dir='{self.input_dir.path}',")
                if self.input_dir.url:
                    lines.append(f"{indent}remote_dir='{self.input_dir.url}',")
            elif isinstance(self.input_dir, str):
                # String: could be local path or remote URL
                if self.input_dir.startswith(("s3://", "gs://", "http://", "https://", "hf://")):
                    lines.append(f"{indent}remote_dir='{self.input_dir}',")
                    lines.append(f"{indent}cache_dir='~/.lightning/ (LitData default)',")
                else:
                    lines.append(f"{indent}local_dir='{self.input_dir}',")

        if self.version:
            lines.append(f"{indent}version='{self.version}',")

        # Field info
        lines.append(f"{indent}fields={{")
        for key, field_type in self.field_types.items():
            type_str = field_type if isinstance(field_type, str) else field_type.__name__
            lines.append(f"{indent}{indent}'{key}': {type_str},")
        lines.append(f"{indent}}},")

        # Options
        lines.append(f"{indent}decode_images={self.decode_images},")
        if self.decode_images:
            lines.append(f"{indent}to_pil={self.to_pil},")

        # Pipelines
        if self.pipelines:
            lines.append(f"{indent}pipelines={{")
            for key, pipeline in self.pipelines.items():
                # Indent multi-line repr properly
                pipeline_repr = repr(pipeline)
                if "\n" in pipeline_repr:
                    # Indent continuation lines
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
