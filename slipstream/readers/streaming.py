"""Streaming reader backed by LitData's StreamingDataset.

Handles LitData format datasets and HuggingFace datasets (via hf:// URIs).
This reader encapsulates all LitData-specific logic: initialization, field
type detection, HuggingFace image normalization, cache path resolution, etc.

The reader is used internally by SlipstreamDataset and should not normally
be instantiated directly.
"""

from __future__ import annotations

import io
import os
import pathlib
from typing import Any

import numpy as np
from litdata import StreamingDataset as LitDataStreamingDataset
from litdata.streaming.resolver import Dir
from litdata.utilities.dataset_utilities import _read_updated_at
from PIL import Image

from slipstream.dataset import (
    ensure_lightning_symlink_on_cluster,
    extract_hf_image_bytes,
    is_hf_image_dict,
    is_image_bytes,
)

__all__ = ["StreamingReader"]


class StreamingReader(LitDataStreamingDataset):
    """LitData-backed reader for streaming and HuggingFace datasets.

    Wraps LitData's StreamingDataset with field type detection and
    HuggingFace image normalization. Returns raw samples (no decode/
    transform/pipeline processing — that is handled by SlipstreamDataset).

    Args:
        remote_dir: Remote URL (S3, GCS, HuggingFace, etc.).
        cache_dir: Local cache directory.
        local_dir: Local-only dataset path.
        input_dir: Direct LitData Dir object or URI string.
        profile: Storage profile name ('wasabi', etc.).
        storage_options: Custom storage options for S3/cloud.
        max_cache_size: Maximum cache size (e.g., '350GB').
        expected_version: Expected dataset version string.
        **kwargs: Additional arguments passed to LitData StreamingDataset.
    """

    def __init__(
        self,
        remote_dir: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        input_dir: Dir | str | None = None,
        profile: str | None = None,
        storage_options: dict[str, Any] | None = None,
        max_cache_size: str = "350GB",
        expected_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Ensure cluster symlinks are set up
        ensure_lightning_symlink_on_cluster()

        # Resolve input_dir from intuitive API if not provided directly
        if input_dir is None:
            input_dir = self._resolve_input_dir(remote_dir, cache_dir, local_dir)

        # Set up storage options
        storage_options = self._resolve_storage_options(storage_options, profile)

        # Initialize LitData parent class
        super().__init__(
            input_dir=input_dir,
            storage_options=storage_options,
            max_cache_size=max_cache_size,
            **kwargs,
        )

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
    def litdata_cache_path(self) -> pathlib.Path | None:
        """Get the LitData cache directory path (where chunks are stored)."""
        parent_cache = getattr(super(), 'cache_dir', None)
        if parent_cache is not None:
            if hasattr(parent_cache, 'path'):
                return pathlib.Path(parent_cache.path)
            if isinstance(parent_cache, str):
                return pathlib.Path(parent_cache)

        if self.input_dir is None:
            return None

        if hasattr(self.input_dir, "path") and self.input_dir.path:
            return pathlib.Path(self.input_dir.path)

        if isinstance(self.input_dir, str):
            if not self.input_dir.startswith(("s3://", "gs://", "http://", "https://", "hf://")):
                return pathlib.Path(self.input_dir)

        return None

    @property
    def dataset_hash(self) -> str | None:
        """Get a unique hash for this dataset version.

        LitData stores data at ~/.lightning/chunks/{hash}/{timestamp}.
        This hash uniquely identifies the dataset content/configuration.
        Returns the first 8 characters of the hash for brevity.
        """
        cache_path = self.litdata_cache_path
        if cache_path is None:
            return None

        try:
            parts = cache_path.parts
            if 'chunks' in parts:
                chunks_idx = parts.index('chunks')
                if chunks_idx + 1 < len(parts):
                    return parts[chunks_idx + 1][:8]
        except Exception:
            pass

        # Fallback: hash the remote_dir if available
        if self.remote_dir:
            import hashlib
            return hashlib.md5(self.remote_dir.encode()).hexdigest()[:8]

        return None

    @property
    def cache_path(self) -> pathlib.Path | None:
        """Get the default SlipCache directory path.

        Returns a versioned path that includes the dataset hash to prevent
        stale cache issues when the source dataset changes.

        Path format: {litdata_cache_path}/slipcache-{hash[:8]}/
        """
        base_path = self.litdata_cache_path
        if base_path is None:
            return None

        dataset_hash = self.dataset_hash
        if dataset_hash:
            return base_path / f"slipcache-{dataset_hash}"

        return base_path / "slipcache"

    @property
    def remote_dir(self) -> str | None:
        """Get the remote URL if using remote storage."""
        if self.input_dir is None:
            return None

        if hasattr(self.input_dir, "url") and self.input_dir.url:
            return self.input_dir.url

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
        """Resolve input_dir from intuitive API parameters."""
        if local_dir is not None:
            if remote_dir is not None:
                raise ValueError(
                    "Cannot specify both 'local_dir' and 'remote_dir'. "
                    "Use 'local_dir' for local datasets, or "
                    "'remote_dir' (with optional 'cache_dir') for remote datasets."
                )
            return str(local_dir)

        if remote_dir is not None:
            if cache_dir is not None:
                return Dir(path=str(cache_dir), url=remote_dir)
            return remote_dir

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
        """Resolve storage options based on profile."""
        if storage_options is None:
            storage_options = {}

        if profile == "wasabi" and not storage_options:
            storage_options = {
                "AWS_NO_SIGN_REQUEST": "yes",
                "S3_ENDPOINT_URL": "https://s3.wasabisys.com",
            }

        return storage_options

    def _set_field_types(self) -> None:
        """Detect field types by inspecting the first sample."""
        raw_sample = super().__getitem__(0)

        self.image_fields: list[str | int] = []
        self.field_types: dict[str | int, str | type] = {}

        if hasattr(raw_sample, "items"):
            for key, value in raw_sample.items():
                if isinstance(value, bytes):
                    if is_image_bytes(value):
                        self.field_types[key] = "ImageBytes"
                        self.image_fields.append(key)
                    else:
                        self.field_types[key] = bytes
                elif isinstance(value, dict) and is_hf_image_dict(value):
                    if is_image_bytes(value):
                        self.field_types[key] = "HFImageDict"
                        self.image_fields.append(key)
                    else:
                        self.field_types[key] = dict
                else:
                    self.field_types[key] = type(value)
        else:
            for idx, value in enumerate(raw_sample):
                if isinstance(value, bytes):
                    if is_image_bytes(value):
                        self.field_types[idx] = "ImageBytes"
                        self.image_fields.append(idx)
                    else:
                        self.field_types[idx] = bytes
                elif isinstance(value, dict) and is_hf_image_dict(value):
                    if is_image_bytes(value):
                        self.field_types[idx] = "HFImageDict"
                        self.image_fields.append(idx)
                    else:
                        self.field_types[idx] = dict
                else:
                    self.field_types[idx] = type(value)

    def __getitem__(self, idx: int) -> dict[str, Any] | tuple[Any, ...]:
        """Get a raw sample with HF image normalization applied."""
        sample = super().__getitem__(idx)
        sample = self._normalize_hf_images(sample)
        return sample

    def _normalize_hf_images(self, sample: dict | tuple) -> dict | tuple:
        """Convert HuggingFace image dicts to raw bytes."""
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
