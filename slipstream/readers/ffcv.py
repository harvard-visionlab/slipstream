"""Reader for FFCV .beton/.ffcv files.

Provides FFCVFileReader, which reads .ffcv files and exposes the protocol
needed by SlipstreamLoader and OptimizedCache.build():
- cache_path, field_types, __len__, __getitem__
- read_all_fields() fast path for bulk cache building

Usage:
    from slipstream.readers import FFCVFileReader

    reader = FFCVFileReader("/path/to/file.ffcv")
    print(reader.field_types)   # {'image': 'ImageBytes', 'label': 'int'}
    print(len(reader))          # 50000
    print(reader[0])            # {'image': b'\\xff\\xd8...', 'label': 42}

    # Use with SlipstreamLoader (builds OptimizedCache automatically)
    from slipstream import SlipstreamLoader
    loader = SlipstreamLoader(reader, batch_size=256, pipelines={...})
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from slipstream.backends.ffcv_file import (
    FFCV_ALLOC_ENTRY_DTYPE,
    FFCV_FIELD_DESC_DTYPE,
    FFCV_HEADER_DTYPE,
    FFCV_TYPE_BYTES,
    FFCV_TYPE_FLOAT,
    FFCV_TYPE_INT,
    FFCV_TYPE_JSON,
    FFCV_TYPE_RGB_IMAGE,
    FFCV_VARIABLE_TYPES,
)


# Map FFCV type IDs to slipstream field type strings
FFCV_TYPE_TO_FIELD_TYPE = {
    FFCV_TYPE_RGB_IMAGE: "ImageBytes",
    FFCV_TYPE_INT: "int",
    FFCV_TYPE_FLOAT: "float",
    FFCV_TYPE_BYTES: "bytes",   # default, overridden by auto-detect
    FFCV_TYPE_JSON: "str",      # JSON is always text
}


def _get_field_metadata_dtype(type_id: int) -> np.dtype:
    """Get the numpy dtype for a field's per-sample metadata.

    Must match FFCV's field implementations exactly.
    See: ffcv/fields/*.py -> metadata_type property
    """
    if type_id == FFCV_TYPE_RGB_IMAGE:
        # ffcv/fields/rgb_image.py RGBImageField.metadata_type
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])
    elif type_id == FFCV_TYPE_INT:
        # ffcv/fields/basics.py IntField.metadata_type
        return np.dtype('<i8')
    elif type_id == FFCV_TYPE_FLOAT:
        # ffcv/fields/basics.py FloatField.metadata_type
        return np.dtype('<f8')
    elif type_id == FFCV_TYPE_BYTES:
        # ffcv/fields/bytes.py BytesField.metadata_type
        return np.dtype([('ptr', '<u8'), ('size', '<u8')])
    elif type_id == FFCV_TYPE_JSON:
        # ffcv/fields/json.py JSONField.metadata_type
        return np.dtype([('ptr', '<u8'), ('size', '<u8')])
    elif type_id == 4:  # NDArrayField
        # ffcv/fields/ndarray.py NDArrayField.metadata_type
        return np.dtype('<u8')
    else:
        return np.dtype('<u8')


def _is_jupyter() -> bool:
    """Detect if running in a Jupyter notebook or Google Colab."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "Shell":
            return True   # Google Colab
        return False
    except (ImportError, NameError, AttributeError):
        return False


def _download_s5cmd(
    remote_path: str,
    local_path: Path,
    *,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    expected_size: int | None = None,
    verbose: bool = True,
) -> bool:
    """Download a file using s5cmd with environment-aware progress.

    Terminal: lets s5cmd print its native --show-progress bar.
    Jupyter: monitors output file size with tqdm (s5cmd suppresses
    progress on non-TTY stderr).

    Returns True if successful, False if s5cmd is not available.
    """
    import shutil
    import subprocess

    if shutil.which("s5cmd") is None:
        if verbose:
            print("  s5cmd not found, falling back to fsspec")
        return False

    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += ["--numworkers", str(numworkers), "cp"]

    if verbose:
        print(f"  Using s5cmd ({numworkers} workers)")

    if _is_jupyter():
        full_cmd = cmd + [remote_path, str(local_path)]
        return _s5cmd_jupyter_monitored(full_cmd, local_path, expected_size, verbose)

    # Terminal: let s5cmd handle its own progress natively
    full_cmd = cmd + ["--show-progress", remote_path, str(local_path)]
    try:
        subprocess.run(full_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"  s5cmd failed: {e}")
        return False


def _s5cmd_jupyter_monitored(
    cmd: list[str],
    local_path: Path,
    expected_size: int | None,
    verbose: bool,
) -> bool:
    """Run s5cmd and update a tqdm bar by watching the local file size."""
    import subprocess
    import time

    from tqdm.auto import tqdm

    local_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    desc = f"  Downloading {local_path.name}"
    with tqdm(total=expected_size, unit="B", unit_scale=True,
              desc=desc, disable=not verbose) as pbar:
        while process.poll() is None:
            if local_path.exists():
                pbar.n = local_path.stat().st_size
                pbar.refresh()
            time.sleep(0.5)

        # Final update
        if local_path.exists():
            pbar.n = local_path.stat().st_size
            pbar.refresh()

    if process.returncode != 0:
        error_msg = process.stderr.read().decode()
        if verbose:
            print(f"  s5cmd error: {error_msg}")
        return False
    return True


def _download_fsspec(
    remote_path: str,
    local_path: Path,
    *,
    file_size: int | None = None,
    verbose: bool = True,
) -> None:
    """Download a file using fsspec (streaming, with tqdm progress bar)."""
    import fsspec

    fs, resolved_path = fsspec.core.url_to_fs(remote_path)

    if file_size is None:
        file_info = fs.info(resolved_path)
        file_size = file_info.get("size", 0) or 0
    if verbose:
        print("  Using fsspec")

    if verbose and file_size:
        from tqdm import tqdm

        with (
            fs.open(resolved_path, "rb") as remote_f,
            open(local_path, "wb") as local_f,
            tqdm(total=file_size, unit="B", unit_scale=True,
                 desc="  Downloading") as pbar,
        ):
            chunk_size = 64 * 1024 * 1024  # 64 MB
            while True:
                chunk = remote_f.read(chunk_size)
                if not chunk:
                    break
                local_f.write(chunk)
                pbar.update(len(chunk))
    else:
        fs.get(resolved_path, str(local_path))


def _find_image_end(data: bytes | np.ndarray, max_len: int) -> int:
    """Find actual image end, trimming FFCV page-alignment padding.

    FFCV rounds allocations up to page_size boundaries, so image bytes
    include garbage padding after the actual data. For JPEG, we find the
    FFD9 end-of-image marker. For other formats, return max_len.
    """
    if isinstance(data, np.ndarray):
        data = bytes(data[:max_len])
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        # JPEG: find FFD9 end-of-image marker
        eoi = data.find(b'\xff\xd9')
        if eoi != -1:
            return eoi + 2
    return max_len


class FFCVFileReader:
    """Reader for FFCV .beton/.ffcv files.

    Satisfies the dataset protocol needed by SlipstreamLoader:
    - cache_path: where to store the optimized cache
    - field_types: dict of field name -> type string
    - __len__: number of samples
    - __getitem__: per-sample access (returns dict)
    - read_all_fields(): bulk fast path for cache building

    Args:
        ffcv_path: Path to .ffcv or .beton file
        cache_dir: Where to store the optimized cache. Defaults to
            a directory next to the .ffcv file.
        image_field: Name to use for the image field (default "image")
        label_field: Name to use for the label/int field (default "label")
        verbose: Print loading information

    Example:
        reader = FFCVFileReader("imagenet_val.ffcv")
        sample = reader[0]  # {'image': b'...', 'label': 0}
    """

    def __init__(
        self,
        ffcv_path: str | Path,
        cache_dir: str | Path | None = None,
        image_field: str = "image",
        label_field: str = "label",
        verbose: bool = True,
    ) -> None:
        self._remote_path = str(ffcv_path)
        self._image_field = image_field
        self._label_field = label_field
        self._verbose = verbose

        # Handle remote paths (s3://, gs://, etc.) by downloading locally
        self._path = self._resolve_path(str(ffcv_path), verbose=verbose)

        if not self._path.exists():
            raise FileNotFoundError(f"FFCV file not found: {self._path}")

        # Parse file structure
        self._read_header()
        self._read_field_descriptors()
        self._read_compound_metadata()
        self._read_alloc_table()

        # Memory-map the file
        self._mmap = np.memmap(str(self._path), dtype=np.uint8, mode='r')

        # Extract contiguous arrays for image field
        self._alloc_ptr = np.ascontiguousarray(self._alloc_table['ptr'])
        self._alloc_size = np.ascontiguousarray(self._alloc_table['size'])

        self.num_samples = int(self._header['num_samples'])

        # Set up base cache_path
        if cache_dir is not None:
            self._base_cache_path = Path(cache_dir)
        else:
            # Store optimized cache next to the local .ffcv file
            self._base_cache_path = self._path.parent / f"{self._path.stem}-slipstream"

        # Build field_types from parsed descriptors
        self._build_field_types()

        if verbose:
            print(f"FFCVFileReader: {self._path.name}")
            print(f"  Samples: {self.num_samples:,}")
            print(f"  Fields: {self.field_types}")
            print(f"  Cache: {self.cache_path}")

    @staticmethod
    def _resolve_path(ffcv_path: str, verbose: bool = True) -> Path:
        """Resolve a local or remote path to a local file.

        For remote paths (s3://, gs://, etc.), downloads the file to the
        slipstream cache directory using fsspec. Skips download if the
        file already exists locally.

        Returns:
            Path to the local .ffcv file.
        """
        # Check if this is a remote path
        if "://" not in ffcv_path or ffcv_path.startswith("file://"):
            # Local path
            local = ffcv_path.removeprefix("file://") if ffcv_path.startswith("file://") else ffcv_path
            return Path(local)

        # Remote path — download to cache dir
        from slipstream.dataset import get_default_cache_dir

        # Deterministic local path based on remote URL
        url_hash = hashlib.sha256(ffcv_path.encode()).hexdigest()[:12]
        filename = ffcv_path.rsplit("/", 1)[-1]
        local_dir = get_default_cache_dir() / "ffcv" / f"{Path(filename).stem}-{url_hash}"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        if local_path.exists():
            if verbose:
                size_gb = local_path.stat().st_size / 1e9
                print(f"Using cached FFCV file: {local_path} ({size_gb:.2f} GB)")
            return local_path

        # Get file size for progress reporting
        import fsspec
        fs, resolved_path = fsspec.core.url_to_fs(ffcv_path)
        file_info = fs.info(resolved_path)
        file_size = file_info.get("size", 0)

        if verbose:
            print(f"Downloading FFCV file: {ffcv_path}")
            print(f"  → {local_path}")
            if file_size:
                print(f"  Size: {file_size / 1e9:.2f} GB")

        # Download to a temp file first, then rename (atomic)
        tmp_path = local_path.with_suffix(".tmp")
        try:
            downloaded = _download_s5cmd(
                ffcv_path, tmp_path,
                expected_size=file_size or None,
                verbose=verbose,
            )
            if not downloaded:
                _download_fsspec(
                    ffcv_path, tmp_path,
                    file_size=file_size or None,
                    verbose=verbose,
                )
            os.rename(tmp_path, local_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        if verbose:
            print(f"  Download complete: {local_path}")

        # Compute and cache content hash after download
        from slipstream.utils.hash import get_or_compute_file_hash
        get_or_compute_file_hash(local_path, verbose=verbose)

        return local_path

    @property
    def dataset_hash(self) -> str:
        """Get content-based hash for this dataset.

        Uses SHA256 hash of the .ffcv file content, cached in a sidecar file
        with mtime-based invalidation. Returns first 8 characters.
        """
        from slipstream.utils.hash import get_or_compute_file_hash
        return get_or_compute_file_hash(self._path, length=12, verbose=False)[:8]

    @property
    def cache_path(self) -> Path:
        """Path where optimized SlipCache will be stored.

        Returns a versioned path that includes the dataset hash to prevent
        stale cache issues when the source changes.

        Path format: {base_cache_path}/slipcache-{hash[:8]}/
        """
        return self._base_cache_path / f"slipcache-{self.dataset_hash}"

    @property
    def image_fields(self) -> list[str]:
        """List of image field names."""
        return [name for name, ft in self.field_types.items() if ft == "ImageBytes"]

    def _read_header(self) -> None:
        self._header = np.fromfile(
            str(self._path), dtype=FFCV_HEADER_DTYPE, count=1
        )[0]
        version = self._header['version']
        if version != 2:
            raise ValueError(f"Unsupported FFCV version: {version} (expected 2)")

    def _read_field_descriptors(self) -> None:
        offset = FFCV_HEADER_DTYPE.itemsize
        num_fields = self._header['num_fields']

        self._field_descriptors = np.fromfile(
            str(self._path),
            dtype=FFCV_FIELD_DESC_DTYPE,
            count=num_fields,
            offset=offset,
        )

        # Decode field names
        self._raw_field_names: list[str] = []
        self._field_type_ids: list[int] = []
        for desc in self._field_descriptors:
            name_bytes = desc['name']
            null_idx = np.where(name_bytes == 0)[0]
            if len(null_idx) > 0:
                name = bytes(name_bytes[:null_idx[0]]).decode('ascii')
            else:
                name = bytes(name_bytes).decode('ascii')
            self._raw_field_names.append(name)
            self._field_type_ids.append(int(desc['type_id']))

    def _read_compound_metadata(self) -> None:
        """Read per-sample metadata using compound dtype (handles multi-field files)."""
        metadata_offset = FFCV_HEADER_DTYPE.itemsize + self._field_descriptors.nbytes
        num_samples = int(self._header['num_samples'])

        # Build compound dtype matching FFCV's layout
        field_dtypes = []
        for ix, desc in enumerate(self._field_descriptors):
            type_id = int(desc['type_id'])
            field_dtype = _get_field_metadata_dtype(type_id)
            field_dtypes.append((f'f{ix}', field_dtype))

        self._compound_dtype = np.dtype(field_dtypes, align=True)

        # Read all metadata
        self._all_metadata = np.fromfile(
            str(self._path),
            dtype=self._compound_dtype,
            count=num_samples,
            offset=metadata_offset,
        )

        # Find image field index
        self._image_field_idx = None
        for i, type_id in enumerate(self._field_type_ids):
            if type_id == FFCV_TYPE_RGB_IMAGE:
                self._image_field_idx = i
                break

        if self._image_field_idx is None:
            self._image_field_idx = 0

        # Extract image dimensions (stored as u2 in FFCV, upcast to u4)
        image_meta = self._all_metadata[f'f{self._image_field_idx}']
        self._heights = np.ascontiguousarray(image_meta['height'].astype(np.uint32))
        self._widths = np.ascontiguousarray(image_meta['width'].astype(np.uint32))
        # Image data pointers are also in metadata (for RGBImageField)
        self._image_data_ptrs = np.ascontiguousarray(image_meta['data_ptr'])

    def _read_alloc_table(self) -> None:
        offset = int(self._header['alloc_table_ptr'])
        num_samples = int(self._header['num_samples'])

        full_alloc_table = np.fromfile(
            str(self._path),
            dtype=FFCV_ALLOC_ENTRY_DTYPE,
            offset=offset,
        )

        total_entries = len(full_alloc_table)
        num_var_fields = total_entries // num_samples

        # Identify variable-length fields
        self._var_field_indices: list[int] = []
        for i, type_id in enumerate(self._field_type_ids):
            if type_id in FFCV_VARIABLE_TYPES:
                self._var_field_indices.append(i)

        # Build per-variable-field alloc arrays, sorted by sample_id.
        # The alloc table is in page-write order, NOT sample order.
        self._var_field_alloc: dict[int, np.ndarray] = {}
        for var_pos, field_idx in enumerate(self._var_field_indices):
            # Extract entries for this variable field (interleaved)
            entries = full_alloc_table[var_pos::num_var_fields]
            # Sort by sample_id so index i corresponds to sample i
            sort_order = np.argsort(entries['sample_id'])
            self._var_field_alloc[field_idx] = entries[sort_order]

        # Convenience: image alloc arrays
        if self._image_field_idx not in self._var_field_indices:
            raise ValueError(
                f"Image field (index {self._image_field_idx}) is not variable-length"
            )

        image_alloc = self._var_field_alloc[self._image_field_idx]
        self._alloc_table = image_alloc

    def _build_field_types(self) -> None:
        """Build field_types dict using the field names stored in the file."""
        self.field_types: dict[str, str] = {}
        self._field_name_map: dict[str, int] = {}  # field name -> field index

        for i, (raw_name, type_id) in enumerate(
            zip(self._raw_field_names, self._field_type_ids)
        ):
            field_type = FFCV_TYPE_TO_FIELD_TYPE.get(type_id)
            if field_type is None:
                continue  # Skip unsupported field types

            # Use the name stored in the FFCV file as-is
            self.field_types[raw_name] = field_type
            self._field_name_map[raw_name] = i

        # Auto-detect text in "bytes" fields by probing first sample
        for name, ft in list(self.field_types.items()):
            if ft != "bytes":
                continue
            field_idx = self._field_name_map[name]
            meta = self._all_metadata[f'f{field_idx}']
            ptr = int(meta[0]['ptr'])
            size = int(meta[0]['size'])
            sample_bytes = bytes(self._mmap[ptr:ptr + size])
            try:
                sample_bytes.decode('utf-8')
                self.field_types[name] = "str"
            except UnicodeDecodeError:
                pass  # Keep as "bytes" (binary data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample as a dict.

        Returns:
            Dict with field values, e.g. {'image': bytes, 'label': int, ...}
        """
        sample: dict[str, Any] = {}

        for name, field_type in self.field_types.items():
            field_idx = self._field_name_map[name]

            if field_type == "ImageBytes":
                # Use data_ptr from per-sample metadata (actual JPEG start),
                # not alloc_ptr (page-aligned allocation start)
                data_ptr = int(self._image_data_ptrs[idx])
                alloc_end = int(self._alloc_ptr[idx]) + int(self._alloc_size[idx])
                max_size = alloc_end - data_ptr
                data = bytes(self._mmap[data_ptr:data_ptr + max_size])
                actual_size = _find_image_end(data, max_size)
                sample[name] = data[:actual_size]
            elif field_type == "str":
                meta = self._all_metadata[f'f{field_idx}']
                ptr = int(meta[idx]['ptr'])
                size = int(meta[idx]['size'])
                raw = bytes(self._mmap[ptr:ptr + size])
                text = raw.decode('utf-8')
                # Strip null terminator (FFCV JSONField appends \0)
                sample[name] = text.rstrip('\x00')
            elif field_type == "bytes":
                meta = self._all_metadata[f'f{field_idx}']
                ptr = int(meta[idx]['ptr'])
                size = int(meta[idx]['size'])
                sample[name] = bytes(self._mmap[ptr:ptr + size])
            elif field_type == "int":
                meta = self._all_metadata[f'f{field_idx}']
                sample[name] = int(meta[idx])
            elif field_type == "float":
                meta = self._all_metadata[f'f{field_idx}']
                sample[name] = float(meta[idx])

        return sample

    def read_all_fields(self) -> dict[str, list] | None:
        """Read all fields in bulk for fast cache building.

        Returns:
            Dict mapping field names to lists of values, with image metadata
            stored under __<field>_sizes, __<field>_heights, __<field>_widths.
        """
        n = self.num_samples
        result: dict[str, list] = {}

        for name, field_type in self.field_types.items():
            field_idx = self._field_name_map[name]

            if field_type == "ImageBytes":
                images = []
                sizes = []
                for i in range(n):
                    # Use data_ptr from per-sample metadata (actual JPEG start),
                    # not alloc_ptr (page-aligned allocation start)
                    data_ptr = int(self._image_data_ptrs[i])
                    alloc_end = int(self._alloc_ptr[i]) + int(self._alloc_size[i])
                    max_size = alloc_end - data_ptr
                    data = bytes(self._mmap[data_ptr:data_ptr + max_size])
                    actual_size = _find_image_end(data, max_size)
                    images.append(data[:actual_size])
                    sizes.append(actual_size)

                result[name] = images
                result[f"__{name}_sizes"] = sizes
                result[f"__{name}_heights"] = self._heights.tolist()
                result[f"__{name}_widths"] = self._widths.tolist()

            elif field_type == "str":
                meta = self._all_metadata[f'f{field_idx}']
                values = []
                for i in range(n):
                    ptr = int(meta[i]['ptr'])
                    size = int(meta[i]['size'])
                    raw = bytes(self._mmap[ptr:ptr + size])
                    text = raw.decode('utf-8')
                    values.append(text.rstrip('\x00'))
                result[name] = values

            elif field_type == "bytes":
                meta = self._all_metadata[f'f{field_idx}']
                values = []
                for i in range(n):
                    ptr = int(meta[i]['ptr'])
                    size = int(meta[i]['size'])
                    values.append(bytes(self._mmap[ptr:ptr + size]))
                result[name] = values

            elif field_type == "int":
                meta = self._all_metadata[f'f{field_idx}']
                result[name] = [int(meta[i]) for i in range(n)]

            elif field_type == "float":
                meta = self._all_metadata[f'f{field_idx}']
                result[name] = [float(meta[i]) for i in range(n)]

        return result

    def __repr__(self) -> str:
        fields_str = ", ".join(f"'{k}': {v}" for k, v in self.field_types.items())
        return (
            f"FFCVFileReader(\n"
            f"    path='{self._path}',\n"
            f"    num_samples={self.num_samples:,},\n"
            f"    fields={{{fields_str}}},\n"
            f"    cache_path='{self.cache_path}',\n"
            f")"
        )


__all__ = ["FFCVFileReader"]
