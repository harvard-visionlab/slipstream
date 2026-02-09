"""ImageFolder reader with S3 tar archive support.

Provides SlipstreamImageFolder, a subclass of torchvision.datasets.ImageFolder
that returns dict samples with (image, label, index, path) fields, and
open_imagefolder() for seamless local or S3 tar archive access.

Usage:
    from slipstream.readers import SlipstreamImageFolder, open_imagefolder

    # Local ImageFolder
    dataset = SlipstreamImageFolder("/path/to/imagenet/val")
    print(dataset[0])  # {'image': b'\\xff\\xd8...', 'label': 0, 'index': 0, 'path': 'val/n01440764/ILSVRC...'}

    # S3 tar archive (auto-download, hash, extract)
    dataset = open_imagefolder("s3://bucket/imagenet/val.tar.gz")

    # Use with SlipstreamLoader
    from slipstream import SlipstreamLoader
    loader = SlipstreamLoader(dataset, batch_size=256, pipelines={...})
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder

__all__ = ["SlipstreamImageFolder", "open_imagefolder"]


# Supported image extensions (matching torchvision.datasets.folder.IMG_EXTENSIONS)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def _is_jupyter() -> bool:
    """Detect if running in a Jupyter notebook or Google Colab."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "Shell":
            return True  # Google Colab
        return False
    except (ImportError, NameError, AttributeError):
        return False


def _compute_file_hash(file_path: Path, length: int = 12) -> str:
    """Compute SHA256 hash of file, returning first `length` characters."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:length]


def _get_cached_hash(tar_path: Path) -> str | None:
    """Return cached hash if tar file hasn't been modified, else None.

    Uses a sidecar file (e.g., val.tar.gz.sha256) containing the hash and
    the tar file's mtime at the time of hashing. If the mtime matches,
    the cached hash is still valid.
    """
    sidecar = tar_path.with_suffix(tar_path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    try:
        content = sidecar.read_text().strip().split()
        if len(content) != 2:
            return None
        cached_hash, cached_mtime = content[0], float(content[1])
        if tar_path.stat().st_mtime == cached_mtime:
            return cached_hash
    except (ValueError, OSError):
        pass
    return None


def _save_hash_cache(tar_path: Path, file_hash: str) -> None:
    """Save hash to sidecar file alongside the tar."""
    sidecar = tar_path.with_suffix(tar_path.suffix + ".sha256")
    sidecar.write_text(f"{file_hash} {tar_path.stat().st_mtime}")


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
    Jupyter: monitors output file size with tqdm.

    Returns True if successful, False if s5cmd is not available.
    """
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
    timeout: int = 30,
) -> bool:
    """Run s5cmd and update a tqdm bar by watching the local file size.

    Args:
        cmd: Full s5cmd command to run
        local_path: Path where file will be downloaded
        expected_size: Expected file size in bytes (for progress bar)
        verbose: Print progress
        timeout: Seconds to wait for file to appear before giving up
    """
    import time

    from tqdm.auto import tqdm

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Start s5cmd process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    desc = f"  Downloading {local_path.name}"
    last_size = 0
    start_time = time.time()
    file_appeared = False
    download_dir = local_path.parent
    base_name = local_path.stem  # e.g., "val.tar" from "val.tar.tmp"

    def _find_download_file() -> Path | None:
        """Find the file being downloaded (s5cmd may add random suffix)."""
        # Check for exact path first
        if local_path.exists():
            return local_path
        # Check for s5cmd temp files (they add random suffixes)
        for f in download_dir.iterdir():
            if f.name.startswith(base_name) and f.is_file():
                return f
        return None

    with tqdm(
        total=expected_size,
        unit="B",
        unit_scale=True,
        desc=desc,
        disable=not verbose,
        miniters=1,
        mininterval=0.5,
    ) as pbar:
        while process.poll() is None:
            download_file = _find_download_file()
            if download_file:
                file_appeared = True
                try:
                    current_size = download_file.stat().st_size
                    if current_size != last_size:
                        pbar.n = current_size
                        pbar.refresh()
                        last_size = current_size
                except (FileNotFoundError, OSError):
                    pass  # File may have been renamed/moved
            elif not file_appeared and (time.time() - start_time) > timeout:
                # File hasn't appeared after timeout, something is wrong
                process.terminate()
                if verbose:
                    print(f"\n  Timeout: file not created after {timeout}s, falling back to fsspec")
                return False
            time.sleep(0.5)

        # Final update
        if local_path.exists():
            pbar.n = local_path.stat().st_size
            pbar.refresh()

    # Check for errors
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""

    if process.returncode != 0:
        if verbose:
            if stderr:
                print(f"  s5cmd stderr: {stderr}")
            if stdout:
                print(f"  s5cmd stdout: {stdout}")
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
            tqdm(
                total=file_size, unit="B", unit_scale=True, desc="  Downloading"
            ) as pbar,
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


def _extract_tar(tar_path: Path, output_dir: Path, verbose: bool = True) -> Path:
    """Extract tar archive, return path to extracted folder.

    Handles .tar, .tar.gz, .tgz, .tar.bz2, .tbz2 formats.

    Returns:
        Path to the extracted top-level directory.
    """
    from tqdm.auto import tqdm

    with tarfile.open(tar_path, "r:*") as tar:
        # Get all members for progress tracking
        members = tar.getmembers()
        if not members:
            raise ValueError(f"Empty tar archive: {tar_path}")

        # Find the common top-level directory
        top_dirs = set()
        for member in members:
            parts = member.name.split("/")
            if parts[0]:
                top_dirs.add(parts[0])

        if len(top_dirs) == 1:
            top_dir = top_dirs.pop()
            extract_path = output_dir / top_dir
        else:
            # Multiple top-level items, use tar filename as directory
            top_dir = tar_path.stem
            if top_dir.endswith(".tar"):
                top_dir = top_dir[:-4]
            extract_path = output_dir / top_dir

        if not extract_path.exists():
            # Extract with progress bar
            desc = f"  Extracting {tar_path.name}"
            for member in tqdm(members, desc=desc, disable=not verbose):
                tar.extract(member, output_dir, filter="data")

    if verbose:
        print(f"  Extracted to {extract_path}")

    return extract_path


def _download_and_extract_s3_tar(
    s3_uri: str,
    cache_dir: Path,
    s3_config: dict | None = None,
    verbose: bool = True,
) -> Path:
    """Download tar from S3, hash it, extract to cache_dir/hashid/<hash>/

    Args:
        s3_uri: S3 URI to tar archive (e.g., s3://bucket/data/val.tar.gz)
        cache_dir: Base cache directory
        s3_config: Optional S3 configuration (endpoint_url, profile, region)
        verbose: Print progress information

    Returns:
        Path to extracted ImageFolder directory.
    """
    import fsspec

    # Parse S3 URI
    filename = s3_uri.rsplit("/", 1)[-1]

    # Create temp download path
    download_dir = cache_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    tar_path = download_dir / filename

    # Check if already downloaded
    if not tar_path.exists():
        if verbose:
            print(f"Downloading: {s3_uri}")

        # Get file size for progress
        endpoint_url = s3_config.get("endpoint_url") if s3_config else None
        storage_options = {}
        if endpoint_url:
            storage_options["endpoint_url"] = endpoint_url

        fs, resolved_path = fsspec.core.url_to_fs(s3_uri, **storage_options)
        file_info = fs.info(resolved_path)
        file_size = file_info.get("size", 0)

        if verbose and file_size:
            print(f"  Size: {file_size / 1e9:.2f} GB")

        # Download to temp file first
        tmp_path = tar_path.with_suffix(".tmp")
        try:
            downloaded = _download_s5cmd(
                s3_uri,
                tmp_path,
                endpoint_url=endpoint_url,
                expected_size=file_size or None,
                verbose=verbose,
            )
            if not downloaded:
                _download_fsspec(
                    s3_uri,
                    tmp_path,
                    file_size=file_size or None,
                    verbose=verbose,
                )
            os.rename(tmp_path, tar_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        if verbose:
            print(f"  Downloaded: {tar_path}")
    else:
        if verbose:
            size_gb = tar_path.stat().st_size / 1e9
            print(f"Using cached tar: {tar_path} ({size_gb:.2f} GB)")

    # Get hash (from cache if available, otherwise compute and cache)
    file_hash = _get_cached_hash(tar_path)
    if file_hash is not None:
        if verbose:
            print(f"  Using cached hash: {file_hash}")
    else:
        if verbose:
            print("  Computing hash...")
        file_hash = _compute_file_hash(tar_path)
        _save_hash_cache(tar_path, file_hash)
        if verbose:
            print(f"  Hash: {file_hash}")

    # Check if already extracted
    extract_base = cache_dir / "hashid" / file_hash
    extract_base.mkdir(parents=True, exist_ok=True)

    # Look for existing extracted directory
    existing = list(extract_base.iterdir())
    if existing:
        extract_path = existing[0]
        if verbose:
            print(f"Using cached extraction: {extract_path}")
        return extract_path

    # Extract tar
    extract_path = _extract_tar(tar_path, extract_base, verbose=verbose)

    return extract_path


class SlipstreamImageFolder(ImageFolder):
    """ImageFolder subclass that returns dict samples with richer metadata.

    Returns samples as dicts with keys:
        - 'image': Raw image bytes (or decoded tensor/PIL if decode_images=True)
        - 'label': Integer class index
        - 'index': Sample index in dataset
        - 'path': Relative path (split/class/filename.jpg)

    This format is compatible with SlipstreamLoader and OptimizedCache.build().

    Args:
        root: Root directory of ImageFolder (contains class subdirectories)
        cache_dir: Where to store the optimized cache. Defaults to root/.slipstream-cache
        decode_images: If True, decode image bytes to tensors/PIL images.
        to_pil: If True and decode_images=True, return PIL Images instead of tensors.
        pipelines: Dict mapping field names to transform functions.
        **kwargs: Additional arguments passed to ImageFolder

    Example:
        dataset = SlipstreamImageFolder("/path/to/imagenet/val")
        sample = dataset[0]
        # {'image': b'\\xff\\xd8...', 'label': 0, 'index': 0, 'path': 'val/n01440764/...'}

        # With decoding
        dataset = SlipstreamImageFolder("/path/to/imagenet/val", decode_images=True)
        sample = dataset[0]
        # {'image': PIL.Image, 'label': 0, 'index': 0, 'path': 'val/n01440764/...'}
    """

    def __init__(
        self,
        root: str | Path,
        cache_dir: str | Path | None = None,
        decode_images: bool = False,
        to_pil: bool = True,
        pipelines: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize parent ImageFolder
        super().__init__(str(root), **kwargs)

        self._root_path = Path(root)

        # Set up base cache path (user-specified or derived from root)
        if cache_dir is not None:
            self._base_cache_path = Path(cache_dir)
        else:
            self._base_cache_path = self._root_path / ".slipstream-cache"

        # Store field types for compatibility with SlipstreamLoader
        self._field_types = {
            "image": "ImageBytes",
            "label": "int",
            "index": "int",
            "path": "str",
        }

        # Store processing options
        self.decode_images = decode_images
        self.to_pil = to_pil
        self.pipelines = dict(pipelines) if pipelines is not None else None
        self.image_fields = ["image"]  # For compatibility with SlipstreamDataset

    @property
    def dataset_hash(self) -> str:
        """Get content-based hash for this dataset.

        For S3 tar archives: extracts the file hash from the extraction path
        (e.g., /hashid/abc123def456/val → abc123de).

        For local directories: computes hash from file listing metadata
        (relative paths, mtimes, sizes), cached in .slipstream-hash sidecar.

        Returns first 8 characters.
        """
        from slipstream.utils.hash import (
            extract_hash_from_path,
            get_or_compute_directory_hash,
            IMG_EXTENSIONS,
        )

        # Check if path contains hash from S3 tar extraction
        extracted_hash = extract_hash_from_path(self._root_path)
        if extracted_hash:
            return extracted_hash[:8]

        # Local directory: compute hash from file listing metadata
        return get_or_compute_directory_hash(
            self._root_path,
            extensions=IMG_EXTENSIONS,
            length=12,
            verbose=False,
        )[:8]

    @property
    def cache_path(self) -> Path:
        """Path where optimized SlipCache will be stored.

        Returns a versioned path that includes the dataset hash to prevent
        stale cache issues when the source changes.

        Path format: {base_cache_path}/slipcache-{hash[:8]}/
        """
        return self._base_cache_path / f"slipcache-{self.dataset_hash}"

    @property
    def field_types(self) -> dict[str, str]:
        """Field types for SlipstreamLoader compatibility."""
        return self._field_types

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a sample as a dict with image bytes and metadata.

        Args:
            index: Sample index

        Returns:
            Dict with keys: 'image', 'label', 'index', 'path'
        """
        path, label = self.samples[index]

        # Read raw image bytes
        with open(path, "rb") as f:
            image_data = f.read()

        # Compute relative path (last 3 parts: split/class/filename or class/filename)
        path_obj = Path(path)
        try:
            rel_path = path_obj.relative_to(self._root_path.parent)
        except ValueError:
            # Fallback: use last 3 parts
            rel_path = Path(*path_obj.parts[-3:])

        sample = {
            "image": image_data,
            "label": label,
            "index": index,
            "path": str(rel_path),
        }

        # Decode images if requested
        if self.decode_images:
            from slipstream.dataset import decode_image
            sample["image"] = decode_image(sample["image"], to_pil=self.to_pil)

        # Apply per-field pipelines
        if self.pipelines is not None:
            for key, pipeline in self.pipelines.items():
                if key in sample:
                    sample[key] = pipeline(sample[key])

        return sample

    def read_all_fields(self) -> dict[str, list]:
        """Read all fields in bulk for fast cache building.

        Returns:
            Dict mapping field names to lists of values, with image metadata
            stored under __image_sizes, __image_heights, __image_widths.
        """
        from slipstream.utils.image_header import read_image_dimensions

        n = len(self.samples)
        images: list[bytes] = []
        labels: list[int] = []
        indices: list[int] = []
        paths: list[str] = []
        sizes: list[int] = []
        heights: list[int] = []
        widths: list[int] = []

        for idx, (img_path, label) in enumerate(self.samples):
            # Read raw bytes - files from disk are complete, no trimming needed
            # (find_image_end is only for FFCV files with page-aligned padding)
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            images.append(img_bytes)
            labels.append(label)
            indices.append(idx)

            # Compute relative path
            path_obj = Path(img_path)
            try:
                rel_path = path_obj.relative_to(self._root_path.parent)
            except ValueError:
                rel_path = Path(*path_obj.parts[-3:])
            paths.append(str(rel_path))

            # Get dimensions from header (fast, no decode)
            try:
                w, h = read_image_dimensions(img_bytes)
            except Exception:
                # Fallback: decode to get dimensions
                from PIL import Image
                import io
                with Image.open(io.BytesIO(img_bytes)) as img:
                    w, h = img.size

            sizes.append(len(img_bytes))
            heights.append(h)
            widths.append(w)

        return {
            "image": images,
            "__image_sizes": sizes,
            "__image_heights": heights,
            "__image_widths": widths,
            "label": labels,
            "index": indices,
            "path": paths,
        }

    def __repr__(self) -> str:
        fields_str = ", ".join(f"'{k}': {v}" for k, v in self._field_types.items())
        return (
            f"SlipstreamImageFolder(\n"
            f"    root='{self._root_path}',\n"
            f"    num_samples={len(self):,},\n"
            f"    num_classes={len(self.classes)},\n"
            f"    fields={{{fields_str}}},\n"
            f"    cache_path='{self.cache_path}',\n"
            f")"
        )


def open_imagefolder(
    source: str | Path,
    cache_dir: str | Path | None = None,
    s3_config: dict | None = None,
    verbose: bool = True,
    decode_images: bool = False,
    to_pil: bool = True,
    pipelines: dict[str, Any] | None = None,
) -> SlipstreamImageFolder:
    """Open ImageFolder from local path or S3 tar archive.

    Supports:
        - Local directories with class subdirectories
        - S3 tar archives (.tar, .tar.gz, .tgz, .tar.bz2, .tbz2)

    For S3 archives:
        1. Downloads tar to cache_dir/downloads/
        2. Computes SHA256 hash
        3. Extracts to cache_dir/hashid/<hash>/
        4. Returns SlipstreamImageFolder pointing to extracted directory

    Args:
        source: Local path or s3:// URI to tar archive
        cache_dir: Where to cache downloaded/extracted data.
            Defaults to ~/.lightning/slipstream/imagefolder/
        s3_config: S3 credentials (endpoint_url, profile, region)
        verbose: Print progress information
        decode_images: If True, decode image bytes to tensors/PIL images.
        to_pil: If True and decode_images=True, return PIL Images instead of tensors.
        pipelines: Dict mapping field names to transform functions.

    Returns:
        SlipstreamImageFolder ready for use with SlipstreamLoader

    Example:
        # Local directory
        dataset = open_imagefolder("/path/to/imagenet/val")

        # S3 tar archive
        dataset = open_imagefolder("s3://visionlab-datasets/imagenet1k-raw/val.tar.gz")

        # With decoding
        dataset = open_imagefolder("/path/to/data", decode_images=True)
    """
    source_str = str(source)

    # Resolve cache_dir
    if cache_dir is None:
        from slipstream.dataset import get_default_cache_dir

        cache_dir = get_default_cache_dir() / "slipstream" / "imagefolder"
    else:
        cache_dir = Path(cache_dir)

    # Handle S3 tar archives
    if source_str.startswith("s3://"):
        local_path = _download_and_extract_s3_tar(
            source_str,
            cache_dir,
            s3_config=s3_config,
            verbose=verbose,
        )
    else:
        local_path = Path(source_str)

    if not local_path.exists():
        raise FileNotFoundError(f"ImageFolder not found: {local_path}")

    # Check for ImageFolder structure
    if not _is_imagefolder_structure(local_path):
        raise ValueError(
            f"Not a valid ImageFolder structure: {local_path}\n"
            f"Expected subdirectories with image files."
        )

    return SlipstreamImageFolder(
        local_path,
        cache_dir=cache_dir / "cache",
        decode_images=decode_images,
        to_pil=to_pil,
        pipelines=pipelines,
    )


def _is_imagefolder_structure(path: Path) -> bool:
    """Check if path has ImageFolder structure (class subdirs with images).

    Args:
        path: Directory path to check

    Returns:
        True if path contains subdirectories with image files.
    """
    if not path.is_dir():
        return False

    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return False

    # Check if subdirs contain image files
    for subdir in subdirs[:3]:  # Sample first 3 subdirs
        files = list(subdir.iterdir())[:10]  # Sample first 10 files
        if any(f.suffix.lower() in IMG_EXTENSIONS for f in files):
            return True

    return False
