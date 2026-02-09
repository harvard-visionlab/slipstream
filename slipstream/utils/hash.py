"""Hash utilities for content-based cache versioning.

Provides sidecar-based file hashing with mtime-based cache invalidation.
Used by readers to generate stable dataset hashes for SlipCache versioning.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


def compute_file_hash(file_path: Path, length: int = 12) -> str:
    """Compute SHA256 hash of file, returning first `length` characters.

    Args:
        file_path: Path to file to hash
        length: Number of hex characters to return (default 12)

    Returns:
        First `length` characters of SHA256 hex digest
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:length]


def get_cached_hash(file_path: Path) -> str | None:
    """Return cached hash if file hasn't been modified, else None.

    Uses a sidecar file (e.g., file.ffcv.sha256) containing the hash and
    the file's mtime at the time of hashing. If the mtime matches,
    the cached hash is still valid.

    Args:
        file_path: Path to the file (sidecar is file_path + ".sha256")

    Returns:
        Cached hash string if valid, None if stale or missing
    """
    sidecar = file_path.with_suffix(file_path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    try:
        content = sidecar.read_text().strip().split()
        if len(content) != 2:
            return None
        cached_hash, cached_mtime = content[0], float(content[1])
        if file_path.stat().st_mtime == cached_mtime:
            return cached_hash
    except (ValueError, OSError):
        pass
    return None


def save_hash_cache(file_path: Path, file_hash: str) -> None:
    """Save hash to sidecar file alongside the original file.

    Args:
        file_path: Path to the original file
        file_hash: Hash string to cache
    """
    sidecar = file_path.with_suffix(file_path.suffix + ".sha256")
    sidecar.write_text(f"{file_hash} {file_path.stat().st_mtime}")


def get_or_compute_file_hash(
    file_path: Path,
    length: int = 12,
    verbose: bool = False,
) -> str:
    """Get cached hash or compute and cache it.

    Args:
        file_path: Path to file to hash
        length: Number of hex characters to return
        verbose: Print progress messages

    Returns:
        Hash string (first `length` characters of SHA256)
    """
    file_hash = get_cached_hash(file_path)
    if file_hash is not None:
        if verbose:
            print(f"  Using cached hash: {file_hash}")
        return file_hash

    if verbose:
        print(f"  Computing hash for {file_path.name}...")
    file_hash = compute_file_hash(file_path, length=length)
    save_hash_cache(file_path, file_hash)
    if verbose:
        print(f"  Hash: {file_hash}")
    return file_hash


def extract_hash_from_path(path: Path, pattern: str = r"/hashid/([a-f0-9]{12})/") -> str | None:
    """Extract a hash from a path if it matches the pattern.

    Used for ImageFolder paths that were extracted from S3 tar archives,
    where the extraction path includes the file hash.

    Args:
        path: Path to search for hash
        pattern: Regex pattern with one capture group for the hash

    Returns:
        Extracted hash string if found, None otherwise
    """
    match = re.search(pattern, str(path))
    if match:
        return match.group(1)
    return None


def compute_directory_hash(
    root: Path,
    extensions: Iterable[str] | None = None,
    length: int = 12,
) -> str:
    """Compute hash of directory contents based on file metadata.

    Hashes a sorted list of (relative_path, mtime, size) for all files.
    This detects added/removed files and modified files (mtime/size changes)
    without reading file contents.

    Args:
        root: Root directory to hash
        extensions: File extensions to include (e.g., {".jpg", ".png"}).
                   If None, includes all files.
        length: Number of hex characters to return

    Returns:
        Hash string based on directory contents metadata
    """
    if extensions is not None:
        extensions = {ext.lower() for ext in extensions}

    # Collect file metadata
    entries = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if extensions is not None and file_path.suffix.lower() not in extensions:
            continue

        stat = file_path.stat()
        rel_path = file_path.relative_to(root)
        entries.append(f"{rel_path}:{stat.st_mtime}:{stat.st_size}")

    # Hash the combined metadata
    content = "\n".join(entries)
    return hashlib.sha256(content.encode()).hexdigest()[:length]


def get_or_compute_directory_hash(
    root: Path,
    extensions: Iterable[str] | None = None,
    length: int = 12,
    verbose: bool = False,
) -> str:
    """Get cached directory hash or compute and cache it.

    Uses a sidecar file at root/.slipstream-hash containing the hash
    and the directory's latest mtime. Recomputes if any file is newer
    than the sidecar.

    Args:
        root: Root directory to hash
        extensions: File extensions to include
        length: Number of hex characters to return
        verbose: Print progress messages

    Returns:
        Hash string based on directory contents metadata
    """
    sidecar = root / ".slipstream-hash"

    # Check if sidecar exists and is still valid
    if sidecar.exists():
        try:
            content = sidecar.read_text().strip().split()
            if len(content) == 2:
                cached_hash, cached_mtime = content[0], float(content[1])
                sidecar_mtime = sidecar.stat().st_mtime

                # Check if any file in the directory is newer than sidecar
                needs_recompute = False
                for file_path in root.rglob("*"):
                    if file_path.is_file() and file_path != sidecar:
                        if file_path.stat().st_mtime > sidecar_mtime:
                            needs_recompute = True
                            break

                if not needs_recompute:
                    if verbose:
                        print(f"  Using cached directory hash: {cached_hash}")
                    return cached_hash
        except (ValueError, OSError):
            pass

    if verbose:
        print(f"  Computing directory hash for {root}...")

    dir_hash = compute_directory_hash(root, extensions=extensions, length=length)

    # Save to sidecar
    import time
    sidecar.write_text(f"{dir_hash} {time.time()}")

    if verbose:
        print(f"  Directory hash: {dir_hash}")

    return dir_hash


# Image extensions for ImageFolder hashing
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
