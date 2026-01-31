"""Fast S3 sync for LitData streaming datasets.

Uses s5cmd for parallel S3→local sync, with tqdm fallback for Jupyter.

Usage:
    from slipstream import sync_s3_dataset

    local_path = sync_s3_dataset("s3://bucket/dataset/train/", "/local/cache")
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _is_jupyter() -> bool:
    """Detect if running in a Jupyter notebook or Google Colab."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except (ImportError, NameError, AttributeError):
        return False


def _deterministic_local_dir(remote_dir: str, cache_dir: Path) -> Path:
    """Generate a deterministic local path for an S3 dataset."""
    url_hash = hashlib.sha256(remote_dir.encode()).hexdigest()[:12]
    # Extract dataset name from URL (last non-empty path component)
    parts = remote_dir.rstrip("/").split("/")
    dataset_name = parts[-1] if parts else "dataset"
    return cache_dir / "s3-sync" / f"{dataset_name}-{url_hash}"


def _count_files(directory: Path) -> int:
    """Count files in a directory (non-recursive for speed)."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.iterdir() if _.is_file())


def _s5cmd_sync_jupyter(
    cmd: list[str],
    local_dir: Path,
    verbose: bool,
) -> bool:
    """Run s5cmd sync with tqdm progress by monitoring local dir file count."""
    import time

    from tqdm.auto import tqdm

    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    desc = f"  Syncing → {local_dir.name}"
    with tqdm(unit=" files", desc=desc, disable=not verbose) as pbar:
        while process.poll() is None:
            n = _count_files(local_dir)
            pbar.n = n
            pbar.refresh()
            time.sleep(0.5)

        # Final update
        pbar.n = _count_files(local_dir)
        pbar.refresh()

    if process.returncode != 0:
        error_msg = process.stderr.read().decode()
        if verbose:
            print(f"  s5cmd error: {error_msg}")
        return False
    return True


def sync_s3_dataset(
    remote_dir: str,
    cache_dir: str | Path | None = None,
    *,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    verbose: bool = True,
) -> Path:
    """Sync an S3 dataset to local disk using s5cmd.

    Uses ``s5cmd sync`` which skips unchanged files on re-runs (incremental).
    Much faster than LitData's built-in download for large datasets.

    Args:
        remote_dir: S3 URL of the dataset directory.
            Example: ``"s3://bucket/dataset/train/"``
        cache_dir: Local parent directory for synced data. Defaults to
            the slipstream cache directory.
        endpoint_url: S3-compatible endpoint URL (e.g. for Wasabi).
        numworkers: Number of parallel s5cmd workers.
        verbose: Print progress information.

    Returns:
        Path to the local synced directory.

    Raises:
        RuntimeError: If s5cmd is not installed.
    """
    if shutil.which("s5cmd") is None:
        raise RuntimeError(
            "s5cmd is required for fast S3 sync but was not found.\n"
            "Install it:\n"
            "  macOS:  brew install peak/tap/s5cmd\n"
            "  Linux:  https://github.com/peak/s5cmd#installation\n"
            "  pip:    pip install s5cmd"
        )

    if cache_dir is None:
        from slipstream.dataset import get_default_cache_dir
        cache_dir = get_default_cache_dir()
    cache_dir = Path(cache_dir)

    local_dir = _deterministic_local_dir(remote_dir, cache_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Normalize remote_dir to ensure trailing wildcard
    remote = remote_dir.rstrip("/") + "/*"

    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += ["--numworkers", str(numworkers), "sync", remote, str(local_dir) + "/"]

    if verbose:
        print(f"Syncing S3 dataset: {remote_dir}")
        print(f"  → {local_dir}")
        print(f"  Workers: {numworkers}")

    if _is_jupyter():
        success = _s5cmd_sync_jupyter(cmd, local_dir, verbose)
    else:
        # Terminal: use --show-progress for native progress
        # s5cmd sync doesn't support --show-progress, but cp does.
        # For sync, just run and let stderr show file-by-file output.
        try:
            subprocess.run(cmd, check=True)
            success = True
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"  s5cmd sync failed: {e}")
            success = False

    if not success:
        raise RuntimeError(f"Failed to sync {remote_dir} to {local_dir}")

    if verbose:
        num_files = _count_files(local_dir)
        print(f"  Sync complete: {num_files} files in {local_dir}")

    return local_dir


__all__ = ["sync_s3_dataset"]
