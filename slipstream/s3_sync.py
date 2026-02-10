"""Fast S3 sync for LitData streaming datasets.

Uses s5cmd for parallel S3→local sync, with tqdm fallback for Jupyter.

Usage:
    from slipstream import sync_s3_dataset

    local_path = sync_s3_dataset("s3://bucket/dataset/train/", "/local/cache")
"""

from __future__ import annotations

import hashlib
import os
import pty
import re
import select
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


def run_s5cmd_with_progress(cmd: list[str], verbose: bool = True) -> int:
    """Run s5cmd with real-time progress display.

    Uses a pseudo-TTY to trick s5cmd into outputting progress (it normally
    disables progress when stdout is not a TTY). Handles carriage returns
    manually for clean display in both Jupyter notebooks and terminals.

    Args:
        cmd: Command list (e.g., ["s5cmd", "cp", "--show-progress", ...])
        verbose: If True, display progress. If False, run silently.

    Returns:
        Process return code
    """
    # Create pseudo-terminal to trick s5cmd into showing progress
    master_fd, slave_fd = pty.openpty()

    process = subprocess.Popen(
        cmd,
        stdout=slave_fd,
        stderr=slave_fd,
        stdin=slave_fd,
        close_fds=True
    )

    os.close(slave_fd)  # Close slave in parent

    # Strip ANSI color codes for cleaner display
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    line_buffer = ""

    # Import clear_output only if in Jupyter
    clear_output = None
    if _is_jupyter() and verbose:
        try:
            from IPython.display import clear_output as _clear_output
            clear_output = _clear_output
        except ImportError:
            pass

    try:
        while True:
            # Check if there's data to read (with timeout)
            if select.select([master_fd], [], [], 0.1)[0]:
                try:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break

                    text = data.decode('utf-8', errors='replace')

                    for char in text:
                        if char == '\r':
                            # Carriage return - update display in place
                            if line_buffer.strip() and verbose:
                                clean = ansi_escape.sub('', line_buffer)
                                if clear_output:
                                    clear_output(wait=True)
                                    print(clean)
                                else:
                                    # Terminal mode: use actual carriage return
                                    print(f"\r{clean}", end='', flush=True)
                            line_buffer = ""
                        elif char == '\n':
                            # Newline - print and continue
                            if line_buffer.strip() and verbose:
                                clean = ansi_escape.sub('', line_buffer)
                                print(clean)
                            line_buffer = ""
                        else:
                            line_buffer += char

                except OSError:
                    break

            # Check if process finished
            if process.poll() is not None:
                # Drain any remaining output
                try:
                    while select.select([master_fd], [], [], 0.1)[0]:
                        data = os.read(master_fd, 1024)
                        if not data:
                            break
                        if verbose:
                            text = data.decode('utf-8', errors='replace')
                            clean = ansi_escape.sub('', text)
                            print(clean, end='')
                except OSError:
                    pass
                break

    except KeyboardInterrupt:
        process.terminate()
        if verbose:
            print("\nInterrupted by user")
    finally:
        os.close(master_fd)

    # Print newline after progress bar in terminal mode
    if verbose and not _is_jupyter():
        print()

    # Ensure we have the return code (wait if process is still running)
    if process.returncode is None:
        process.wait()

    return process.returncode or 0


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


def _s5cmd_upload_jupyter(
    cmd: list[str],
    total_files: int,
    verbose: bool,
) -> bool:
    """Run s5cmd upload in Jupyter, showing s5cmd's native output."""
    import sys

    # s5cmd buffers its output, so we can't get real-time progress.
    # Just run it and let output pass through.
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()

    if process.returncode != 0:
        if verbose:
            print(f"  s5cmd error (exit code {process.returncode})")
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


def s3_path_exists(
    s3_path: str,
    endpoint_url: str | None = None,
) -> bool:
    """Check if an S3 path exists.

    Uses s5cmd ls to check for existence. Supports both directory paths
    (ending with /) and file paths.

    Args:
        s3_path: S3 URL to check (e.g., "s3://bucket/path/")
        endpoint_url: S3-compatible endpoint URL (e.g., for Wasabi)

    Returns:
        True if the path exists, False otherwise

    Raises:
        RuntimeError: If s5cmd is not installed
    """
    if shutil.which("s5cmd") is None:
        raise RuntimeError(
            "s5cmd is required for S3 operations but was not found.\n"
            "Install it:\n"
            "  macOS:  brew install peak/tap/s5cmd\n"
            "  Linux:  https://github.com/peak/s5cmd#installation\n"
            "  pip:    pip install s5cmd"
        )

    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += ["ls", s3_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        # s5cmd ls returns 0 and non-empty output if path exists
        # Returns 0 with empty output or non-zero if path doesn't exist
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


def download_s3_cache(
    remote_cache_path: str,
    local_cache_path: Path,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    verbose: bool = True,
) -> bool:
    """Download cache directory from S3 to local path.

    Downloads the .slipstream subdirectory from S3 to the local cache path.

    Args:
        remote_cache_path: S3 URL of the cache directory
            (e.g., "s3://bucket/caches/slipcache-abc123/")
        local_cache_path: Local directory to download to
        endpoint_url: S3-compatible endpoint URL (e.g., for Wasabi)
        numworkers: Number of parallel s5cmd workers
        verbose: Print progress information

    Returns:
        True if successful, False if download failed

    Raises:
        RuntimeError: If s5cmd is not installed
    """
    if shutil.which("s5cmd") is None:
        raise RuntimeError(
            "s5cmd is required for S3 operations but was not found.\n"
            "Install it:\n"
            "  macOS:  brew install peak/tap/s5cmd\n"
            "  Linux:  https://github.com/peak/s5cmd#installation\n"
            "  pip:    pip install s5cmd"
        )

    local_cache_path = Path(local_cache_path)
    local_slipstream = local_cache_path / ".slipstream"
    local_slipstream.mkdir(parents=True, exist_ok=True)

    # Ensure remote path has proper format
    remote = remote_cache_path.rstrip("/") + "/.slipstream/*"
    local = str(local_slipstream) + "/"

    # Use cp with --show-progress for progress display
    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += [
        "--numworkers", str(numworkers),
        "cp", "--show-progress",
        remote, local
    ]

    if verbose:
        print(f"Downloading cache from S3: {remote_cache_path}")

    # Run with PTY-based progress
    returncode = run_s5cmd_with_progress(cmd, verbose=verbose)
    success = returncode == 0

    if not success and verbose:
        print(f"  Download failed (exit code {returncode})")

    if success and verbose:
        num_files = _count_files(local_slipstream)
        print(f"  Download complete: {num_files} files")

    return success


def upload_s3_cache(
    local_cache_path: Path,
    remote_cache_path: str,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    verbose: bool = True,
) -> bool:
    """Upload local cache directory to S3.

    Uploads the .slipstream subdirectory from local to S3.

    Args:
        local_cache_path: Local cache directory containing .slipstream/
        remote_cache_path: S3 URL to upload to
            (e.g., "s3://bucket/caches/slipcache-abc123/")
        endpoint_url: S3-compatible endpoint URL (e.g., for Wasabi)
        numworkers: Number of parallel s5cmd workers
        verbose: Print progress information

    Returns:
        True if successful, False if upload failed

    Raises:
        RuntimeError: If s5cmd is not installed
    """
    if shutil.which("s5cmd") is None:
        raise RuntimeError(
            "s5cmd is required for S3 operations but was not found.\n"
            "Install it:\n"
            "  macOS:  brew install peak/tap/s5cmd\n"
            "  Linux:  https://github.com/peak/s5cmd#installation\n"
            "  pip:    pip install s5cmd"
        )

    local_cache_path = Path(local_cache_path)
    local_slipstream = local_cache_path / ".slipstream"

    if not local_slipstream.exists():
        if verbose:
            print(f"  No .slipstream directory found at {local_cache_path}")
        return False

    # Source and destination paths
    local = str(local_slipstream) + "/*"
    remote = remote_cache_path.rstrip("/") + "/.slipstream/"

    # Use cp with --show-progress and sync flags for progress display
    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += [
        "--numworkers", str(numworkers),
        "cp", "--show-progress", "--if-size-differ",
        local, remote
    ]

    num_files = _count_files(local_slipstream)
    total_size = sum(f.stat().st_size for f in local_slipstream.iterdir() if f.is_file())
    if verbose:
        print(f"Uploading cache to S3: {remote_cache_path}")
        size_str = f"{total_size / 1e9:.2f} GB" if total_size > 1e9 else f"{total_size / 1e6:.1f} MB"
        print(f"  {num_files} files ({size_str})")

    # Run with PTY-based progress
    returncode = run_s5cmd_with_progress(cmd, verbose=verbose)
    success = returncode == 0

    if not success and verbose:
        print(f"  Upload failed (exit code {returncode})")

    if success and verbose:
        print(f"  Upload complete")

    return success


def sync_s3_cache(
    local_cache_path: Path,
    remote_cache_path: str,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    verbose: bool = True,
) -> tuple[int, int]:
    """Bidirectional sync between local and remote cache.

    Syncs the .slipstream subdirectory in both directions:
    1. Download files from remote that don't exist locally
    2. Upload files from local that don't exist remotely

    This ensures that indexes, stats, and other derived files are shared
    across machines using the same remote_cache.

    Args:
        local_cache_path: Local cache directory containing .slipstream/
        remote_cache_path: S3 URL of the cache directory
            (e.g., "s3://bucket/caches/slipcache-abc123/")
        endpoint_url: S3-compatible endpoint URL (e.g., for Wasabi)
        numworkers: Number of parallel s5cmd workers
        verbose: Print progress information

    Returns:
        Tuple of (downloaded_count, uploaded_count) indicating files transferred

    Raises:
        RuntimeError: If s5cmd is not installed
    """
    if shutil.which("s5cmd") is None:
        raise RuntimeError(
            "s5cmd is required for S3 operations but was not found.\n"
            "Install it:\n"
            "  macOS:  brew install peak/tap/s5cmd\n"
            "  Linux:  https://github.com/peak/s5cmd#installation\n"
            "  pip:    pip install s5cmd"
        )

    local_cache_path = Path(local_cache_path)
    local_slipstream = local_cache_path / ".slipstream"

    if not local_slipstream.exists():
        if verbose:
            print(f"  No .slipstream directory found at {local_cache_path}")
        return (0, 0)

    remote = remote_cache_path.rstrip("/") + "/.slipstream/"
    local = str(local_slipstream) + "/"

    # Count files before sync
    local_before = set(f.name for f in local_slipstream.iterdir() if f.is_file())

    # Get remote file list
    cmd_ls = ["s5cmd"]
    if endpoint_url:
        cmd_ls += ["--endpoint-url", endpoint_url]
    cmd_ls += ["ls", remote + "*"]

    try:
        result = subprocess.run(cmd_ls, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            # Parse s5cmd ls output: "2024-01-01 12:00:00  12345 s3://bucket/path/file.ext"
            remote_before = set()
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        # Last part is the full S3 path, extract filename
                        s3_path = parts[-1]
                        filename = s3_path.split("/")[-1]
                        remote_before.add(filename)
        else:
            remote_before = set()
    except Exception:
        remote_before = set()

    downloaded_count = 0
    uploaded_count = 0

    # 1. Download: remote → local (files in remote but not local)
    to_download = remote_before - local_before
    if to_download:
        if verbose:
            print(f"Syncing from remote: {len(to_download)} new files")

        # Download only the specific missing files
        for filename in to_download:
            cmd_dl = ["s5cmd"]
            if endpoint_url:
                cmd_dl += ["--endpoint-url", endpoint_url]
            cmd_dl += [
                "cp", "--show-progress",
                remote + filename, local
            ]

            returncode = run_s5cmd_with_progress(cmd_dl, verbose=verbose)
            if returncode == 0:
                downloaded_count += 1
            elif verbose:
                print(f"  Download failed for {filename} (exit code {returncode})")

    # 2. Upload: local → remote (files in local but not remote)
    to_upload = local_before - remote_before
    if to_upload:
        # Calculate total size of files to upload
        upload_size = sum(
            (local_slipstream / f).stat().st_size
            for f in to_upload
            if (local_slipstream / f).exists()
        )
        if verbose:
            size_str = f"{upload_size / 1e9:.2f} GB" if upload_size > 1e9 else f"{upload_size / 1e6:.1f} MB"
            print(f"Syncing to remote: {len(to_upload)} files ({size_str})")

        # Upload only the specific missing files (not all files with skip flags,
        # which would re-upload if local is newer due to file move/copy)
        for filename in to_upload:
            local_file = local_slipstream / filename
            if not local_file.exists():
                continue

            cmd_ul = ["s5cmd"]
            if endpoint_url:
                cmd_ul += ["--endpoint-url", endpoint_url]
            cmd_ul += [
                "cp", "--show-progress",
                str(local_file), remote + filename
            ]

            returncode = run_s5cmd_with_progress(cmd_ul, verbose=verbose)
            if returncode == 0:
                uploaded_count += 1
            elif verbose:
                print(f"  Upload failed for {filename} (exit code {returncode})")

    if verbose and (downloaded_count > 0 or uploaded_count > 0):
        print(f"  Sync complete: {downloaded_count} downloaded, {uploaded_count} uploaded")
    elif verbose and not to_download and not to_upload:
        pass  # Don't print anything if already in sync

    return (downloaded_count, uploaded_count)


__all__ = [
    "sync_s3_dataset",
    "s3_path_exists",
    "download_s3_cache",
    "upload_s3_cache",
    "sync_s3_cache",
    "run_s5cmd_with_progress",
]
