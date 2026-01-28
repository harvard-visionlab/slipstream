"""Benchmark utilities for slipstream.

Provides machine info collection, timing helpers, and result formatting.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MachineInfo:
    """Machine information for benchmark tracking."""

    hostname: str
    machine_name: str  # User-provided name (e.g., "nolan-25")
    platform: str
    platform_version: str
    architecture: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_gb: float
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    drive_type: str | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.hostname,
            "machine_name": self.machine_name,
            "platform": self.platform,
            "platform_version": self.platform_version,
            "architecture": self.architecture,
            "cpu_model": self.cpu_model,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "ram_gb": self.ram_gb,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "drive_type": self.drive_type,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "MACHINE INFO",
            "=" * 60,
            f"Machine:       {self.machine_name}",
            f"Hostname:      {self.hostname}",
            f"Platform:      {self.platform} {self.platform_version}",
            f"Architecture:  {self.architecture}",
            f"CPU:           {self.cpu_model}",
            f"CPU Cores:     {self.cpu_cores_physical} physical, {self.cpu_cores_logical} logical",
            f"RAM:           {self.ram_gb:.1f} GB",
            f"Python:        {self.python_version}",
            f"PyTorch:       {self.pytorch_version}",
        ]

        if self.cuda_available:
            lines.append(f"CUDA:          {self.cuda_version}")
            lines.append(f"GPU:           {self.gpu_name}")
            if self.gpu_memory_gb:
                lines.append(f"GPU Memory:    {self.gpu_memory_gb:.1f} GB")
        else:
            lines.append("CUDA:          Not available")

        if self.drive_type:
            lines.append(f"Drive Type:    {self.drive_type}")

        lines.append(f"Timestamp:     {self.timestamp}")
        lines.append("=" * 60)

        return "\n".join(lines)


def get_cpu_model() -> str:
    """Get CPU model name."""
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        # Try for Apple Silicon
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.chip"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown"


def get_cpu_cores() -> tuple[int, int]:
    """Get (physical, logical) CPU core counts."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1, psutil.cpu_count(logical=True) or 1
    except ImportError:
        pass

    logical = os.cpu_count() or 1

    # Try to get physical cores
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()), logical
        except Exception:
            pass
    elif system == "Linux":
        try:
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Core(s) per socket:" in line:
                        cores_per_socket = int(line.split(":")[1].strip())
                    if "Socket(s):" in line:
                        sockets = int(line.split(":")[1].strip())
                return cores_per_socket * sockets, logical
        except Exception:
            pass

    return logical, logical


def get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal:" in line:
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass

    return 0.0


def get_drive_info(path: str = ".") -> dict[str, Any]:
    """Get detailed drive information for a given path.

    Returns dict with:
        - type: "NVMe SSD", "SSD", "HDD", or None
        - device: device path (e.g., /dev/nvme0n1p1)
        - mount_point: mount point path
        - filesystem: filesystem type
        - size_gb: total size in GB (if available)
        - free_gb: free space in GB (if available)
    """
    info = {
        "type": None,
        "device": None,
        "mount_point": None,
        "filesystem": None,
        "size_gb": None,
        "free_gb": None,
    }

    system = platform.system()

    # Resolve to absolute path
    try:
        path = os.path.realpath(path)
    except Exception:
        pass

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["df", "-k", path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    info["device"] = parts[0]
                    info["mount_point"] = parts[8] if len(parts) > 8 else parts[5]
                    # Size and free in 1K blocks
                    if len(parts) >= 4:
                        info["size_gb"] = int(parts[1]) / (1024 * 1024)
                        info["free_gb"] = int(parts[3]) / (1024 * 1024)

                    device = parts[0]
                    if "nvme" in device.lower():
                        info["type"] = "NVMe SSD"
                    else:
                        result2 = subprocess.run(
                            ["diskutil", "info", device],
                            capture_output=True,
                            text=True,
                        )
                        if result2.returncode == 0:
                            if "Solid State" in result2.stdout:
                                if "Yes" in result2.stdout.split("Solid State")[1][:20]:
                                    info["type"] = "SSD"
                            if "NVMe" in result2.stdout:
                                info["type"] = "NVMe SSD"
                        if info["type"] is None:
                            info["type"] = "SSD (assumed)"
        except Exception:
            pass

    elif system == "Linux":
        try:
            # Get device and mount info
            result = subprocess.run(
                ["df", "-T", "-k", path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    info["device"] = parts[0]
                    info["filesystem"] = parts[1]
                    info["mount_point"] = parts[6] if len(parts) > 6 else None
                    # Size and free in 1K blocks
                    if len(parts) >= 5:
                        info["size_gb"] = int(parts[2]) / (1024 * 1024)
                        info["free_gb"] = int(parts[4]) / (1024 * 1024)

                    # Extract base device name for drive type detection
                    device = parts[0]
                    # Handle overlay/docker filesystems
                    if device == "overlay" or device.startswith("/dev/loop"):
                        # Try to find the underlying device from mount info
                        try:
                            with open("/proc/mounts") as f:
                                for line in f:
                                    mount_parts = line.split()
                                    if len(mount_parts) >= 2:
                                        if mount_parts[1] == info["mount_point"]:
                                            # For overlay, check the upper dir
                                            pass
                        except Exception:
                            pass
                        info["type"] = "Unknown (container)"
                    else:
                        device_name = os.path.basename(device).rstrip("0123456789p")
                        if device_name.startswith("nvme"):
                            info["type"] = "NVMe SSD"
                        else:
                            # Check rotational flag
                            rotational_path = f"/sys/block/{device_name}/queue/rotational"
                            if os.path.exists(rotational_path):
                                with open(rotational_path) as f:
                                    if f.read().strip() == "0":
                                        info["type"] = "SSD"
                                    else:
                                        info["type"] = "HDD"
        except Exception:
            pass

    return info


def get_drive_type(path: str = ".") -> str | None:
    """Attempt to detect drive type (SSD/HDD/NVMe) for given path."""
    return get_drive_info(path).get("type")


def get_gpu_info() -> tuple[bool, str | None, str | None, float | None]:
    """Get GPU info: (cuda_available, cuda_version, gpu_name, gpu_memory_gb)."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, cuda_version, gpu_name, gpu_memory
    except Exception:
        pass

    return False, None, None, None


def get_machine_info(machine_name: str | None = None) -> MachineInfo:
    """Collect all machine information.

    Args:
        machine_name: User-provided machine name (e.g., "nolan-25").
                      If None, uses the hostname.
    """
    import torch

    hostname = socket.gethostname()
    physical_cores, logical_cores = get_cpu_cores()
    cuda_available, cuda_version, gpu_name, gpu_memory = get_gpu_info()

    return MachineInfo(
        hostname=hostname,
        machine_name=machine_name or hostname,
        platform=platform.system(),
        platform_version=platform.release(),
        architecture=platform.machine(),
        cpu_model=get_cpu_model(),
        cpu_cores_physical=physical_cores,
        cpu_cores_logical=logical_cores,
        ram_gb=get_ram_gb(),
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        drive_type=get_drive_type(),
        timestamp=datetime.now().isoformat(),
    )


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    samples_per_sec: float
    total_samples: int
    elapsed_sec: float
    num_epochs: int
    warmup_epochs: int
    per_epoch_results: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.samples_per_sec:,.0f} samples/sec"


def run_epoch(
    iterator,
    count_fn=None,
    desc: str = "",
) -> dict[str, float]:
    """Run one epoch and return timing results.

    Args:
        iterator: Iterable to benchmark
        count_fn: Function to count samples from each batch. If None, counts batches.
        desc: Description for progress bar

    Returns:
        Dict with samples_per_sec, elapsed_sec, total_samples
    """
    from tqdm import tqdm

    total_samples = 0
    start = time.perf_counter()

    for batch in tqdm(iterator, desc=desc, leave=False):
        if count_fn:
            total_samples += count_fn(batch)
        else:
            total_samples += 1

    elapsed = time.perf_counter() - start
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    return {
        "samples_per_sec": samples_per_sec,
        "elapsed_sec": elapsed,
        "total_samples": total_samples,
    }


def run_benchmark(
    name: str,
    iterator_fn,
    count_fn=None,
    num_epochs: int = 3,
    num_warmup: int = 1,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run a complete benchmark with warmup and multiple epochs.

    Args:
        name: Benchmark name
        iterator_fn: Callable that returns an iterator (called fresh each epoch)
        count_fn: Function to count samples from each batch
        num_epochs: Number of timed epochs
        num_warmup: Number of warmup epochs
        metadata: Additional metadata to store

    Returns:
        BenchmarkResult with averaged results
    """
    print(f"\n{name}:")

    # Warmup
    print(f"  Warmup ({num_warmup} epoch(s)):")
    warmup_results = []
    for i in range(num_warmup):
        result = run_epoch(iterator_fn(), count_fn, desc=f"Warmup {i+1}")
        warmup_results.append(result)
        print(f"    Warmup {i + 1}: {result['samples_per_sec']:,.0f} samples/sec ({result['elapsed_sec']:.2f}s)")

    # Timed epochs
    epoch_results = []
    for epoch in range(num_epochs):
        result = run_epoch(iterator_fn(), count_fn, desc=f"Epoch {epoch+1}")
        epoch_results.append(result)
        print(f"  Epoch {epoch + 1}: {result['samples_per_sec']:,.0f} samples/sec ({result['elapsed_sec']:.2f}s)")

    avg_samples_per_sec = np.mean([r["samples_per_sec"] for r in epoch_results])
    total_elapsed = sum(r["elapsed_sec"] for r in epoch_results)
    total_samples = epoch_results[0]["total_samples"] if epoch_results else 0

    print(f"  Average: {avg_samples_per_sec:,.0f} samples/sec")

    return BenchmarkResult(
        name=name,
        samples_per_sec=avg_samples_per_sec,
        total_samples=total_samples,
        elapsed_sec=total_elapsed,
        num_epochs=num_epochs,
        warmup_epochs=num_warmup,
        per_epoch_results=warmup_results + epoch_results,
        metadata=metadata or {},
    )


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format results as a markdown table."""
    lines = [
        "| Benchmark | Samples/sec | Epoch Time (s) |",
        "|-----------|-------------|----------------|",
    ]

    for r in results:
        epoch_time = r.total_samples / r.samples_per_sec if r.samples_per_sec > 0 else 0
        lines.append(f"| {r.name} | {r.samples_per_sec:,.0f} | {epoch_time:.2f} |")

    return "\n".join(lines)


def save_results(
    results: list[BenchmarkResult],
    machine_info: MachineInfo,
    output_path: str | Path,
    benchmark_name: str,
) -> None:
    """Save benchmark results to a JSON file."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": benchmark_name,
        "machine": machine_info.to_dict(),
        "results": [
            {
                "name": r.name,
                "samples_per_sec": r.samples_per_sec,
                "total_samples": r.total_samples,
                "elapsed_sec": r.elapsed_sec,
                "num_epochs": r.num_epochs,
                "warmup_epochs": r.warmup_epochs,
                "metadata": r.metadata,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
