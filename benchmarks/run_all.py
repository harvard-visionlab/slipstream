#!/usr/bin/env python3
"""Run all benchmarks and generate a summary.

Usage:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py --quick  # Fewer epochs for quick test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.utils import get_machine_info


def run_benchmark(script: str, extra_args: list[str] | None = None) -> int:
    """Run a benchmark script and return exit code."""
    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def collate_results(results_dir: Path, hostname: str) -> dict:
    """Collate all JSON results for a hostname."""
    results = {}

    for pattern in ["raw_io", "decode", "loader"]:
        json_path = results_dir / f"{pattern}_{hostname}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[pattern] = json.load(f)

    return results


def format_markdown_results(results: dict, machine_info) -> str:
    """Format results as markdown for results.md."""
    lines = [
        f"### {machine_info.machine_name}",
        "",
        "**Machine Info:**",
        f"- Platform: {machine_info.platform} {machine_info.platform_version}",
        f"- CPU: {machine_info.cpu_model}",
        f"- Cores: {machine_info.cpu_cores_physical} physical, {machine_info.cpu_cores_logical} logical",
        f"- RAM: {machine_info.ram_gb:.1f} GB",
    ]

    if machine_info.cuda_available:
        lines.append(f"- GPU: {machine_info.gpu_name} ({machine_info.gpu_memory_gb:.1f} GB)")
    else:
        lines.append("- GPU: None")

    if machine_info.drive_type:
        lines.append(f"- Drive: {machine_info.drive_type}")

    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # Raw I/O results
    if "raw_io" in results:
        lines.append("**Raw I/O:**")
        lines.append("| Benchmark | Samples/sec |")
        lines.append("|-----------|-------------|")

        raw_results = {r["name"]: r["samples_per_sec"] for r in results["raw_io"]["results"]}

        for name, rate in raw_results.items():
            lines.append(f"| {name} | {rate:,.0f} |")

        # Calculate speedup
        slipstream = raw_results.get("SlipstreamLoader (raw, no pipelines)", 0)
        streaming = [v for k, v in raw_results.items() if "StreamingDataLoader" in k]
        if streaming and streaming[0] > 0:
            speedup = slipstream / streaming[0]
            lines.append(f"| **Speedup** | **{speedup:.1f}x** |")

        lines.append("")

    # Decode results
    if "decode" in results:
        lines.append("**Decode:**")
        lines.append("| Benchmark | Samples/sec |")
        lines.append("|-----------|-------------|")

        for r in results["decode"]["results"]:
            lines.append(f"| {r['name']} | {r['samples_per_sec']:,.0f} |")

        lines.append("")

    # Loader results
    if "loader" in results:
        lines.append("**Full Pipeline:**")
        lines.append("| Benchmark | Samples/sec |")
        lines.append("|-----------|-------------|")

        for r in results["loader"]["results"]:
            lines.append(f"| {r['name']} | {r['samples_per_sec']:,.0f} |")

        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results (e.g., 'nolan-25')")
    parser.add_argument("--skip-raw-io", action="store_true", help="Skip raw I/O benchmark")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode benchmark")
    parser.add_argument("--skip-loader", action="store_true", help="Skip loader benchmark")
    args = parser.parse_args()

    # Get machine info
    machine_info = get_machine_info(args.machine_name)
    print(machine_info)

    machine_name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
    benchmarks_dir = Path(__file__).parent
    results_dir = benchmarks_dir / "results"

    # Build common args
    extra_args = []
    if args.quick:
        extra_args.extend(["--epochs", "1", "--warmup", "1"])
    if args.cache_dir:
        extra_args.extend(["--cache-dir", args.cache_dir])
    if args.machine_name:
        extra_args.extend(["--machine-name", args.machine_name])

    # Run benchmarks
    exit_codes = []

    if not args.skip_raw_io:
        code = run_benchmark(str(benchmarks_dir / "benchmark_raw_io.py"), extra_args)
        exit_codes.append(("raw_io", code))

    if not args.skip_decode:
        code = run_benchmark(str(benchmarks_dir / "benchmark_decode.py"), extra_args)
        exit_codes.append(("decode", code))

    if not args.skip_loader:
        code = run_benchmark(str(benchmarks_dir / "benchmark_loader.py"), extra_args)
        exit_codes.append(("loader", code))

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for name, code in exit_codes:
        status = "✓ PASS" if code == 0 else "✗ FAIL"
        print(f"  {name}: {status}")

    # Collate results
    results = collate_results(results_dir, machine_name)

    if results:
        print("\n" + "=" * 60)
        print("RESULTS (copy to results.md)")
        print("=" * 60)
        print(format_markdown_results(results, machine_info))

    # Check for failures
    if any(code != 0 for _, code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
