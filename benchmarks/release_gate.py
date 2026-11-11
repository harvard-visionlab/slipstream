"""Release gate: loader throughput must not regress between tags.

Run on the reference machine (machina) from a clean checkout of the commit to be tagged:

    uv run python -m benchmarks.release_gate [--append-md] [--wait 30]

1. Refuses to start unless the machine is quiet (1-min load <= --max-load, every GPU <= 10 % busy);
   ``--wait N`` polls for up to N minutes. The load before/after is recorded with the results.
2. Runs the ablation (benchmarks/loader_ablation.py: A-G, G@160/192, multicrop M/Mf) for jpeg and
   yuv420 as full passes over imagenet1k val (50k): per step the cache is evicted from the page cache,
   then --epochs (default 3) epochs run; epoch 1 is cold, the best later epoch is warm.
3. Saves ``benchmarks/results/gate-full/<host>/v<version>.json`` and compares cold and warm for every
   step with the latest earlier version saved for this host. FAIL (exit 1) if either drops more than
   --tolerance (default 10 %), or if threaded prefetch (B) is below 75 % of simple (A) -- the 0.9.x
   regression. (``benchmarks/results/gate/`` holds the older warm-only, partial-epoch results.)
4. Times DecodeVideoWindow augmentation, aug_mode="frame" vs "clip" (benchmarks/video_aug_modes.py), on
   each device: FAIL if frame mode is more than --tolerance slower than clip mode (George 2026-11-10).
5. ``--append-md`` appends the table to BENCHMARKS.md ("Release gate" section).

Commit the results file (and BENCHMARKS.md) with the release.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from datetime import date
from pathlib import Path

from benchmarks.loader_ablation import MULTICROP_STEPS, STEPS, run_epochs
from benchmarks.video_aug_modes import CHAIN as VIDEO_CHAIN
from benchmarks.video_aug_modes import run as run_video_aug

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "gate-full"
THREAD_FLOOR = 0.75          # B (threaded) must reach this fraction of A (simple)


def _gpu_busy() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def _quiet(max_load: float) -> tuple[bool, str]:
    load = os.getloadavg()[0]
    gpus = _gpu_busy()
    ok = load <= max_load and all(g <= 10 for g in gpus)
    return ok, f"load1={load:.1f} gpu%={gpus or '-'}"


def _vkey(name: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", name)[:3])


def _meta(host: str | None) -> dict:
    import numba
    import numpy
    import torch

    from slipstream.version import __version__

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"version": __version__, "git": sha + ("-dirty" if dirty else ""), "date": date.today().isoformat(),
            "host": host or socket.gethostname().split(".")[0], "cpus": cpus, "numba": numba.__version__,
            "numpy": numpy.__version__, "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _baseline(host_dir: Path, version: str) -> tuple[str, dict] | None:
    earlier = [p for p in host_dir.glob("v*.json") if _vkey(p.stem) < _vkey(version)]
    if not earlier:
        return None
    p = max(earlier, key=lambda q: _vkey(q.stem))
    return p.stem, json.loads(p.read_text())


def _table(res: dict, base: dict | None, tol: float) -> tuple[list[str], list[str]]:
    lines, fails = [], []
    for fmt, steps in res["results"].items():
        b = (base or {}).get("results", {}).get(fmt, {})
        extra = [(k, f"+ RandomHorizontalFlip @ {k[2:]}px") for k in steps if k.startswith("G@")]
        labels = [(k, lbl) for k, lbl, *_ in STEPS] + extra + [(k, lbl) for k, lbl, _ in MULTICROP_STEPS]
        for key, label in labels:
            if key not in steps:
                continue
            cells = []
            for kind in ("cold", "warm"):
                now, was = steps[key][kind], b.get(key, {}).get(kind)
                ratio = now / was if was else None
                bad = ratio is not None and ratio < 1 - tol
                if bad:
                    fails.append(f"{fmt} {key} {kind} ({label}): {now:,.0f} vs {was:,.0f} img/s ({ratio:.2f}x)")
                cells += [f"{was:,.0f}" if was else "—", f"{now:,.0f}",
                          ("—" if ratio is None else f"{ratio:.2f}x") + (" ✗" if bad else "")]
            lines.append(f"| {fmt} | {key} | {label} | " + " | ".join(cells) + " |")
        if "A" in steps and "B" in steps and steps["B"]["warm"] < THREAD_FLOOR * steps["A"]["warm"]:
            fails.append(f"{fmt}: threaded B {steps['B']['warm']:,.0f} < {THREAD_FLOOR:.0%} of simple A "
                         f"{steps['A']['warm']:,.0f}")
    return lines, fails


def main() -> int:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path,
                    default=Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "~/.slipstream")).expanduser())
    ap.add_argument("--host", default=os.environ.get("SLIPSTREAM_BENCH_HOST"),
                    help="results name for this machine (in a container the hostname is the container id)")
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--wait", type=float, default=0, help="minutes to wait for a quiet machine")
    ap.add_argument("--epochs", type=int, default=3, help="full passes per step (>= 2: one cold + warm)")
    ap.add_argument("--force", action="store_true", help="run even if the machine is busy (results marked)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--append-md", action="store_true")
    a = ap.parse_args()

    deadline = time.time() + a.wait * 60
    ok, state = _quiet(a.max_load)
    while not ok and time.time() < deadline:
        time.sleep(30)
        ok, state = _quiet(a.max_load)
    if not ok and not a.force:
        print(f"gate: machine busy ({state}); not running. Use --wait or --force.")
        return 2

    meta = _meta(a.host)
    meta["load_before"] = state
    print(f"gate: slipstream {meta['version']} ({meta['git']}) on {meta['host']}, {meta['cpus']} CPUs, "
          f"numba {meta['numba']}, torch {meta['torch']}, {state}")
    results = {f: run_epochs(a.cache_root, f, epochs=a.epochs) for f in ("jpeg", "yuv420")}
    import torch
    video = {d: run_video_aug(a.cache_root, d, iters=100 if d == "cuda" else 30) for d in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]}
    meta["load_after"] = _quiet(a.max_load)[1]
    meta["forced_busy"] = not ok
    res = {"meta": meta, "params": {"epochs": a.epochs, "batch_size": 512, "samples": "imagenet1k val, full"},
           "results": results, "video_aug": video}

    host_dir = RESULTS / meta["host"]
    base = _baseline(host_dir, meta["version"])
    lines, fails = _table(res, base[1] if base else None, a.tolerance)
    was = base[0] if base else "baseline"
    header = [f"| fmt | step | pipeline | cold {was} | cold v{meta['version']} | ratio "
              f"| warm {was} | warm v{meta['version']} | ratio |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    vlines = [f"DecodeVideoWindow aug_mode, {VIDEO_CHAIN} (frames/s, transforms only):", "",
              "| device | B x T | clip | frame | frame/clip |", "| --- | --- | ---: | ---: | ---: |"]
    for dev, r in video.items():
        bad = r["ratio"] < 1 - a.tolerance
        if bad:
            fails.append(f"video aug {dev}: frame {r['frame']:,.0f} vs clip {r['clip']:,.0f} frames/s ({r['ratio']:.2f}x)")
        vlines.append(f"| {dev} | {r['B']} x {r['T']} | {r['clip']:,.0f} | {r['frame']:,.0f} | {r['ratio']:.3f}"
                      + (" ✗" if bad else "") + " |")
    print("\n".join(header + lines + [""] + vlines))
    if not a.no_save:
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / f"v{meta['version']}.json").write_text(json.dumps(res, indent=2) + "\n")
    if a.append_md:
        md = ROOT / "BENCHMARKS.md"
        text = md.read_text()
        if "## Release gate (full passes)" not in text:
            text += ("\n## Release gate (full passes)\n\nimagenet1k val (50k), batch 512; per step the cache "
                     "is evicted from the page cache, then 3 full epochs: cold = epoch 1, warm = best of epochs 2-3 "
                     "(img/s, loader start-up included). `benchmarks/release_gate.py` "
                     "(steps: `benchmarks/loader_ablation.py`).\n")
        text += (f"\n### v{meta['version']} ({meta['git']}), {meta['host']}, {meta['date']}\n\n"
                 f"{meta['cpus']} CPUs, numba {meta['numba']}, torch {meta['torch']}, {meta['cuda']}; "
                 f"before: {meta['load_before']}, after: {meta['load_after']}\n\n" + "\n".join(header + lines + [""] + vlines) + "\n")
        md.write_text(text)
    if fails:
        print("GATE FAIL:\n  " + "\n  ".join(fails))
        return 1
    print("GATE PASS" + ("" if base else " (no earlier baseline on this host; this run is the baseline)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
