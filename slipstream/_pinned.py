"""Pinned output buffers for decoders followed by ``ToTorchImage`` on CUDA (0.18.0).

A decoder whose next pipeline stage is ``ToTorchImage(<cuda>)`` writes its output into pinned host memory
(the loader enables this; raw decoder output never reaches user code then). ``to_torch_image`` copies such
a buffer to the GPU asynchronously on a side stream and records an event; the decoder calls :func:`fence`
before writing its output buffers again, so a buffer is never overwritten while a copy still reads it.
Only memory allocated here takes the async path: user tensors, pinned or not, keep the synchronous copy.

Exit (0.18.1): torch's host allocator caches freed pinned blocks and never returns them before the process
ends, leaving gigabytes (a video ring) to driver teardown at exit. An atexit hook stops the loaders that use
pinned buffers, drops every holder's buffers, and returns the cached blocks with cudaFreeHost while the
process is still healthy; it prints one line (bytes released, seconds) to stderr.
"""

from __future__ import annotations

import atexit
import gc
import sys
import threading
import time
import weakref

import numpy as np
import torch

_lock = threading.RLock()                     # re-entrant: _forget can run from a GC pass inside a locked region
_ranges: list[tuple[int, int]] = []           # (start, end) of every pinned buffer allocated here
_pending: list = []                           # events of async copies that may still read those buffers
_streams: dict[int, torch.cuda.Stream] = {}   # side stream per CUDA device
async_copies = 0                              # count of async copies issued (tests check the path was taken)
_holders: weakref.WeakSet = weakref.WeakSet()   # objects with release_pinned() (video stages)
_loaders: weakref.WeakSet = weakref.WeakSet()   # loaders whose decoders write pinned buffers


def register_holder(obj) -> None:
    """`obj.release_pinned()` drops its pinned buffers; called at interpreter exit."""
    _holders.add(obj)


def register_loader(loader) -> None:
    """`loader.shutdown()` (stops prefetching, releases its stages' pinned buffers) runs at interpreter exit."""
    _loaders.add(loader)


def _rss_mb() -> float:
    """Resident set size in MB (Linux; pinned pages count, VmPin does not), or nan."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def empty_host_cache() -> None:
    """Return torch's cached (free) pinned host blocks to the driver."""
    fn = (getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
          or getattr(torch._C, "_host_emptyCache", None))
    if fn is not None:
        fn()


def release_all() -> None:
    """Stop registered loaders, drop all pinned buffers, wait for the GPU, free the cached pinned blocks."""
    if not (_loaders or _holders) or not torch.cuda.is_initialized():
        return
    t, rss = time.perf_counter(), _rss_mb()
    for ld in list(_loaders):
        try:
            ld.shutdown()
        except Exception:
            pass
    for h in list(_holders):
        try:
            h.release_pinned()
        except Exception:
            pass
    torch.cuda.synchronize()
    with _lock:
        _pending.clear()
        _streams.clear()
    gc.collect()
    empty_host_cache()
    print(f"slipstream: released pinned host memory at exit (RSS {rss:.0f} -> {_rss_mb():.0f} MB) in "
          f"{time.perf_counter() - t:.2f} s",
          file=sys.stderr, flush=True)


def _release_at_exit() -> None:
    try:
        release_all()
    except BaseException:                     # never block or fail interpreter exit
        pass


atexit.register(_release_at_exit)


def zeros(shape) -> np.ndarray:
    """A zeroed uint8 numpy array in pinned host memory (registered as slipstream-owned)."""
    t = torch.zeros(shape, dtype=torch.uint8, pin_memory=True)
    r = (t.data_ptr(), t.data_ptr() + t.numel())
    with _lock:
        _ranges.append(r)
    a = t.numpy()                              # its .base keeps the pinned storage alive
    weakref.finalize(a.base, _forget, r)       # freed buffer: its address range is no longer ours
    return a


def _forget(r) -> None:
    with _lock:
        if r in _ranges:
            _ranges.remove(r)


def owned(t: torch.Tensor) -> bool:
    """Whether `t` lies in a pinned buffer allocated by :func:`zeros`."""
    if t.device.type != "cpu":
        return False
    p, n = t.data_ptr(), t.numel() * t.element_size()
    with _lock:
        return any(s <= p and p + n <= e for s, e in _ranges)


def fence() -> None:
    """Block until every async copy out of slipstream's pinned buffers has finished."""
    with _lock:
        evs = list(_pending)
    for ev in evs:                            # wait first, remove after: a concurrent fence() in another thread
        ev.synchronize()                      # must never find the list empty while a copy still runs
    with _lock:
        for ev in evs:
            try:
                _pending.remove(ev)
            except ValueError:                # another thread's fence() already removed it
                pass


def to_device_async(t: torch.Tensor, device) -> torch.Tensor:
    """Copy a slipstream-owned pinned tensor to `device` on a side stream; the caller's current stream
    waits for it (GPU-side) and the allocator learns that stream uses the result."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with _lock:
        s = _streams.get(device.index)
        if s is None:
            s = _streams[device.index] = torch.cuda.Stream(device)
    with torch.cuda.stream(s):
        out = t.to(device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(s)
    global async_copies
    with _lock:
        _pending.append(ev)
        async_copies += 1
    cur = torch.cuda.current_stream(device)
    cur.wait_event(ev)
    out.record_stream(cur)
    return out
