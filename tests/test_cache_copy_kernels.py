"""Cache read kernels: identical bytes on both paths, and no slow copy in the serial one.

numba 0.67 lowers `dest[i, :n] = src[p:p + n]` in a non-parallel njit function to a copy ~20x
slower than a plain loop (1.8 vs ~45 GB/s), which throttled the loader's prefetch thread
(use_threading=True, parallel=False reads) 10-20x in 0.9.4. Uses the real imagenet10 cache.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from slipstream.cache import (
    OptimizedCache,
    _load_variable_batch_parallel,
    _load_variable_batch_sequential,
)

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-{fmt}-val"


def _storage(fmt):
    path = Path(str(CACHE).format(fmt=fmt))
    if not path.exists():
        pytest.skip(f"{path} not present")
    cache = OptimizedCache.load(path, verbose=False)
    return cache, cache.fields["image"]


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_serial_and_parallel_reads_identical(fmt):
    cache, st = _storage(fmt)
    idx = np.random.default_rng(0).permutation(cache.num_samples)[:64].astype(np.int64)
    out = []
    for par in (False, True):
        dest = np.zeros((len(idx), st.max_size), np.uint8)
        sizes = np.zeros(len(idx), np.uint64)
        st.load_batch_into(idx, dest, sizes, np.zeros(len(idx), np.uint32), np.zeros(len(idx), np.uint32),
                           parallel=par)
        out.append((dest, sizes))
    assert np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][0], out[1][0])
    for i, s in enumerate(out[0][1]):                            # bytes match the mmap'd records
        p = int(st._metadata[idx[i]]["data_ptr"])
        assert np.array_equal(out[0][0][i, : int(s)], st._data_array[p: p + int(s)])


def test_serial_kernel_copies_at_memcpy_speed():
    cache, st = _storage("yuv420")
    idx = np.random.default_rng(1).permutation(cache.num_samples)[:256].astype(np.int64)
    dest = np.zeros((len(idx), st.max_size), np.uint8)
    sizes = np.zeros(len(idx), np.uint64)
    meta, data = st._metadata, st._data_array
    _load_variable_batch_parallel(idx, meta, data, dest, sizes)     # warm pages + JIT
    _load_variable_batch_sequential(idx, meta, data, dest, sizes)

    def best(fn, reps=5):
        t = []
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            t.append(time.perf_counter() - s)
        return min(t)

    ptrs = [(int(meta[i]["data_ptr"]), int(meta[i]["data_size"])) for i in idx]

    def numpy_copy():                                                 # single-threaded memcpy reference
        for k, (p, n) in enumerate(ptrs):
            dest[k, :n] = data[p:p + n]

    kernel = best(lambda: _load_variable_batch_sequential(idx, meta, data, dest, sizes))
    ref = best(numpy_copy)
    assert kernel < 4 * ref, f"serial kernel {kernel * 1e3:.1f} ms vs numpy copy {ref * 1e3:.1f} ms"
