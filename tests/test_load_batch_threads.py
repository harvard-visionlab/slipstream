"""ImageBytesStorage.load_batch from many threads (0.11.1).

Before 0.11.1 every call returned views into ONE per-field scratch buffer, so concurrent calls
(e.g. a ThreadPoolExecutor reading video blobs) overwrote each other's bytes. Uses the real
imagenet10 val cache under ~/.slipstream.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from slipstream.cache import OptimizedCache

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val"


@pytest.fixture(scope="module")
def field():
    if not CACHE.exists():
        pytest.skip(f"{CACHE} not present")
    return OptimizedCache.load(CACHE, verbose=False).fields["image"]


def _truth(f, i):
    p, n = int(f._metadata[i]["data_ptr"]), int(f._metadata[i]["data_size"])
    return bytes(f._data_array[p:p + n])


def test_concurrent_load_batch_returns_each_records_bytes(field):
    def blob(i):
        out = field.load_batch(np.array([i]), parallel=False)
        return bytes(out["data"][0][: int(out["sizes"][0])])

    idx = [i % 500 for i in range(3000)]
    with ThreadPoolExecutor(48) as ex:
        got = list(ex.map(blob, idx))
    assert all(g == _truth(field, i) for g, i in zip(got, idx))


def test_concurrent_batches(field):
    rng = np.random.default_rng(0)
    batches = [rng.integers(0, 500, 16) for _ in range(200)]

    def load(b):
        out = field.load_batch(b, parallel=False)
        return [bytes(out["data"][k][: int(out["sizes"][k])]) for k in range(len(b))]

    with ThreadPoolExecutor(16) as ex:
        for b, rows in zip(batches, ex.map(load, batches)):
            assert rows == [_truth(field, int(i)) for i in b]


def test_storage_pickles_without_thread_local(field):
    clone = pickle.loads(pickle.dumps(field))
    out = clone.load_batch(np.array([3]), parallel=False)
    assert bytes(out["data"][0][: int(out["sizes"][0])]) == _truth(field, 3)
