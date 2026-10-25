"""SlipstreamDataset item reads inside forked torch DataLoader workers (0.11.3).

Before 0.11.3 each ds[i] ran load_batch(parallel=True), starting numba's parallel pool in the parent;
forked workers then aborted (GNU OpenMP: "fork() called from a process already using GNU OpenMP") or
hung/crawled (other layers). Runs in a fresh interpreter so other tests can't have started the pool.
Uses a real jpeg val cache (skipped when absent).
"""

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

CANDIDATES = [Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val",
              Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent")) / "imagenet1k-s256_l512-jpeg-val"]

SCRIPT = r"""
import sys, numba, torch
from slipstream import SlipstreamDataset
ds = SlipstreamDataset(input_dir=sys.argv[1])
want = [bytes(ds[i]["image"]) for i in range(64)]                  # parent reads first
try:
    numba.threading_layer()
    sys.exit("parent started numba's parallel pool")
except ValueError:
    pass
class Raw(torch.utils.data.Dataset):
    def __len__(self): return 64
    def __getitem__(self, i): return torch.frombuffer(bytearray(ds[i]["image"]), dtype=torch.uint8)
dl = torch.utils.data.DataLoader(Raw(), batch_size=None, num_workers=2, multiprocessing_context="fork", timeout=60)
assert [bytes(t.numpy()) for t in dl] == want
print("OK")
"""


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_item_reads_in_forked_workers():
    cache = next((p for p in CANDIDATES if p.exists()), None)
    if cache is None:
        pytest.skip("no jpeg val cache present")
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(cache)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout[-2000:] + r.stderr[-2000:]
