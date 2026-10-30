"""store_identity / OptimizedCache.identity (0.13.0): moved from visionlab-datasets (cache_identity).

The value is frozen: evals pins identities computed by the datasets implementation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slipstream.identity import store_identity

FILES = {"image.bin": "ab" * 32, "label.npy": "cd" * 32}


def _manifest(tmp_path, **m):
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return tmp_path


def test_frozen_value_and_only_count_and_hashes_matter(tmp_path):
    want = "sha256:5bc40b6a6f54ca641b0bc9bbe79ea730fc0494e6cb3910b3ed9be440cd8d3547"
    assert store_identity(_manifest(tmp_path, num_samples=3, version="x", file_sha256=FILES)) == want
    other = tmp_path / "b"
    other.mkdir()
    # other keys, key order and paths do not enter the identity
    assert store_identity(_manifest(other, file_sha256=dict(reversed(FILES.items())), num_samples=3,
                                    fields={"image": {}}, version="y")) == want


def test_none_without_hashes(tmp_path):
    assert store_identity(_manifest(tmp_path, num_samples=3)) is None
    assert store_identity(_manifest(tmp_path, num_samples=3, file_sha256={})) is None


def test_real_store_matches_evals_pin():
    cache = Path.home() / ".slipstream" / "imagenet100-original-val"
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    from slipstream.cache import OptimizedCache

    want = "sha256:f358feb3953248ad50e67eac646d24f1ee14c1a010b9c0043c32aed600dee29b"
    assert store_identity(cache) == want
    assert OptimizedCache.load(cache, verbose=False).identity == want


def test_import_is_light():
    code = "import sys, slipstream.identity; print(any(m in sys.modules for m in ('torch', 'numba')))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "False"
