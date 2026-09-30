"""0.10.0 cache integrity for shared read-only caches (e.g. one lab copy on netscratch).

- file_sha256 in the manifest + check_integrity(deep=True) + OptimizedCache.add_hashes / `slipstream hash`
- manifest.json written last and atomically
- on_invalid_cache="raise" (and automatic "raise" when the dataset reads the cache directly):
  never delete or rebuild; a manifest-less directory with data files is never rebuilt in place

Every test works on a copy of the real imagenet10 val cache (35 MB) under ~/.slipstream.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from slipstream.cache import CacheIntegrityError, OptimizedCache, write_manifest_atomic

SRC = Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val"


@pytest.fixture
def cache(tmp_path):
    if not SRC.exists():
        pytest.skip(f"{SRC} not present")
    dst = tmp_path / "cache"
    shutil.copytree(SRC, dst)
    return dst


def _manifest(d):
    return json.loads((d / "manifest.json").read_text())


def _flip_byte(path: Path, offset: int = 1000):
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


def test_deep_check_needs_hashes(cache):
    assert OptimizedCache.check_integrity(cache) == (True, [])
    ok, problems = OptimizedCache.check_integrity(cache, deep=True)
    assert not ok and any("no file_sha256" in p for p in problems)


def test_add_hashes_then_deep_check_catches_same_size_corruption(cache):
    hashes = OptimizedCache.add_hashes(cache)
    m = _manifest(cache)
    assert set(m["file_sha256"]) == set(m["file_sizes"]) == set(hashes)          # same files as file_sizes
    assert all(len(h) == 64 and h == h.lower() for h in hashes.values())
    assert OptimizedCache.check_integrity(cache, deep=True) == (True, [])
    _flip_byte(cache / "image.bin")                                               # size unchanged
    assert OptimizedCache.check_integrity(cache)[0]                               # cheap check can't see it
    ok, problems = OptimizedCache.check_integrity(cache, deep=True)
    assert not ok and problems == ["sha256 mismatch: image.bin"]


def test_add_hashes_refuses_a_damaged_copy(cache):
    with open(cache / "path.bin", "ab") as f:
        f.write(b"x")
    with pytest.raises(CacheIntegrityError):
        OptimizedCache.add_hashes(cache)
    assert "file_sha256" not in _manifest(cache)


def test_manifest_write_is_atomic(cache):
    m = _manifest(cache)
    m["marker"] = 1
    write_manifest_atomic(cache, m)
    assert _manifest(cache)["marker"] == 1
    assert not [p for p in cache.iterdir() if p.name.startswith(".manifest.json.tmp")]


def test_build_records_hashes(cache, tmp_path):
    from slipstream import SlipstreamDataset

    out = tmp_path / "rebuilt"
    OptimizedCache.build(SlipstreamDataset(input_dir=str(cache)), out, verbose=False)
    m = _manifest(out)
    assert set(m["file_sha256"]) == set(m["file_sizes"])
    assert OptimizedCache.check_integrity(out, deep=True) == (True, [])


def test_cli_hash(cache):
    from slipstream.cli import main

    assert main(["hash", str(cache)]) == 0
    assert OptimizedCache.check_integrity(cache, deep=True)[0]


def _loader(dataset, **kw):
    from slipstream.loader import SlipstreamLoader

    return SlipstreamLoader(dataset, batch_size=16, verbose=False, **kw)


def test_raise_mode_never_deletes(cache):
    (cache / "path.bin").write_bytes(b"truncated")
    before = sorted(p.name for p in cache.iterdir())
    ds = SimpleNamespace(cache_path=cache, remote_dir=None)          # a source the loader could rebuild from
    with pytest.raises(CacheIntegrityError, match="size mismatch: path.bin"):
        _loader(ds, on_invalid_cache="raise")
    assert sorted(p.name for p in cache.iterdir()) == before


def test_reading_the_cache_directly_always_raises(cache):
    from slipstream import SlipstreamDataset

    ds = SlipstreamDataset(input_dir=str(cache))
    (cache / "path.bin").write_bytes(b"truncated")
    with pytest.raises(CacheIntegrityError, match="reads this cache directly"):
        _loader(ds)                                                   # default "rebuild" is overridden
    assert (cache / "image.bin").exists() and (cache / "manifest.json").exists()
    with pytest.raises(ValueError, match="no source to rebuild"):
        _loader(ds, force_rebuild=True)


def test_missing_manifest_with_data_is_never_rebuilt(cache):
    from slipstream import SlipstreamDataset

    (cache / "manifest.json").unlink()
    before = sorted(p.name for p in cache.iterdir())
    with pytest.raises(CacheIntegrityError, match="manifest.json missing"):
        _loader(SimpleNamespace(cache_path=cache, remote_dir=None))    # even in "rebuild" mode
    with pytest.raises(CacheIntegrityError, match="no manifest.json"):
        SlipstreamDataset(input_dir=str(cache))
    assert sorted(p.name for p in cache.iterdir()) == before


def test_invalid_mode_value(cache):
    with pytest.raises(ValueError, match="on_invalid_cache"):
        _loader(SimpleNamespace(cache_path=cache, remote_dir=None), on_invalid_cache="ignore")


def test_empty_directory_is_reported_as_purged_cache(tmp_path):
    from slipstream import SlipstreamDataset

    empty = tmp_path / "purged"
    empty.mkdir()
    (empty / ".hidden").write_text("")                              # dotfiles don't count
    with pytest.raises(CacheIntegrityError, match="empty directory"):
        SlipstreamDataset(input_dir=str(empty))
