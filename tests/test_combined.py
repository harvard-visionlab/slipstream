"""0.19.0 CombinedDataset: several real caches mixed in one loader.

Sources: imagenet10 val (500 records, s256) and imagenet100 val at s292 (5000 records): different images,
sizes and label spaces, so a read from the wrong source cannot pass. Every combined sample is compared with
the same record read by a single-source loader.
"""

import collections
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream import CombinedDataset, SlipstreamDataset, SlipstreamLoader
from slipstream.decoders import DecodeCenterCrop

A, B = "imagenet10-s256_l512-{fmt}-val", "imagenet100-s292_l584-{fmt}-val"
B2 = "imagenet100-s256_l512-{fmt}-val"


def _dir(name, fmt="jpeg"):
    for root in (Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent")), Path.home() / ".slipstream"):
        p = root / name.format(fmt=fmt)
        if p.exists():
            return p
    pytest.skip(f"no real cache {name.format(fmt=fmt)}")


def _ds(name, fmt="jpeg"):
    return SlipstreamDataset(input_dir=str(_dir(name, fmt)))


def _loader(ds, fmt="jpeg", **kw):
    kw.setdefault("batch_size", 64)
    kw.setdefault("seed", 0)
    kw.setdefault("drop_last", False)
    return SlipstreamLoader(ds, verbose=False, exclude_fields=["path"], image_format=fmt,
                            pipelines={"image": [DecodeCenterCrop(48)]}, **kw)


def _all(loader, keys=("image", "label")):
    out = collections.defaultdict(list)
    try:
        for b in loader:
            for k in keys:
                out[k].append(torch.as_tensor(np.asarray(b[k])).clone())
    finally:
        loader.shutdown()
    return {k: torch.cat(v) for k, v in out.items()}


def _reference(name, fmt="jpeg"):
    """Every record of one source in record order (single-source loader)."""
    return _all(_loader(_ds(name, fmt), fmt, shuffle=False))


def _combined(fmt="jpeg", **kw):
    kw.setdefault("labels", "disjoint")
    kw.setdefault("num_classes", [10, 100])
    return CombinedDataset([_ds(A, fmt), _ds(B, fmt)], **kw)


@pytest.mark.parametrize("fmt,b", [("jpeg", B), ("yuv420", B2)])
def test_samples_match_single_source_reads(fmt, b):
    a_name = A if fmt == "jpeg" else "imagenet100-s292_l584-{fmt}-val"
    if fmt == "yuv420":                                 # yuv420 val caches present on every test host
        ds = CombinedDataset([_ds(a_name, fmt), _ds(b, fmt)], weights=[0.3, 0.7], labels="shared",
                             epoch_length=3000)
    else:
        ds = _combined(weights=[0.3, 0.7], epoch_length=3000)
    refs = [_reference(a_name, fmt), _reference(b, fmt)]
    L = _loader(ds, fmt)
    got = _all(L, keys=("image", "label", "source_label", "source_id", "_indices"))
    src, local = L.cache.locate(got["_indices"].numpy())
    assert torch.equal(got["source_id"], torch.as_tensor(src))
    assert len(src) == 3000 and np.bincount(src).tolist() == [900, 2100]
    off = [0, 10] if fmt == "jpeg" else [0, 0]
    for i in range(len(src)):
        r = refs[src[i]]
        assert torch.equal(got["image"][i], r["image"][local[i]]), i
        assert int(got["source_label"][i]) == int(r["label"][local[i]])
        assert int(got["label"][i]) == int(r["label"][local[i]]) + off[src[i]]


def test_epoch_modes_counts_repeats_and_coverage():
    sizes = [500, 5000]
    short = _combined(weights=[0.25, 0.75], epoch_length="shortest")
    assert short.epoch_size(sizes) == 2000 and short.counts(sizes, 2000).tolist() == [500, 1500]
    order = short.epoch_order(sizes, 0, 0, True)
    assert len(np.unique(order)) == len(order)                         # shortest: nothing repeats
    long = _combined(weights=[0.5, 0.5], epoch_length="longest")
    assert long.epoch_size(sizes) == 10000
    order = long.epoch_order(sizes, 0, 0, True)
    a, b = order[order < 500], order[order >= 500]
    assert len(a) == len(b) == 5000
    assert np.array_equal(np.sort(b), np.arange(500, 5500))           # the large source exactly once
    assert np.all(np.bincount(a, minlength=500) == 10)                 # the small one: 10 full passes
    assert not np.array_equal(a[:500], a[500:1000])                    # each pass reshuffled
    default = _combined()                                              # weights=None: every record once
    o = default.epoch_order(sizes, 0, 0, True)
    assert np.array_equal(np.sort(o), np.arange(5500))
    plain = default.epoch_order(sizes, 0, 0, False)                    # shuffle=False: record order, interleaved
    assert np.array_equal(plain[plain < 500], np.arange(500))
    assert np.array_equal(plain[plain >= 500], np.arange(500, 5500))
    assert abs(np.flatnonzero(plain < 500).mean() - 2750) < 30          # spread evenly


def test_order_is_a_function_of_seed_and_epoch_and_shards_cover_it():
    L = _loader(_combined(weights=[0.5, 0.5], epoch_length=1200), batch_size=50)
    try:
        e0, e1 = L._generate_indices(0), L._generate_indices(1)
        assert not np.array_equal(e0, e1)
        L2 = _loader(_combined(weights=[0.5, 0.5], epoch_length=1200), batch_size=50)
        assert np.array_equal(L2._generate_indices(1), e1)
        L2.shutdown()
        L.distributed, L.world_size = True, 4
        shards = []
        for r in range(4):
            L.rank = r
            shards.append(L._generate_indices(1))
        assert sorted(np.concatenate(shards).tolist()) == sorted(e1.tolist())
        assert all(len(s) == 300 for s in shards)
    finally:
        L.shutdown()


@pytest.mark.parametrize("drop_last", [True, False])
def test_len_matches_batches(drop_last):
    L = _loader(_combined(weights=[0.5, 0.5], epoch_length=1000), batch_size=64, drop_last=drop_last)
    try:
        n = sum(1 for _ in L)
    finally:
        L.shutdown()
    assert n == len(L) == (1000 // 64 if drop_last else -(-1000 // 64))


def test_rename_and_shared_labels():
    ren = {"label": "local_label", "in1000_label": "label"}
    ds = CombinedDataset([_ds(A), _ds(B)], labels="shared", rename_fields=[ren, ren], epoch_length=600)
    L = _loader(ds)
    got = _all(L, keys=("label", "source_label", "local_label", "_indices"))
    assert "in1000_label" not in L.cache.field_types
    src, local = L.cache.locate(got["_indices"].numpy())
    for s, name in enumerate((A, B)):
        ref = _all(SlipstreamLoader(_ds(name), batch_size=500, shuffle=False, drop_last=False, verbose=False,
                                    exclude_fields=["path", "image"]), keys=("in1000_label", "label"))
        m = src == s
        assert torch.equal(got["label"][m], ref["in1000_label"][local[m]])
        assert torch.equal(got["local_label"][m], ref["label"][local[m]])
    assert torch.equal(got["label"], got["source_label"])


def test_errors():
    with pytest.raises(ValueError, match="labels="):
        _loader(CombinedDataset([_ds(A), _ds(B)]))                    # shared label field, no labels=
    with pytest.raises(ValueError, match="num_classes"):
        CombinedDataset([_ds(A), _ds(B)], labels="disjoint")
    with pytest.raises(ValueError, match="outside num_classes"):
        _loader(_combined(num_classes=[10, 50]))                       # imagenet100 labels reach 99
    with pytest.raises(ValueError, match="rename_fields"):
        _loader(_combined(rename_fields=[{"nope": "x"}, None]))
    with pytest.raises(NotImplementedError, match="window"):
        _loader(_combined(), window=2)
    with pytest.raises(NotImplementedError, match="group"):
        _loader(_combined(), group=2, batch_size=64)
    with pytest.raises(ValueError, match="weights"):
        CombinedDataset([_ds(A), _ds(B)], weights=[1, 0], labels="shared")
