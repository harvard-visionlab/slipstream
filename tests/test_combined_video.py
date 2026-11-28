"""0.19.0 CombinedDataset with a raw-bytes video primary: procthor-routes val, objects + empty scenes.

Real caches (machina only; skipped elsewhere). Paired sources: same clip_id per record index, different scenes.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream import CombinedDataset, SlipstreamDataset, SlipstreamLoader

OBJ, EMPTY = "procthor-routes-objects-lossless-160x120-val", "procthor-routes-empty-lossless-160x120-val"


def _ds(name):
    root = Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent"))
    if not (root / name).exists():
        pytest.skip(f"no real cache {name}")
    return SlipstreamDataset(input_dir=str(root / name))


def _combined(**kw):
    kw.setdefault("weights", [0.5, 0.5])
    kw.setdefault("epoch_length", 96)
    return CombinedDataset([_ds(OBJ), _ds(EMPTY)], **kw)


def test_video_bytes_match_single_source_reads():
    ds = _combined()
    L = SlipstreamLoader(ds, batch_size=32, seed=0, drop_last=False, verbose=False,
                         exclude_fields=["positions", "headings"])
    blobs, idx, src_ids = [], [], []
    try:
        for b in L:
            v = b["video"]
            blobs += [bytes(v["data"][i, : int(v["sizes"][i])]) for i in range(len(v["sizes"]))]
            idx.append(np.asarray(b["_indices"]))
            src_ids.append(np.asarray(b["source_id"]))
    finally:
        L.shutdown()
    src, local = L.cache.locate(np.concatenate(idx))
    assert np.array_equal(src, np.concatenate(src_ids)) and np.bincount(src).tolist() == [48, 48]
    for s, name in enumerate((OBJ, EMPTY)):
        m = np.flatnonzero(src == s)
        ref = SlipstreamLoader(_ds(name), batch_size=len(m), shuffle=False, drop_last=False, verbose=False,
                               indices=local[m], exclude_fields=["positions", "headings"])
        try:
            rb = next(iter(ref))["video"]
        finally:
            ref.shutdown()
        for j, i in enumerate(m):       # ref reads local[m] in the given order (shuffle=False)
            assert blobs[i] == bytes(rb["data"][j, : int(rb["sizes"][j])]), (s, int(local[i]))


def test_video_decode_through_combined_loader():
    pytest.importorskip("torchcodec")
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401  (libtorchcodec must load)
    except Exception as exc:
        pytest.skip(f"torchcodec not usable here: {str(exc)[:120]}")
    from slipstream.decoders import DecodeVideoWindow

    def run():
        stage = DecodeVideoWindow(T=16, rate_hz=15.0, seed=0, num_workers=8)
        L = SlipstreamLoader(_combined(), batch_size=24, seed=0, drop_last=True, verbose=False,
                             pipelines={"video": [stage]})
        out = []
        try:
            for b in L:
                out.append((b["video"].clone(), b["source_id"].clone(), b["positions"].clone()))
        finally:
            L.shutdown()
        return out

    a, b = run(), run()
    assert len(a) == 4
    for (va, sa, pa), (vb, sb, pb) in zip(a, b):
        assert va.shape[:2] == (24, 16) and va.dtype == torch.uint8
        assert set(sa.tolist()) <= {0, 1} and pa.shape == (24, 125, 2)
        assert torch.equal(va, vb) and torch.equal(sa, sb)        # seed-determined order and windows
    assert {int(s) for _, sb, _ in a for s in sb} == {0, 1}
