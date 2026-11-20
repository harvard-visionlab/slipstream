"""0.18.0: SlipstreamLoader(clone=True) yields batches that own their memory.

Raw decoder outputs live in reused buffers (image decoders: overwritten on the next batch), so kept batches
went stale silently (tjepa-redux, 2026-11-19). Real data: the imagenet10 val caches (skipped when absent).
"""

import numpy as np
import pytest
import torch

from slipstream.decoders import DecodeMultiRandomResizedCrop, DecodeRandomResizedCrop
from tests.test_ssl_params import _cache


def _collect(fmt, pipe, clone, threaded=True, n=6):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    path = _cache(fmt)
    if path is None:
        pytest.skip(f"no real {fmt} val cache")
    loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=64, shuffle=False, verbose=False,
                              exclude_fields=["path"], image_format=fmt, pipelines={"image": pipe},
                              use_threading=threaded, clone=clone)
    kept, snaps = [], []
    try:
        for i, b in enumerate(loader):
            kept.append(b)
            snaps.append({k: (v.clone() if isinstance(v, torch.Tensor) else np.copy(v) if isinstance(v, np.ndarray)
                              else v) for k, v in b.items()})
            if i + 1 == n:
                break
    finally:
        loader.shutdown()
    return kept, snaps


def _img(b, key):
    return torch.as_tensor(b[key])


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
@pytest.mark.parametrize("threaded", [True, False])
def test_clone_keeps_batches_intact(fmt, threaded):
    kept, snaps = _collect(fmt, [DecodeRandomResizedCrop(size=64, seed=1)], clone=True, threaded=threaded)
    for b, s in zip(kept, snaps):
        assert torch.equal(_img(b, "image"), _img(s, "image"))
        assert torch.equal(torch.as_tensor(b["label"]), torch.as_tensor(s["label"]))
    # same content as the default loader, batch by batch (cloning changes ownership, not data)
    ref, ref_snaps = _collect(fmt, [DecodeRandomResizedCrop(size=64, seed=1)], clone=False, threaded=threaded)
    for b, s in zip(kept, ref_snaps):
        assert torch.equal(_img(b, "image"), _img(s, "image"))


def test_default_aliases_reused_buffers():
    """Documents why clone exists: without it a kept raw-decoder batch is overwritten by later batches."""
    kept, snaps = _collect("jpeg", [DecodeRandomResizedCrop(size=64, seed=1)], clone=False)
    assert any(not torch.equal(_img(b, "image"), _img(s, "image")) for b, s in zip(kept, snaps))


def test_clone_nested_views():
    kept, snaps = _collect("jpeg", [DecodeMultiRandomResizedCrop({"a": dict(size=64, seed=1),
                                                                  "b": dict(size=48, seed=2)})], clone=True)
    for b, s in zip(kept, snaps):
        for k in ("a", "b"):
            assert torch.equal(_img(b, k), _img(s, k))
