"""Regressions found against 0.7.0 by the model-rearing bug hunt.

1. affine_transform / RandomRotate crashed on bf16 images (fp32 grid vs bf16 input).
2. DecodeMultiResizeCropEmbed with image_format="yuv420" had no
   ``decode_batch_resize_short_crop_long`` on the inner YUV decoder.
3. ``set_epoch`` did not reach DecodeMultiResizeCropEmbed's inner crop decoder
   nor its embed counter, so a resumed run replayed epoch 0's crops/placements.

The loader tests use the real imagenet10 val caches under ~/.slipstream and skip
when they are not present.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slipstream.transforms.functional import affine_transform
from slipstream.transforms.geometric import RandomRotate

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-{fmt}-val"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kw", [{}, {"sz": 16}, {"align_corners": True}])
def test_affine_transform_half_dtypes(dtype, kw):
    x = torch.rand(2, 3, 32, 32)
    mat = torch.eye(3)[None].repeat(2, 1, 1)
    mat[:, 0, 1] = 0.3
    mat = mat[:, :2]
    ref = affine_transform(x, mat.clone(), **kw)
    out = affine_transform(x.to(dtype), mat.clone(), **kw)
    assert out.dtype == dtype
    assert (out.float() - ref).abs().max() < 0.02


@pytest.mark.parametrize("kw", [dict(max_deg=45), dict(angles=[90])])
def test_random_rotate_bf16(kw):
    x = torch.rand(4, 3, 32, 32)
    ref = RandomRotate(p=1.0, seed=0, **kw)(x)
    out = RandomRotate(p=1.0, seed=0, **kw)(x.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert (out.float() - ref).abs().mean() < 0.01     # bf16 matrix: a few edge pixels move


def _embed_loader(fmt, size):
    from slipstream.decoders.multicrop import DecodeMultiResizeCropEmbed
    from slipstream.loader import SlipstreamLoader

    cache = Path(str(CACHE).format(fmt=fmt))
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    dec = DecodeMultiResizeCropEmbed(
        {"a": dict(size=size, seed=1)}, canvas_size=160,
        embed_x_range=(0, 1), embed_y_range=(0, 1), embed_seed=7,
    )
    ds = SimpleNamespace(cache_path=cache, remote_dir=None)
    loader = SlipstreamLoader(ds, batch_size=32, shuffle=True, seed=0, image_format=fmt,
                              verbose=False, pipelines={"image": [dec]})
    return loader, dec


def _epoch(loader, epoch, n=3):
    loader.set_epoch(epoch)
    out = []
    for i, b in enumerate(loader):
        if i < n:
            out.append((np.asarray(b["a"]).copy(), np.asarray(b["_indices"]).copy()))
    return out


@pytest.mark.parametrize("size", [(64, 128), (64, 64)])   # per-image sizes / fixed size
def test_embed_yuv420_decodes_like_jpeg(size):
    jl, _ = _embed_loader("jpeg", size)
    yl, _ = _embed_loader("yuv420", size)
    for (j, ji), (y, yi) in zip(_epoch(jl, 0), _epoch(yl, 0)):
        assert np.array_equal(ji, yi)
        assert np.array_equal(j[..., 3], y[..., 3])            # same crop sizes + placements
        assert np.abs(j[..., :3].astype(int) - y[..., :3].astype(int)).mean() < 3


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_embed_set_epoch_resumes(fmt):
    fresh_loader, dec = _embed_loader(fmt, (64, 128))
    _epoch(fresh_loader, 0)
    fresh = _epoch(fresh_loader, 1)
    resumed_loader, dec2 = _embed_loader(fmt, (64, 128))
    resumed = _epoch(resumed_loader, 1)
    for (a, ai), (b, bi) in zip(fresh, resumed):
        assert np.array_equal(ai, bi)
        assert np.array_equal(a, b)
    resumed_loader.set_epoch(5)
    # 0.9.0: set_epoch restarts streams at (rank, epoch) with counters back at 0.
    assert dec2._embed_seed_counter == dec2._inner._decoder._seed_counter == 0
    assert dec2._seed_key == dec2._inner._decoder._seed_key == (0, 5)
