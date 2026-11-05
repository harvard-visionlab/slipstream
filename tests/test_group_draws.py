"""0.16.0: SlipstreamLoader(group=G): every G consecutive samples share one draw of every random step.

Paired augmentation (ce_ssl): different images, identical augmentation. Decoders share the per-sample
draw (not the pixel box), so crops on images of different sizes match in relative terms and stay in bounds.
Real data: the imagenet10 / imagenet1k s256_l512 val caches (skipped when absent).
"""

import numpy as np
import pytest
import torch

from slipstream.decoders import DecodeMultiRandomResizedCrop, DecodeMultiRandomResizeShortCropLong, DecodeRandomResizedCrop
from slipstream.transforms import (
    CollectParams, RandomColorJitterHSV, RandomGaussianBlur, RandomGrayscale, RandomHorizontalFlip, RandomRotate,
    RandomSolarization, ToTorchImage,
)
from tests.test_ssl_params import _cache


def _loader(fmt, pipelines, group, after=None, batch_size=64, **kw):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    path = _cache(fmt)
    if path is None:
        pytest.skip(f"no real {fmt} val cache")
    return SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=batch_size, shuffle=True, seed=0,
                            verbose=False, exclude_fields=["path"], image_format=fmt, pipelines=pipelines,
                            after_batch_transforms=after or [], group=group, **kw)


def _first_batches(loader, n=2):
    try:
        it = iter(loader)
        return [next(it) for _ in range(n)]
    finally:
        loader.shutdown()


def _check_boxes(box, W, H, paired):
    """All boxes in bounds; with `paired`, pairs share area fraction + aspect ratio (identical box on equal sizes)."""
    box, W, H = (np.asarray(a, dtype=np.float64) for a in (box, W, H))
    x, y, w, h = box.T
    assert (x >= 0).all() and (y >= 0).all() and (x + w <= W).all() and (y + h <= H).all()
    area, logr = w * h / (W * H), np.log(w / h)
    match = (np.abs(area[0::2] - area[1::2]) < 0.02) & (np.abs(logr[0::2] - logr[1::2]) < 0.03)
    same_size = (W[0::2] == W[1::2]) & (H[0::2] == H[1::2])
    if paired:
        np.testing.assert_array_equal(box[0::2][same_size], box[1::2][same_size])
        assert match.mean() >= 0.8, match.mean()     # the rest: a crop that did not fit one image (retry/fallback)
    else:
        assert match.mean() <= 0.2, match.mean()
    return same_size


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
@pytest.mark.parametrize("group", [1, 2])
def test_rrc_group_shares_the_relative_crop(fmt, group):
    dec = DecodeRandomResizedCrop(size=64, scale=(0.08, 0.5), seed=1)
    for batch in _first_batches(_loader(fmt, {"image": [dec]}, group)):
        ind = batch["_indices"].numpy()
        assert len(set(ind.tolist())) == len(ind)          # a group holds different images
        lp = dec.last_params()
        _check_boxes(lp["crop_params"], lp["widths"], lp["heights"], paired=group > 1)


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_multicrop_group(fmt):
    dec = DecodeMultiRandomResizedCrop({"a": dict(size=64, scale=(0.08, 0.5), seed=1),
                                        "b": dict(size=48, scale=(0.05, 0.3), seed=2)})
    (batch,) = _first_batches(_loader(fmt, {"image": [dec]}, 2), 1)
    lp = dec.last_params()
    for view in ("a", "b"):
        _check_boxes(lp["crop_params"][view], lp["widths"], lp["heights"], paired=True)
    assert not np.array_equal(lp["crop_params"]["a"], lp["crop_params"]["b"])


def test_short_crop_long_group():
    dec = DecodeMultiRandomResizeShortCropLong({"a": dict(size=64, x_range=(0, 1), seed=1),
                                                "b": dict(size=(48, 80), x_range=(0, 1), seed=2)})
    (batch,) = _first_batches(_loader("jpeg", {"image": [dec]}, 2), 1)
    lp = dec.last_params
    W, H = lp["widths"].astype(np.float64), lp["heights"].astype(np.float64)
    for view in ("a", "b"):
        c = lp["crops"][view]
        for k in ("x_pos", "y_pos", "target_sizes"):
            np.testing.assert_array_equal(c[k][0::2], c[k][1::2])
        x, y, w, h = c["crop_params"].astype(np.float64).T
        assert (x >= 0).all() and (y >= 0).all() and (x + w <= W).all() and (y + h <= H).all()
        assert np.std(c["x_pos"][0::2]) > 0


def test_gpu_transforms_share_draws():
    """Every random step's params are identical within a pair, and differ across pairs."""
    steps = [ToTorchImage("cpu"), RandomHorizontalFlip(seed=3), RandomColorJitterHSV(p=0.8, hue=0.1, seed=4),
             RandomGrayscale(p=0.2, seed=5), RandomGaussianBlur(p=0.5, seed=6), RandomSolarization(p=0.2, seed=7),
             RandomRotate(p=0.5, max_deg=30, seed=8)]
    pipelines = {"image": [DecodeRandomResizedCrop(size=64, seed=1), *steps]}
    for batch in _first_batches(_loader("jpeg", pipelines, 2, after=[CollectParams(pipelines)])):
        P = batch["_params"]["image"]
        assert len(P) == 7
        for name, params in P.items():
            for k, v in params.items():
                if name.startswith("DecodeRandomResizedCrop") or not isinstance(v, torch.Tensor) or v.shape[:1] != (64,):
                    continue
                torch.testing.assert_close(v[0::2], v[1::2], equal_nan=True, msg=f"{name}.{k}")
        assert P["RandomHorizontalFlip"]["applied"][0::2].float().std() > 0   # draws still vary across pairs
        img = batch["image"]
        assert not torch.equal(img[0::2], img[1::2])


def test_group_validation():
    with pytest.raises(ValueError, match="multiple of group"):
        _loader("jpeg", {"image": [DecodeRandomResizedCrop(size=64, seed=1)]}, 2, batch_size=15)
    with pytest.raises(ValueError, match="group must be"):
        _loader("jpeg", {"image": [DecodeRandomResizedCrop(size=64, seed=1)]}, 0)
