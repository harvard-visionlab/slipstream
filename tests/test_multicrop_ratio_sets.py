"""DecodeMultiRandomResizedCrop(ratios=...) (0.13.0): discrete aspect-ratio sets for SimCLR hard negatives.

Real imagenet10 val caches (jpeg + yuv420) under ~/.slipstream; skipped when absent.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop, MultiCropPipeline

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-{fmt}-val"
B = 32
SET = (1 / 2, 3 / 5, 5 / 3, 2.0)


def _batch(crops, fmt="jpeg", post=None, **kw):
    from slipstream.loader import SlipstreamLoader

    cache = Path(str(CACHE).format(fmt=fmt))
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    pipe = [DecodeMultiRandomResizedCrop(crops, **kw)] + ([post] if post is not None else [])
    loader = SlipstreamLoader(SimpleNamespace(cache_path=cache, remote_dir=None), batch_size=B, shuffle=False,
                              verbose=False, image_format=fmt, pipelines={"image": pipe}, seed=0)
    b = next(iter(loader))
    loader.shutdown()
    return {k: np.asarray(v) if not isinstance(v, torch.Tensor) else v for k, v in b.items()}


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
@pytest.mark.parametrize("mode", ["standard", "direct"])
def test_single_ratio_set_equals_fixed_range(fmt, mode):
    # the grouped path with one ratio must reproduce the plain path with ratio=(r, r) exactly
    b = _batch({"fixed": dict(size=64, seed=7, ratio=(2.0, 2.0)),
                "set": dict(size=64, seed=7, ratios=[2.0])}, fmt=fmt, crop_mode=mode)
    assert np.array_equal(b["fixed"], b["set"])


@pytest.mark.parametrize("mode", ["standard", "direct"])
def test_each_image_uses_one_ratio_from_the_set_with_the_positive_draws(mode):
    crops = {"hn": dict(size=64, seed=42, ratios=SET)}
    crops |= {f"r{i}": dict(size=64, seed=42, ratio=(r, r)) for i, r in enumerate(SET)}
    b = _batch(crops, crop_mode=mode)
    which = []
    for j in range(B):
        hits = [i for i in range(len(SET)) if np.array_equal(b["hn"][j], b[f"r{i}"][j])]
        assert hits, f"image {j}: hard negative matches no fixed-ratio crop with the same seed"
        which.append(hits[0])
    assert len(set(which)) >= 3                                  # the choice actually varies per image


def test_choice_is_seeded_and_shared_by_same_seed_crops():
    a = _batch({"hn": dict(size=64, seed=42, ratios=SET), "twin": dict(size=64, seed=42, ratios=SET),
                "other": dict(size=64, seed=43, ratios=SET)})
    again = _batch({"hn": dict(size=64, seed=42, ratios=SET)})
    assert np.array_equal(a["hn"], a["twin"]) and np.array_equal(a["hn"], again["hn"])
    assert not np.array_equal(a["hn"], a["other"])


def test_validation():
    with pytest.raises(ValueError, match="not both"):
        DecodeMultiRandomResizedCrop({"x": dict(size=64, ratio=(0.5, 2), ratios=[1.0])})
    for bad in ([], [0.0], [-1.0]):
        with pytest.raises(ValueError, match="positive"):
            DecodeMultiRandomResizedCrop({"x": dict(size=64, ratios=bad)})


def test_same_seed_transform_chains_draw_identically():
    # model-rearing Q2: a hard negative's pixel augmentations equal its positive's when every transform in
    # its chain has the same seed (separate instances; one shared instance would advance twice per batch)
    from slipstream.transforms import RandomColorJitter, RandomGrayscale, RandomHorizontalFlip, ToTorchImage

    def chain(seed):
        return [ToTorchImage("cpu"), RandomHorizontalFlip(seed=seed),
                RandomColorJitter(p=0.8, value=0.4, contrast=0.4, saturation=0.2, hue=0.1, seed=seed + 1),
                RandomGrayscale(p=0.2, seed=seed + 2)]

    post = MultiCropPipeline({"view_0": chain(100), "hn_0": chain(100), "view_1": chain(200)})
    b = _batch({"view_0": dict(size=64, seed=42, ratio=(1.0, 1.0)),
                "hn_0": dict(size=64, seed=42, ratios=[1.0]),          # same pixels in, so outputs must match
                "view_1": dict(size=64, seed=42, ratio=(1.0, 1.0))}, post=post)
    assert torch.equal(b["view_0"], b["hn_0"])
    assert not torch.equal(b["view_0"], b["view_1"])               # different transform seeds -> different draws
