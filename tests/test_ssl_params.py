"""0.15.0: crop boxes on RRC decoders, RGBToLab, CollectParams (params of every random step in the batch).

Real data: the imagenet10 / imagenet1k s256_l512 val caches (skipped when absent).
"""

import io
import os
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torch

from slipstream.decoders import DecodeMultiRandomResizedCrop, DecodeRandomResizedCrop, MultiCropPipeline, NamedCopies
from slipstream.transforms import (
    LAB_CMC_MEAN, LAB_CMC_STD, CollectParams, Normalize, RandomApply, RandomColorJitterHSV, RandomGaussianBlur,
    RandomGrayscale, RandomHorizontalFlip, RandomRotate, RandomSolarization, RGBToLab, ToTorchImage,
)
from slipstream.transforms import functional as F


def _cache(fmt):
    """The same dataset for both formats (the yuv420 test compares against the jpeg records)."""
    roots = [Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent")), Path.home() / ".slipstream"]
    for root in roots:
        for ds in ("imagenet10", "imagenet1k"):
            if all((root / f"{ds}-s256_l512-{f}-val").exists() for f in ("jpeg", "yuv420")):
                return root / f"{ds}-s256_l512-{fmt}-val"
    return None


def _loader(fmt, pipelines, after=None, batch_size=16):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    path = _cache(fmt)
    if path is None:
        pytest.skip(f"no real {fmt} val cache")
    return SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=batch_size, shuffle=False,
                            verbose=False, exclude_fields=["path"], image_format=fmt, pipelines=pipelines,
                            after_batch_transforms=after or [])


def _jpeg_images(indices):
    """Full-resolution PIL decodes of the jpeg cache's records (the reference for both formats)."""
    from slipstream import SlipstreamDataset, SlipstreamLoader
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=1, verbose=False)
    try:
        r = loader.cache.fields["image"].load_batch(np.asarray(indices))
    finally:
        loader.shutdown()
    return [np.asarray(PIL.Image.open(io.BytesIO(r["data"][i, :r["sizes"][i]].tobytes())).convert("RGB"))
            for i in range(len(indices))]


def _crop_mae(full, box, out):
    x, y, w, h = (int(v) for v in box)
    ref = PIL.Image.fromarray(full[y:y + h, x:x + w]).resize(out.shape[:2][::-1], PIL.Image.BILINEAR)
    return np.abs(np.asarray(ref, np.float32) - out.astype(np.float32)).mean()


# ── crop boxes ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_multicrop_boxes_locate_the_crops(fmt):
    """Each reported box, cut from the full image and resized, reproduces the decoded crop."""
    dec = DecodeMultiRandomResizedCrop({"a": dict(size=96, scale=(0.2, 1.0), seed=1),
                                        "b": dict(size=64, scale=(0.05, 0.3), seed=2)})
    loader = _loader(fmt, {"image": [dec]})
    try:
        batch = next(iter(loader))
    finally:
        loader.shutdown()
    lp = dec.last_params()
    fulls = _jpeg_images(batch["_indices"].numpy())
    for view in ("a", "b"):
        boxes = lp["crop_params"][view]
        assert boxes.shape == (16, 4) and boxes.dtype == np.int32
        for i, full in enumerate(fulls):
            H, W = int(lp["heights"][i]), int(lp["widths"][i])
            # yuv420 stores even dims: up to 1 px of edge padding at the right / bottom
            assert 0 <= H - full.shape[0] <= (fmt == "yuv420") and 0 <= W - full.shape[1] <= (fmt == "yuv420")
            full = np.pad(full, ((0, H - full.shape[0]), (0, W - full.shape[1]), (0, 0)), mode="edge")
            fulls[i] = full
            x, y, w, h = boxes[i]
            assert 0 <= x and 0 <= y and x + w <= full.shape[1] and y + h <= full.shape[0]
            out = np.asarray(batch[view][i])      # named views are merged into the batch
            right = _crop_mae(full, boxes[i], out)
            wrong = _crop_mae(full, boxes[(i + 1) % 16] if (i + 1) % 16 != i else boxes[i], out)
            assert right < 6.0, (view, i, right)
            assert right < 0.5 * wrong or wrong < 6.0, (view, i, right, wrong)


def test_rrc_boxes_locate_the_crop():
    dec = DecodeRandomResizedCrop(size=112, seed=7)
    loader = _loader("jpeg", {"image": [dec]})
    try:
        batch = next(iter(loader))
    finally:
        loader.shutdown()
    lp = dec.last_params()
    fulls = _jpeg_images(batch["_indices"].numpy())
    for i, full in enumerate(fulls):
        assert _crop_mae(full, lp["crop_params"][i], np.asarray(batch["image"][i])) < 6.0


# ── RGBToLab ────────────────────────────────────────────────────────────────

def _lab_reference(rgb):
    """Textbook sRGB (D65) -> CIE Lab in float64, written independently of slipstream."""
    rgb = rgb.astype(np.float64)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    xyz = np.einsum("oc,...chw->...ohw", m, lin) / np.array([0.95047, 1.0, 1.08883])[:, None, None]
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1, :, :] - 16, 500 * (f[..., 0, :, :] - f[..., 1, :, :]),
                     200 * (f[..., 1, :, :] - f[..., 2, :, :])], -3)


def test_lab_known_colours():
    # skimage.color.rgb2lab values
    rgb = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    lab = RGBToLab()(rgb.T[:, :, None]).squeeze(-1).T
    want = torch.tensor([[100, 0, 0], [0, 0, 0], [53.2408, 80.0925, 67.2032],
                         [87.7347, -86.1827, 83.1793], [32.2970, 79.1875, -107.8602]])
    torch.testing.assert_close(lab, want, atol=1e-2, rtol=0)   # white's a/b are +-0.005 in skimage too


def test_lab_matches_reference_on_real_images():
    loader = _loader("jpeg", {"image": [DecodeRandomResizedCrop(size=128, seed=3), ToTorchImage("cpu")]})
    try:
        x = next(iter(loader))["image"]
    finally:
        loader.shutdown()
    t = RGBToLab()
    lab = t(x)
    assert lab.shape == x.shape and lab.dtype == x.dtype
    np.testing.assert_allclose(lab.numpy(), _lab_reference(x.numpy()), atol=2e-3, rtol=0)
    torch.testing.assert_close(t.apply_last(x), lab)
    assert t.last_params() == {}
    n = Normalize(LAB_CMC_MEAN, LAB_CMC_STD, device="cpu")(lab.clone())
    assert n.abs().max() < 1.1                       # CMC normalisation maps Lab to about [-1, 1]


def test_lab_rejects_uint8():
    with pytest.raises(TypeError):
        F.srgb_to_lab(torch.zeros(1, 3, 4, 4, dtype=torch.uint8))


# ── CollectParams ───────────────────────────────────────────────────────────

def _view_pipe(s):
    return [ToTorchImage("cpu"), RandomHorizontalFlip(seed=s), RandomColorJitterHSV(p=0.5, hue=0.1, saturation=(0.6, 1.4), seed=s + 1),
            RandomGrayscale(p=0.3, seed=s + 2), RandomGaussianBlur(p=0.5, seed=s + 3), RandomSolarization(p=0.3, seed=s + 4),
            RandomRotate(p=0.5, max_deg=30, seed=s + 5), RGBToLab()]


def test_collect_params_multicrop():
    dec = DecodeMultiRandomResizedCrop({"g0": dict(size=96, seed=1), "g1": dict(size=96, seed=2)})
    mcp = MultiCropPipeline({"g0": _view_pipe(10), "g1": _view_pipe(20)})
    pipelines = {"image": [dec, mcp]}
    loader = _loader("jpeg", pipelines, after=[CollectParams(pipelines)])
    try:
        it = iter(loader)
        batch = next(it)
        batch = next(it)                               # second batch: params must follow the batch
    finally:
        loader.shutdown()
    P = batch["_params"]["image"]
    assert set(P) == {"g0", "g1"}
    for view in ("g0", "g1"):
        p = P[view]
        assert list(p) == ["DecodeMultiRandomResizedCrop", "RandomHorizontalFlip", "RandomColorJitterHSV",
                           "RandomGrayscale", "RandomGaussianBlur", "RandomSolarization", "RandomRotate"]
        crop = p["DecodeMultiRandomResizedCrop"]
        np.testing.assert_array_equal(crop["box"].numpy(), dec.last_params()["crop_params"][view])
        assert crop["heights"].shape == (16,)
        t = dict(zip(["flip", "jit", "gray", "blur", "sol", "rot"], mcp.pipelines[view][1:7]))
        assert torch.equal(p["RandomHorizontalFlip"]["applied"], t["flip"].do > 0)
        j = p["RandomColorJitterHSV"]
        assert j["applied"].shape == (16,) and j["h"].shape == (16,)
        assert torch.equal(torch.where(j["applied"])[0], t["jit"].idx)
        torch.testing.assert_close(j["h"][t["jit"].idx], t["jit"].h)
        assert (j["h"][~j["applied"]] == 0).all() and (j["s"][~j["applied"]] == 1).all()
        assert torch.equal(torch.where(p["RandomGrayscale"]["applied"])[0], t["gray"].idx)
        b = p["RandomGaussianBlur"]
        assert (b["sigma"][~b["applied"]] == 0).all() and (b["kernel"][~b["applied"]] == -1).all()
        torch.testing.assert_close(b["sigma"][b["applied"]], t["blur"].sigmas[b["kernel"][b["applied"]]])
        r = p["RandomRotate"]
        assert torch.equal(r["applied"], t["rot"].do > 0) and (r["deg"][~r["applied"]] == 0).all()
    assert not torch.equal(P["g0"]["DecodeMultiRandomResizedCrop"]["box"], P["g1"]["DecodeMultiRandomResizedCrop"]["box"])


def test_collect_params_shared_steps_and_random_apply():
    """A single decoder + NamedCopies: the shared crop is reported in every view; skipped RandomApply."""
    dec = DecodeRandomResizedCrop(size=64, seed=5)
    flip = RandomHorizontalFlip(seed=9)
    never = RandomApply([RandomColorJitterHSV(p=1.0, seed=3)], p=0.0)
    pipelines = {"image": [dec, NamedCopies(["v1", "v2"]),
                           MultiCropPipeline({"v1": [ToTorchImage("cpu"), flip, never], "v2": [ToTorchImage("cpu")]})]}
    loader = _loader("jpeg", pipelines, after=[CollectParams(pipelines, key="aug")])
    try:
        batch = next(iter(loader))
    finally:
        loader.shutdown()
    P = batch["aug"]["image"]
    assert list(P["v2"]) == ["DecodeRandomResizedCrop"]
    assert torch.equal(P["v1"]["DecodeRandomResizedCrop"]["box"], P["v2"]["DecodeRandomResizedCrop"]["box"])
    assert list(P["v1"]) == ["DecodeRandomResizedCrop", "RandomHorizontalFlip", "RandomColorJitterHSV"]
    assert P["v1"]["RandomColorJitterHSV"] == {"applied": P["v1"]["RandomColorJitterHSV"]["applied"]}
    assert not P["v1"]["RandomColorJitterHSV"]["applied"].any()


def test_without_collect_params_batch_is_unchanged():
    loader = _loader("jpeg", {"image": [DecodeRandomResizedCrop(size=64, seed=5)]})
    try:
        batch = next(iter(loader))
    finally:
        loader.shutdown()
    assert "_params" not in batch


def test_collect_params_window_fold_and_device():
    """[B*T] params fold to [B, T] like the loader's own fields; device= moves every tensor."""
    flip = RandomHorizontalFlip(seed=1)
    pipelines = {"image": [flip]}
    flip(torch.zeros(6, 3, 4, 4))
    batch = CollectParams(pipelines, device="cpu")({"_indices": torch.zeros(2, 3, dtype=torch.long)})
    assert batch["_params"]["image"]["RandomHorizontalFlip"]["applied"].shape == (2, 3)
