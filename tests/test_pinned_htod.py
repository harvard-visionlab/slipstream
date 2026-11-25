"""0.18.0: a decoder followed by ToTorchImage(<cuda>) writes into pinned memory and the copy runs async on a
side stream. Output must be byte-identical to the synchronous path: real data, past buffer reuse, partial last
batch, 2 epochs, with the GPU deliberately backed up so copies lag. Needs CUDA (machina)."""

import pytest
import torch

from slipstream import SlipstreamLoader
from slipstream.decoders import DecodeCenterCrop, DecodeRandomResizedCrop
from slipstream.transforms import ToTorchImage
from tests.test_ssl_params import _cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(fmt, make, pinned, dtype=torch.float32):
    from slipstream import SlipstreamDataset
    path = _cache(fmt)
    if path is None:
        pytest.skip(f"no real {fmt} val cache")
    orig = SlipstreamLoader._enable_pinned_outputs
    if not pinned:
        SlipstreamLoader._enable_pinned_outputs = lambda self: None
    try:
        loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=96, shuffle=True, seed=0,
                                  drop_last=False, verbose=False, exclude_fields=["path"], image_format=fmt,
                                  pipelines={"image": [make(), ToTorchImage("cuda", dtype=dtype)]})
    finally:
        SlipstreamLoader._enable_pinned_outputs = orig
    dec = loader.pipelines["image"][0]._decoder
    a = torch.randn(2048, 2048, device="cuda")
    out = []
    try:
        for _ in range(2):
            for b in loader:
                for _ in range(30):
                    a @ a                                  # GPU busy: the next decode races the copy
                out.append(b["image"].cpu())
    finally:
        loader.shutdown()
    return out, dec.pin_output


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
@pytest.mark.parametrize("make", [lambda: DecodeRandomResizedCrop(size=64, seed=1), lambda: DecodeCenterCrop(64)],
                         ids=["rrc", "center"])
def test_pinned_async_matches_sync(fmt, make):
    ref, p0 = _run(fmt, make, pinned=False)
    got, p1 = _run(fmt, make, pinned=True)
    assert (p0, p1) == (False, True)
    assert len(ref) == len(got) and ref[-1].shape[0] != 96          # partial last batch (imagenet10 val: 500)
    for r, g in zip(ref, got):
        assert torch.equal(r, g)


def test_pinned_bf16_matches_sync():
    make = lambda: DecodeRandomResizedCrop(size=64, seed=1)  # noqa: E731
    ref, _ = _run("jpeg", make, pinned=False, dtype=torch.bfloat16)
    got, _ = _run("jpeg", make, pinned=True, dtype=torch.bfloat16)
    assert all(torch.equal(r, g) for r, g in zip(ref, got))


def test_not_pinned_without_cuda_consumer():
    """Raw decoder output (no ToTorchImage on CUDA next) stays pageable: user code may copy it itself."""
    from slipstream import SlipstreamDataset
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    for pipe in ([DecodeRandomResizedCrop(size=64, seed=1)],
                 [DecodeRandomResizedCrop(size=64, seed=1), ToTorchImage("cpu")]):
        loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=64, verbose=False,
                                  exclude_fields=["path"], pipelines={"image": pipe})
        assert not pipe[0]._decoder.pin_output
        loader.shutdown()
