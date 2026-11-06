"""0.16.1: grayscale transforms never return a stride-0 expanded tensor.

When every row of a batch was selected (p=1, or a 1-row tail batch with an unlucky draw), RandomGrayscale returned
`to_grayscale(...).expand(...)`, and the in-place Normalize after it raised "more than one element of the
written-to tensor refers to a single memory location". Real data: imagenet10 / imagenet1k val (skipped when absent).
"""

import pytest
import torch

from slipstream.decoders import DecodeRandomResizedCrop
from slipstream.transforms import Normalize, RandomGrayscale, ToTorchImage
from slipstream.transforms import functional as F
from slipstream.transforms.grayscale import ToGrayscale, ToGrayscaleTorch
from tests.test_ssl_params import _cache


def _batch(gray, batch_size):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    raw = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=batch_size, shuffle=False, verbose=False,
                           exclude_fields=["path"],
                           pipelines={"image": [DecodeRandomResizedCrop(size=64, seed=0), ToTorchImage("cpu")]})
    L = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=batch_size, shuffle=False, verbose=False,
                         exclude_fields=["path"],
                         pipelines={"image": [DecodeRandomResizedCrop(size=64, seed=0), ToTorchImage("cpu"), gray,
                                              Normalize([0.5] * 3, [0.2] * 3, device="cpu")]})
    try:
        return next(iter(raw))["image"], next(iter(L))["image"]
    finally:
        raw.shutdown()
        L.shutdown()


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("make", [lambda: RandomGrayscale(p=1.0, seed=1), lambda: ToGrayscale(num_output_channels=3),
                                  lambda: ToGrayscaleTorch(num_output_channels=3)], ids=["random", "to", "torch"])
def test_grayscale_then_inplace_normalize(make, batch_size):
    rgb, out = _batch(make(), batch_size)
    ref = (F.to_grayscale(rgb, num_output_channels=3) - 0.5) / 0.2
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)     # ToGrayscaleTorch: torchvision's luma weights (rounded)
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 0], out[:, 2])


def test_random_grayscale_partial_unchanged():
    b = torch.rand(6, 3, 8, 8)
    out = F.random_grayscale(b.clone(), torch.tensor([1, 4]), num_output_channels=3)
    gray = F.to_grayscale(b, num_output_channels=3)
    keep = torch.tensor([0, 2, 3, 5])
    torch.testing.assert_close(out[keep], b[keep])
    torch.testing.assert_close(out[[1, 4]], gray[[1, 4]])
    assert out.stride()[1] != 0
