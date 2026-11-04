"""0.15.1: colour-space transforms must not sync the GPU (srgb_to_lrgb's range check did, every batch)."""

import pytest
import torch

from slipstream.transforms import RGBToLab, SRGBToLMS
from slipstream.transforms import functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("t", [SRGBToLMS(), RGBToLab()], ids=lambda t: type(t).__name__)
def test_no_host_sync_on_cuda(t):
    x = torch.rand(4, 3, 32, 32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        t(x)
        F.srgb_to_lrgb(x * 2)                      # out of range: no warning, and no sync, on CUDA
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_cpu_still_warns_out_of_range():
    with pytest.warns(UserWarning, match="outside the"):
        F.srgb_to_lrgb(torch.full((1, 3, 2, 2), 2.0))


def test_values_unchanged():
    x = torch.rand(2, 3, 8, 8)
    want = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    torch.testing.assert_close(F.srgb_to_lrgb(x), want)
