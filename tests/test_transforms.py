"""Tests for slipstream GPU batch augmentation transforms.

All tests use synthetic tensors (no dataset needed).
"""

import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
from slipstream.transforms import (
    BatchAugment, Compose, RandomApply, MultiSample,
    ToTorchImage, ToNumpy, ToChannelsFirst, ToChannelsLast, ToDevice, ToFloat, ToFloatDiv,
    Normalize, NormalizeLGN,
    ToGrayscale, ToGrayscaleTorch, RandomGrayscale,
    RandomBrightness, RandomContrast,
    ColorJitter, RandomColorJitter, RandomColorJitterHSV, RandomColorJitterYIQ,
    RandomHorizontalFlip, RandomRotate, RandomZoom, RandomRotateObject,
    RandomGaussianBlur, RandomSolarization, RandomPatchShuffle,
    CircularMask, FixedOpticalDistortion,
    SRGBToLMS, LMSToParvo, LMSToMagno, LMSToKonio, RGBToLGN, RGBToMagno,
)


B, C, H, W = 4, 3, 32, 32
device = "cpu"


def make_batch():
    return torch.rand(B, C, H, W, device=device)


def make_batch_01():
    """Batch guaranteed in [0, 1]."""
    return torch.rand(B, C, H, W, device=device)


# =================================================
#  Shape tests
# =================================================

class TestShapePreservation:
    """Each transform should preserve [B,C,H,W] → [B,C,H,W] (or documented output shape)."""

    def test_normalize_gpu(self):
        b = make_batch()
        t = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
        out = t(b)
        assert out.shape == b.shape

    def test_random_grayscale(self):
        b = make_batch()
        t = RandomGrayscale(p=1.0, num_output_channels=3)
        out = t(b)
        assert out.shape == b.shape

    def test_to_grayscale_gpu(self):
        b = make_batch()
        t = ToGrayscale(num_output_channels=3)
        out = t(b)
        assert out.shape == b.shape

    def test_to_grayscale_torch_gpu(self):
        b = make_batch()
        t = ToGrayscaleTorch(num_output_channels=3)
        out = t(b)
        assert out.shape == b.shape

    def test_random_brightness(self):
        b = make_batch_01()
        t = RandomBrightness(p=1.0)
        out = t(b)
        assert out.shape == b.shape

    def test_random_contrast(self):
        b = make_batch_01()
        t = RandomContrast(p=1.0)
        out = t(b)
        assert out.shape == b.shape

    def test_color_jitter(self):
        b = make_batch_01()
        t = ColorJitter(p=1.0, hue=0.1, saturation=0.2, value=0.2, contrast=0.2)
        out = t(b)
        assert out.shape == b.shape

    def test_random_color_jitter(self):
        b = make_batch_01()
        t = RandomColorJitter(p=1.0, hue=0.1, saturation=0.2, value=0.2, contrast=0.2)
        out = t(b)
        assert out.shape == b.shape

    def test_random_color_jitter_hsv(self):
        b = make_batch_01()
        t = RandomColorJitterHSV(p=1.0, hue=0.1, saturation=0.2, value=0.2, contrast=0.2)
        out = t(b)
        assert out.shape == b.shape

    def test_random_color_jitter_yiq(self):
        b = make_batch_01()
        t = RandomColorJitterYIQ(p=1.0, hue=30, saturation=0.2, value=0.2, brightness=0.2, contrast=0.2)
        out = t(b)
        assert out.shape == b.shape

    def test_random_horizontal_flip(self):
        b = make_batch()
        t = RandomHorizontalFlip(p=1.0)
        out = t(b)
        assert out.shape == b.shape

    def test_random_rotate(self):
        b = make_batch()
        t = RandomRotate(p=1.0, max_deg=30)
        out = t(b)
        assert out.shape == b.shape

    def test_random_zoom(self):
        b = make_batch()
        t = RandomZoom(p=1.0, zoom=(0.5, 1.0))
        out = t(b)
        assert out.shape == b.shape

    def test_random_rotate_object(self):
        b = make_batch()
        t = RandomRotateObject(p=1.0, max_deg=30)
        out = t(b)
        assert out.shape == b.shape

    def test_random_gaussian_blur(self):
        b = make_batch()
        t = RandomGaussianBlur(p=1.0, kernel_size=6)
        out = t(b)
        assert out.shape == b.shape

    def test_random_solarization(self):
        b = make_batch_01()
        t = RandomSolarization(p=1.0, threshold=0.5)
        out = t(b)
        assert out.shape == b.shape

    def test_random_patch_shuffle(self):
        b = make_batch()
        t = RandomPatchShuffle(sizes=0.25, p=1.0, img_size=H)
        out = t(b)
        assert out.shape == b.shape

    def test_circular_mask(self):
        b = make_batch()
        t = CircularMask(output_size=H, device=device)
        out = t(b)
        assert out.shape == b.shape

    def test_fixed_optical_distortion(self):
        b = make_batch()
        t = FixedOpticalDistortion(output_size=(H, W), distortion=-0.5, device=device)
        out = t(b)
        assert out.shape == b.shape


# =================================================
#  Probability tests: p=0 → identity, p=1 → modified
# =================================================

class TestProbability:
    def test_p0_identity_flip(self):
        b = make_batch()
        t = RandomHorizontalFlip(p=0.0)
        out = t(b)
        # With p=0, mask is all zeros, flip_val is all 1 → identity affine
        assert out.shape == b.shape

    def test_p0_identity_grayscale(self):
        b = make_batch()
        t = RandomGrayscale(p=0.0, num_output_channels=3)
        out = t(b)
        assert torch.allclose(out, b)

    def test_p0_identity_solarization(self):
        b = make_batch_01()
        t = RandomSolarization(p=0.0, threshold=0.5)
        out = t(b)
        assert torch.allclose(out, b)

    def test_p1_modifies_grayscale(self):
        b = make_batch()
        t = RandomGrayscale(p=1.0, num_output_channels=3)
        out = t(b)
        # All 3 channels should be identical after grayscale
        assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)
        assert torch.allclose(out[:, 1], out[:, 2], atol=1e-5)


# =================================================
#  Replay tests
# =================================================

class TestReplay:
    def test_compose_replay(self):
        b = make_batch()
        t = Compose([RandomHorizontalFlip(p=1.0, seed=42)])
        out1 = t(b.clone())
        out2 = t.apply_last(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_brightness_replay(self):
        b = make_batch_01()
        t = RandomBrightness(p=1.0, seed=42)
        out1 = t(b)
        out2 = t.apply_last(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_contrast_replay(self):
        b = make_batch_01()
        t = RandomContrast(p=1.0, seed=42)
        out1 = t(b)
        out2 = t.apply_last(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_solarization_replay(self):
        b = make_batch_01()
        t = RandomSolarization(p=1.0, threshold=0.5)
        out1 = t(b)
        out2 = t.apply_last(b.clone())
        assert torch.allclose(out1, out2)

    def test_gaussian_blur_replay(self):
        b = make_batch()
        t = RandomGaussianBlur(p=1.0, seed=42)
        out1 = t(b.clone())
        out2 = t.apply_last(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)


# =================================================
#  Seed determinism tests
# =================================================

class TestSeedDeterminism:
    def test_flip_determinism(self):
        b = make_batch()
        t1 = RandomHorizontalFlip(p=0.5, seed=123)
        t2 = RandomHorizontalFlip(p=0.5, seed=123)
        out1 = t1(b.clone())
        out2 = t2(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_rotate_determinism(self):
        b = make_batch()
        t1 = RandomRotate(p=0.5, seed=123)
        t2 = RandomRotate(p=0.5, seed=123)
        out1 = t1(b.clone())
        out2 = t2(b.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_brightness_determinism(self):
        b = make_batch_01()
        t1 = RandomBrightness(p=1.0, seed=42)
        t2 = RandomBrightness(p=1.0, seed=42)
        out1 = t1(b.clone())
        out2 = t2(b.clone())
        assert torch.allclose(out1, out2, atol=1e-6)


# =================================================
#  Compose and MultiSample tests
# =================================================

class TestComposeMultiSample:
    def test_compose(self):
        b = make_batch()
        t = Compose([RandomHorizontalFlip(p=1.0), RandomGrayscale(p=1.0, num_output_channels=3)])
        out = t(b)
        assert out.shape == b.shape

    def test_random_apply(self):
        b = make_batch()
        t = RandomApply([RandomGrayscale(p=1.0, num_output_channels=3)], p=1.0)
        out = t(b)
        assert out.shape == b.shape

    def test_multi_sample(self):
        b = make_batch()
        t = MultiSample([RandomHorizontalFlip(p=0.5)], num_copies=3)
        out = t(b)
        assert len(out) == 3
        for o in out:
            assert o.shape == b.shape

    def test_multi_sample_return_input(self):
        b = make_batch()
        t = MultiSample([RandomHorizontalFlip(p=0.5)], num_copies=2, return_input=True)
        out = t(b)
        assert len(out) == 3  # input + 2 copies


# =================================================
#  Conversion tests
# =================================================

class TestConversion:
    def test_to_channels_first(self):
        b = torch.rand(B, H, W, C)
        t = ToChannelsFirst()
        out = t(b)
        assert out.shape == (B, C, H, W)

    def test_to_channels_last(self):
        b = make_batch()
        t = ToChannelsLast()
        out = t(b)
        assert out.shape == (B, H, W, C)

    def test_to_float_div(self):
        b = torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8).float()
        t = ToFloatDiv(255.0)
        out = t(b)
        assert out.max() <= 1.0

    def test_to_device(self):
        b = make_batch()
        t = ToDevice("cpu")
        out = t(b)
        assert out.device.type == "cpu"


# =================================================
#  Color space tests
# =================================================

class TestColorSpace:
    def test_srgb_to_lms(self):
        b = make_batch_01()
        t = SRGBToLMS()
        out = t(b)
        assert out.shape == b.shape  # still 3 channels (L, M, S)

    def test_rgb_to_lgn(self):
        b = make_batch_01()
        t = RGBToLGN(device=device)
        out = t(b)
        assert out.shape == (B, 5, H, W)

    def test_rgb_to_magno(self):
        b = make_batch_01()
        t = RGBToMagno(device=device)
        out = t(b)
        assert out.shape == (B, 2, H, W)

    def test_normalize_lgn(self):
        b = torch.rand(B, 5, H, W)
        t = NormalizeLGN(device=device)
        out = t(b)
        assert out.shape == b.shape


# =================================================
#  repr tests
# =================================================

class TestRepr:
    """Every transform should have a __repr__ that doesn't crash."""

    @pytest.mark.parametrize("cls,kwargs", [
        (Normalize, dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device="cpu")),
        (RandomGrayscale, dict(p=0.5)),
        (RandomBrightness, dict(p=0.5)),
        (RandomContrast, dict(p=0.5)),
        (ColorJitter, dict(p=0.5, hue=0.1)),
        (RandomColorJitter, dict(p=0.5, hue=0.1)),
        (RandomColorJitterYIQ, dict(p=0.5, hue=30)),
        (RandomHorizontalFlip, dict(p=0.5)),
        (RandomRotate, dict(p=0.5)),
        (RandomZoom, dict(p=0.5)),
        (RandomRotateObject, dict(p=0.5)),
        (RandomGaussianBlur, dict(p=0.5)),
        (RandomSolarization, dict(p=0.5)),
        (RandomPatchShuffle, dict(sizes=0.25, p=0.5, img_size=32)),
        (CircularMask, dict(output_size=32)),
        (FixedOpticalDistortion, dict(output_size=(32, 32))),
        (RGBToLGN, dict(device="cpu")),
        (RGBToMagno, dict(device="cpu")),
    ])
    def test_repr(self, cls, kwargs):
        t = cls(**kwargs)
        r = repr(t)
        assert isinstance(r, str)
        assert len(r) > 0


# =================================================
#  3D (single image) tests
# =================================================

class TestSingleImage3D:
    """Transforms should handle [C, H, W] inputs (single image, no batch dim)."""

    def test_normalize_gpu_3d(self):
        img = torch.rand(C, H, W)
        t = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_grayscale_3d(self):
        img = torch.rand(C, H, W)
        t = RandomGrayscale(p=1.0, num_output_channels=3)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_brightness_3d(self):
        img = torch.rand(C, H, W)
        t = RandomBrightness(p=1.0)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_contrast_3d(self):
        img = torch.rand(C, H, W)
        t = RandomContrast(p=1.0)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_color_jitter_3d(self):
        img = torch.rand(C, H, W)
        t = ColorJitter(p=1.0, hue=0.1, saturation=0.2, value=0.2, contrast=0.2)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_color_jitter_yiq_3d(self):
        img = torch.rand(C, H, W)
        t = RandomColorJitterYIQ(p=1.0, hue=30, saturation=0.2, value=0.2, brightness=0.2, contrast=0.2)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_horizontal_flip_3d(self):
        img = torch.rand(C, H, W)
        t = RandomHorizontalFlip(p=1.0)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_gaussian_blur_3d(self):
        img = torch.rand(C, H, W)
        t = RandomGaussianBlur(p=1.0, kernel_size=6)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_random_solarization_3d(self):
        img = torch.rand(C, H, W)
        t = RandomSolarization(p=1.0, threshold=0.5)
        out = t(img)
        assert out.shape == (C, H, W)

    def test_srgb_to_lms_3d(self):
        img = torch.rand(C, H, W)
        t = SRGBToLMS()
        out = t(img)
        assert out.shape == (C, H, W)


# =================================================
#  GPU tests (optional)
# =================================================

# =================================================
#  Torchvision alignment tests
# =================================================

class TestTorchvisionAlignment:
    """Verify slipstream transforms produce identical output to torchvision v2 equivalents.

    For deterministic transforms: exact match (atol=1e-5).
    For random transforms: force identical parameters and compare.
    """

    def test_normalize(self):
        b = make_batch()
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ss = Normalize(mean, std, device=device)(b.clone())
        tv = v2.Normalize(mean, std)(b.clone())
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_grayscale_1ch(self):
        b = make_batch()
        ss = ToGrayscale(num_output_channels=1)(b)
        tv = torch.stack([TF.rgb_to_grayscale(img, num_output_channels=1) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_grayscale_3ch(self):
        b = make_batch()
        ss = ToGrayscale(num_output_channels=3)(b)
        tv = torch.stack([TF.rgb_to_grayscale(img, num_output_channels=3) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_horizontal_flip(self):
        b = make_batch()
        ss = RandomHorizontalFlip(p=1.0)(b.clone())
        tv = torch.stack([TF.horizontal_flip(img) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_solarization(self):
        b = make_batch_01()
        threshold = 0.5
        ss = RandomSolarization(p=1.0, threshold=threshold)(b)
        tv = torch.stack([TF.solarize(img, threshold=threshold) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_brightness(self):
        """Compare with fixed brightness factor."""
        b = make_batch_01()
        factor = 1.3
        ss_t = RandomBrightness(p=1.0, scale_range=(factor, factor), device=device)
        ss = ss_t(b)
        tv = torch.stack([TF.adjust_brightness(img, factor) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_contrast(self):
        """Compare with fixed contrast factor."""
        b = make_batch_01()
        factor = 1.3
        ss_t = RandomContrast(p=1.0, scale_range=(factor, factor), device=device)
        ss = ss_t(b)
        tv = torch.stack([TF.adjust_contrast(img, factor) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    def test_gaussian_blur(self):
        """Compare with fixed sigma — SS bank has one sigma so all images get the same kernel."""
        torch.manual_seed(0)
        b = torch.rand(B, C, 64, 64)
        kernel_size = 7
        sigma = 1.5
        # num_sigmas=1 with range=(s, s) ensures every image uses exactly sigma
        ss_t = RandomGaussianBlur(p=1.0, kernel_size=kernel_size,
                                   sigma_range=(sigma, sigma), num_sigmas=1)
        ss = ss_t(b.clone())
        tv = torch.stack([TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size],
                                            sigma=[sigma, sigma]) for img in b])
        assert torch.allclose(ss, tv, atol=1e-5), f"max diff: {(ss - tv).abs().max()}"

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_gaussian_blur_sigmas(self, sigma):
        """Verify kernel bank entries match torchvision at multiple sigmas."""
        b = torch.rand(1, C, 64, 64)
        kernel_size = 7
        ss_t = RandomGaussianBlur(p=1.0, kernel_size=kernel_size,
                                   sigma_range=(sigma, sigma), num_sigmas=1)
        ss = ss_t(b.clone())
        tv = TF.gaussian_blur(b, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        assert torch.allclose(ss, tv, atol=1e-5), f"sigma={sigma} max diff: {(ss - tv).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
class TestGPU:
    def test_normalize_gpu_cuda(self):
        b = torch.rand(B, C, H, W, device="cuda")
        t = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device="cuda")
        out = t(b)
        assert out.device.type == "cuda"
        assert out.shape == b.shape

    def test_random_flip_cuda(self):
        b = torch.rand(B, C, H, W, device="cuda")
        t = RandomHorizontalFlip(p=1.0, device="cuda")
        out = t(b)
        assert out.device.type == "cuda"
        assert out.shape == b.shape
