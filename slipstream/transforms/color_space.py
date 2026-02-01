"""Biological color space transforms (sRGB → LMS → Parvo/Magno/Konio)."""

import torch
from typing import Callable
from .base import BatchAugment
from . import functional as F


class SRGBToLMS(BatchAugment):
    """Convert sRGB to CAT02 LMS color space."""

    def apply_last(self, b):
        return F.srgb_to_lms(b)

    def __call__(self, b, **kwargs):
        return F.srgb_to_lms(b)


class LMSToParvo(BatchAugment):
    """Compute parvocellular (red-green opponent) response from LMS channels."""

    def __init__(self, sigma_c: float = 1.5, sigma_s: float = 5.0,
                 nonlin=None, device=None):
        self.sigma_c = sigma_c
        self.sigma_s = sigma_s
        self.size_c = F.kernel_size_for_sigma(self.sigma_c)
        self.size_s = F.kernel_size_for_sigma(self.sigma_s)
        self.kernel_c = F.make_gaussian_kernel(self.size_c, self.sigma_c, device=device)
        self.kernel_s = F.make_gaussian_kernel(self.size_s, self.sigma_s, device=device)
        self.nonlin = nonlin if nonlin is not None else torch.nn.functional.relu

    def __call__(self, L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        self.kernel_c = self.kernel_c.to(device=L.device, dtype=L.dtype)
        self.kernel_s = self.kernel_s.to(device=L.device, dtype=L.dtype)
        L_resp = F.dog_filter(L, self.kernel_c, self.kernel_s)
        M_resp = F.dog_filter(M, self.kernel_c, self.kernel_s)
        parvo1 = self.nonlin(L_resp - M_resp)
        parvo2 = self.nonlin(M_resp - L_resp)
        return torch.cat([parvo1, parvo2], dim=-3)

    def __repr__(self):
        nonlin_name = self.nonlin.__name__
        return f"{self.__class__.__name__}(σc={self.sigma_c}, σs={self.sigma_s}, nonlin={nonlin_name})"


class LMSToMagno(BatchAugment):
    """Compute magnocellular (luminance) response from LMS channels."""

    def __init__(self, sigma_c: float = 2.5, sigma_s: float = 5.0,
                 nonlin=None, device=None):
        self.sigma_c = sigma_c
        self.sigma_s = sigma_s
        self.size_c = F.kernel_size_for_sigma(self.sigma_c)
        self.size_s = F.kernel_size_for_sigma(self.sigma_s)
        self.kernel_c = F.make_gaussian_kernel(self.size_c, self.sigma_c, device=device)
        self.kernel_s = F.make_gaussian_kernel(self.size_s, self.sigma_s, device=device)
        self.nonlin = nonlin if nonlin is not None else torch.nn.functional.relu

    def __call__(self, L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        self.kernel_c = self.kernel_c.to(device=L.device, dtype=L.dtype)
        self.kernel_s = self.kernel_s.to(device=L.device, dtype=L.dtype)
        lum = L + M
        lum_resp = F.dog_filter(lum, self.kernel_c, self.kernel_s)
        mag_on = self.nonlin(lum_resp.clone())
        mag_off = self.nonlin(-lum_resp)
        return torch.cat([mag_on, mag_off], dim=-3)

    def __repr__(self):
        nonlin_name = self.nonlin.__name__
        return f"{self.__class__.__name__}(σc={self.sigma_c}, σs={self.sigma_s}, nonlin={nonlin_name})"


class LMSToKonio(BatchAugment):
    """Compute koniocellular (blue-yellow opponent) response from LMS channels."""

    def __init__(self, sigma_c: float = 2.5, sigma_s: float = 6.5,
                 nonlin=None, device=None):
        self.sigma_c = sigma_c
        self.sigma_s = sigma_s
        self.size_c = F.kernel_size_for_sigma(self.sigma_c)
        self.size_s = F.kernel_size_for_sigma(self.sigma_s)
        self.kernel_c = F.make_gaussian_kernel(self.size_c, self.sigma_c, device=device)
        self.kernel_s = F.make_gaussian_kernel(self.size_s, self.sigma_s, device=device)
        self.nonlin = nonlin if nonlin is not None else torch.nn.functional.relu

    def __call__(self, L: torch.Tensor, M: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        self.kernel_c = self.kernel_c.to(device=L.device, dtype=L.dtype)
        self.kernel_s = self.kernel_s.to(device=L.device, dtype=L.dtype)
        s_resp = F.dog_filter(S, self.kernel_c, self.kernel_s)
        lm_sum = L + M
        lm_resp = F.dog_filter(lm_sum, self.kernel_c, self.kernel_s)
        return self.nonlin(s_resp - lm_resp)

    def __repr__(self):
        nonlin_name = self.nonlin.__name__
        return f"{self.__class__.__name__}(σc={self.sigma_c}, σs={self.sigma_s}, nonlin={nonlin_name})"


class RGBToLGN(BatchAugment):
    """Convert RGB to 5-channel LGN output (Parvo + Magno + Konio)."""

    def __init__(self,
                 parvo_sigma_c: float = 1.5, parvo_sigma_s: float = 5.0,
                 magno_sigma_c: float = 2.5, magno_sigma_s: float = 5.0,
                 konio_sigma_c: float = 2.5, konio_sigma_s: float = 6.5,
                 nonlin: Callable = torch.nn.functional.relu,
                 device=None):
        self.parvo = LMSToParvo(sigma_c=parvo_sigma_c, sigma_s=parvo_sigma_s, nonlin=nonlin, device=device)
        self.magno = LMSToMagno(sigma_c=magno_sigma_c, sigma_s=magno_sigma_s, nonlin=nonlin, device=device)
        self.konio = LMSToKonio(sigma_c=konio_sigma_c, sigma_s=konio_sigma_s, nonlin=nonlin, device=device)

    def __call__(self, b, **kwargs):
        """Input: RGB batch (*,3,H,W) [0..1]. Output: LGN (*,5,H,W)."""
        lms = F.srgb_to_lms(b)
        L = lms[..., 0:1, :, :]
        M = lms[..., 1:2, :, :]
        S = lms[..., 2:3, :, :]
        parvo_out = self.parvo(L, M)
        magno_out = self.magno(L, M)
        konio_out = self.konio(L, M, S)
        return torch.cat([parvo_out, magno_out, konio_out], dim=-3)

    def apply_last(self, b):
        return self(b)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"   (parvo): {self.parvo}\n"
            f"   (magno): {self.magno}\n"
            f"   (konio): {self.konio}\n)"
        )


class RGBToMagno(BatchAugment):
    """Convert RGB to 2-channel Magno output."""

    def __init__(self,
                 magno_sigma_c: float = 2.5, magno_sigma_s: float = 5.0,
                 nonlin: Callable = torch.nn.functional.relu,
                 device=None):
        self.magno = LMSToMagno(sigma_c=magno_sigma_c, sigma_s=magno_sigma_s, nonlin=nonlin, device=device)

    def __call__(self, b, **kwargs):
        """Input: RGB batch (*,3,H,W) [0..1]. Output: Magno (*,2,H,W)."""
        lms = F.srgb_to_lms(b)
        L = lms[..., 0:1, :, :]
        M = lms[..., 1:2, :, :]
        return self.magno(L, M)

    def apply_last(self, b):
        return self(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n   (magno): {self.magno}\n)"
