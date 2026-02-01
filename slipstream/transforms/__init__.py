"""GPU batch augmentation transforms for slipstream.

Ported from lrm-ssl fastaugs. Tensor-only (no numpy/PIL), no kornia/albumentations/cv2.
"""

# Base classes
from .base import BatchAugment, Compose, RandomApply, MultiSample

# Conversion transforms
from .conversion import ToTorchImage, ToNumpy, ToChannelsFirst, ToChannelsLast, ToDevice, ToFloat, ToFloatDiv

# Normalization
from .normalization import Normalize, NormalizeLGN

# Grayscale
from .grayscale import ToGrayscale, ToGrayscaleTorch, RandomGrayscale

# Brightness/Contrast
from .brightness_contrast import RandomBrightness, RandomContrast

# Color jitter
from .color_jitter import ColorJitter, RandomColorJitter, RandomColorJitterHSV, RandomColorJitterYIQ

# Geometric
from .geometric import RandomHorizontalFlip, RandomRotate, RandomZoom, RandomRotateObject

# Effects
from .effects import (
    RandomGaussianBlur,
    RandomSolarization,
    RandomPatchShuffle,
    CircularMask,
    FixedOpticalDistortion,
)

# Color space
from .color_space import SRGBToLMS, LMSToParvo, LMSToMagno, LMSToKonio, RGBToLGN, RGBToMagno

__all__ = [
    # Base
    "BatchAugment",
    "Compose",
    "RandomApply",
    "MultiSample",
    # Conversion
    "ToTorchImage",
    "ToNumpy",
    "ToChannelsFirst",
    "ToChannelsLast",
    "ToDevice",
    "ToFloat",
    "ToFloatDiv",
    # Normalization
    "Normalize",
    "NormalizeLGN",
    # Grayscale
    "ToGrayscale",
    "ToGrayscaleTorch",
    "RandomGrayscale",
    # Brightness/Contrast
    "RandomBrightness",
    "RandomContrast",
    # Color jitter
    "ColorJitter",
    "RandomColorJitter",
    "RandomColorJitterHSV",
    "RandomColorJitterYIQ",
    # Geometric
    "RandomHorizontalFlip",
    "RandomRotate",
    "RandomZoom",
    "RandomRotateObject",
    # Effects
    "RandomGaussianBlur",
    "RandomSolarization",
    "RandomPatchShuffle",
    "CircularMask",
    "FixedOpticalDistortion",
    # Color space
    "SRGBToLMS",
    "LMSToParvo",
    "LMSToMagno",
    "LMSToKonio",
    "RGBToLGN",
    "RGBToMagno",
]
