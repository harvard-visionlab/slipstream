"""Color jitter tensor operations for GPU batch augmentations.

Ported from lrm-ssl fastaugs/functional_tensor.py. Kornia dependency removed.
"""

import math
import torch
from torch import Tensor
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad
from typing import Optional, List

M_PI = math.pi


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def get_image_size(img: Tensor) -> List[int]:
    """Returns (w, h) of tensor image."""
    _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


def get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]
    raise TypeError(f"Input ndim should be 2 or more. Got {img.ndim}")


def _max_value(dtype: torch.dtype) -> float:
    a = torch.tensor(2, dtype=dtype)
    signed = 1 if torch.tensor(0, dtype=dtype).is_signed() else 0
    bits = 1
    max_value = torch.tensor(-signed, dtype=torch.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            break
    return max_value.item()


def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = get_image_num_channels(img)
    if c not in permitted:
        raise TypeError(f"Input image tensor permitted channel values are {permitted}, but found {c}")


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():
        if torch.tensor(0, dtype=dtype).is_floating_point():
            return image.to(dtype)
        eps = 1e-3
        max_val = _max_value(dtype)
        result = image.mul(max_val + 1.0 - eps)
        return result.to(dtype)
    else:
        input_max = _max_value(image.dtype)
        if torch.tensor(0, dtype=dtype).is_floating_point():
            image = image.to(dtype)
            return image / input_max
        output_max = _max_value(dtype)
        if input_max > output_max:
            factor = int((input_max + 1) // (output_max + 1))
            image = torch.div(image, factor, rounding_mode="floor")
            return image.to(dtype)
        else:
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(dtype)
            return image * factor


def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)
    return img.flip(-1)


def vflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)
    return img.flip(-2)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)
    w, h = get_image_size(img)
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)]
        return pad(img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0)
    return img[..., top:bottom, left:right]


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)
    return l_img


def _format_param(img: Tensor, param: float, expand_dims: int) -> Tensor:
    param = torch.tensor(param) if isinstance(param, float) else param
    if isinstance(param, (list,)):
        param = torch.tensor(param)
    if len(img.shape) >= 3:  # CxHxW or BxCxHxW
        dims_to_expand = expand_dims if len(img.shape) == 4 else max(0, expand_dims - 1)
        for _ in range(dims_to_expand):
            param = param.unsqueeze(-1)
    param = param.to(img.device, non_blocking=True)
    return param


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    brightness_factor = _format_param(img, brightness_factor, 3)
    if (brightness_factor < 0).any():
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")
    _assert_image_tensor(img)
    _assert_channels(img, [1, 3])
    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    contrast_factor = _format_param(img, contrast_factor, 3)
    if (contrast_factor < 0).any():
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    _assert_image_tensor(img)
    _assert_channels(img, [3, 1])
    c = get_image_num_channels(img)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    if c == 3:
        mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    else:
        mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return _blend(img, mean, contrast_factor)


def adjust_hue_fast(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue using fast in-place HSV conversion (no kornia dependency)."""
    hue_factor = _format_param(img, hue_factor, 2)

    if not ((-0.5 <= hue_factor) & (hue_factor <= 0.5)).all():
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    _assert_image_tensor(img)
    _assert_channels(img, [1, 3])
    if get_image_num_channels(img) == 1:
        return img

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    # TODO(BUG): In-place HSV conversion has numerical edge cases and can corrupt input.
    img = _rgb2hsv_fast(img)
    h = img.select(dim=-3, index=0)
    h.add_(hue_factor).fmod_(1.0)
    img_hue_adj = _hsv2rgb_fast(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    saturation_factor = _format_param(img, saturation_factor, 3)
    if (saturation_factor < 0).any():
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")
    _assert_image_tensor(img)
    _assert_channels(img, [1, 3])
    if get_image_num_channels(img) == 1:
        return img
    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def color_jitter(
    img: Tensor, hue_factor, saturation_factor, brightness_factor, contrast_factor
) -> Tensor:
    """Apply color jitter: brightness, contrast, saturation, hue."""
    if brightness_factor is not None:
        img = adjust_brightness(img, brightness_factor)
    if contrast_factor is not None:
        img = adjust_contrast(img, contrast_factor)
    if saturation_factor is not None:
        img = adjust_saturation(img, saturation_factor)
    if hue_factor is not None:
        img = adjust_hue_fast(img, hue_factor)
    return img


def random_color_jitter(
    b: Tensor, idx, hue_factor, saturation_factor, brightness_factor, contrast_factor
):
    """Run color jitter on subset of images specified by idx."""
    if len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return (
            color_jitter(b, hue_factor, saturation_factor, brightness_factor, contrast_factor)
            if do
            else b
        )
    elif len(b.shape) == 4:
        return b.index_copy(
            0,
            idx,
            color_jitter(
                b.index_select(0, idx),
                hue_factor,
                saturation_factor,
                brightness_factor,
                contrast_factor,
            ).to(dtype=b.dtype),
        )


# =================================================
#  HSV conversion
# =================================================


def _rgb2hsv(img: Tensor) -> Tensor:
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


# TODO(BUG): In-place HSV conversion has numerical edge cases and can corrupt input.
def _rgb2hsv_fast(img: Tensor) -> Tensor:
    r, g, b = img.select(dim=-3, index=0), img.select(dim=-3, index=1), img.select(dim=-3, index=2)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    img[..., 0, :, :] = h
    img[..., 1, :, :] = s
    img[..., 2, :, :] = maxc
    return img


def _hsv2rgb(img: Tensor) -> Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


# TODO(BUG): In-place HSV conversion has numerical edge cases and can corrupt input.
def _hsv2rgb_fast(img: Tensor) -> Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    was_3d = img.ndim == 3
    if was_3d:
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        v, q, p, t = v.unsqueeze(0), q.unsqueeze(0), p.unsqueeze(0), t.unsqueeze(0)

    bs, _, h_dim, w_dim = img.shape
    a4 = torch.zeros((bs, 3, 6, h_dim, w_dim), device=img.device)
    a4[:, 0, 0] = v;  a4[:, 0, 1] = q;  a4[:, 0, 2] = p
    a4[:, 0, 3] = p;  a4[:, 0, 4] = t;  a4[:, 0, 5] = v
    a4[:, 1, 0] = t;  a4[:, 1, 1] = v;  a4[:, 1, 2] = v
    a4[:, 1, 3] = q;  a4[:, 1, 4] = p;  a4[:, 1, 5] = p
    a4[:, 2, 0] = p;  a4[:, 2, 1] = p;  a4[:, 2, 2] = t
    a4[:, 2, 3] = v;  a4[:, 2, 4] = v;  a4[:, 2, 5] = q

    out = (mask.to(dtype=img.dtype).unsqueeze(-4) * a4).sum(dim=-3)
    return out.squeeze(0) if was_3d else out


# =================================================
#  Padding utilities
# =================================================


def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        crop_left, crop_right, crop_top, crop_bottom = [-min(x, 0) for x in padding]
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()
    _x_indices = [i for i in range(in_sizes[-1])]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]
    right_indices = [-(i + 1) for i in range(padding[1])]
    x_indices = torch.tensor(left_indices + _x_indices + right_indices, device=img.device)

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + _y_indices + bottom_indices, device=img.device)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    _assert_image_tensor(img)

    if isinstance(padding, tuple):
        padding = list(padding)
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (torch.float32, torch.float64):
        need_cast = True
        img = img.to(torch.float32)

    img = torch_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)
    if need_cast:
        img = img.to(out_dtype)

    return img
