"""Tensor-only functional operations for GPU batch augmentations.

Ported from lrm-ssl fastaugs/functional.py. All numpy/PIL dispatch paths removed.
"""

import math
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor

from ._compat import mask_batch, mask_tensor  # noqa: F401


# =================================================
#  normalize
# =================================================

def normalize(x: torch.Tensor, mean, std, inplace=False):
    if not inplace:
        x = x.clone()

    if len(x.shape) == 4:
        x.sub_(mean[None, ..., None, None]).div_(std[None, ..., None, None])
    elif len(x.shape) == 3:
        x.sub_(mean[..., None, None]).div_(std[..., None, None])
    else:
        raise TypeError(f"unsupported shape, expected 3 or 4 dimensions, got: {x.shape}")

    return x


# =================================================
#  to_device, to_float, div_
# =================================================

def to_device(x, device):
    """Move tensor (or sequence of tensors) to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, (list, tuple)):
        return [to_device(item, device) for item in x]
    else:
        raise TypeError(f"unsupported type: {type(x)}")


def to_float(x):
    """Convert tensor to float."""
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, (list, tuple)):
        return [to_float(item) for item in x]
    else:
        raise TypeError(f"unsupported type: {type(x)}")


def div_(x, val=255.0, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return (x / val).to(dtype)
    elif isinstance(x, (list, tuple)):
        return [div_(samples, val).to(dtype) for samples in x]
    else:
        raise TypeError(f"unsupported type: {type(x)}")


# =================================================
#  to_channels_first, to_channels_last
# =================================================

def to_channels_first(x: torch.Tensor):
    if len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    elif len(x.shape) == 3:
        x = x.permute(2, 0, 1)  # HWC -> CHW
    return x


def to_channels_last(x: torch.Tensor):
    if len(x.shape) == 4:
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
    elif len(x.shape) == 3:
        x = x.permute(1, 2, 0)  # CHW -> HWC
    return x


# =================================================
#  to_torch_image
# =================================================

def _max_value(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 255
    elif dtype == torch.int16:
        return 32767
    elif dtype == torch.int32:
        return 2147483647
    else:
        return 1


def _is_hwc(b: torch.Tensor) -> bool:
    """Check if tensor is HWC/BHWC format (channels last).

    HWC: [H, W, 3] or [B, H, W, 3] where the last dimension is 3 (RGB channels).
    CHW: [3, H, W] or [B, 3, H, W] where channels are in dim -3.
    """
    if b.ndim == 3:
        # [H, W, 3] vs [3, H, W]
        return b.shape[-1] == 3 and b.shape[0] != 3
    elif b.ndim == 4:
        # [B, H, W, 3] vs [B, 3, H, W]
        return b.shape[-1] == 3 and b.shape[1] != 3
    return False


def to_torch_image(b, device="cpu", dtype=torch.float32, **kwargs):
    """Convert numpy array or tensor to torch image (CxHxW or BxCxHxW; 0-1).

    Accepts:
        - numpy array [H, W, 3] or [B, H, W, 3] (HWC)
        - numpy array [3, H, W] or [B, 3, H, W] (CHW)
        - torch tensor in any of the above formats

    Returns:
        torch.Tensor [C, H, W] or [B, C, H, W] float in [0, 1]

    Note:
        For GPU targets with HWC input, the order of operations is optimized:
        1. Transfer contiguous HWC data to GPU (fast DMA)
        2. Permute on GPU (essentially free)
        3. Cast to float and divide
        This avoids the slow path of transferring non-contiguous data.
    """
    import numpy as np

    # Handle numpy input
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    max_val = _max_value(b.dtype)
    needs_permute = _is_hwc(b)

    # Optimization: for GPU targets with HWC input, transfer contiguous data first,
    # then permute on GPU. Permuting on CPU creates a non-contiguous view, and
    # transferring non-contiguous data to GPU is very slow (can't use DMA).
    is_gpu = device != "cpu" and device is not None and str(device) != "cpu"

    if is_gpu and needs_permute:
        # GPU + HWC: transfer contiguous HWC first, permute on GPU
        b = b.to(device=device)
        b = to_channels_first(b)
        # Cast and normalize in one step
        return (b / max_val).to(dtype=dtype)
    else:
        # CPU path or already CHW
        if needs_permute:
            b = to_channels_first(b)
        # Transfer (if needed) and cast, then normalize
        b = b.to(device=device, dtype=dtype)
        return b / max_val


# =================================================
#  grayscale
# =================================================

def to_grayscale(x: torch.Tensor, num_output_channels=1):
    """Convert to grayscale using ITU-R 601-2 luma transform.

    Uses in-place add_ to minimize allocations (1 vs 5 intermediate tensors).
    r.mul() returns a new tensor; subsequent add_() ops are in-place on that
    new tensor, so the input x is never modified.
    Matches torchvision's _rgb_to_grayscale_image.
    """
    r, g, b = x.unbind(dim=-3)
    L = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    L = L.unsqueeze(dim=-3).to(x.dtype)
    if num_output_channels == 3:
        L = L.expand(x.shape)
    return L


def random_grayscale(b: torch.Tensor, idx, num_output_channels=3):
    """Convert random set of images in batch to grayscale.

    Strategy: compute grayscale for the full batch, then assign back only the
    selected images. This is ~1.8x faster than index_select + to_grayscale +
    index_copy because to_grayscale is cheap (scalar mul + in-place add, no
    allocation beyond the result), while index_select/index_copy each do a full
    copy of the selected images. PyTorch has no "index_view" — all advanced
    indexing (integer or boolean) returns a copy, not a view.
    """
    if len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return to_grayscale(b, num_output_channels=num_output_channels) if do else b
    elif len(b.shape) == 4:
        if len(idx) == 0:
            return b
        gray = to_grayscale(b, num_output_channels=num_output_channels)
        if len(idx) == b.shape[0]:
            return gray
        b[idx] = gray[idx]
        return b


# =================================================
#  YIQ color jitter
# =================================================

M_PI = math.pi

Rgb2Yiq = torch.tensor(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
    dtype=torch.float64,
).float()

# Exact inverse (DALI computes inverse(Rgb2Yiq); the old hardcoded approximation
# had ~0.3% error per roundtrip which corrupts colors on repeated application).
Yiq2Rgb = torch.inverse(
    torch.tensor(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
        dtype=torch.float64,
    )
).float()


def mat3(value):
    """Identity matrix scaled by value (per-batch)."""
    n = len(value)
    ret = torch.eye(3, device=value.device).float().unsqueeze(0).repeat(n, 1, 1)
    ret = ret * value.view(n, 1, 1)
    return ret


def hue_mat(hue):
    """Transformation matrix for hue rotation in YIQ space."""
    n = len(hue)
    h_rad = hue.mul(M_PI).div(180)
    ret = torch.eye(3, device=hue.device).float().unsqueeze(0).repeat(n, 1, 1)
    ret[:, 1, 1] = h_rad.cos()
    ret[:, 2, 2] = h_rad.cos()
    ret[:, 1, 2] = h_rad.sin()
    ret[:, 2, 1] = -h_rad.sin()
    return ret


def sat_mat(saturation):
    """Transformation matrix for saturation in YIQ space."""
    n = len(saturation)
    ret = torch.eye(3, device=saturation.device).float().unsqueeze(0).repeat(n, 1, 1)
    ret[:, 1, 1] = saturation
    ret[:, 2, 2] = saturation
    return ret


def val_mat(value):
    """Transformation matrix for value/brightness in YIQ space."""
    n = len(value)
    ret = torch.eye(3, device=value.device).float().unsqueeze(0).repeat(n, 1, 1)
    ret[:, 0, 0] = value
    return ret


def _get_hsv_mat(h, s, v):
    h = _ensure_1d_tensor(h)
    s = _ensure_1d_tensor(s)
    v = _ensure_1d_tensor(v)
    return Yiq2Rgb @ hue_mat(h.float()) @ sat_mat(s.float()) @ val_mat(v.float()) @ Rgb2Yiq


def _get_hsv_mat2(h, s, v, b, c):
    """Build fused color twist matrix: b * c * Yiq2Rgb * hue * sat * val * Rgb2Yiq.

    Fuses hue*sat*val into a single YIQ-space matrix to reduce the number of
    matmuls from 6 to 3. The brightness*contrast scalar is applied after.
    """
    h = _ensure_1d_tensor(h).float()
    s = _ensure_1d_tensor(s).float()
    v = _ensure_1d_tensor(v).float()
    b = _ensure_1d_tensor(b).float()
    c = _ensure_1d_tensor(c).float()
    device = h.device
    n = len(h)

    # Fuse hue_rot * sat * val into one matrix in YIQ space:
    # [v, 0, 0]   [1,    0,    0 ]   [v,         0,          0         ]
    # [0, s, 0] * [0, cos_h, sin_h] = [0, s*cos_h,  s*sin_h            ]
    # [0, 0, s]   [0,-sin_h, cos_h]   [0, -s*sin_h, s*cos_h            ]
    h_rad = h * (M_PI / 180.0)
    cos_h = h_rad.cos()
    sin_h = h_rad.sin()

    yiq_mat = torch.zeros(n, 3, 3, device=device)
    yiq_mat[:, 0, 0] = v
    yiq_mat[:, 1, 1] = s * cos_h
    yiq_mat[:, 1, 2] = s * sin_h
    yiq_mat[:, 2, 1] = -s * sin_h
    yiq_mat[:, 2, 2] = s * cos_h

    # Full transform: (b*c) * Yiq2Rgb @ yiq_mat @ Rgb2Yiq
    bc = (b * c).view(n, 1, 1)
    rgb2yiq = Rgb2Yiq.to(device)
    yiq2rgb = Yiq2Rgb.to(device)
    return bc * (yiq2rgb @ yiq_mat @ rgb2yiq)


def _ensure_1d_tensor(x):
    if torch.is_tensor(x):
        return x.unsqueeze(0) if x.ndim == 0 else x
    if isinstance(x, (int, float)):
        return torch.tensor([x])
    return torch.tensor(x)


def hsv_jitter_tensor(x, mat):
    if len(x.shape) == 4 and mat.shape[0] == x.shape[0]:
        out = (mat @ x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
    elif len(x.shape) == 4 and mat.shape[0] == 1:
        mat = mat.expand(x.size(0), 3, 3).contiguous()
        out = (mat @ x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
    elif len(x.shape) == 3 and mat.shape[0] == 1:
        out = (mat.squeeze() @ x.view(x.shape[0], -1)).view(x.shape)
    else:
        raise ValueError(f"Unsupported shapes: x={x.shape}, mat={mat.shape}")
    return out


def hsv_jitter(x, h, s, v):
    """HSV jitter via YIQ transform. Input: [0,1] float tensor, channels first."""
    mat = _get_hsv_mat(h, s, v)
    mat = mat.to(x.device, dtype=x.dtype)
    out = hsv_jitter_tensor(x, mat)
    return torch.clamp(out, 0, 1)


def hsv_jitter2(x: torch.Tensor, h, s, v, b, c):
    """Color jitter via YIQ transform (matches DALI color_twist).

    Applies hue rotation, saturation scaling, value scaling, contrast, and
    brightness as a single 3x3 matrix multiply + offset per pixel.

    output = M @ pixel + offset
    M = brightness * contrast * Yiq2Rgb * hue_rot * sat * val * Rgb2Yiq
    offset = (0.5 - 0.5 * contrast) * brightness  (for float [0,1] images)
    """
    was_3d = x.ndim == 3
    if was_3d:
        x = x.unsqueeze(0)

    mat = _get_hsv_mat2(h, s, v, b, c)
    mat = mat.to(x.device, dtype=torch.float32)

    # DALI contrast offset: adjust around midpoint (0.5), scaled by brightness.
    c_dev = c.to(x.device, dtype=torch.float32)
    b_dev = b.to(x.device, dtype=torch.float32)
    offset = (0.5 - 0.5 * c_dev) * b_dev
    offset = offset.view(-1, 1, 1, 1)

    out = hsv_jitter_tensor(x.to(torch.float32), mat) + offset
    out = torch.clamp(out, 0, 1.0).to(dtype=x.dtype)
    return out.squeeze(0) if was_3d else out


def random_hsv_jitter2(x: torch.Tensor, idx, h, s, v, b, c):
    """Randomly apply HSV jitter to each image in a batch."""
    if len(idx) == 0:
        return x
    if len(x.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return hsv_jitter2(x, h, s, v, b, c) if do else x
    elif len(x.shape) == 4:
        return x.index_copy(0, idx, hsv_jitter2(x.index_select(0, idx), h, s, v, b, c))


# =================================================
#  brightness
# =================================================

def _get_scale_factor(x: torch.Tensor, sf):
    if torch.is_tensor(sf):
        sf = sf.unsqueeze(0) if sf.ndim == 0 else sf
    else:
        sf = [sf] if isinstance(sf, (int, float)) else sf
        sf = torch.tensor(sf)

    if len(x.shape) == 4:
        n = x.shape[0]
        sf = sf.repeat(n) if len(sf) == 1 else sf

    return sf.to(x.device, non_blocking=True)


def adjust_brightness(x: torch.Tensor, scale_factor, max_value=255.0):
    scale_factor = _get_scale_factor(x, scale_factor)
    if len(x.shape) == 4:
        out = torch.clamp((x * scale_factor[:, None, None, None]), 0, max_value)
    elif len(x.shape) == 3:
        out = torch.clamp((x * scale_factor), 0, max_value)
    else:
        raise TypeError(f"unsupported shape: {x.shape}")
    return out


def random_adjust_brightness(b: torch.Tensor, scale_factor, idx, max_value):
    if len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return adjust_brightness(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4:
        return b.index_copy(
            0, idx, adjust_brightness(b.index_select(0, idx), scale_factor, max_value=max_value)
        )


# =================================================
#  contrast
# =================================================

def adjust_contrast(x: torch.Tensor, scale_factor, max_value=1.0):
    scale_factor = _get_scale_factor(x, scale_factor)
    if len(x.shape) == 4:
        # Per-image mean (matching torchvision's adjust_contrast)
        mean = to_grayscale(x, num_output_channels=1).mean(dim=(-3, -2, -1), keepdim=True)
        sf = scale_factor[:, None, None, None]
        out = torch.clamp(mean * (1.0 - sf) + x * sf, 0, max_value)
    elif len(x.shape) == 3:
        mean = to_grayscale(x, num_output_channels=1).mean()
        out = torch.clamp((mean * (1.0 - scale_factor) + x * scale_factor), 0, max_value)
    else:
        raise TypeError(f"unsupported shape: {x.shape}")
    return out


def random_adjust_contrast(b: torch.Tensor, scale_factor, idx, max_value):
    if len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return adjust_contrast(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4:
        return b.index_copy(
            0, idx, adjust_contrast(b.index_select(0, idx), scale_factor, max_value=max_value)
        )


# =================================================
#  affine transforms
# =================================================

_pi = torch.tensor([math.pi])

# Cache for base grids keyed by (h, w, dtype, device) to avoid reallocation.
_base_grid_cache: dict[tuple, torch.Tensor] = {}


def _prepare_mat(x, mat):
    """Prepare transformation matrix for grid_sample.

    Takes a [N, 3, 3] matrix built in normalized [-1,1] coordinates and applies
    aspect ratio correction, then slices to [N, 2, 3] for affine_grid/grid_sample.
    """
    h, w = x.shape[-2:]
    mat[:, 0, 1] *= h / w
    mat[:, 1, 0] *= w / h
    return mat[:, :2].to(x.device, dtype=x.dtype)


def _grid_sample(x, coords, mode="bilinear", padding_mode="reflection", align_corners=None):
    """Resample pixels using grid_sample with optional anti-aliasing."""
    needs_precision_fix = x.dtype == torch.float16 and x.device.type == "cpu"
    original_dtype = x.dtype if needs_precision_fix else None

    if needs_precision_fix:
        x = x.float()
        coords = coords.float()

    if mode == "bilinear":
        # Anti-aliasing: only needed when downsampling (output smaller than input).
        # Compute d from integer shape metadata first (free) to skip expensive
        # coords.min()/max() tensor reductions when not downsampling.
        d = min(x.shape[-2] / coords.shape[-2], x.shape[-1] / coords.shape[-1]) / 2
        if d > 1:
            mn, mx = coords.min(), coords.max()
            z = 1 / (mx - mn).item() * 2
            if d > z:
                x = F.interpolate(x, scale_factor=1 / d, mode="area")

    result = F.grid_sample(x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    if needs_precision_fix:
        result = result.to(original_dtype)

    return result


def _get_base_grid(h: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get or create a cached [1, h*w, 3] base grid for affine_grid via bmm.

    Uses torchvision's pixel-center coordinate convention (align_corners=False).
    """
    key = (h, w, dtype, device)
    grid = _base_grid_cache.get(key)
    if grid is not None:
        return grid
    base = torch.empty(1, h, w, 3, dtype=dtype, device=device)
    x_grid = torch.linspace((1.0 - w) * 0.5, (w - 1.0) * 0.5, steps=w, dtype=dtype, device=device)
    base[..., 0].copy_(x_grid)
    y_grid = torch.linspace((1.0 - h) * 0.5, (h - 1.0) * 0.5, steps=h, dtype=dtype, device=device).unsqueeze_(-1)
    base[..., 1].copy_(y_grid)
    base[..., 2].fill_(1)
    grid = base.view(1, h * w, 3)
    _base_grid_cache[key] = grid
    return grid


def affine_transform(x: torch.Tensor, mat, sz=None, align_corners=False, mode="bilinear", pad_mode="reflection"):
    """Apply affine transformation matrix using grid_sample.

    Uses a cached base grid + bmm for grid generation (like torchvision's _affine_grid),
    avoiding F.affine_grid overhead. For same-size transforms (sz=None), calls F.grid_sample
    directly without anti-aliasing checks.
    """
    expanded = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        expanded = True
    if len(mat.shape) == 2:
        mat = mat.unsqueeze(0)

    h, w = x.shape[-2:]
    oh, ow = (h, w) if sz is None else ((sz, sz) if isinstance(sz, int) else tuple(sz))

    if not align_corners and sz is None:
        # Fast path: cached base grid + bmm (matches torchvision's _affine_grid).
        # rescaled_theta normalizes pixel coords by image dimensions.
        dtype = x.dtype
        device = x.device
        base_grid = _get_base_grid(oh, ow, dtype, device)
        rescale = torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device)
        # mat is [B, 2, 3], transpose to [B, 3, 2], divide by rescale
        rescaled_theta = mat.transpose(1, 2) / rescale
        # Translation row (index 2) must stay in normalized [-1,1] space —
        # it multiplies with the homogeneous coordinate (1), not pixel coords.
        rescaled_theta[:, 2, :] = mat[:, :, 2]
        # bmm: [1, h*w, 3] x [B, 3, 2] -> [B, h*w, 2]  (broadcast on batch dim)
        if mat.shape[0] == 1:
            coords = base_grid.bmm(rescaled_theta).view(1, oh, ow, 2)
        else:
            coords = base_grid.expand(mat.shape[0], -1, -1).bmm(rescaled_theta).view(mat.shape[0], oh, ow, 2)
        # Same-size transform: anti-aliasing never triggers (d=0.5 < 1), skip _grid_sample.
        out = F.grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=False)
    else:
        # Fallback: use F.affine_grid for align_corners=True or resize cases
        size = x.shape[:2] + (oh, ow)
        coords = F.affine_grid(mat, size, align_corners=align_corners)
        out = _grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners)

    if expanded:
        out = out.squeeze()

    return out


# =================================================
#  flip matrix
# =================================================

def flip_mat(flip: torch.Tensor):
    """Create transformation matrix for horizontal flip."""
    if not torch.is_tensor(flip):
        flip = torch.tensor([flip]).float() if isinstance(flip, (int, float)) else torch.tensor(flip).float()
    if flip.ndim == 0:
        flip = flip.unsqueeze(0)
    n = len(flip)
    mat = torch.eye(3, device=flip.device, dtype=flip.dtype).unsqueeze(0).repeat(n, 1, 1)
    mat[:, 0, 0] = flip
    return mat


# =================================================
#  rotation matrix
# =================================================

def _prepare_thetas(thetas):
    """Convert degrees to radians tensor."""
    if not torch.is_tensor(thetas):
        if isinstance(thetas, (int, float)):
            thetas = torch.tensor([math.radians(thetas)]).float()
        elif isinstance(thetas, list):
            thetas = torch.tensor([math.radians(t) for t in thetas]).float()
        else:
            thetas = torch.tensor(thetas).float()
            thetas = thetas * _pi.to(thetas.device) / 180.0
    else:
        if thetas.ndim == 0:
            thetas = thetas.unsqueeze(0)
        thetas = thetas * _pi.to(thetas.device) / 180.0
    if thetas.ndim == 0:
        thetas = thetas.unsqueeze(0)
    return thetas


def _prepare_coords(coords):
    """Convert [0,1] coordinates to [-1,1]."""
    if not torch.is_tensor(coords):
        if isinstance(coords, (int, float)):
            coords = torch.tensor([coords])
        else:
            coords = torch.tensor(coords)
    if coords.ndim == 0:
        coords = coords.unsqueeze(0)
    return 2 * coords.float() - 1


def _prepare_param(param):
    """Ensure param is a tensor."""
    if torch.is_tensor(param):
        return param
    if isinstance(param, (int, float)):
        return torch.tensor([param]).float()
    return torch.tensor(param).float()


def rotate_mat(degrees, x=0.5, y=0.5):
    """Create transformation matrix for rotation by degrees around point (x, y).

    Convention: positive degrees = counter-clockwise (matches torchvision).
    """
    thetas = -_prepare_thetas(degrees)  # negate so positive = CCW (torchvision convention)
    x = _prepare_coords(x)
    y = _prepare_coords(y)

    n = len(thetas)
    mat = torch.eye(3, device=thetas.device).float().unsqueeze(0).repeat(n, 1, 1)
    cos_theta = thetas.cos()
    sin_theta = thetas.sin()

    mat[:, 0, 0] = cos_theta
    mat[:, 0, 1] = sin_theta
    mat[:, 0, 2] = x - cos_theta * x - sin_theta * y
    mat[:, 1, 0] = -sin_theta
    mat[:, 1, 1] = cos_theta
    mat[:, 1, 2] = y + sin_theta * x - cos_theta * y

    return mat


def zoom_mat(scale, col_pct=0.5, row_pct=0.5):
    """Create transformation matrix for zoom/scale around (col_pct, row_pct)."""
    scale = _prepare_param(scale)
    col_pct = _prepare_param(col_pct)
    row_pct = _prepare_param(row_pct)
    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    n = len(scale)
    mat = torch.eye(3, device=scale.device).float().unsqueeze(0).repeat(n, 1, 1)

    mat[:, 0, 0] = scale
    mat[:, 0, 2] = (1 - scale) * (2 * col_pct - 1)
    mat[:, 1, 1] = scale
    mat[:, 1, 2] = (1 - scale) * (2 * row_pct - 1)

    return mat


def translate_mat(tx, ty):
    """Create transformation matrix for translation."""
    tx = _prepare_param(tx) * 2
    ty = _prepare_param(ty) * 2
    n = len(tx)
    mat = torch.eye(3, device=tx.device).float().unsqueeze(0).repeat(n, 1, 1)
    mat[:, 0, 2] = tx
    mat[:, 1, 2] = ty
    return mat


def rotate_object_mat(degrees, ctr_x, ctr_y, scale, dest_x, dest_y):
    """Create transformation matrix for object rotation + scale + translation."""
    mat = rotate_mat(degrees, ctr_x, ctr_y)
    mat = mat @ zoom_mat(scale, col_pct=ctr_x, row_pct=ctr_y)
    if dest_x is not None and dest_y is not None:
        mat = mat @ translate_mat(ctr_x - dest_x, ctr_y - dest_y)
    return mat


# =================================================
#  grid_sample wrappers
# =================================================

def _check_precision(b):
    needs_precision_fix = b.dtype == torch.float16 and b.device.type == "cpu"
    original_dtype = b.dtype if needs_precision_fix else None
    return needs_precision_fix, original_dtype


def grid_sample(b: torch.Tensor, grid: torch.Tensor, align_corners=False):
    """Apply grid_sample to batch or single image."""
    needs_precision_fix, original_dtype = _check_precision(b)
    if needs_precision_fix:
        b = b.float()
        grid = grid.float()

    if len(b.shape) == 4:
        if grid.shape[0] == 1:
            grid = grid.expand(b.shape[0], *grid.shape[1:])
        x = F.grid_sample(b, grid, align_corners=align_corners)
    elif len(b.shape) == 3:
        x = F.grid_sample(b.unsqueeze(0), grid, align_corners=align_corners).squeeze()
    else:
        raise TypeError(f"unsupported shape: {b.shape}")

    if needs_precision_fix:
        x = x.to(original_dtype)
    return x


def random_grid_sample(b: torch.Tensor, idx, grid, align_corners=True):
    """Apply grid_sample to random subset of images in batch."""
    needs_precision_fix, original_dtype = _check_precision(b)
    if needs_precision_fix:
        b = b.float()
        grid = grid.float()

    if len(b.shape) == 4:
        b.index_copy_(0, idx, F.grid_sample(b.index_select(0, idx), grid, align_corners=align_corners))
        result = b
    elif len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        result = F.grid_sample(b.unsqueeze(0), grid, align_corners=align_corners).squeeze(0) if do else b
    else:
        raise TypeError(f"unsupported shape: {b.shape}")

    if needs_precision_fix:
        result = result.to(original_dtype)
    return result


# =================================================
#  apply_mask
# =================================================

def apply_mask(b: torch.Tensor, mask: torch.Tensor):
    if len(b.shape) == 4:
        return b * mask
    elif len(b.shape) == 3:
        return (b.unsqueeze(0) * mask).squeeze()
    else:
        raise TypeError(f"unsupported shape: {b.shape}")


# =================================================
#  solarization
# =================================================

def solarization(image, threshold):
    mask = image >= threshold
    if image.dtype == torch.uint8:
        return torch.where(mask, 255 - image, image)
    else:
        return torch.where(mask, 1 - image, image)


def random_solarization(b: torch.Tensor, idx, threshold):
    if len(b.shape) == 3:
        do = len(idx) == 1 and idx[0] == 0
        return solarization(b, threshold) if do else b
    elif len(b.shape) == 4:
        return b.index_copy(0, idx, solarization(b.index_select(0, idx), threshold))


# =================================================
#  gaussian blur
# =================================================

def random_gaussian_blur2d(x, blur_indices, selected_kernels, kernels, padding_mode="reflect"):
    if len(blur_indices) == 0:
        return x

    was_3d = x.ndim == 3
    if was_3d:
        x = x.unsqueeze(0)

    bs, c, h, w = x.shape
    blur_indices = blur_indices.to(x.device)
    selected_kernels = selected_kernels.to(x.device)
    kernels = kernels.to(x.device).to(x.dtype)

    kh, kw = kernels.shape[-2:]
    pad_top, pad_bottom = kh // 2, (kh - 1) // 2
    pad_left, pad_right = kw // 2, (kw - 1) // 2

    n_blur = blur_indices.shape[0]

    if n_blur == 1:
        # Single image: efficient depthwise conv, no reshape gymnastics
        kernel = kernels[selected_kernels[0]].unsqueeze(0).unsqueeze(0).expand(c, 1, kh, kw)
        sub = F.pad(x[blur_indices], (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)
        x[blur_indices] = F.conv2d(sub, kernel, groups=c, padding=0)
    else:
        # Batch: build per-image depthwise kernels, single grouped conv.
        # Each image's kernel is repeated C times for depthwise conv with groups=n_blur*C.
        # kernel_stack: [n_blur, kH, kW] -> [n_blur*C, 1, kH, kW]
        kernel_stack = kernels[selected_kernels]  # [n_blur, kH, kW]
        kernel_stack = kernel_stack.unsqueeze(1).expand(n_blur, c, kh, kw).reshape(n_blur * c, 1, kh, kw)
        x_blur = F.pad(x[blur_indices], (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)
        x_blur = x_blur.reshape(1, n_blur * c, h + pad_top + pad_bottom, w + pad_left + pad_right)
        x_blur = F.conv2d(x_blur, kernel_stack, groups=n_blur * c, padding=0)
        x[blur_indices] = x_blur.view(n_blur, c, h, w)

    return x.squeeze(0) if was_3d else x


# =================================================
#  batch permutations
# =================================================

def generate_batch_permutations(batch_size, N, rng=None):
    """Generate a batch of random permutations, each of size N."""
    base = torch.arange(N).repeat(batch_size, 1)
    random_rows = torch.rand(batch_size, N, generator=rng)
    _, indices = torch.sort(random_rows, dim=1)
    return torch.gather(base, 1, indices)


# =================================================
#  color space conversions
# =================================================

def srgb_to_lrgb(image: torch.Tensor, inplace=True) -> torch.Tensor:
    """Convert sRGB [0,1] to linear RGB (inverse gamma)."""
    if image.max() > 1:
        warnings.warn("srgb_to_lrgb: srgb appears to be outside the (0,1) range")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input must have shape (*, 3, H, W). Got {image.shape}")

    # torch.where avoids boolean indexing overhead (no masked assignment)
    low = image / 12.92
    high = ((image + 0.055) / 1.055).pow(2.4)
    return torch.where(image > 0.04045, high, low)


def lrgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to CIE XYZ."""
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    return torch.stack([x, y, z], -3)


def xyz_to_lms(image: torch.Tensor) -> torch.Tensor:
    """Convert CIE XYZ to CAT02 LMS."""
    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]
    l = 0.7328 * x + 0.4296 * y + -0.1624 * z
    m = -0.7036 * x + 1.6975 * y + 0.0061 * z
    s = 0.0030 * x + 0.0136 * y + 0.9834 * z
    return torch.stack([l, m, s], -3)


# Pre-computed sRGB→LMS matrix: CAT02_LMS @ sRGB_to_XYZ
# Fuses lrgb_to_xyz and xyz_to_lms into a single 3x3 matmul.
_SRGB_TO_LMS_MAT = torch.tensor([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
], dtype=torch.float32)
_CAT02_MAT = torch.tensor([
    [ 0.7328,  0.4296, -0.1624],
    [-0.7036,  1.6975,  0.0061],
    [ 0.0030,  0.0136,  0.9834],
], dtype=torch.float32)
_SRGB_TO_LMS_FUSED = (_CAT02_MAT @ _SRGB_TO_LMS_MAT)


def srgb_to_lms(image: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to CAT02 LMS via linear RGB.

    Uses a pre-computed fused 3x3 matrix (sRGB→XYZ→LMS) and einsum
    to avoid intermediate tensors and unbind/stack overhead.
    """
    linrgb = srgb_to_lrgb(image)
    # einsum: ...cHW, oc -> ...oHW (3x3 color matrix applied per-pixel)
    mat = _SRGB_TO_LMS_FUSED.to(device=linrgb.device, dtype=linrgb.dtype)
    return torch.einsum('...chw, oc -> ...ohw', linrgb, mat)


# =================================================
#  Gaussian kernels for DoG filtering
# =================================================

def make_gaussian_kernel(size: int, sigma: float, device=None) -> torch.Tensor:
    """Create a 2D Gaussian kernel normalized to sum=1."""
    if device is None:
        device = torch.device("cpu")
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    x_coords = coords.view(1, -1).repeat(size, 1)
    y_coords = coords.view(-1, 1).repeat(1, size)
    exponent = -(x_coords**2 + y_coords**2) / (2 * sigma**2)
    kernel = torch.exp(exponent)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def kernel_size_for_sigma(sigma: float) -> int:
    """Compute an odd kernel size ~6*sigma + 1."""
    size = int(6 * sigma) + 1
    return size if (size % 2 == 1) else size + 1


def dog_filter(b, kernel_c, kernel_s, pad_c=None, pad_s=None):
    """Difference of Gaussians filter."""
    pad_c = kernel_c.shape[-1] // 2 if pad_c is None else pad_c
    pad_s = kernel_s.shape[-1] // 2 if pad_s is None else pad_s
    center_out = F.conv2d(b, kernel_c, padding=pad_c)
    surround_out = F.conv2d(b, kernel_s, padding=pad_s)
    return center_out - surround_out
