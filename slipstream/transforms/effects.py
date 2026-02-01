"""Effect transforms: blur, solarization, patch shuffle, circular mask, optical distortion."""

import math
import torch
import torch.nn.functional as torch_F
from torch.nn.modules.utils import _pair
from .base import BatchAugment
from . import functional as F
from ._compat import mask_batch, mask_tensor


class RandomGaussianBlur(BatchAugment):
    """Gaussian blur augmentation (SimCLR style).

    Pre-computes a bank of kernels at init time for efficiency.
    """

    def __init__(self, p=0.5, kernel_size=6, sigma_range=(0.1, 2.0), num_sigmas=10,
                 seed=None, device=None):
        self.p = p
        self.kernel_size = _pair(kernel_size)
        self.sigma_range = sigma_range
        self.num_sigmas = num_sigmas

        self.sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_sigmas)
        if device is not None:
            self.sigmas = self.sigmas.to(device, non_blocking=True)
        self.kernels = self._get_gaussian_kernels(self.kernel_size, self.sigmas)
        if device is not None:
            self.kernels = self.kernels.to(device, non_blocking=True)

        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def _get_gaussian_kernel1d(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        return pdf / pdf.sum()

    def _get_gaussian_kernel2d(self, kernel_size, sigma):
        kernel1d_x = self._get_gaussian_kernel1d(kernel_size[0], sigma[0])
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size[1], sigma[1])
        return torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])

    def _get_gaussian_kernels(self, kernel_size, sigmas):
        kernels = [
            self._get_gaussian_kernel2d(self.kernel_size, _pair(sigma.item()))
            for sigma in sigmas
        ]
        return torch.stack(kernels)

    def before_call(self, b, **kwargs):
        _, self.idx = mask_batch(b, p=self.p, rng=self.rng)
        self.selected_kernels = torch.randint(
            0, self.kernels.shape[0], (self.idx.shape[0],),
            generator=self.rng, device=b.device,
        )
        if self.kernels.device != b.device:
            self.kernels = self.kernels.to(b.device, non_blocking=True)
        if self.kernels.dtype != b.dtype:
            self.kernels = self.kernels.to(b.dtype, non_blocking=True)

    def last_params(self):
        return {"idx": self.idx, "selected_kernels": self.selected_kernels}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_gaussian_blur2d(b, params["idx"], params["selected_kernels"], self.kernels)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.random_gaussian_blur2d(b, self.idx, self.selected_kernels, self.kernels)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, kernel_size={self.kernel_size}, "
            f"sigma_range={self.sigma_range}, num_sigmas={self.num_sigmas}, seed={self.seed})"
        )


class RandomSolarization(BatchAugment):
    """Randomly solarize images by inverting pixels above a threshold."""

    def __init__(self, p=0.5, threshold=0.5, seed=None, device=None):
        self.p = p
        self.threshold = threshold
        self.seed = seed
        self.rng = None
        self._init_rng(device)

    def _init_rng(self, device):
        if self.seed is not None and device is not None and self.rng is None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        self._init_rng(b.device)
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)

    def last_params(self):
        return {"do": self.do, "idx": self.idx}

    def apply_last(self, b):
        return F.random_solarization(b, self.last_params()["idx"], self.threshold)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.random_solarization(b, self.idx, self.threshold)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, threshold={self.threshold}, seed={self.seed})"


class RandomPatchShuffle(BatchAugment):
    """Randomly shuffle image patches (assumes square images)."""

    def __init__(self, sizes, p=0.5, seed=None, img_size=224, device=None):
        self.sizes = [sizes] if isinstance(sizes, (float, int)) else list(sizes)
        self.p = p
        self.img_size = img_size
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)
        self.base_grids = {}
        self.patches = {}
        self._init_grid_patches(img_size)

    def _init_grid_patches(self, out_size):
        if out_size in self.base_grids:
            return
        base_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, out_size, dtype=torch.float),
                torch.linspace(-1, 1, out_size, dtype=torch.float),
                indexing="xy",
            ),
            dim=-1,
        )
        self.base_grids[out_size] = base_grid
        self.patches[out_size] = {}
        for patch_size_pct in self.sizes:
            patch_size = int(patch_size_pct * out_size)
            patches = base_grid.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            patches = patches.reshape(-1, 2, patch_size, patch_size)
            self.patches[out_size][patch_size] = patches

    def _get_grid_patches(self, out_size, device, dtype):
        if out_size not in self.base_grids:
            self._init_grid_patches(out_size)
        for k, v in self.patches[out_size].items():
            self.patches[out_size][k] = v.to(device, dtype=dtype, non_blocking=True)
        return self.patches[out_size]

    def _get_shuffled_grids(self, b, rand_patch_sizes):
        H, W = b.shape[-2:]
        assert H == W, f"RandomPatchShuffle requires square inputs, got H={H}, W={W}"
        img_size = H
        patches = self._get_grid_patches(img_size, b.device, b.dtype)

        grid = torch.zeros(len(rand_patch_sizes), H, W, 2, device=b.device, dtype=b.dtype)
        for patch_size_pct in rand_patch_sizes.unique().tolist():
            patch_size = int(patch_size_pct * img_size)
            loc = torch.where(rand_patch_sizes == patch_size_pct)[0]
            if len(loc) == 0:
                continue

            n_patches = int(H / patch_size) * int(W / patch_size)
            shuffled_patch_idxs = F.generate_batch_permutations(len(loc), n_patches, rng=self.rng)
            shuffled_patches = patches[patch_size][shuffled_patch_idxs]

            B, nP, C, pH, pW = shuffled_patches.shape
            input_patches = shuffled_patches.reshape(B, -1, C * pH * pW).permute(0, 2, 1)

            output = torch_F.fold(
                input_patches,
                output_size=(H, W),
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
            )
            output = output.permute(0, 2, 3, 1)
            grid[loc] = output

        return grid

    def before_call(self, b, **kwargs):
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)
        n = len(self.idx)
        rand_patch_idxs = torch.randint(0, len(self.sizes), (n,), generator=self.rng)
        self.rand_patch_sizes = torch.tensor([self.sizes[idx] for idx in rand_patch_idxs])
        self.rand_grids = self._get_shuffled_grids(b, self.rand_patch_sizes)

    def last_params(self):
        return {
            "do": self.do, "idx": self.idx,
            "rand_patch_sizes": self.rand_patch_sizes, "rand_grids": self.rand_grids,
        }

    def apply_last(self, b):
        params = self.last_params()
        return F.random_grid_sample(b, idx=params["idx"], grid=params["rand_grids"], align_corners=True)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return self.apply_last(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, sizes={self.sizes}, seed={self.seed})"


def _cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi


class CircularMask(BatchAugment):
    """Apply a circular mask with smooth edges to each image."""

    def __init__(self, output_size, blur_span=8.0, tol=0.0005, device="cpu"):
        self.p = 1.0
        self.output_size = _pair(output_size)
        self.blur_span = blur_span
        self.tol = tol
        self.device = device
        self._gen_mask(self.output_size)

    @property
    def blur_radius(self):
        return self.output_size[0] / 2 - self.blur_span

    def _gen_mask(self, output_size):
        blur_span = torch.tensor(self.blur_span).float()
        zero = torch.tensor(0.0).float()

        y_vals = torch.tensor(range(output_size[0])).float()
        x_vals = torch.tensor(range(output_size[1])).float()
        y, x = torch.meshgrid(y_vals, x_vals, indexing="ij")
        rho, _ = _cart2pol(x - x_vals.mean(), y - y_vals.mean())

        scale = blur_span.clone()
        normal = torch.distributions.normal.Normal(loc=0, scale=scale)
        v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()
        while v > self.tol:
            scale = scale * 0.975
            normal = torch.distributions.normal.Normal(loc=0, scale=scale)
            v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()

        self.mask = normal.log_prob(rho - self.blur_radius).exp() / normal.log_prob(zero).exp()
        self.mask[rho < self.blur_radius] = 1.0
        self.mask = self.mask.view(1, 1, *self.mask.shape).to(self.device, non_blocking=True)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        self.do = mask_tensor(b.new_ones(n), p=self.p)

        if self.mask.shape[-2:] != b.shape[-2:]:
            self.output_size = tuple(b.shape[-2:])
            self._gen_mask(b.shape[-2:])

        if self.mask.device != b.device or self.mask.dtype != b.dtype:
            self.mask = self.mask.to(b.device, dtype=b.dtype)

    def last_params(self):
        return {"do": self.do, "mask": self.mask}

    def apply_last(self, b):
        return F.apply_mask(b, self.last_params()["mask"])

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.apply_mask(b, self.mask)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"blur_span={self.blur_span}, tol={self.tol}, device='{self.device}')"
        )


class FixedOpticalDistortion(BatchAugment):
    """Fixed optical (barrel/pincushion) distortion using pure-torch coordinate grid.

    Replaces cv2.initUndistortRectifyMap with a radial distortion formula.
    """

    def __init__(self, output_size, distortion=-0.5, dx=0, dy=0, device="cpu"):
        self.output_size = _pair(output_size)
        self.p = 1.0
        self.distortion = distortion
        self.dx = dx
        self.dy = dy
        self.device = device
        self.grid = self._gen_grid(self.distortion, self.dx, self.dy, self.width, self.height)

    @property
    def width(self):
        return self.output_size[1]

    @property
    def height(self):
        return self.output_size[0]

    def _gen_grid(self, k, dx, dy, width, height):
        """Generate distortion grid using radial distortion formula.

        This replaces cv2.initUndistortRectifyMap with a pure-torch implementation.
        """
        H, W = height, width
        # Normalized coordinates centered at (cx, cy)
        cx = width * 0.5 + dx
        cy = height * 0.5 + dy

        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32)
        x_coords = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Normalize to focal-length units
        x_norm = (xx - cx) / W
        y_norm = (yy - cy) / H

        # Radial distance squared
        r2 = x_norm**2 + y_norm**2

        # Apply radial distortion: x_distorted = x * (1 + k1*r^2 + k2*r^4)
        # Matches cv2.initUndistortRectifyMap with distortion=[k, k, 0, 0, 0]
        distortion_factor = 1.0 + k * r2 + k * r2**2
        x_distorted = x_norm * distortion_factor
        y_distorted = y_norm * distortion_factor

        # Convert back to pixel coordinates then normalize to [-1, 1] for grid_sample
        map_x = x_distorted * W + cx
        map_y = y_distorted * H + cy

        nmap1 = (map_x - W / 2) / (W / 2)
        nmap2 = (map_y - H / 2) / (H / 2)

        grid = torch.stack([nmap1, nmap2], dim=2).unsqueeze(0).to(self.device, non_blocking=True)
        return grid

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        self.do = mask_tensor(b.new_ones(n), p=self.p)

        if self.grid.shape[1:3] != b.shape[2:]:
            self.output_size = tuple(b.shape[-2:])
            self.grid = self._gen_grid(self.distortion, self.dx, self.dy, b.shape[-1], b.shape[-2])

        if self.grid.device != b.device or self.grid.dtype != b.dtype:
            self.grid = self.grid.to(b.device, dtype=b.dtype)

    def last_params(self):
        return {"do": self.do, "grid": self.grid}

    def apply_last(self, b):
        return F.grid_sample(b, self.last_params()["grid"], align_corners=False)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.grid_sample(b, self.grid, align_corners=False)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"distortion={self.distortion}, dx={self.dx}, dy={self.dy}, device='{self.device}')"
        )
