"""Multi-crop decode stages for SSL.

- DecodeUniformMultiRandomResizedCrop: decode-once + N uniform random crops (legacy)
- DecodeMultiRandomResizedCrop: decode-once + N named crops with per-crop params
- MultiCropPipeline: apply per-crop transform chains to named crop dict
- NamedCopies: duplicate a single array/tensor into a named dict for MultiCropPipeline
"""

from __future__ import annotations

from typing import Any

import copy

import numpy as np
import torch

from numba import njit

from slipstream.decoders.base import BatchTransform
from slipstream.decoders.numba_decoder import NumbaBatchDecoder


@njit(cache=True)
def _embed_batch_rgba(
    crop: np.ndarray,       # [B, crop_h, crop_w, 3] uint8
    canvas: np.ndarray,     # [B, cs, cs, 4] uint8 (zeros, modified in-place)
    rects: np.ndarray,      # [B, 4] int32 (output: x0, y0, copy_w, copy_h)
    seed: int,
    x_range_min: float,
    x_range_max: float,
    y_range_min: float,
    y_range_max: float,
) -> None:
    """Embed a uniform-size crop batch onto RGBA canvases (Numba JIT)."""
    B = crop.shape[0]
    crop_h = crop.shape[1]
    crop_w = crop.shape[2]
    cs = canvas.shape[1]

    for i in range(B):
        np.random.seed((seed + i) % 2147483647)
        x_frac = np.random.uniform(x_range_min, x_range_max)
        y_frac = np.random.uniform(y_range_min, y_range_max)

        slack_x = cs - crop_w
        if slack_x < 0:
            slack_x = 0
        slack_y = cs - crop_h
        if slack_y < 0:
            slack_y = 0

        x0 = int(x_frac * slack_x)
        y0 = int(y_frac * slack_y)

        copy_h = crop_h
        if y0 + copy_h > cs:
            copy_h = cs - y0
        copy_w = crop_w
        if x0 + copy_w > cs:
            copy_w = cs - x0

        for row in range(copy_h):
            for col in range(copy_w):
                canvas[i, y0 + row, x0 + col, 0] = crop[i, row, col, 0]
                canvas[i, y0 + row, x0 + col, 1] = crop[i, row, col, 1]
                canvas[i, y0 + row, x0 + col, 2] = crop[i, row, col, 2]
                canvas[i, y0 + row, x0 + col, 3] = 255

        rects[i, 0] = x0
        rects[i, 1] = y0
        rects[i, 2] = copy_w
        rects[i, 3] = copy_h


def _get_yuv420_decoder_class() -> type:
    from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
    return YUV420NumbaBatchDecoder


def _swap_yuv420_if_needed(decoder, image_format: str):
    if image_format == "yuv420" and not isinstance(decoder, _get_yuv420_decoder_class()):
        nt = decoder.num_threads
        decoder.shutdown()
        return _get_yuv420_decoder_class()(num_threads=nt)
    return decoder


class DecodeUniformMultiRandomResizedCrop(BatchTransform):
    """Decode-once + N random crops with uniform parameters (legacy).

    Decodes each JPEG once, then applies N different random crops from the
    same decoded image. All crops share the same size, scale, and ratio.
    Much faster than N separate decode stages since JPEG decode (~80-92%
    of per-image time) happens only once.

    For per-crop parameters (different sizes, scales), use
    :class:`DecodeMultiRandomResizedCrop` instead.

    Args:
        num_crops: Number of random crop views per image.
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seeds: Per-crop seeds for reproducibility. None = auto.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        List of num_crops arrays, each [B, size, size, 3] uint8 (HWC, default)
        or [B, 3, size, size] uint8 (CHW if permute=True).

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.

    Example:
        multi_crop = DecodeUniformMultiRandomResizedCrop(num_crops=2, size=224)
        views = multi_crop(batch_data)  # [array1, array2]
    """

    def __init__(
        self,
        num_crops: int = 2,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seeds: list[int | None] | None = None,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        self.num_crops = num_crops
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seeds = seeds
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> list[torch.Tensor] | list[np.ndarray]:
        crops = self._decoder.decode_batch_multi_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            num_crops=self.num_crops, target_size=self.size,
            scale=self.scale, ratio=self.ratio, seeds=self.seeds,
        )
        if self.permute:
            crops = self._decoder.multi_hwc_to_chw(crops)
        if self.to_tensor:
            return [torch.from_numpy(c) for c in crops]
        return crops

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return (
            f"DecodeUniformMultiRandomResizedCrop(num_crops={self.num_crops}, size={self.size}, "
            f"scale={self.scale}, ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}){extra})"
        )


class DecodeMultiRandomResizedCrop(BatchTransform):
    """Decode-once + N named crops with per-crop parameters.

    Each crop can have a different target size, scale range, ratio range,
    and seed. Decodes each JPEG once, then applies all crops from the same
    decoded image data. Returns a dict of named arrays.

    Args:
        crops: Dict mapping crop names to parameter dicts.
            Required key: ``size`` (int).
            Optional keys: ``scale`` (tuple), ``ratio`` (tuple), ``seed`` (int).
        ratio: Default aspect ratio range (used if not specified per crop).
        crop_mode: ``"standard"`` (torchvision-compatible, 10-attempt rejection
            sampling) or ``"direct"`` (analytic, no rejection loop).
        num_threads: Parallel decode threads. 0 = auto.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        Dict[str, np.ndarray] or Dict[str, torch.Tensor] — named crops,
        each [B, size, size, 3] uint8 (HWC, default) or [B, 3, size, size] (CHW).

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.

    Yoked crops:
        Crops with the **same seed** share the same random number sequence,
        producing crops centered on the same point. Combined with different
        scale ranges, this gives a zoomed-in / zoomed-out pair::

            DecodeMultiRandomResizedCrop({
                "zoom_out": dict(size=224, scale=(0.4, 1.0), seed=42),
                "zoom_in":  dict(size=224, scale=(0.05, 0.4), seed=42),
            })

    Example:
        multi = DecodeMultiRandomResizedCrop({
            "global_0": dict(size=224, scale=(0.4, 1.0), seed=42),
            "global_1": dict(size=224, scale=(0.4, 1.0), seed=43),
            "local_0":  dict(size=96,  scale=(0.05, 0.4), seed=44),
            "local_1":  dict(size=96,  scale=(0.05, 0.4), seed=45),
        })
        named_crops = multi(batch_data)  # {"global_0": array, ...}
    """

    def __init__(
        self,
        crops: dict[str, dict],
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        crop_mode: str = "standard",
        num_threads: int = 0,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        if crop_mode not in ("standard", "direct"):
            raise ValueError(f"crop_mode must be 'standard' or 'direct', got '{crop_mode}'")
        self.ratio = ratio
        self.crop_mode = crop_mode
        self.num_threads = num_threads
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

        self._crop_names: list[str] = []
        self._crop_sizes: list[int] = []
        self._crop_scales: list[tuple[float, float]] = []
        self._crop_ratios: list[tuple[float, float]] = []
        self._crop_seeds: list[int | None] = []

        for name, params in crops.items():
            if 'size' not in params:
                raise ValueError(f"Crop '{name}' must specify 'size'")
            self._crop_names.append(name)
            self._crop_sizes.append(params['size'])
            self._crop_scales.append(params.get('scale', (0.08, 1.0)))
            self._crop_ratios.append(params.get('ratio', ratio))
            self._crop_seeds.append(params.get('seed', None))

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> dict[str, torch.Tensor]:
        import math
        from slipstream.decoders.numba_decoder import (
            _generate_random_crop_params_batch,
            _generate_direct_random_crop_params_batch,
        )

        _generate_fn = (
            _generate_direct_random_crop_params_batch
            if self.crop_mode == "direct"
            else _generate_random_crop_params_batch
        )

        data = batch_data['data']
        sizes = batch_data['sizes']
        heights = batch_data['heights']
        widths = batch_data['widths']

        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        num_crops = len(self._crop_names)

        self._decoder._seed_counter += 1
        batch_offset = self._decoder._seed_counter
        batch_size = len(batch_data['sizes'])

        crop_params_list = []
        for c in range(num_crops):
            scale = self._crop_scales[c]
            ratio = self._crop_ratios[c]
            seed = self._crop_seeds[c]

            log_ratio_min = math.log(ratio[0])
            log_ratio_max = math.log(ratio[1])

            if seed is not None:
                batch_seed = (seed + batch_size * batch_offset) % 2147483647
            else:
                batch_seed = (batch_size * (batch_offset * num_crops + c)) % 2147483647

            params = _generate_fn(
                widths_i32, heights_i32,
                scale[0], scale[1],
                log_ratio_min, log_ratio_max,
                batch_seed,
            )
            crop_params_list.append(params)

        all_same_size = len(set(self._crop_sizes)) == 1

        crops_hwc = self._decoder.decode_batch_multi_crop_varied(
            data, sizes, heights, widths,
            crop_params_list=crop_params_list,
            target_sizes=self._crop_sizes,
        )

        if self.permute:
            if all_same_size:
                crops_out = self._decoder.multi_hwc_to_chw(crops_hwc)
            else:
                crops_out = self._decoder.multi_hwc_to_chw_varied(crops_hwc)
        else:
            crops_out = crops_hwc

        result = {}
        for c, name in enumerate(self._crop_names):
            if self.to_tensor:
                result[name] = torch.from_numpy(crops_out[c])
            else:
                result[name] = crops_out[c]
        return result

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        crop_strs = []
        for c, name in enumerate(self._crop_names):
            crop_strs.append(
                f"'{name}': size={self._crop_sizes[c]}, "
                f"scale={self._crop_scales[c]}"
            )
        mode = f", crop_mode='{self.crop_mode}'" if self.crop_mode != "standard" else ""
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return f"DecodeMultiRandomResizedCrop({{{', '.join(crop_strs)}}}{mode}{extra})"


class DecodeMultiRandomResizeShortCropLong(BatchTransform):
    """Decode-once + N named resize-short-crop-long views.

    Each crop resizes the short edge to its target size (preserving aspect
    ratio) and crops the long edge, with controllable crop position. Decodes
    each JPEG once, then applies all crops from the same decoded image.

    Args:
        crops: Dict mapping crop names to parameter dicts.
            Required key: ``size`` (int or tuple ``(min, max)`` for random
            size sampling).
            Optional keys: ``x_range`` (float or tuple), ``y_range``
            (float or tuple), ``seed`` (int), ``size_mode`` (str).
        size_mode: Default size mode for all crops. ``"per_batch"`` samples
            one size shared by all images (output is a stacked array).
            ``"per_image"`` samples independently per image (output is a
            list of arrays). Can be overridden per crop. Ignored when
            ``size`` is a fixed int. Default: ``"per_batch"``.
        num_threads: Parallel decode threads. 0 = auto.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC->CHW. If False, keep HWC layout.

    Returns:
        Fixed size or per_batch mode:
            Dict[str, np.ndarray] or Dict[str, torch.Tensor] — named crops,
            each [B, size, size, 3] uint8 (HWC, default) or [B, 3, size, size] (CHW).
        Per_image mode:
            Dict[str, list] — named crops, each a list of per-image arrays.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.

    Yoked crops:
        Crops with the **same seed** share the same random number sequence,
        producing crops at the same position. Combined with different sizes,
        this gives a multi-scale view of the same region::

            DecodeMultiRandomResizeShortCropLong({
                "large": dict(size=224, x_range=(0, 1), seed=42),
                "small": dict(size=96, x_range=(0, 1), seed=42),
            })

    Example::

        multi = DecodeMultiRandomResizeShortCropLong({
            "view_224": dict(size=224, x_range=(0.0, 1.0), y_range=(0.0, 1.0), seed=42),
            "view_448": dict(size=448, x_range=0.5, y_range=0.5),
        })
        named = multi(batch_data)  # {"view_224": array, "view_448": array}
    """

    def __init__(
        self,
        crops: dict[str, dict],
        size_mode: str = "per_batch",
        num_threads: int = 0,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        if size_mode not in ("per_batch", "per_image"):
            raise ValueError(f"size_mode must be 'per_batch' or 'per_image', got '{size_mode}'")
        self.size_mode = size_mode
        self.num_threads = num_threads
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)
        self._last_params: dict = {}

        self._crop_names: list[str] = []
        self._crop_size_ranges: list[tuple[int, int]] = []
        self._crop_size_modes: list[str] = []
        self._crop_x_ranges: list[tuple[float, float]] = []
        self._crop_y_ranges: list[tuple[float, float]] = []
        self._crop_seeds: list[int | None] = []

        for name, params in crops.items():
            if 'size' not in params:
                raise ValueError(f"Crop '{name}' must specify 'size'")
            self._crop_names.append(name)
            s = params['size']
            self._crop_size_ranges.append((s, s) if isinstance(s, int) else tuple(s))
            sm = params.get('size_mode', size_mode)
            if sm not in ("per_batch", "per_image"):
                raise ValueError(f"Crop '{name}': size_mode must be 'per_batch' or 'per_image', got '{sm}'")
            self._crop_size_modes.append(sm)
            xr = params.get('x_range', (0.5, 0.5))
            yr = params.get('y_range', (0.5, 0.5))
            self._crop_x_ranges.append(
                (xr, xr) if isinstance(xr, (int, float)) else tuple(xr)
            )
            self._crop_y_ranges.append(
                (yr, yr) if isinstance(yr, (int, float)) else tuple(yr)
            )
            self._crop_seeds.append(params.get('seed', None))

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any] | bytes | bytearray | memoryview) -> dict[str, torch.Tensor | np.ndarray]:
        if isinstance(batch_data, (bytes, bytearray, memoryview)):
            from slipstream.decoders.base import _bytes_to_batch_dict, _unwrap_single_result
            return _unwrap_single_result(self(_bytes_to_batch_dict(batch_data)))

        from slipstream.decoders.numba_decoder import (
            _compute_resize_short_crop_long_params,
            _generate_resize_short_crop_long_params_batch,
        )

        data = batch_data['data']
        sizes = batch_data['sizes']
        heights = batch_data['heights']
        widths = batch_data['widths']
        batch_size = len(sizes)
        num_crops = len(self._crop_names)

        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        self._decoder._seed_counter += 1
        batch_offset = self._decoder._seed_counter

        # Check if any crop has variable per-image sizes
        has_variable_sizes = any(
            sr[0] != sr[1] and sm == "per_image"
            for sr, sm in zip(self._crop_size_ranges, self._crop_size_modes)
        )

        # Generate crop_params for each named crop
        crop_params_list = []
        x_pos_list = []
        y_pos_list = []
        target_sizes_list = []
        for c in range(num_crops):
            size_range = self._crop_size_ranges[c]
            x_range = self._crop_x_ranges[c]
            y_range = self._crop_y_ranges[c]
            seed = self._crop_seeds[c]

            # Compute per-image seed (same formula as DecodeMultiRandomResizedCrop)
            if seed is not None:
                batch_seed = (seed + batch_size * batch_offset) % 2147483647
            else:
                batch_seed = (batch_size * (batch_offset * num_crops + c)) % 2147483647

            # Determine target size for this crop
            crop_size_mode = self._crop_size_modes[c]
            if size_range[0] == size_range[1]:
                # Fixed size
                target_size = size_range[0]
            elif crop_size_mode == "per_batch":
                # One random size for the whole batch
                rng = np.random.RandomState(
                    (batch_seed + batch_size) % 2147483647
                )
                target_size = int(rng.randint(size_range[0], size_range[1] + 1))
            else:
                target_size = None  # per_image — handled in slow path

            if target_size is not None and not has_variable_sizes:
                # Fast path: JIT-compiled RNG + param computation
                x_pos = np.empty(batch_size, dtype=np.float64)
                y_pos = np.empty(batch_size, dtype=np.float64)
                params = _generate_resize_short_crop_long_params_batch(
                    widths_i32, heights_i32,
                    target_size,
                    x_range[0], x_range[1],
                    y_range[0], y_range[1],
                    batch_seed,
                    x_pos, y_pos,
                )
                target_sizes_arr = np.full(batch_size, target_size, dtype=np.int32)
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)
            else:
                # Slow path: per-image variable sizes need Python RNG
                x_pos = np.empty(batch_size, dtype=np.float64)
                y_pos = np.empty(batch_size, dtype=np.float64)
                for i in range(batch_size):
                    rng_i = np.random.RandomState((batch_seed + i) % 2147483647)
                    x_pos[i] = rng_i.uniform(x_range[0], x_range[1])
                    y_pos[i] = rng_i.uniform(y_range[0], y_range[1])

                if target_size is not None:
                    target_sizes_arr = np.full(batch_size, target_size, dtype=np.int32)
                else:
                    # Per-image random sizes
                    target_sizes_arr = np.empty(batch_size, dtype=np.int32)
                    for i in range(batch_size):
                        rng_i = np.random.RandomState(
                            (batch_seed + batch_size + i) % 2147483647
                        )
                        target_sizes_arr[i] = int(
                            rng_i.randint(size_range[0], size_range[1] + 1)
                        )

                params = _compute_resize_short_crop_long_params(
                    heights, widths, target_sizes_arr, x_pos, y_pos,
                )
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)

            crop_params_list.append(params)
            target_sizes_list.append(target_sizes_arr)

        # Store last_params for inspection
        self._last_params = {
            "heights": np.asarray(heights, dtype=np.int32),
            "widths": np.asarray(widths, dtype=np.int32),
            "crops": {
                name: {
                    "target_sizes": target_sizes_list[c],
                    "x_pos": x_pos_list[c],
                    "y_pos": y_pos_list[c],
                    "crop_params": crop_params_list[c],
                }
                for c, name in enumerate(self._crop_names)
            },
        }

        if has_variable_sizes:
            # Per-image variable sizes: decode each image individually
            crops_hwc = self._decode_variable_sizes(
                data, sizes, heights, widths,
                target_sizes_list, x_pos_list, y_pos_list,
            )
        else:
            # All crops have uniform size within each batch (fixed or per_batch)
            # Use the actual sampled size (first element, since all are equal)
            fixed_sizes = [int(ts[0]) for ts in target_sizes_list]
            all_same_size_flag = len(set(fixed_sizes)) == 1

            crops_hwc = self._decoder.decode_batch_multi_crop_varied(
                data, sizes, heights, widths,
                crop_params_list=crop_params_list,
                target_sizes=fixed_sizes,
            )

        if has_variable_sizes:
            # Variable sizes → each crop is a list of arrays, not a stacked array
            if self.permute:
                crops_out = [
                    [np.ascontiguousarray(img.transpose(2, 0, 1)) for img in crop]
                    for crop in crops_hwc
                ]
            else:
                crops_out = crops_hwc

            result = {}
            for c, name in enumerate(self._crop_names):
                if self.to_tensor:
                    result[name] = [torch.from_numpy(img) for img in crops_out[c]]
                else:
                    result[name] = crops_out[c]
            return result
        else:
            if self.permute:
                if all_same_size_flag:
                    crops_out = self._decoder.multi_hwc_to_chw(crops_hwc)
                else:
                    crops_out = self._decoder.multi_hwc_to_chw_varied(crops_hwc)
            else:
                crops_out = crops_hwc

            result = {}
            for c, name in enumerate(self._crop_names):
                if self.to_tensor:
                    result[name] = torch.from_numpy(crops_out[c])
                else:
                    result[name] = crops_out[c]
            return result

    def _decode_variable_sizes(
        self,
        data, sizes, heights, widths,
        target_sizes_list, x_pos_list, y_pos_list,
    ) -> list[list[np.ndarray]]:
        """Decode with per-image variable target sizes.

        Returns a list of crops, where each crop is a list of per-image arrays
        (since images may have different sizes and can't be stacked).
        """
        num_crops = len(self._crop_names)

        # Decode each crop using the existing per-image variable-size path
        result: list[list[np.ndarray]] = []
        for c in range(num_crops):
            crop_result = self._decoder.decode_batch_resize_short_crop_long(
                data, sizes, heights, widths,
                target_sizes=target_sizes_list[c],
                x_pos=x_pos_list[c],
                y_pos=y_pos_list[c],
            )
            # If all sizes happened to be equal, it returns a stacked array;
            # convert to list for uniform handling
            if isinstance(crop_result, np.ndarray) and crop_result.ndim == 4:
                crop_result = [crop_result[i] for i in range(crop_result.shape[0])]
            result.append(crop_result)
        return result

    @property
    def last_params(self) -> dict:
        """Parameters from the most recent ``__call__``."""
        return self._last_params

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        crop_strs = []
        for c, name in enumerate(self._crop_names):
            sr = self._crop_size_ranges[c]
            size_str = sr[0] if sr[0] == sr[1] else sr
            parts = [f"size={size_str}"]
            if sr[0] != sr[1]:
                parts.append(f"size_mode='{self._crop_size_modes[c]}'")
            if self._crop_x_ranges[c] != (0.5, 0.5):
                parts.append(f"x_range={self._crop_x_ranges[c]}")
            if self._crop_y_ranges[c] != (0.5, 0.5):
                parts.append(f"y_range={self._crop_y_ranges[c]}")
            if self._crop_seeds[c] is not None:
                parts.append(f"seed={self._crop_seeds[c]}")
            crop_strs.append(f"'{name}': {', '.join(parts)}")
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return f"DecodeMultiRandomResizeShortCropLong({{{', '.join(crop_strs)}}}{extra})"


class MultiCropPipeline(BatchTransform):
    """Apply per-crop transform pipelines to a dict of named values.

    Takes a dict of named values (e.g., from DecodeMultiRandomResizedCrop) and
    applies a separate pipeline to each. Generic — works with any transforms.

    Args:
        pipelines: Dict mapping names to lists of transforms (any callable).

    Example:
        from slipstream.transforms import ToTorchImage, Normalize

        pipe = MultiCropPipeline({
            "global_0": [ToTorchImage('cuda'), Normalize(mean, std)],
            "local_0":  [ToTorchImage('cuda'), Normalize(mean, std)],
        })
        result = pipe(named_crops)  # {"global_0": tensor, "local_0": tensor}
    """

    def __init__(self, pipelines: dict[str, list]) -> None:
        self.pipelines = pipelines

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for name, pipeline in self.pipelines.items():
            val = data[name]
            for transform in pipeline:
                val = transform(val)
            result[name] = val
        for name in data:
            if name not in self.pipelines:
                result[name] = data[name]
        return result

    def set_image_format(self, image_format: str) -> None:
        for pipeline in self.pipelines.values():
            for transform in pipeline:
                if hasattr(transform, 'set_image_format'):
                    transform.set_image_format(image_format)

    def shutdown(self) -> None:
        for pipeline in self.pipelines.values():
            for transform in pipeline:
                if hasattr(transform, 'shutdown'):
                    transform.shutdown()

    def __repr__(self) -> str:
        pipe_strs = []
        for name, pipeline in self.pipelines.items():
            transforms = [type(t).__name__ for t in pipeline]
            pipe_strs.append(f"'{name}': [{', '.join(transforms)}]")
        return f"MultiCropPipeline({{{', '.join(pipe_strs)}}})"


class NamedCopies(BatchTransform):
    """Duplicate a single value into a named dict for MultiCropPipeline.

    Takes a single array or tensor (e.g., from DecodeCenterCrop or
    DecodeResizeCrop) and produces a dict of deep copies keyed by name.
    This bridges single-output decoders with MultiCropPipeline, which
    expects a dict input.

    Each copy is independent (deep-copied) so that downstream transforms
    can safely mutate them in-place.

    Args:
        names: List of names for the copies (e.g., ``['view1', 'view2']``).

    Returns:
        Dict mapping each name to a deep copy of the input.

    Example:
        Decode once, create two named views, apply different transforms::

            pipelines = {'image': [
                DecodeResizeCrop(resize_size=256, crop_size=224),
                NamedCopies(['view1', 'view2']),
                MultiCropPipeline({
                    'view1': [ToTorchImage(device='cuda'), RandomZoom(zoom=(1.0, 1.0))],
                    'view2': [ToTorchImage(device='cuda'), RandomZoom(zoom=(0.5, 0.5))],
                }),
            ]}
    """

    def __init__(self, names: list[str]) -> None:
        if len(names) < 1:
            raise ValueError("names must contain at least one entry")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate names: {names}")
        self.names = list(names)

    def __call__(self, data: Any) -> dict[str, Any]:
        if isinstance(data, np.ndarray):
            return {name: data.copy() for name in self.names}
        if isinstance(data, torch.Tensor):
            return {name: data.clone() for name in self.names}
        return {name: copy.deepcopy(data) for name in self.names}

    def __repr__(self) -> str:
        return f"NamedCopies({self.names})"


class DecodeMultiResizeCropEmbed(BatchTransform):
    """Decode + resize-short + crop-long + embed onto a fixed-size RGBA canvas.

    Wraps :class:`DecodeMultiRandomResizeShortCropLong` and places each crop
    onto a zero-filled canvas of ``canvas_size``, adding a binary alpha channel.
    This guarantees uniform output size regardless of per-image crop size
    variation, enabling fast batched GPU pipelines even with ``size=(min, max)``.

    The output for each named crop is ``[B, canvas_size, canvas_size, 4]``
    uint8 HWC (RGBA), where the alpha channel is 255 inside the image
    region and 0 outside.

    Downstream, use :class:`~slipstream.transforms.RandomBackgroundBlend` to
    blend the embedded image with a generated background, optionally applying
    circular or cosine edge fading.

    Args:
        crops: Dict mapping crop names to parameter dicts.
            Required key: ``size`` (int or tuple ``(min, max)``).
            Optional keys: ``x_range``, ``y_range``, ``seed``, ``size_mode``,
            ``embed_x_range``, ``embed_y_range``, ``embed_seed``, ``coords``
            (per-crop overrides for canvas placement; fall back to global
            defaults).  ``coords`` is a list of ``(x, y)`` tuples in [0, 1]
            specifying explicit fractional placement positions.  When set, it
            overrides ``embed_x_range``/``embed_y_range`` for that crop.  A
            single coord gives deterministic placement; multiple coords cause
            a random choice per image.
        canvas_size: Output spatial size (square). Must be >= largest crop size.
        embed_x_range: Horizontal placement on canvas.  Float or ``(min, max)``
            in [0, 1] where 0 = left, 0.5 = center, 1 = right.
        embed_y_range: Vertical placement on canvas.  Float or ``(min, max)``.
        embed_seed: Seed for placement RNG (separate from per-crop seeds).
        size_mode: Default size mode (``"per_batch"`` or ``"per_image"``).
            Default: ``"per_image"`` — each image gets an independently
            sampled crop size, which is natural since the canvas guarantees
            uniform output shape regardless.
        num_threads: Parallel decode threads. 0 = auto.

    Returns:
        Dict[str, np.ndarray] — named crops, each ``[B, canvas_size, canvas_size, 4]``
        uint8 HWC (RGBA).

    Example::

        decoder = DecodeMultiResizeCropEmbed({
            "small": dict(size=(64, 112), x_range=(0, 1), seed=42),
            "large": dict(size=(112, 224), x_range=(0, 1), seed=43),
        }, canvas_size=320, embed_x_range=(0, 1), embed_y_range=(0, 1))
        result = decoder(batch_data)  # {"small": [B,320,320,4], "large": [B,320,320,4]}
    """

    def __init__(
        self,
        crops: dict[str, dict],
        canvas_size: int = 224,
        embed_x_range: float | tuple[float, float] = 0.5,
        embed_y_range: float | tuple[float, float] = 0.5,
        embed_seed: int | None = None,
        size_mode: str = "per_image",
        num_threads: int = 0,
    ) -> None:
        self.canvas_size = canvas_size
        self.embed_x_range = (
            (embed_x_range, embed_x_range) if isinstance(embed_x_range, (int, float))
            else tuple(embed_x_range)
        )
        self.embed_y_range = (
            (embed_y_range, embed_y_range) if isinstance(embed_y_range, (int, float))
            else tuple(embed_y_range)
        )
        self.embed_seed = embed_seed
        self._embed_seed_counter = 0

        # Inner decoder — always HWC, no permute/tensor (we handle that)
        self._inner = DecodeMultiRandomResizeShortCropLong(
            crops,
            size_mode=size_mode,
            num_threads=num_threads,
            to_tensor=False,
            permute=False,
        )

        # Per-crop embed placement (overrides global defaults)
        self._embed_x_ranges: list[tuple[float, float]] = []
        self._embed_y_ranges: list[tuple[float, float]] = []
        self._embed_seeds: list[int | None] = []
        self._embed_coords: list[list[tuple[float, float]] | None] = []
        for name in self._inner._crop_names:
            params = crops[name]
            exr = params.get('embed_x_range', embed_x_range)
            exr = (exr, exr) if isinstance(exr, (int, float)) else tuple(exr)
            self._embed_x_ranges.append(exr)
            eyr = params.get('embed_y_range', embed_y_range)
            eyr = (eyr, eyr) if isinstance(eyr, (int, float)) else tuple(eyr)
            self._embed_y_ranges.append(eyr)
            self._embed_seeds.append(params.get('embed_seed', embed_seed))
            embed_coords = params.get('coords', None)
            if embed_coords is not None:
                embed_coords = [(float(x), float(y)) for x, y in embed_coords]
            self._embed_coords.append(embed_coords)

    def set_image_format(self, image_format: str) -> None:
        self._inner.set_image_format(image_format)

    def __call__(
        self, batch_data: dict[str, Any] | bytes | bytearray | memoryview,
    ) -> dict[str, np.ndarray]:
        if isinstance(batch_data, (bytes, bytearray, memoryview)):
            from slipstream.decoders.base import _bytes_to_batch_dict, _unwrap_single_result
            return _unwrap_single_result(self(_bytes_to_batch_dict(batch_data)))

        # Decode crops (HWC uint8)
        crops_rgb = self._inner(batch_data)
        batch_size = batch_data['sizes'].shape[0]

        # Advance embed RNG
        self._embed_seed_counter += 1

        result = {}
        embed_rects = {}
        for c, (name, crop_data) in enumerate(crops_rgb.items()):
            is_list = isinstance(crop_data, list)

            cs = self.canvas_size
            canvas = np.zeros((batch_size, cs, cs, 4), dtype=np.uint8)
            rects = np.zeros((batch_size, 4), dtype=np.int32)

            # Compute embed seed for this crop
            crop_embed_seed = self._embed_seeds[c]
            if crop_embed_seed is not None:
                embed_batch_seed = (
                    crop_embed_seed + batch_size * self._embed_seed_counter
                    + c * 7919  # prime offset per crop for independence
                ) % 2147483647
            else:
                embed_batch_seed = (
                    batch_size * self._embed_seed_counter + c * 7919
                ) % 2147483647

            embed_coords = self._embed_coords[c]

            if embed_coords is not None:
                # Coords mode: explicit fractional positions
                if len(embed_coords) == 1 and not is_list:
                    # Single coord + uniform size → fast path with fixed range
                    xf, yf = embed_coords[0]
                    _embed_batch_rgba(
                        crop_data, canvas, rects,
                        embed_batch_seed,
                        xf, xf, yf, yf,
                    )
                else:
                    # Multi-coord or variable-size: slow path with per-image
                    rng = np.random.RandomState(embed_batch_seed)
                    for i in range(batch_size):
                        idx = rng.randint(0, len(embed_coords))
                        x_frac, y_frac = embed_coords[idx]
                        img = crop_data[i]  # [h, w, 3]
                        h, w = img.shape[0], img.shape[1]

                        slack_x = max(0, cs - w)
                        slack_y = max(0, cs - h)
                        x0 = int(x_frac * slack_x)
                        y0 = int(y_frac * slack_y)

                        copy_h = min(h, cs - y0)
                        copy_w = min(w, cs - x0)

                        canvas[i, y0:y0 + copy_h, x0:x0 + copy_w, :3] = img[:copy_h, :copy_w]
                        canvas[i, y0:y0 + copy_h, x0:x0 + copy_w, 3] = 255
                        rects[i] = [x0, y0, copy_w, copy_h]
            elif not is_list:
                # Fast path: stacked [B, h, w, 3] — use Numba JIT
                _embed_batch_rgba(
                    crop_data, canvas, rects,
                    embed_batch_seed,
                    self._embed_x_ranges[c][0], self._embed_x_ranges[c][1],
                    self._embed_y_ranges[c][0], self._embed_y_ranges[c][1],
                )
            else:
                # Slow path: per_image variable sizes (list of arrays)
                for i in range(batch_size):
                    img = crop_data[i]  # [h, w, 3]
                    h, w = img.shape[0], img.shape[1]

                    rng_i = np.random.RandomState(
                        (embed_batch_seed + i) % 2147483647
                    )
                    x_frac = rng_i.uniform(self._embed_x_ranges[c][0], self._embed_x_ranges[c][1])
                    y_frac = rng_i.uniform(self._embed_y_ranges[c][0], self._embed_y_ranges[c][1])

                    slack_x = max(0, cs - w)
                    slack_y = max(0, cs - h)
                    x0 = int(x_frac * slack_x)
                    y0 = int(y_frac * slack_y)

                    copy_h = min(h, cs - y0)
                    copy_w = min(w, cs - x0)

                    canvas[i, y0:y0 + copy_h, x0:x0 + copy_w, :3] = img[:copy_h, :copy_w]
                    canvas[i, y0:y0 + copy_h, x0:x0 + copy_w, 3] = 255
                    rects[i] = [x0, y0, copy_w, copy_h]

            result[name] = canvas
            embed_rects[name] = rects

        # Store params for inspection
        self._last_params = {
            **self._inner.last_params,
            "embed_rects": embed_rects,
        }

        return result

    @property
    def last_params(self) -> dict:
        """Parameters from the most recent ``__call__``."""
        return getattr(self, '_last_params', {})

    def shutdown(self) -> None:
        self._inner.shutdown()

    def __repr__(self) -> str:
        inner_crops = []
        for c, name in enumerate(self._inner._crop_names):
            sr = self._inner._crop_size_ranges[c]
            size_str = sr[0] if sr[0] == sr[1] else sr
            parts = [f"size={size_str}"]
            if sr[0] != sr[1]:
                parts.append(f"size_mode='{self._inner._crop_size_modes[c]}'")
            if self._inner._crop_x_ranges[c] != (0.5, 0.5):
                parts.append(f"x_range={self._inner._crop_x_ranges[c]}")
            if self._inner._crop_y_ranges[c] != (0.5, 0.5):
                parts.append(f"y_range={self._inner._crop_y_ranges[c]}")
            if self._inner._crop_seeds[c] is not None:
                parts.append(f"seed={self._inner._crop_seeds[c]}")
            if self._embed_x_ranges[c] != self.embed_x_range:
                parts.append(f"embed_x_range={self._embed_x_ranges[c]}")
            if self._embed_y_ranges[c] != self.embed_y_range:
                parts.append(f"embed_y_range={self._embed_y_ranges[c]}")
            if self._embed_seeds[c] != self.embed_seed:
                parts.append(f"embed_seed={self._embed_seeds[c]}")
            if self._embed_coords[c] is not None:
                parts.append(f"coords={self._embed_coords[c]}")
            inner_crops.append(f"'{name}': {', '.join(parts)}")
        extras = [f"canvas_size={self.canvas_size}"]
        if self.embed_x_range != (0.5, 0.5):
            extras.append(f"embed_x_range={self.embed_x_range}")
        if self.embed_y_range != (0.5, 0.5):
            extras.append(f"embed_y_range={self.embed_y_range}")
        if self.embed_seed is not None:
            extras.append(f"embed_seed={self.embed_seed}")
        return f"DecodeMultiResizeCropEmbed({{{', '.join(inner_crops)}}}, {', '.join(extras)})"


# Backward-compatible aliases (deprecated)
MultiCropRandomResizedCrop = DecodeUniformMultiRandomResizedCrop
MultiRandomResizedCrop = DecodeMultiRandomResizedCrop
MultiRandomResizeShortCropLong = DecodeMultiRandomResizeShortCropLong
