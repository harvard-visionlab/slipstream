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

from slipstream.decoders.base import BatchTransform
from slipstream.decoders.numba_decoder import NumbaBatchDecoder


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


# Backward-compatible aliases (deprecated)
MultiCropRandomResizedCrop = DecodeUniformMultiRandomResizedCrop
MultiRandomResizedCrop = DecodeMultiRandomResizedCrop
