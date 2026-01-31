"""YUV420P batch decoder using Numba prange + libslipstream C extension.

Converts raw YUV420P (I420) images to RGB using a C kernel.
No bitstream parsing — just a fixed-point matrix multiply per pixel.

Usage:
    decoder = YUV420NumbaBatchDecoder(num_threads=12)
    images = decoder.decode_batch_to_buffer(data, sizes, heights, widths, dest)
    images = decoder.decode_batch_center_crop(data, sizes, heights, widths, crop_size=224)
    images = decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
"""

from __future__ import annotations

from ctypes import c_int, c_uint32, c_uint64, c_void_p
from multiprocessing import cpu_count
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slipstream.decoders.numba_decoder import (
    Compiler,
    _generate_center_crop_params_batch,
    _generate_random_crop_params_batch,
    _transpose_hwc_to_chw,
    load_library,
    resize_crop_numba,
)

__all__ = ["YUV420NumbaBatchDecoder"]

_ctypes_yuv420_decode: Any = None


def _setup_yuv420_ctypes() -> None:
    global _ctypes_yuv420_decode
    if _ctypes_yuv420_decode is not None:
        return
    lib = load_library()
    lib.yuv420p_to_rgb_buffer.argtypes = [
        c_void_p, c_uint64, c_void_p, c_uint32, c_uint32
    ]
    lib.yuv420p_to_rgb_buffer.restype = c_int
    _ctypes_yuv420_decode = lib.yuv420p_to_rgb_buffer


def yuv420_decode_numba(
    source: np.ndarray,
    dst: np.ndarray,
    height: int,
    width: int,
) -> int:
    global _ctypes_yuv420_decode
    if _ctypes_yuv420_decode is None:
        _setup_yuv420_ctypes()
    return _ctypes_yuv420_decode(
        source.ctypes.data, source.size,
        dst.ctypes.data,
        height, width,
    )


def _create_yuv420_decode_function() -> Any:
    yuv_c = Compiler.compile(yuv420_decode_numba)
    my_range = Compiler.get_iterator()

    def decode_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        destination: np.ndarray,
    ) -> np.ndarray:
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])
            source = yuv_data[i, :size]
            dst = destination[i, :h, :w, :]
            yuv_c(source, dst, h, w)
        return destination[:batch_size]

    decode_batch.is_parallel = True
    return Compiler.compile(decode_batch)


def _create_yuv420_decode_with_crop_function() -> Any:
    yuv_c = Compiler.compile(yuv420_decode_numba)
    resize_crop_c = Compiler.compile(resize_crop_numba)
    my_range = Compiler.get_iterator()

    def decode_crop_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        crop_params: np.ndarray,
        temp_buffer: np.ndarray,
        destination: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])
            source = yuv_data[i, :size]
            temp = temp_buffer[i, :h, :w, :]
            yuv_c(source, temp, h, w)

            crop_x = int(crop_params[i, 0])
            crop_y = int(crop_params[i, 1])
            crop_w = int(crop_params[i, 2])
            crop_h = int(crop_params[i, 3])

            dest = destination[i, :, :, :]
            resize_crop_c(
                temp, h, w,
                crop_y, crop_x, crop_h, crop_w,
                dest, target_h, target_w,
            )

        return destination[:batch_size]

    decode_crop_batch.is_parallel = True
    return Compiler.compile(decode_crop_batch)


def _create_yuv420_decode_multi_crop_function() -> Any:
    yuv_c = Compiler.compile(yuv420_decode_numba)
    resize_crop_c = Compiler.compile(resize_crop_numba)
    my_range = Compiler.get_iterator()

    def decode_multi_crop_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        all_crop_params: np.ndarray,
        temp_buffer: np.ndarray,
        destinations: np.ndarray,
        target_h: int,
        target_w: int,
        num_crops: int,
    ) -> np.ndarray:
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])

            source = yuv_data[i, :size]
            temp = temp_buffer[i, :h, :w, :]
            yuv_c(source, temp, h, w)

            for c in range(num_crops):
                crop_x = int(all_crop_params[c, i, 0])
                crop_y = int(all_crop_params[c, i, 1])
                crop_w = int(all_crop_params[c, i, 2])
                crop_h = int(all_crop_params[c, i, 3])

                dest = destinations[c, i, :, :, :]
                resize_crop_c(
                    temp, h, w,
                    crop_y, crop_x, crop_h, crop_w,
                    dest, target_h, target_w,
                )

        return destinations

    decode_multi_crop_batch.is_parallel = True
    return Compiler.compile(decode_multi_crop_batch)


_yuv420_decode_compiled: Any = None
_yuv420_decode_crop_compiled: Any = None
_yuv420_decode_multi_crop_compiled: Any = None


def _get_yuv420_decode() -> Any:
    global _yuv420_decode_compiled
    if _yuv420_decode_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_decode_compiled = _create_yuv420_decode_function()
    return _yuv420_decode_compiled


def _get_yuv420_decode_crop() -> Any:
    global _yuv420_decode_crop_compiled
    if _yuv420_decode_crop_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_decode_crop_compiled = _create_yuv420_decode_with_crop_function()
    return _yuv420_decode_crop_compiled


def _get_yuv420_decode_multi_crop() -> Any:
    global _yuv420_decode_multi_crop_compiled
    if _yuv420_decode_multi_crop_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_decode_multi_crop_compiled = _create_yuv420_decode_multi_crop_function()
    return _yuv420_decode_multi_crop_compiled


class YUV420NumbaBatchDecoder:
    """High-performance batch YUV420P→RGB decoder using Numba prange + C extension.

    Same API as NumbaBatchDecoder but for YUV420P-encoded images.
    """

    def __init__(self, num_threads: int = 0) -> None:
        if num_threads < 1:
            num_threads = cpu_count()
        self.num_threads = num_threads
        Compiler.set_num_threads(num_threads)

        _setup_yuv420_ctypes()
        self._decode_fn = _get_yuv420_decode()
        self._decode_crop_fn = _get_yuv420_decode_crop()
        self._decode_multi_crop_fn = _get_yuv420_decode_multi_crop()
        self._seed_counter = 0

        self._temp_buffer: np.ndarray | None = None
        self._dest_buffer: np.ndarray | None = None
        self._chw_buffer: np.ndarray | None = None
        self._multi_crop_buffer: np.ndarray | None = None

    def _ensure_temp_buffer(self, batch_size: int, max_h: int, max_w: int) -> np.ndarray:
        if (self._temp_buffer is None or
            self._temp_buffer.shape[0] < batch_size or
            self._temp_buffer.shape[1] < max_h or
            self._temp_buffer.shape[2] < max_w):
            self._temp_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
        return self._temp_buffer

    def _ensure_dest_buffer(self, batch_size: int, target_h: int, target_w: int) -> np.ndarray:
        if (self._dest_buffer is None or
            self._dest_buffer.shape[0] < batch_size or
            self._dest_buffer.shape[1] != target_h or
            self._dest_buffer.shape[2] != target_w):
            self._dest_buffer = np.zeros((batch_size, target_h, target_w, 3), dtype=np.uint8)
        return self._dest_buffer

    def decode_batch_to_buffer(
        self,
        yuv_data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        destination: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        sizes = np.ascontiguousarray(sizes, dtype=np.uint64)
        heights = np.ascontiguousarray(heights, dtype=np.uint32)
        widths = np.ascontiguousarray(widths, dtype=np.uint32)
        return self._decode_fn(yuv_data, sizes, heights, widths, destination)

    def decode_batch_center_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        crop_size: int = 224,
    ) -> NDArray[np.uint8]:
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        crop_params = _generate_center_crop_params_batch(widths_i32, heights_i32, crop_size)
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, crop_size, crop_size)

        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            crop_size, crop_size,
        )
        return dest_buffer[:batch_size]

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        seed: int | None = None,
    ) -> NDArray[np.uint8]:
        import math

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        log_ratio_min = math.log(ratio[0])
        log_ratio_max = math.log(ratio[1])

        self._seed_counter += 1
        batch_seed = (seed + self._seed_counter) if seed is not None else self._seed_counter

        crop_params = _generate_random_crop_params_batch(
            widths_i32, heights_i32,
            scale[0], scale[1],
            log_ratio_min, log_ratio_max,
            batch_seed,
        )

        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, target_size, target_size)

        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            target_size, target_size,
        )
        return dest_buffer[:batch_size]

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray | None = None,
        widths: NDArray | None = None,
        destination: NDArray[np.uint8] | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch of YUV420P images to list of numpy arrays."""
        if heights is None or widths is None:
            raise ValueError("heights and widths are required for YUV420NumbaBatchDecoder")

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        if destination is None:
            destination = self._ensure_temp_buffer(batch_size, max_h, max_w)

        self.decode_batch_to_buffer(data, sizes, heights, widths, destination)

        results = []
        for i in range(batch_size):
            h, w = int(heights[i]), int(widths[i])
            results.append(destination[i, :h, :w, :].copy())

        return results

    def decode_batch_resize_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        resize_size: int = 256,
        crop_size: int = 224,
        destination: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Decode batch with resize shortest edge + center crop."""
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)

        crop_params = np.zeros((batch_size, 4), dtype=np.int32)
        for i in range(batch_size):
            h = int(heights[i])
            w = int(widths[i])

            if h < w:
                scale = resize_size / h
                new_h = resize_size
                new_w = int(w * scale + 0.5)
            else:
                scale = resize_size / w
                new_w = resize_size
                new_h = int(h * scale + 0.5)

            crop_h_resized = min(crop_size, new_h)
            crop_w_resized = min(crop_size, new_w)
            start_y_resized = (new_h - crop_h_resized) // 2
            start_x_resized = (new_w - crop_w_resized) // 2

            crop_x = int(start_x_resized / scale + 0.5)
            crop_y = int(start_y_resized / scale + 0.5)
            crop_w_orig = int(crop_w_resized / scale + 0.5)
            crop_h_orig = int(crop_h_resized / scale + 0.5)

            crop_x = max(0, min(crop_x, w - 1))
            crop_y = max(0, min(crop_y, h - 1))
            crop_w_orig = max(1, min(crop_w_orig, w - crop_x))
            crop_h_orig = max(1, min(crop_h_orig, h - crop_y))

            crop_params[i, 0] = crop_x
            crop_params[i, 1] = crop_y
            crop_params[i, 2] = crop_w_orig
            crop_params[i, 3] = crop_h_orig

        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, crop_size, crop_size)

        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            crop_size, crop_size,
        )

        return dest_buffer[:batch_size]

    def decode_batch_multi_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        num_crops: int = 2,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        seeds: list[int | None] | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch once, then apply N random crops from decoded data."""
        import math

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        log_ratio_min = math.log(ratio[0])
        log_ratio_max = math.log(ratio[1])

        all_crop_params = np.zeros((num_crops, batch_size, 4), dtype=np.int32)
        for c in range(num_crops):
            if seeds is not None and seeds[c] is not None:
                self._seed_counter += 1
                batch_seed = seeds[c] + self._seed_counter
            else:
                self._seed_counter += 1
                batch_seed = self._seed_counter

            all_crop_params[c] = _generate_random_crop_params_batch(
                widths_i32, heights_i32,
                scale[0], scale[1],
                log_ratio_min, log_ratio_max,
                batch_seed,
            )

        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)

        if (self._multi_crop_buffer is None or
            self._multi_crop_buffer.shape[0] < num_crops or
            self._multi_crop_buffer.shape[1] < batch_size or
            self._multi_crop_buffer.shape[2] != target_size or
            self._multi_crop_buffer.shape[3] != target_size):
            self._multi_crop_buffer = np.zeros(
                (num_crops, batch_size, target_size, target_size, 3), dtype=np.uint8)

        self._decode_multi_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            all_crop_params, temp_buffer, self._multi_crop_buffer,
            target_size, target_size, num_crops,
        )

        return [self._multi_crop_buffer[c, :batch_size] for c in range(num_crops)]

    def _ensure_chw_buffer(self, batch_size: int, target_h: int, target_w: int) -> np.ndarray:
        if (self._chw_buffer is None or
            self._chw_buffer.shape[0] < batch_size or
            self._chw_buffer.shape[2] != target_h or
            self._chw_buffer.shape[3] != target_w):
            self._chw_buffer = np.zeros((batch_size, 3, target_h, target_w), dtype=np.uint8)
        return self._chw_buffer

    def hwc_to_chw(
        self,
        hwc: NDArray[np.uint8],
        batch_size: int | None = None,
    ) -> NDArray[np.uint8]:
        """Transpose [B, H, W, 3] -> [B, 3, H, W] using parallel per-image copy."""
        if batch_size is None:
            batch_size = hwc.shape[0]
        H, W = hwc.shape[1], hwc.shape[2]
        chw_buffer = self._ensure_chw_buffer(batch_size, H, W)
        _transpose_hwc_to_chw(hwc, chw_buffer, batch_size)
        return chw_buffer[:batch_size]

    def shutdown(self) -> None:
        self._temp_buffer = None
        self._dest_buffer = None
        self._chw_buffer = None
        self._multi_crop_buffer = None

    def __repr__(self) -> str:
        return f"YUV420NumbaBatchDecoder(num_threads={self.num_threads})"
