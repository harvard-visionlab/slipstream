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
from slipstream.decoders.numba_decoder import _available_cpus
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slipstream.decoders.numba_decoder import (
    Compiler,
    _generate_center_crop_params_batch,
    _generate_direct_random_crop_params_batch,
    _generate_random_crop_params_batch,
    _transpose_hwc_to_chw,
    load_library,
    resize_crop_numba,
)

__all__ = ["YUV420NumbaBatchDecoder"]

_ctypes_yuv420_decode: Any = None
_ctypes_yuv420_to_yuv_fullres: Any = None
_ctypes_yuv420_extract_planes: Any = None


def _setup_yuv420_ctypes() -> None:
    global _ctypes_yuv420_decode, _ctypes_yuv420_to_yuv_fullres, _ctypes_yuv420_extract_planes
    if _ctypes_yuv420_decode is not None:
        return
    lib = load_library()
    lib.yuv420p_to_rgb_buffer.argtypes = [
        c_void_p, c_uint64, c_void_p, c_uint32, c_uint32, c_uint32
    ]
    lib.yuv420p_to_rgb_buffer.restype = c_int
    _ctypes_yuv420_decode = lib.yuv420p_to_rgb_buffer

    lib.yuv420p_to_yuv_fullres.argtypes = [
        c_void_p, c_uint64, c_void_p, c_uint32, c_uint32, c_uint32
    ]
    lib.yuv420p_to_yuv_fullres.restype = c_int
    _ctypes_yuv420_to_yuv_fullres = lib.yuv420p_to_yuv_fullres

    lib.yuv420p_extract_planes.argtypes = [
        c_void_p, c_uint64, c_void_p, c_void_p, c_void_p, c_uint32, c_uint32,
        c_uint32, c_uint32
    ]
    lib.yuv420p_extract_planes.restype = c_int
    _ctypes_yuv420_extract_planes = lib.yuv420p_extract_planes


def _make_yuv420_fullres_wrapper() -> Any:
    """Create YUV420→fullres YUV wrapper after ctypes are initialized."""
    fn = _ctypes_yuv420_to_yuv_fullres

    def yuv420_to_yuv_fullres_numba(
        source: np.ndarray,
        dst: np.ndarray,
        height: int,
        width: int,
        out_stride: int,
    ) -> int:
        return fn(
            source.ctypes.data, source.size,
            dst.ctypes.data,
            height, width, out_stride,
        )
    return yuv420_to_yuv_fullres_numba


def _make_yuv420_extract_planes_wrapper() -> Any:
    """Create YUV420 plane extraction wrapper after ctypes are initialized."""
    fn = _ctypes_yuv420_extract_planes

    def yuv420_extract_planes_numba(
        source: np.ndarray,
        y_out: np.ndarray,
        u_out: np.ndarray,
        v_out: np.ndarray,
        height: int,
        width: int,
        y_out_stride: int,
        uv_out_stride: int,
    ) -> int:
        return fn(
            source.ctypes.data, source.size,
            y_out.ctypes.data, u_out.ctypes.data, v_out.ctypes.data,
            height, width, y_out_stride, uv_out_stride,
        )
    return yuv420_extract_planes_numba


def yuv420_decode_numba(
    source: np.ndarray,
    dst: np.ndarray,
    height: int,
    width: int,
    out_stride: int,
) -> int:
    global _ctypes_yuv420_decode
    if _ctypes_yuv420_decode is None:
        _setup_yuv420_ctypes()
    return _ctypes_yuv420_decode(
        source.ctypes.data, source.size,
        dst.ctypes.data,
        height, width, out_stride,
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
        out_stride = destination.shape[2]
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])
            source = yuv_data[i, :size]
            dst = destination[i, :h, :w, :]
            yuv_c(source, dst, h, w, out_stride)
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
            yuv_c(source, temp, h, w, w)

            crop_x = int(crop_params[i, 0])
            crop_y = int(crop_params[i, 1])
            crop_w = int(crop_params[i, 2])
            crop_h = int(crop_params[i, 3])

            dest = destination[i, :, :, :]
            resize_crop_c(
                temp, h, w,
                crop_y, crop_x, crop_h, crop_w,
                dest, target_h, target_w,
                w, 0,
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
            yuv_c(source, temp, h, w, w)

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
                    w, 0,
                )

        return destinations

    decode_multi_crop_batch.is_parallel = True
    return Compiler.compile(decode_multi_crop_batch)


def _create_yuv420_decode_multi_crop_varied_function() -> Any:
    """Create YUV420 decode-once + varied-size multi-crop function."""
    yuv_c = Compiler.compile(yuv420_decode_numba)
    resize_crop_c = Compiler.compile(resize_crop_numba)
    my_range = Compiler.get_iterator()

    def decode_multi_crop_varied_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        all_crop_params: np.ndarray,
        temp_buffer: np.ndarray,
        destinations: np.ndarray,
        target_sizes: np.ndarray,
        num_crops: int,
    ) -> np.ndarray:
        batch_size = len(sizes)
        max_target = destinations.shape[2]
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])

            source = yuv_data[i, :size]
            temp = temp_buffer[i, :h, :w, :]
            yuv_c(source, temp, h, w, w)

            for c in range(num_crops):
                crop_x = int(all_crop_params[c, i, 0])
                crop_y = int(all_crop_params[c, i, 1])
                crop_w = int(all_crop_params[c, i, 2])
                crop_h = int(all_crop_params[c, i, 3])
                ts = int(target_sizes[c])

                dest = destinations[c, i, :ts, :ts, :]
                resize_crop_c(
                    temp, h, w,
                    crop_y, crop_x, crop_h, crop_w,
                    dest, ts, ts,
                    w, max_target,
                )

        return destinations

    decode_multi_crop_varied_batch.is_parallel = True
    return Compiler.compile(decode_multi_crop_varied_batch)


def _create_yuv420_to_yuv_fullres_function() -> Any:
    yuv_fullres_c = Compiler.compile(_make_yuv420_fullres_wrapper())
    my_range = Compiler.get_iterator()

    def decode_yuv_fullres_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        destination: np.ndarray,
    ) -> np.ndarray:
        batch_size = len(sizes)
        out_stride = destination.shape[2]
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])
            source = yuv_data[i, :size]
            dst = destination[i, :h, :w, :]
            yuv_fullres_c(source, dst, h, w, out_stride)
        return destination[:batch_size]

    decode_yuv_fullres_batch.is_parallel = True
    return Compiler.compile(decode_yuv_fullres_batch)


def _create_yuv420_extract_planes_function() -> Any:
    extract_c = Compiler.compile(_make_yuv420_extract_planes_wrapper())
    my_range = Compiler.get_iterator()

    def extract_planes_batch(
        yuv_data: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        y_dest: np.ndarray,
        u_dest: np.ndarray,
        v_dest: np.ndarray,
    ) -> int:
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])
            source = yuv_data[i, :size]
            y_out = y_dest[i, :h, :w]
            u_out = u_dest[i, :h // 2, :w // 2]
            v_out = v_dest[i, :h // 2, :w // 2]
            extract_c(source, y_out, u_out, v_out, h, w, y_dest.shape[2], u_dest.shape[2])
        return batch_size

    extract_planes_batch.is_parallel = True
    return Compiler.compile(extract_planes_batch)


_yuv420_decode_compiled: Any = None
_yuv420_decode_crop_compiled: Any = None
_yuv420_decode_multi_crop_compiled: Any = None
_yuv420_decode_multi_crop_varied_compiled: Any = None
_yuv420_yuv_fullres_compiled: Any = None
_yuv420_extract_planes_compiled: Any = None


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


def _get_yuv420_decode_multi_crop_varied() -> Any:
    global _yuv420_decode_multi_crop_varied_compiled
    if _yuv420_decode_multi_crop_varied_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_decode_multi_crop_varied_compiled = _create_yuv420_decode_multi_crop_varied_function()
    return _yuv420_decode_multi_crop_varied_compiled


def _get_yuv420_yuv_fullres() -> Any:
    global _yuv420_yuv_fullres_compiled
    if _yuv420_yuv_fullres_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_yuv_fullres_compiled = _create_yuv420_to_yuv_fullres_function()
    return _yuv420_yuv_fullres_compiled


def _get_yuv420_extract_planes() -> Any:
    global _yuv420_extract_planes_compiled
    if _yuv420_extract_planes_compiled is None:
        _setup_yuv420_ctypes()
        _yuv420_extract_planes_compiled = _create_yuv420_extract_planes_function()
    return _yuv420_extract_planes_compiled


class YUV420NumbaBatchDecoder:
    """High-performance batch YUV420P→RGB decoder using Numba prange + C extension.

    Same API as NumbaBatchDecoder but for YUV420P-encoded images.
    """

    def __init__(self, num_threads: int = 0) -> None:
        if num_threads < 1:
            num_threads = _available_cpus()
        self.num_threads = num_threads
        Compiler.set_num_threads(num_threads)

        _setup_yuv420_ctypes()
        self._decode_fn = _get_yuv420_decode()
        self._decode_crop_fn = _get_yuv420_decode_crop()
        self._decode_multi_crop_fn = _get_yuv420_decode_multi_crop()
        self._decode_multi_crop_varied_fn = _get_yuv420_decode_multi_crop_varied()
        self._yuv_fullres_fn = _get_yuv420_yuv_fullres()
        self._extract_planes_fn = _get_yuv420_extract_planes()
        self._seed_counter = 0

        self._temp_buffer: np.ndarray | None = None
        self._dest_buffer: np.ndarray | None = None
        self._chw_buffer: np.ndarray | None = None
        self._multi_crop_buffer: np.ndarray | None = None
        self._multi_chw_buffer: np.ndarray | None = None
        self._multi_crop_varied_buffer: np.ndarray | None = None
        self._varied_chw_buffers: dict[tuple[int, int], np.ndarray] = {}
        self._yuv_fullres_buffer: np.ndarray | None = None
        self._y_plane_buffer: np.ndarray | None = None
        self._u_plane_buffer: np.ndarray | None = None
        self._v_plane_buffer: np.ndarray | None = None

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
        if seed is not None:
            batch_seed = (seed + batch_size * self._seed_counter) % 2147483647
        else:
            batch_seed = (batch_size * self._seed_counter) % 2147483647

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

    def decode_batch_direct_random_crop(
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
        """Decode batch with DirectRandomResizedCrop (analytic, no rejection sampling)."""
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
        if seed is not None:
            batch_seed = (seed + batch_size * self._seed_counter) % 2147483647
        else:
            batch_seed = (batch_size * self._seed_counter) % 2147483647

        crop_params = _generate_direct_random_crop_params_batch(
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
                batch_seed = (seeds[c] + batch_size * self._seed_counter) % 2147483647
            else:
                self._seed_counter += 1
                batch_seed = (batch_size * self._seed_counter) % 2147483647

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

    def decode_batch_multi_crop_varied(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        crop_params_list: list[NDArray[np.int32]],
        target_sizes: list[int],
    ) -> list[NDArray[np.uint8]]:
        """Decode once, apply N crops with potentially different target sizes."""
        num_crops = len(crop_params_list)
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))
        max_target = max(target_sizes)

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)

        all_crop_params = np.stack(crop_params_list, axis=0)
        target_sizes_arr = np.array(target_sizes, dtype=np.int32)

        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)

        if (self._multi_crop_varied_buffer is None or
            self._multi_crop_varied_buffer.shape[0] < num_crops or
            self._multi_crop_varied_buffer.shape[1] < batch_size or
            self._multi_crop_varied_buffer.shape[2] < max_target or
            self._multi_crop_varied_buffer.shape[3] < max_target):
            self._multi_crop_varied_buffer = np.zeros(
                (num_crops, batch_size, max_target, max_target, 3), dtype=np.uint8)

        self._decode_multi_crop_varied_fn(
            data, sizes_u64, heights_u32, widths_u32,
            all_crop_params, temp_buffer, self._multi_crop_varied_buffer,
            target_sizes_arr, num_crops,
        )

        results = []
        for c in range(num_crops):
            ts = target_sizes[c]
            results.append(self._multi_crop_varied_buffer[c, :batch_size, :ts, :ts, :])
        return results

    def multi_hwc_to_chw_varied(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> list[NDArray[np.uint8]]:
        """Transpose N crop arrays of potentially different sizes to CHW."""
        from slipstream.decoders.numba_decoder import _transpose_hwc_to_chw
        results = []
        for i, crop in enumerate(crops):
            B, H, W = crop.shape[0], crop.shape[1], crop.shape[2]
            key = (i, H, W)
            buf = self._varied_chw_buffers.get(key)
            if buf is None or buf.shape[0] < B:
                buf = np.zeros((B, 3, H, W), dtype=np.uint8)
                self._varied_chw_buffers[key] = buf
            _transpose_hwc_to_chw(crop, buf[:B], B)
            results.append(buf[:B])
        return results

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

    def multi_hwc_to_chw(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> list[NDArray[np.uint8]]:
        """Transpose N crop arrays [B, H, W, 3] → [B, 3, H, W] into separate buffers."""
        num_crops = len(crops)
        if not crops:
            return []

        batch_size = crops[0].shape[0]
        H, W = crops[0].shape[1], crops[0].shape[2]

        if (self._multi_chw_buffer is None or
            self._multi_chw_buffer.shape[0] < num_crops or
            self._multi_chw_buffer.shape[1] < batch_size or
            self._multi_chw_buffer.shape[3] != H or
            self._multi_chw_buffer.shape[4] != W):
            self._multi_chw_buffer = np.zeros(
                (num_crops, batch_size, 3, H, W), dtype=np.uint8)

        results = []
        for c in range(num_crops):
            _transpose_hwc_to_chw(crops[c], self._multi_chw_buffer[c], batch_size)
            results.append(self._multi_chw_buffer[c, :batch_size])

        return results

    def decode_batch_yuv_fullres(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
    ) -> list[NDArray[np.uint8]]:
        """Decode YUV420P to full-resolution YUV (nearest-neighbor U/V upsample).

        Returns a list of [H, W, 3] uint8 arrays where channels are (Y, U, V)
        at full resolution. Same shape as RGB output but in YUV colorspace.
        """
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)

        if (self._yuv_fullres_buffer is None or
            self._yuv_fullres_buffer.shape[0] < batch_size or
            self._yuv_fullres_buffer.shape[1] < max_h or
            self._yuv_fullres_buffer.shape[2] < max_w):
            self._yuv_fullres_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)

        self._yuv_fullres_fn(data, sizes_u64, heights_u32, widths_u32, self._yuv_fullres_buffer)

        results = []
        for i in range(batch_size):
            h, w = int(heights[i]), int(widths[i])
            results.append(self._yuv_fullres_buffer[i, :h, :w, :].copy())
        return results

    def decode_batch_yuv_planes(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
    ) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]]:
        """Extract raw YUV420P planes without conversion.

        Returns a list of (Y, U, V) tuples where:
        - Y: [H, W] uint8
        - U: [H/2, W/2] uint8
        - V: [H/2, W/2] uint8

        This is the fastest possible decode — just memcpy of planes.
        """
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)

        # Allocate/reuse plane buffers
        if (self._y_plane_buffer is None or
            self._y_plane_buffer.shape[0] < batch_size or
            self._y_plane_buffer.shape[1] < max_h or
            self._y_plane_buffer.shape[2] < max_w):
            self._y_plane_buffer = np.zeros((batch_size, max_h, max_w), dtype=np.uint8)
            self._u_plane_buffer = np.zeros((batch_size, max_h // 2, max_w // 2), dtype=np.uint8)
            self._v_plane_buffer = np.zeros((batch_size, max_h // 2, max_w // 2), dtype=np.uint8)

        self._extract_planes_fn(
            data, sizes_u64, heights_u32, widths_u32,
            self._y_plane_buffer, self._u_plane_buffer, self._v_plane_buffer,
        )

        results = []
        for i in range(batch_size):
            h, w = int(heights[i]), int(widths[i])
            y = self._y_plane_buffer[i, :h, :w].copy()
            u = self._u_plane_buffer[i, :h // 2, :w // 2].copy()
            v = self._v_plane_buffer[i, :h // 2, :w // 2].copy()
            results.append((y, u, v))
        return results

    def shutdown(self) -> None:
        self._temp_buffer = None
        self._dest_buffer = None
        self._chw_buffer = None
        self._multi_crop_buffer = None
        self._multi_chw_buffer = None
        self._multi_crop_varied_buffer = None
        self._varied_chw_buffers = {}
        self._yuv_fullres_buffer = None
        self._y_plane_buffer = None
        self._u_plane_buffer = None
        self._v_plane_buffer = None

    def __repr__(self) -> str:
        return f"YUV420NumbaBatchDecoder(num_threads={self.num_threads})"
