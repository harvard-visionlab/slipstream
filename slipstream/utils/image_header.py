"""Utilities for reading image dimensions from headers without full decode.

This module provides fast dimension extraction from image bytes by parsing
headers directly, avoiding expensive full-image decoding.
"""

from __future__ import annotations

import struct
from typing import Tuple

__all__ = ["read_image_dimensions"]


def read_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Read image dimensions from bytes without full decode.

    Attempts methods in order of speed:
    1. libslipstream jpeg_header() - fastest, JPEG only
    2. Header parsing - fast, supports JPEG/PNG/GIF/BMP/WEBP
    3. PIL.Image - fallback for other formats

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If image format is not recognized or corrupted
    """
    # Try libslipstream first (fastest for JPEG)
    try:
        from libslipstream._libslipstream import jpeg_header

        # jpeg_header returns (width, height, num_components)
        result = jpeg_header(image_bytes)
        if result is not None:
            return result[0], result[1]
    except (ImportError, Exception):
        pass

    # Try header parsing
    result = _parse_image_header(image_bytes)
    if result is not None:
        return result

    # Fallback to PIL
    return _read_dimensions_pil(image_bytes)


def _parse_image_header(data: bytes) -> Tuple[int, int] | None:
    """Parse image header to extract dimensions.

    Supports: JPEG, PNG, GIF, BMP, WEBP

    Args:
        data: Raw image bytes

    Returns:
        Tuple of (width, height) or None if format not recognized
    """
    if len(data) < 12:
        return None

    # JPEG: starts with 0xFF 0xD8 0xFF
    if data[:3] == b"\xff\xd8\xff":
        return _parse_jpeg_dimensions(data)

    # PNG: starts with 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        # Dimensions in IHDR chunk at offset 16 (width) and 20 (height)
        width = struct.unpack(">I", data[16:20])[0]
        height = struct.unpack(">I", data[20:24])[0]
        return width, height

    # GIF: starts with GIF87a or GIF89a
    if data[:6] in (b"GIF87a", b"GIF89a"):
        width = struct.unpack("<H", data[6:8])[0]
        height = struct.unpack("<H", data[8:10])[0]
        return width, height

    # BMP: starts with BM
    if data[:2] == b"BM":
        # BITMAPINFOHEADER starts at offset 14
        width = struct.unpack("<I", data[18:22])[0]
        height = abs(struct.unpack("<i", data[22:26])[0])  # Height can be negative
        return width, height

    # WEBP: starts with RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _parse_webp_dimensions(data)

    return None


def _parse_jpeg_dimensions(data: bytes) -> Tuple[int, int] | None:
    """Parse JPEG header to extract dimensions.

    Searches for SOF0-SOF3 markers which contain image dimensions.
    """
    i = 2  # Skip SOI marker

    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue

        marker = data[i + 1]

        # Skip padding bytes
        if marker == 0xFF:
            i += 1
            continue

        # SOF markers (0xC0-0xC3, 0xC5-0xC7, 0xC9-0xCB, 0xCD-0xCF)
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            # Dimensions at offset +5 (height) and +7 (width) from marker
            height = struct.unpack(">H", data[i + 5 : i + 7])[0]
            width = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return width, height

        # Skip marker segment
        if marker in (0x00, 0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9):
            # Standalone markers (no length)
            i += 2
        else:
            # Marker with length
            if i + 3 >= len(data):
                break
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length

    return None


def _parse_webp_dimensions(data: bytes) -> Tuple[int, int] | None:
    """Parse WEBP header to extract dimensions.

    Handles both lossy (VP8) and lossless (VP8L) formats.
    """
    if len(data) < 30:
        return None

    # VP8 lossy format
    if data[12:16] == b"VP8 ":
        # Skip to frame header (after signature)
        # Frame header starts at byte 23 for standard VP8
        # Width and height are 14-bit values
        if len(data) < 30:
            return None
        # VP8 bitstream starts at offset 20
        # Dimensions at offset 26-27 (width) and 28-29 (height)
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height

    # VP8L lossless format
    if data[12:16] == b"VP8L":
        # Signature byte at offset 21
        if len(data) < 25:
            return None
        # Read 4 bytes at offset 21
        bits = struct.unpack("<I", data[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height

    # VP8X extended format
    if data[12:16] == b"VP8X":
        if len(data) < 30:
            return None
        # Canvas width and height at offset 24-29 (24-bit values)
        width = struct.unpack("<I", data[24:27] + b"\x00")[0] + 1
        height = struct.unpack("<I", data[27:30] + b"\x00")[0] + 1
        return width, height

    return None


def _read_dimensions_pil(data: bytes) -> Tuple[int, int]:
    """Read image dimensions using PIL (fallback).

    Args:
        data: Raw image bytes

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If image cannot be opened
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception as e:
        raise ValueError(f"Failed to read image dimensions: {e}") from e
