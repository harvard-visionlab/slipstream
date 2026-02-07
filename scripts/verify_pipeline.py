#!/usr/bin/env python3
"""Pipeline verification script - human-readable correctness report.

This script answers the question: "Is slipstream reading and processing
my images correctly?"

Run with:
    uv run python scripts/verify_pipeline.py

What it checks:
    1. JPEG bytes survive cache unchanged (bit-perfect)
    2. Decoded images match PIL reference (within JPEG tolerance)
    3. YUV420 colorspace conversion is correct (BT.601 standard)
    4. Dimensions are parsed correctly (no silent failures)

If all checks pass, your training data will be processed correctly.
"""

import hashlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image


def create_test_jpeg(width: int, height: int, color: tuple) -> bytes:
    """Create a test JPEG image."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def print_result(name: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details and not passed:
        print(f"         {details}")


def check_cache_roundtrip():
    """Verify JPEG bytes survive cache build/load unchanged."""
    print("\n1. CACHE ROUND-TRIP (Are bytes preserved exactly?)")
    print("   If this fails: Your cached images would be corrupted.")

    from slipstream.cache import OptimizedCache

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.cache_path = tmpdir
                self._images = [
                    create_test_jpeg(64, 64, (255, 0, 0)),
                    create_test_jpeg(48, 36, (0, 255, 0)),
                    create_test_jpeg(100, 75, (0, 0, 255)),
                ]
                self._labels = [0, 1, 2]

            @property
            def field_types(self):
                return {"image": "ImageBytes", "label": "int"}

            def __len__(self):
                return len(self._images)

            def __getitem__(self, idx):
                return {"image": self._images[idx], "label": self._labels[idx]}

        dataset = MockDataset()

        # Build and load cache
        OptimizedCache.build(dataset, output_dir=tmpdir, verbose=False)
        loaded = OptimizedCache.load(tmpdir, verbose=False)

        all_passed = True
        for idx in range(len(dataset)):
            original = dataset[idx]['image']

            # Read from cache
            storage = loaded.fields['image']
            meta = storage._metadata[idx]
            ptr, size = int(meta['data_ptr']), int(meta['data_size'])
            cached = bytes(storage._data_mmap[ptr:ptr + size])

            # Compare hashes
            orig_hash = hashlib.sha256(original).hexdigest()[:16]
            cache_hash = hashlib.sha256(cached).hexdigest()[:16]

            passed = orig_hash == cache_hash
            all_passed = all_passed and passed
            print_result(
                f"Image {idx}: {len(original)} bytes",
                passed,
                f"Hash mismatch: {orig_hash} vs {cache_hash}"
            )

        return all_passed


def check_jpeg_decode():
    """Verify JPEG decode matches PIL reference."""
    print("\n2. JPEG DECODE (Do decoded pixels match PIL?)")
    print("   If this fails: Your model would see wrong pixel values.")

    try:
        from slipstream.decoders.numba_decoder import NumbaBatchDecoder
    except RuntimeError:
        print("  ⚠ SKIP: libslipstream not compiled")
        print("         Run: uv run python libslipstream/setup.py build_ext --inplace")
        return None

    decoder = NumbaBatchDecoder(num_threads=1)

    all_passed = True
    test_cases = [
        ("Solid red", (255, 0, 0)),
        ("Solid green", (0, 255, 0)),
        ("Solid blue", (0, 0, 255)),
        ("Gray", (128, 128, 128)),
    ]

    try:
        for name, color in test_cases:
            jpeg_bytes = create_test_jpeg(32, 32, color)

            # PIL decode
            pil_rgb = np.array(PIL.Image.open(io.BytesIO(jpeg_bytes)))

            # Slipstream decode
            batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
            batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
            heights = np.array([32], dtype=np.uint32)
            widths = np.array([32], dtype=np.uint32)

            slip_rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

            # Compare
            diff = np.abs(pil_rgb.astype(int) - slip_rgb.astype(int))
            max_diff = int(np.max(diff))

            # Allow ±2 for JPEG decode variance (different DCT implementations)
            passed = max_diff <= 2
            all_passed = all_passed and passed
            print_result(
                f"{name}: max pixel diff = {max_diff}",
                passed,
                f"Exceeds tolerance of ±2"
            )
    finally:
        decoder.shutdown()

    return all_passed


def check_yuv420_colorspace():
    """Verify BT.601 YUV colorspace conversion."""
    print("\n3. YUV420 COLORSPACE (Is color conversion correct?)")
    print("   If this fails: Colors would be wrong after YUV420 caching.")

    try:
        from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
    except RuntimeError:
        print("  ⚠ SKIP: libslipstream not compiled")
        return None

    decoder = YUV420NumbaBatchDecoder(num_threads=1)

    def create_solid_yuv420(y: int, u: int, v: int, h: int, w: int) -> bytes:
        y_plane = np.full((h, w), y, dtype=np.uint8)
        u_plane = np.full((h // 2, w // 2), u, dtype=np.uint8)
        v_plane = np.full((h // 2, w // 2), v, dtype=np.uint8)
        return y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

    # BT.601 test cases: (Y, U, V) -> expected dominant RGB channel
    # Red: Y≈76, U≈85, V≈255
    # Green: Y≈150, U≈44, V≈21
    # Blue: Y≈29, U≈255, V≈107
    test_cases = [
        ("Red (Y=76, U=85, V=255)", 76, 85, 255, "R"),
        ("Green (Y=150, U=44, V=21)", 150, 44, 21, "G"),
        ("Blue (Y=29, U=255, V=107)", 29, 255, 107, "B"),
        ("White (Y=255, U=128, V=128)", 255, 128, 128, "W"),
        ("Black (Y=0, U=128, V=128)", 0, 128, 128, "K"),
    ]

    all_passed = True
    try:
        for name, y, u, v, expected in test_cases:
            h, w = 16, 16
            yuv_bytes = create_solid_yuv420(y, u, v, h, w)

            batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
            batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
            sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
            heights = np.array([h], dtype=np.uint32)
            widths = np.array([w], dtype=np.uint32)

            rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]
            r, g, b = rgb[h // 2, w // 2]

            # Check expected color
            if expected == "R":
                passed = r >= 245 and g <= 10 and b <= 10
            elif expected == "G":
                passed = g >= 245 and r <= 10 and b <= 10
            elif expected == "B":
                passed = b >= 245 and r <= 10 and g <= 10
            elif expected == "W":
                passed = r >= 250 and g >= 250 and b >= 250
            elif expected == "K":
                passed = r <= 5 and g <= 5 and b <= 5
            else:
                passed = False

            all_passed = all_passed and passed
            print_result(
                f"{name} -> RGB({r}, {g}, {b})",
                passed,
                f"Expected {expected} color"
            )
    finally:
        decoder.shutdown()

    return all_passed


def check_dimension_parsing():
    """Verify image dimensions are parsed correctly."""
    print("\n4. DIMENSION PARSING (Are image sizes read correctly?)")
    print("   If this fails: Crops would be wrong or decode would crash.")

    from slipstream.cache import read_jpeg_dimensions, read_png_dimensions

    all_passed = True

    # Test various JPEG sizes
    for w, h in [(64, 48), (100, 75), (224, 224), (31, 47)]:
        jpeg_bytes = create_test_jpeg(w, h, (100, 100, 100))
        parsed_w, parsed_h = read_jpeg_dimensions(jpeg_bytes)

        passed = parsed_w == w and parsed_h == h
        all_passed = all_passed and passed
        print_result(
            f"JPEG {w}x{h}",
            passed,
            f"Parsed as {parsed_w}x{parsed_h}"
        )

    # Test PNG
    img = PIL.Image.new("RGB", (50, 30), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    parsed_w, parsed_h = read_png_dimensions(png_bytes)
    passed = parsed_w == 50 and parsed_h == 30
    all_passed = all_passed and passed
    print_result(
        "PNG 50x30",
        passed,
        f"Parsed as {parsed_w}x{parsed_h}"
    )

    return all_passed


def main():
    print("=" * 60)
    print("SLIPSTREAM PIPELINE VERIFICATION")
    print("=" * 60)
    print("\nThis checks that your images will be read and decoded correctly.")
    print("All tests use synthetic data - no external files needed.")

    results = {}

    results["cache"] = check_cache_roundtrip()
    results["decode"] = check_jpeg_decode()
    results["yuv420"] = check_yuv420_colorspace()
    results["dimensions"] = check_dimension_parsing()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)

    print(f"\n  Passed:  {passed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")

    if failed > 0:
        print("\n⚠ SOME CHECKS FAILED - Review output above")
        return 1
    elif skipped > 0:
        print("\n✓ All run checks passed (some skipped - compile libslipstream)")
        return 0
    else:
        print("\n✓ ALL CHECKS PASSED - Pipeline is working correctly")
        return 0


if __name__ == "__main__":
    sys.exit(main())
