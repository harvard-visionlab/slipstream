"""Verify YUV→RGB round-trip: DecodeYUVFullRes → manual BT.601 → matches decode_batch RGB."""
import os
from pathlib import Path

import numpy as np
import pytest

from slipstream.cache import OptimizedCache, load_yuv420_cache
from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder

CANDIDATES = [Path.home() / ".slipstream" / "imagenet10-s256_l512-yuv420-val",
              Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent")) / "imagenet1k-s256_l512-yuv420-val"]


def yuv_to_rgb_bt601(yuv: np.ndarray) -> np.ndarray:
    y = yuv[:, :, 0].astype(np.float32)
    u = yuv[:, :, 1].astype(np.float32) - 128.0
    v = yuv[:, :, 2].astype(np.float32) - 128.0
    r = np.clip(y + 1.402 * v, 0, 255).astype(np.uint8)
    g = np.clip(y - 0.344136 * u - 0.714136 * v, 0, 255).astype(np.uint8)
    b = np.clip(y + 1.772 * u, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def test_roundtrip():
    cache_dir = next((p for p in CANDIDATES if p.exists()), None)
    if cache_dir is None:
        pytest.skip("no yuv420 val cache present")
    cache = OptimizedCache.load(cache_dir)
    # a yuv420 store keeps the field as "image"; a jpeg store may carry an image_yuv420 sibling
    if cache.get_image_format("image") == "yuv420":
        storage = cache.fields["image"]
    else:
        storage = load_yuv420_cache(cache.cache_dir, "image")
    indices = np.arange(8, dtype=np.int64)
    batch_data = storage.load_batch(indices)

    decoder = YUV420NumbaBatchDecoder(num_threads=1)
    rgb_images = decoder.decode_batch(
        batch_data['data'], batch_data['sizes'],
        batch_data['heights'], batch_data['widths'],
    )
    yuv_images = decoder.decode_batch_yuv_fullres(
        batch_data['data'], batch_data['sizes'],
        batch_data['heights'], batch_data['widths'],
    )

    for i in range(len(rgb_images)):
        rgb_rt = yuv_to_rgb_bt601(yuv_images[i])
        diff = np.abs(rgb_images[i].astype(np.int16) - rgb_rt.astype(np.int16))
        max_diff = diff.max()
        print(f"  Image {i}: shape={rgb_images[i].shape}, max diff={max_diff}")
        assert max_diff <= 1, f"Image {i}: max diff {max_diff} > 1"

    print("\nAll images: max diff ≤ 1 ✓")
    decoder.shutdown()


if __name__ == "__main__":
    test_roundtrip()
