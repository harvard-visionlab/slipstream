# Slipstream Verification Checklist

End-to-end accuracy verification for all supported data formats.

## Quick Status

| Format | Reader | Cache | Decode | Accuracy |
|--------|--------|-------|--------|----------|
| FFCV (.ffcv/.beton) | ✅ | ⬜ | ✅ | ⬜ |
| LitData (streaming) | ✅ | ⬜ | ✅ | ⬜ |
| ImageFolder | ✅ | ⬜ | ✅ | ⬜ |
| SlipCache (.slipcache) | n/a | ⚠️ | ✅ | ⬜ |

> ⚠️ **SlipCache Cache column**: Tests pass on synthetic data, but **must test all 50,000 ImageNet samples**. See [Known Issues](#known-issues--future-work) for PNG-in-JPEG problem.

**Column key:**
| Column | Description | Pass Criterion |
|--------|-------------|----------------|
| Reader | Raw bytes match reference implementation | SHA256 hash identical |
| Cache | Data survives build→load round-trip | JPEG: byte-identical; YUV420: pixels ±1 |
| Decode | Decoded pixels match PIL/reference | Max diff ≤5, mean diff <1.5 |
| Accuracy | Model accuracy consistent across formats | All within 0.1% of each other |

---

## Layer 1: Reader Correctness

Verify readers return identical bytes to reference implementations.

### FFCV Reader
- [x] Sample count matches native ffcv-ssl — `test_ffcv_verification.py::test_sample_count_matches`
- [x] Image bytes match native (SHA256) — `test_ffcv_verification.py::test_first_100_samples_bytes_match`
- [x] Labels match native — `test_ffcv_verification.py::test_labels_match`
- [x] All samples readable — `test_ffcv_verification.py::test_all_samples_readable`
- [x] Full dataset bytes match — `test_ffcv_verification.py::test_all_samples_bytes_match`

### LitData Reader (uses real ImageNet val from S3)

Tests run twice: once via S3 path (`StreamingReader(remote_dir=...)`), once via local cached path.

- [x] Sample count = 50,000 — `test_sample_count_is_50000[s3]`, `[local]`
- [x] Sample count matches native — `test_sample_count_matches[s3]`, `[local]`
- [x] Field types detected — `test_field_types_detected[s3]`, `[local]`
- [x] Labels in valid range — `test_all_labels_in_range[s3]`, `[local]`
- [x] Labels match native — `test_labels_match_native[s3]`, `[local]`
- [x] Image bytes match native (SHA256) — `test_image_bytes_match_native[s3]`, `[local]`
- [x] Valid image format (JPEG/PNG) — `test_first_100_samples_valid_images[s3]`, `[local]`
- [x] Decoded pixels match native — `test_decoded_images_match[s3]`, `[local]`
- [x] Random samples decodable — `test_random_samples_valid[s3]`, `[local]`
- [x] Index field matches native — `test_indices_match_native[s3]`, `[local]`
- [x] Path field matches native — `test_paths_match_native[s3]`, `[local]`
- [x] All samples readable (slow) — `test_all_samples_readable[s3]`, `[local]`

**24 tests total** (12 core × 2 variants)

### ImageFolder Reader (uses real ImageNet val from S3)

Tests run twice: once via S3 path (`open_imagefolder("s3://...")`), once via local path (`SlipstreamImageFolder("/path")`).

- [x] Sample count = 50,000 — `test_sample_count_matches[s3]`, `[local]`
- [x] Class count = 1,000 — `test_class_count_matches[s3]`, `[local]`
- [x] Class-to-index consistent (0-999) — `test_class_to_idx_consistent[s3]`, `[local]`
- [x] Labels in valid range — `test_all_labels_in_range[s3]`, `[local]`
- [x] Labels match torchvision ordering — `test_labels_match_torchvision[s3]`, `[local]`
- [x] Image bytes match file read (SHA256) — `test_image_bytes_match_file[s3]`, `[local]`
- [x] Valid JPEG markers (SOI/EOI) — `test_first_100_samples_valid_jpeg[s3]`, `[local]`
- [x] Decoded pixels match torchvision — `test_decoded_images_match[s3]`, `[local]`
- [x] Random samples decodable — `test_random_samples_valid[s3]`, `[local]`
- [x] Path structure correct — `test_paths_have_class_and_filename[s3]`, `[local]`
- [x] Path class matches label — `test_path_class_matches_label[s3]`, `[local]`
- [x] S3 cache reuse works — `test_s3_uses_cache_on_second_load`

**23 tests total** (11 core × 2 variants + 1 cache test)

---

## Layer 2: Cache Round-Trip

Verify cache build → load preserves data correctly.

### JPEG Cache
- [x] JPEG bytes identical after round-trip — `test_cache_roundtrip.py::test_jpeg_bytes_identical_after_roundtrip`
- [x] Dimensions stored correctly — `test_cache_roundtrip.py::test_jpeg_dimensions_stored_correctly`
- [x] Labels preserved — `test_cache_roundtrip.py::test_labels_preserved`
- [x] Decodable after round-trip — `test_cache_roundtrip.py::test_jpeg_decodable_after_roundtrip`

### YUV420 Cache (PNG/other → YUV420)
- [x] Dimensions match original — `test_cache_roundtrip.py::test_png_yuv420_dimensions_match`
- [x] Decoded pixels match original (±1) — `test_cache_roundtrip.py::test_png_yuv420_decode_matches_original`
- [x] RGB→YUV420 round-trip correct — `test_cache_roundtrip.py::test_rgb_to_yuv420_roundtrip_solid_color`
- [x] Odd dimensions handled — `test_cache_roundtrip.py::test_rgb_to_yuv420_odd_dimensions`

### Cache Validation
- [x] Zero dimensions detected — `test_cache_roundtrip.py::test_verify_detects_dimension_parse_failure`
- [ ] Manifest corruption detected
- [ ] Field type validation

### Source → SlipCache Conversion (Full Dataset)

These tests build SlipCache from real ImageNet data and verify ALL 50,000 samples.

#### LitData → SlipCache
- [ ] Sample count = 50,000 — `test_litdata_to_slipcache.py::test_sample_count_is_50000`
- [ ] JPEG bytes identical (first 100) — `test_litdata_to_slipcache.py::test_first_100_bytes_comparison`
- [ ] Labels match (first 100) — `test_litdata_to_slipcache.py::test_first_100_labels_match`
- [ ] Decodable (first 100) — `test_litdata_to_slipcache.py::test_first_100_decodable`
- [ ] Format issues detected (all 50k, slow) — `test_litdata_to_slipcache.py::test_all_samples_format_check`
- [ ] Labels match (all 50k, slow) — `test_litdata_to_slipcache.py::test_all_labels_match`
- [ ] Decodable (all 50k, slow) — `test_litdata_to_slipcache.py::test_all_decodable`

#### ImageFolder → SlipCache
- [ ] Sample count = 50,000 — `test_imagefolder_to_slipcache.py::test_sample_count_is_50000`
- [ ] JPEG bytes identical (first 100) — `test_imagefolder_to_slipcache.py::test_first_100_bytes_comparison`
- [ ] Labels match (first 100) — `test_imagefolder_to_slipcache.py::test_first_100_labels_match`
- [ ] Decodable (first 100) — `test_imagefolder_to_slipcache.py::test_first_100_decodable`
- [ ] Format issues detected (all 50k, slow) — `test_imagefolder_to_slipcache.py::test_all_samples_format_check`
- [ ] Labels match (all 50k, slow) — `test_imagefolder_to_slipcache.py::test_all_labels_match`
- [ ] Decodable (all 50k, slow) — `test_imagefolder_to_slipcache.py::test_all_decodable`

#### FFCV → SlipCache
- [ ] Sample count = 50,000 — `test_ffcv_to_slipcache.py::test_sample_count_is_50000`
- [ ] JPEG bytes identical (first 100) — `test_ffcv_to_slipcache.py::test_first_100_bytes_identical`
- [ ] Labels match (first 100) — `test_ffcv_to_slipcache.py::test_first_100_labels_match`
- [ ] Decodable (first 100) — `test_ffcv_to_slipcache.py::test_first_100_decodable`
- [ ] JPEG bytes identical (all 50k, slow) — `test_ffcv_to_slipcache.py::test_all_bytes_identical`
- [ ] Labels match (all 50k, slow) — `test_ffcv_to_slipcache.py::test_all_labels_match`
- [ ] Decodable (all 50k, slow) — `test_ffcv_to_slipcache.py::test_all_decodable`

---

## Layer 3: Decode Correctness

Verify decoded images match reference within tolerance.

### JPEG Decode (TurboJPEG via libslipstream)
- [x] Solid colors match PIL (±2) — `test_decode_correctness.py::test_solid_color_decode`
- [x] Gradients match PIL (±5, mean <1.5) — `test_decode_correctness.py::test_gradient_decode`
- [x] Varying dimensions correct — `test_decode_correctness.py::test_varying_dimensions`
- [x] FFCV decoded matches native — `test_ffcv_verification.py::test_decoded_images_match`

### YUV420 Decode (BT.601)
- [x] Red (Y=76, U=85, V=255) — `test_decode_correctness.py::test_bt601_red`
- [x] Green (Y=150, U=44, V=21) — `test_decode_correctness.py::test_bt601_green`
- [x] Blue (Y=29, U=255, V=107) — `test_decode_correctness.py::test_bt601_blue`
- [x] White/Black neutral — `test_decode_correctness.py::test_bt601_white`, `test_bt601_black`

### Resize/Crop Operations
- [x] Center crop dimensions — `test_decode_correctness.py::test_center_crop_dimensions`
- [x] Random crop dimensions — `test_decode_correctness.py::test_random_crop_dimensions`
- [x] Resize preserves color — `test_decode_correctness.py::test_resize_preserves_color`

### Batch Processing
- [x] Batch images independent — `test_decode_correctness.py::test_batch_decode_all_images_different`
- [x] Varying sizes in batch — `test_decode_correctness.py::test_batch_varying_sizes`

---

## Layer 4: Functional Validation

Verify model accuracy matches across all formats.

### ResNet50 ImageNet-val (expected: 76.1% top-1)
- [ ] FFCV source accuracy
- [ ] LitData source accuracy
- [ ] ImageFolder source accuracy
- [ ] SlipCache (JPEG) accuracy
- [ ] SlipCache (YUV420) accuracy
- [ ] All within 0.1% of each other

### AlexNet ImageNet-val (cross-architecture validation)
- [ ] All formats within 0.1%

---

## Running Tests

```bash
# Local tests (no special setup, no S3 required)
uv run pytest tests/test_cache_roundtrip.py tests/test_decode_correctness.py -v

# Reader verification tests (requires AWS credentials)
uv run pytest tests/test_imagefolder_verification.py -v -s
uv run pytest tests/test_litdata_verification.py -v -s

# Source → SlipCache tests (requires AWS, builds cache on first run)
# Quick tests (first 100 samples)
uv run pytest tests/test_litdata_to_slipcache.py -v -s -m "not slow"
uv run pytest tests/test_imagefolder_to_slipcache.py -v -s -m "not slow"
uv run pytest tests/test_ffcv_to_slipcache.py -v -s -m "not slow"

# Full tests (all 50,000 samples - SLOW but catches PNG-in-JPEG issues)
uv run pytest tests/test_litdata_to_slipcache.py -v -s
uv run pytest tests/test_imagefolder_to_slipcache.py -v -s
uv run pytest tests/test_ffcv_to_slipcache.py -v -s

# Human-readable verification
uv run python scripts/verify_pipeline.py

# FFCV reader verification (requires docker for native ffcv-ssl comparison)
docker build -t slipstream-ffcv -f .devcontainer/Dockerfile .
docker run --rm \
  -v "$(pwd)":/workspace \
  -v ~/.aws:/root/.aws:ro \
  -v "$(pwd)/.devcontainer/cache":/root/.cache \
  -e SLIPSTREAM_CACHE_DIR=/root/.cache/slipstream \
  -w /workspace \
  slipstream-ffcv \
  bash -c "uv venv --clear && uv pip install -r .devcontainer/requirements-ffcv.txt && \
           uv run python libslipstream/setup.py build_ext --inplace && \
           uv run pytest tests/test_ffcv_verification.py -v"
```

---

## Test Files

| File | Purpose |
|------|---------|
| `tests/test_ffcv_verification.py` | FFCV reader vs native ffcv-ssl |
| `tests/test_litdata_verification.py` | LitData reader vs native litdata |
| `tests/test_imagefolder_verification.py` | ImageFolder reader vs torchvision |
| `tests/test_cache_roundtrip.py` | Cache build/load integrity (synthetic data) |
| `tests/test_litdata_to_slipcache.py` | LitData → SlipCache (all 50k samples) |
| `tests/test_imagefolder_to_slipcache.py` | ImageFolder → SlipCache (all 50k samples) |
| `tests/test_ffcv_to_slipcache.py` | FFCV → SlipCache (all 50k samples) |
| `tests/test_decode_correctness.py` | Decoder tolerance tests |
| `scripts/verify_pipeline.py` | Human-readable sanity check |

---

## Known Issues / Future Work

### PNG-in-JPEG Problem (CRITICAL for SlipCache)

**Issue**: ImageNet contains ~1% PNG files with `.JPEG` extension. The source datasets (LitData, ImageFolder, FFCV) correctly return raw bytes regardless of format. However, SlipCache decoders (TurboJPEG/libslipstream) are **JPEG-only**.

**Risk**: If SlipCache stores raw bytes from a PNG source file, the decoder will fail or produce garbage.

**Current test gap**: Cache round-trip tests use synthetic data (100 samples). **Must test all 50,000 ImageNet samples** to catch PNG files.

**Detection**: Check magic bytes:
- JPEG: `\xff\xd8` (SOI marker)
- PNG: `\x89PNG\r\n\x1a\n` (8-byte signature)

**Solution options**:
1. **Transcode PNG→JPEG during cache build** (lossy, but consistent decoder path)
2. **Store as YUV420** (already transcodes via PIL, handles both formats)
3. **Add PNG decoder path** (increases complexity, minimal benefit)

**Recommended**: Option 2 (YUV420) is already implemented and handles this transparently. For JPEG-bytes mode, Option 1 should be implemented.

**Action items**:
- [ ] Add full-dataset SlipCache test (all 50k samples)
- [ ] Detect PNG files during JPEG-bytes cache build and transcode
- [ ] Add warning/error for unsupported source formats
