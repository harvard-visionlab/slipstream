/**
 * libslipstream - Fast parallel JPEG decoding with TurboJPEG + stb resize
 *
 * Based on FFCV's libffcv.cpp and litdata-mmap's libffcv_lite.cpp
 * Attribution: FFCV (https://github.com/libffcv/ffcv) - MIT License
 *
 * Key features:
 * - Thread-local TurboJPEG handles (no lock contention)
 * - stb_image_resize2 for fast crop+resize (no OpenCV dependency)
 * - Designed to be called from Numba prange loops (nogil)
 *
 * Functions:
 * - jpeg_header(): Get JPEG dimensions without full decode
 * - imdecode_simple(): Decode JPEG to full size (fast path)
 * - resize_crop(): Crop and resize in a single operation
 */

#include <cstdint>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>
#include <pthread.h>
#include <time.h>
#include <atomic>

#ifdef USE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

// stb_image_resize2 for crop+resize (header-only, no dependencies)
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Profiling counters (C++ atomics, must be outside extern "C")
static std::atomic<uint64_t> g_decode_ns{0};
static std::atomic<uint64_t> g_resize_ns{0};
static std::atomic<uint64_t> g_decode_count{0};
static std::atomic<uint64_t> g_resize_count{0};

extern "C" {

// Thread-local storage keys for TurboJPEG handles
static pthread_key_t key_tj_decompressor;
static pthread_key_t key_tj_transformer;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

/**
 * Initialize pthread keys for thread-local TurboJPEG handles.
 * Called once per process via pthread_once.
 */
static void make_keys() {
    pthread_key_create(&key_tj_decompressor, NULL);
    pthread_key_create(&key_tj_transformer, NULL);
}

/**
 * Get or create thread-local TurboJPEG decompressor handle.
 */
static tjhandle get_decompressor() {
    pthread_once(&key_once, make_keys);
    tjhandle decompressor = (tjhandle)pthread_getspecific(key_tj_decompressor);
    if (decompressor == NULL) {
        decompressor = tjInitDecompress();
        pthread_setspecific(key_tj_decompressor, decompressor);
    }
    return decompressor;
}

/**
 * Get or create thread-local TurboJPEG transformer handle.
 */
static tjhandle get_transformer() {
    pthread_once(&key_once, make_keys);
    tjhandle transformer = (tjhandle)pthread_getspecific(key_tj_transformer);
    if (transformer == NULL) {
        transformer = tjInitTransform();
        pthread_setspecific(key_tj_transformer, transformer);
    }
    return transformer;
}

/**
 * Decode JPEG header to get dimensions without full decode.
 *
 * @param input_buffer  Raw JPEG bytes
 * @param input_size    Size of JPEG data
 * @param out_width     Output: image width
 * @param out_height    Output: image height
 * @return 0 on success, -1 on error
 */
int jpeg_header(
    unsigned char *input_buffer,
    uint64_t input_size,
    uint32_t *out_width,
    uint32_t *out_height
) {
    tjhandle decompressor = get_decompressor();
    int width, height, subsamp, colorspace;

    int result = tjDecompressHeader3(
        decompressor, input_buffer, input_size,
        &width, &height, &subsamp, &colorspace
    );

    if (result == 0) {
        *out_width = (uint32_t)width;
        *out_height = (uint32_t)height;
    }
    return result;
}

/**
 * Get MCU block size based on JPEG subsampling.
 * MCU (Minimum Coded Unit) alignment is required for tjTransform crop.
 */
static int get_mcu_size(int subsamp) {
    switch (subsamp) {
        case TJSAMP_420:  return 16;  // 4:2:0 - most common
        case TJSAMP_422:  return 16;  // 4:2:2
        case TJSAMP_440:  return 16;  // 4:4:0
        default:          return 8;   // 4:4:4, gray, etc.
    }
}

/**
 * Align value down to nearest multiple of alignment.
 */
static int align_down(int value, int alignment) {
    return (value / alignment) * alignment;
}

/**
 * FFCV-style JPEG decode with crop in compressed domain.
 *
 * This is the key optimization: tjTransform() crops the JPEG data BEFORE
 * decompression, which is much faster than decoding the full image then cropping.
 *
 * The crop must be MCU-aligned (8 or 16 pixels depending on subsampling).
 * We align down to MCU boundaries and track the offset for final pixel-level crop.
 *
 * @param input_buffer   Raw JPEG bytes
 * @param input_size     Size of JPEG data
 * @param output_buffer  Pre-allocated output buffer (must fit final output!)
 * @param temp_buffer    Temporary buffer for decoded image (must fit MCU-aligned crop)
 * @param src_height     Original image height
 * @param src_width      Original image width
 * @param crop_y         Crop Y offset (will be MCU-aligned internally)
 * @param crop_x         Crop X offset (will be MCU-aligned internally)
 * @param crop_h         Crop height
 * @param crop_w         Crop width
 * @param target_h       Final output height (after resize)
 * @param target_w       Final output width (after resize)
 * @param hflip          Horizontal flip (true/false)
 * @return 0 on success, negative on error
 */
int imdecode_crop(
    unsigned char *input_buffer,
    uint64_t input_size,
    unsigned char *output_buffer,
    unsigned char *temp_buffer,
    uint32_t src_height,
    uint32_t src_width,
    uint32_t crop_y,
    uint32_t crop_x,
    uint32_t crop_h,
    uint32_t crop_w,
    uint32_t target_h,
    uint32_t target_w,
    int hflip
) {
    tjhandle decompressor = get_decompressor();
    tjhandle transformer = get_transformer();

    // Get JPEG header info (need subsampling for MCU alignment)
    int width, height, subsamp, colorspace;
    int result = tjDecompressHeader3(
        decompressor, input_buffer, input_size,
        &width, &height, &subsamp, &colorspace
    );
    if (result != 0) return result;

    // Get MCU size for alignment
    int mcu = get_mcu_size(subsamp);

    // Align crop coordinates to MCU boundaries
    int aligned_x = align_down(crop_x, mcu);
    int aligned_y = align_down(crop_y, mcu);

    // Calculate MCU-aligned crop dimensions (extend to cover original crop)
    int aligned_w = align_down(crop_x + crop_w + mcu - 1, mcu) - aligned_x;
    int aligned_h = align_down(crop_y + crop_h + mcu - 1, mcu) - aligned_y;

    // Clamp to image boundaries
    if (aligned_x + aligned_w > width) aligned_w = width - aligned_x;
    if (aligned_y + aligned_h > height) aligned_h = height - aligned_y;

    // Track pixel offset from MCU-aligned crop to actual crop
    int pixel_offset_x = crop_x - aligned_x;
    int pixel_offset_y = crop_y - aligned_y;

    // Set up tjTransform for crop (and optional hflip)
    tjtransform xform;
    memset(&xform, 0, sizeof(tjtransform));
    xform.r.x = aligned_x;
    xform.r.y = aligned_y;
    xform.r.w = aligned_w;
    xform.r.h = aligned_h;
    xform.options = TJXOPT_CROP;
    if (hflip) {
        xform.op = TJXOP_HFLIP;
    }

    // Allocate buffer for transformed (cropped) JPEG
    unsigned char *crop_jpeg = NULL;
    unsigned long crop_jpeg_size = 0;

    // Perform the JPEG-domain crop
    result = tjTransform(
        transformer, input_buffer, input_size,
        1, &crop_jpeg, &crop_jpeg_size,
        &xform, TJFLAG_FASTDCT
    );
    if (result != 0) {
        if (crop_jpeg) tjFree(crop_jpeg);
        return result;
    }

    // Find optimal scale factor for the cropped JPEG
    int num_factors = 0;
    tjscalingfactor *factors = tjGetScalingFactors(&num_factors);
    int best_idx = -1;
    int best_pixels = aligned_w * aligned_h;

    if (factors != NULL) {
        for (int i = 0; i < num_factors; i++) {
            int scaled_w = TJSCALED(aligned_w, factors[i]);
            int scaled_h = TJSCALED(aligned_h, factors[i]);

            // Need enough resolution for the target output
            // Scale the pixel offset region too
            int scaled_crop_w = (int)(crop_w * scaled_w / (float)aligned_w + 0.5f);
            int scaled_crop_h = (int)(crop_h * scaled_h / (float)aligned_h + 0.5f);

            if (scaled_crop_w >= (int)target_w && scaled_crop_h >= (int)target_h) {
                int pixels = scaled_w * scaled_h;
                if (pixels < best_pixels) {
                    best_pixels = pixels;
                    best_idx = i;
                }
            }
        }
    }

    // Calculate decode dimensions
    int decode_w, decode_h;
    tjscalingfactor sf;
    if (best_idx >= 0) {
        sf = factors[best_idx];
        decode_w = TJSCALED(aligned_w, sf);
        decode_h = TJSCALED(aligned_h, sf);
    } else {
        sf.num = 1;
        sf.denom = 1;
        decode_w = aligned_w;
        decode_h = aligned_h;
    }

    // Decompress the cropped JPEG
    result = tjDecompress2(
        decompressor, crop_jpeg, crop_jpeg_size, temp_buffer,
        decode_w, 0, decode_h, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
    );
    tjFree(crop_jpeg);
    if (result != 0) return result;

    // Scale pixel offsets to match decode dimensions
    int scaled_offset_x = (int)(pixel_offset_x * decode_w / (float)aligned_w + 0.5f);
    int scaled_offset_y = (int)(pixel_offset_y * decode_h / (float)aligned_h + 0.5f);
    int final_crop_w = (int)(crop_w * decode_w / (float)aligned_w + 0.5f);
    int final_crop_h = (int)(crop_h * decode_h / (float)aligned_h + 0.5f);

    // Clamp final crop dimensions
    if (scaled_offset_x + final_crop_w > decode_w) {
        final_crop_w = decode_w - scaled_offset_x;
    }
    if (scaled_offset_y + final_crop_h > decode_h) {
        final_crop_h = decode_h - scaled_offset_y;
    }

    // If hflip was applied, adjust the x offset
    if (hflip) {
        scaled_offset_x = decode_w - scaled_offset_x - final_crop_w;
    }

    // Final crop + resize using stb
    int stride = decode_w * 3;
    uint8_t *crop_start = temp_buffer + scaled_offset_y * stride + scaled_offset_x * 3;
    unsigned char *resize_result = stbir_resize_uint8_linear(
        crop_start, final_crop_w, final_crop_h, stride,
        output_buffer, target_w, target_h, 0,
        STBIR_RGB
    );

    return resize_result != NULL ? 0 : -1;
}

/**
 * Simple JPEG decode to full size (fast path, no transforms).
 *
 * @param input_buffer   Raw JPEG bytes
 * @param input_size     Size of JPEG data
 * @param output_buffer  Pre-allocated output buffer [height * width * 3]
 * @param height         Image height (from metadata)
 * @param width          Image width (from metadata)
 * @return 0 on success, negative on error
 */
int imdecode_simple(
    unsigned char *input_buffer,
    uint64_t input_size,
    unsigned char *output_buffer,
    uint32_t height,
    uint32_t width
) {
    tjhandle decompressor = get_decompressor();

    int result = tjDecompress2(
        decompressor, input_buffer, input_size, output_buffer,
        width, 0, height, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
    );

    return result;
}

/**
 * JPEG decode with TurboJPEG scaling (FFCV-style optimization).
 *
 * TurboJPEG can decode directly to smaller sizes (1/2, 1/4, 1/8, etc.) during
 * the IDCT step with almost no extra cost. This is key to FFCV's performance
 * for RandomResizedCrop - decode at a smaller size when the crop is small.
 *
 * @param input_buffer   Raw JPEG bytes
 * @param input_size     Size of JPEG data
 * @param output_buffer  Pre-allocated output buffer (must fit scaled size!)
 * @param src_height     Original image height (from metadata)
 * @param src_width      Original image width (from metadata)
 * @param scale_num      Scale numerator (e.g., 1 for 1/2)
 * @param scale_denom    Scale denominator (e.g., 2 for 1/2)
 * @param out_height     Output: actual decoded height
 * @param out_width      Output: actual decoded width
 * @return 0 on success, negative on error
 */
int imdecode_scaled(
    unsigned char *input_buffer,
    uint64_t input_size,
    unsigned char *output_buffer,
    uint32_t src_height,
    uint32_t src_width,
    uint32_t scale_num,
    uint32_t scale_denom,
    uint32_t *out_height,
    uint32_t *out_width
) {
    tjhandle decompressor = get_decompressor();

    // Calculate scaled dimensions using TurboJPEG's macro
    tjscalingfactor sf = {(int)scale_num, (int)scale_denom};
    int scaled_width = TJSCALED(src_width, sf);
    int scaled_height = TJSCALED(src_height, sf);

    // Output the actual dimensions
    *out_width = (uint32_t)scaled_width;
    *out_height = (uint32_t)scaled_height;

    // Decode at scaled size
    int result = tjDecompress2(
        decompressor, input_buffer, input_size, output_buffer,
        scaled_width, 0, scaled_height, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
    );

    return result;
}

/**
 * Combined scaled decode + crop + resize in one call.
 *
 * This is the FFCV-style optimized path:
 * 1. Choose optimal scale factor based on crop size
 * 2. Decode at that scale (fewer pixels!)
 * 3. Crop from the smaller decoded image
 * 4. Resize to target size
 *
 * @param input_buffer   Raw JPEG bytes
 * @param input_size     Size of JPEG data
 * @param temp_buffer    Temporary buffer for decoded image (must fit scaled size)
 * @param output_buffer  Final output buffer [target_h * target_w * 3]
 * @param src_height     Original image height
 * @param src_width      Original image width
 * @param crop_y         Crop Y offset (in original image coords)
 * @param crop_x         Crop X offset (in original image coords)
 * @param crop_h         Crop height (in original image coords)
 * @param crop_w         Crop width (in original image coords)
 * @param target_h       Final output height
 * @param target_w       Final output width
 * @return 0 on success, negative on error
 */
int decode_crop_resize(
    unsigned char *input_buffer,
    uint64_t input_size,
    unsigned char *temp_buffer,
    unsigned char *output_buffer,
    uint32_t src_height,
    uint32_t src_width,
    uint32_t crop_y,
    uint32_t crop_x,
    uint32_t crop_h,
    uint32_t crop_w,
    uint32_t target_h,
    uint32_t target_w
) {
    tjhandle decompressor = get_decompressor();

    // Get available scaling factors from TurboJPEG
    int num_factors = 0;
    tjscalingfactor *factors = tjGetScalingFactors(&num_factors);
    if (factors == NULL || num_factors == 0) {
        // Fallback: no scaling available, decode at full size
        int result = tjDecompress2(
            decompressor, input_buffer, input_size, temp_buffer,
            src_width, 0, src_height, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
        );
        if (result != 0) return result;

        // Crop and resize from full image
        int stride = src_width * 3;
        uint8_t *crop_start = temp_buffer + crop_y * stride + crop_x * 3;
        unsigned char *resize_result = stbir_resize_uint8_linear(
            crop_start, crop_w, crop_h, stride,
            output_buffer, target_w, target_h, 0,
            STBIR_RGB
        );
        return resize_result != NULL ? 0 : -1;
    }

    // Find the smallest scale that still covers the crop region
    // We need scaled_crop_dim >= target_dim for good quality
    int best_idx = -1;
    int best_pixels = src_width * src_height;  // Start with full size

    for (int i = 0; i < num_factors; i++) {
        int scaled_w = TJSCALED(src_width, factors[i]);
        int scaled_h = TJSCALED(src_height, factors[i]);

        // Calculate where the crop region maps to in scaled coords
        // Scale the crop coordinates proportionally
        int scaled_crop_w = (crop_w * scaled_w + src_width - 1) / src_width;
        int scaled_crop_h = (crop_h * scaled_h + src_height - 1) / src_height;

        // Ensure the scaled crop is at least as large as target for quality
        // (we'll resize down, not up significantly)
        if (scaled_crop_w >= (int)target_w && scaled_crop_h >= (int)target_h) {
            int pixels = scaled_w * scaled_h;
            if (pixels < best_pixels) {
                best_pixels = pixels;
                best_idx = i;
            }
        }
    }

    // Decode at best scale (or full size if no suitable scale found)
    int scaled_w, scaled_h;
    if (best_idx >= 0) {
        scaled_w = TJSCALED(src_width, factors[best_idx]);
        scaled_h = TJSCALED(src_height, factors[best_idx]);
    } else {
        scaled_w = src_width;
        scaled_h = src_height;
    }

    int result = tjDecompress2(
        decompressor, input_buffer, input_size, temp_buffer,
        scaled_w, 0, scaled_h, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
    );
    if (result != 0) return result;

    // Map crop coordinates to scaled image
    int scaled_crop_x = (crop_x * scaled_w) / src_width;
    int scaled_crop_y = (crop_y * scaled_h) / src_height;
    int scaled_crop_w = (crop_w * scaled_w + src_width - 1) / src_width;
    int scaled_crop_h = (crop_h * scaled_h + src_height - 1) / src_height;

    // Clamp to valid bounds
    if (scaled_crop_x + scaled_crop_w > scaled_w) {
        scaled_crop_w = scaled_w - scaled_crop_x;
    }
    if (scaled_crop_y + scaled_crop_h > scaled_h) {
        scaled_crop_h = scaled_h - scaled_crop_y;
    }

    // Crop and resize
    int stride = scaled_w * 3;
    uint8_t *crop_start = temp_buffer + scaled_crop_y * stride + scaled_crop_x * 3;
    unsigned char *resize_result = stbir_resize_uint8_linear(
        crop_start, scaled_crop_w, scaled_crop_h, stride,
        output_buffer, target_w, target_h, 0,
        STBIR_RGB
    );

    return resize_result != NULL ? 0 : -1;
}

/**
 * Crop and resize image region using stb_image_resize2.
 *
 * This is the FFCV-style fused crop+resize operation, designed to be
 * called from within a Numba prange loop for maximum parallelism.
 *
 * @param source_p      Source image buffer pointer (as int64 for Numba compatibility)
 * @param source_h      Source image height
 * @param source_w      Source image width
 * @param crop_y        Crop region start row
 * @param crop_x        Crop region start column
 * @param crop_h        Crop region height
 * @param crop_w        Crop region width
 * @param dest_p        Destination buffer pointer (as int64 for Numba compatibility)
 * @param target_h      Target output height
 * @param target_w      Target output width
 * @return 1 on success, 0 on failure
 */
int resize_crop(
    int64_t source_p,
    int64_t source_h,
    int64_t source_w,
    int64_t crop_y,
    int64_t crop_x,
    int64_t crop_h,
    int64_t crop_w,
    int64_t dest_p,
    int64_t target_h,
    int64_t target_w
) {
    uint8_t *source = (uint8_t *)source_p;
    uint8_t *dest = (uint8_t *)dest_p;

    // Calculate pointer to crop region start
    // Source is RGB with row stride = source_w * 3
    int source_stride = (int)source_w * 3;
    uint8_t *crop_start = source + crop_y * source_stride + crop_x * 3;

    // Use stb_image_resize2 for the resize
    // STBIR_RGB = 3 channels, no alpha
    unsigned char *result = stbir_resize_uint8_linear(
        crop_start,                    // input pixels (start of crop region)
        (int)crop_w,                   // input width
        (int)crop_h,                   // input height
        source_stride,                 // input stride (full source row, not crop width!)
        dest,                          // output pixels
        (int)target_w,                 // output width
        (int)target_h,                 // output height
        0,                             // output stride (0 = packed)
        STBIR_RGB                      // pixel layout
    );

    return result != NULL ? 1 : 0;
}

/**
 * Simple resize without crop (whole image).
 *
 * @param source_p      Source image buffer pointer
 * @param source_h      Source image height
 * @param source_w      Source image width
 * @param dest_p        Destination buffer pointer
 * @param target_h      Target output height
 * @param target_w      Target output width
 * @return 1 on success, 0 on failure
 */
int resize_simple(
    int64_t source_p,
    int64_t source_h,
    int64_t source_w,
    int64_t dest_p,
    int64_t target_h,
    int64_t target_w
) {
    uint8_t *source = (uint8_t *)source_p;
    uint8_t *dest = (uint8_t *)dest_p;

    unsigned char *result = stbir_resize_uint8_linear(
        source,
        (int)source_w,
        (int)source_h,
        0,                             // input stride (0 = packed)
        dest,
        (int)target_w,
        (int)target_h,
        0,                             // output stride (0 = packed)
        STBIR_RGB
    );

    return result != NULL ? 1 : 0;
}

// =============================================================================
// Profiling support
// =============================================================================

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/**
 * Decode + crop + resize with per-step timing.
 * Same signature as resize_crop but also does decode.
 * Accumulates timing into atomic counters readable via get_profile_stats().
 */
int decode_and_resize_profiled(
    int64_t jpeg_p, int64_t jpeg_size,
    int64_t temp_p, int64_t temp_h, int64_t temp_w,
    int64_t crop_y, int64_t crop_x, int64_t crop_h, int64_t crop_w,
    int64_t dest_p, int64_t target_h, int64_t target_w
) {
    uint8_t *jpeg = (uint8_t *)jpeg_p;
    uint8_t *temp = (uint8_t *)temp_p;
    uint8_t *dest = (uint8_t *)dest_p;

    // Time decode
    uint64_t t0 = now_ns();
    tjhandle decompressor = get_decompressor();
    int result = tjDecompress2(
        decompressor, jpeg, jpeg_size, temp,
        (int)temp_w, 0, (int)temp_h, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC
    );
    uint64_t t1 = now_ns();
    if (result != 0) return -1;

    // Time resize
    int source_stride = (int)temp_w * 3;
    uint8_t *crop_start = temp + crop_y * source_stride + crop_x * 3;
    unsigned char *resize_result = stbir_resize_uint8_linear(
        crop_start, (int)crop_w, (int)crop_h, source_stride,
        dest, (int)target_w, (int)target_h, 0, STBIR_RGB
    );
    uint64_t t2 = now_ns();

    g_decode_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
    g_resize_ns.fetch_add(t2 - t1, std::memory_order_relaxed);
    g_decode_count.fetch_add(1, std::memory_order_relaxed);
    g_resize_count.fetch_add(1, std::memory_order_relaxed);

    return resize_result != NULL ? 1 : 0;
}

/**
 * Get profiling stats. Returns total nanoseconds and count for decode and resize.
 * Call reset_profile_stats() between benchmarks.
 */
void get_profile_stats(
    uint64_t *out_decode_ns, uint64_t *out_resize_ns,
    uint64_t *out_decode_count, uint64_t *out_resize_count
) {
    *out_decode_ns = g_decode_ns.load(std::memory_order_relaxed);
    *out_resize_ns = g_resize_ns.load(std::memory_order_relaxed);
    *out_decode_count = g_decode_count.load(std::memory_order_relaxed);
    *out_resize_count = g_resize_count.load(std::memory_order_relaxed);
}

void reset_profile_stats() {
    g_decode_ns.store(0, std::memory_order_relaxed);
    g_resize_ns.store(0, std::memory_order_relaxed);
    g_decode_count.store(0, std::memory_order_relaxed);
    g_resize_count.store(0, std::memory_order_relaxed);
}

// =============================================================================
// OpenCV resize (optional, built with USE_OPENCV=1)
// =============================================================================

#ifdef USE_OPENCV
/**
 * Crop + resize using OpenCV cv::resize with INTER_AREA.
 * Same signature as resize_crop for drop-in replacement.
 */
int resize_crop_cv(
    int64_t source_p,
    int64_t source_h,
    int64_t source_w,
    int64_t crop_y,
    int64_t crop_x,
    int64_t crop_h,
    int64_t crop_w,
    int64_t dest_p,
    int64_t target_h,
    int64_t target_w
) {
    cv::Mat source((int)source_h, (int)source_w, CV_8UC3, (uint8_t *)source_p);
    cv::Mat dest((int)target_h, (int)target_w, CV_8UC3, (uint8_t *)dest_p);
    cv::resize(
        source.rowRange((int)crop_y, (int)(crop_y + crop_h))
              .colRange((int)crop_x, (int)(crop_x + crop_w)),
        dest, dest.size(), 0, 0, cv::INTER_AREA
    );
    return 1;
}
#endif

/**
 * Check if OpenCV support was compiled in.
 * Returns 1 if USE_OPENCV was defined at build time, 0 otherwise.
 */
int has_opencv() {
#ifdef USE_OPENCV
    return 1;
#else
    return 0;
#endif
}

/**
 * memcpy wrapper callable from Numba.
 */
void my_memcpy(void *source, void *dst, uint64_t size) {
    memcpy(dst, source, size);
}

/**
 * Get TurboJPEG scaling factors for a given input/output size.
 *
 * TurboJPEG supports specific scaling factors during IDCT:
 * 1/1, 1/2, 3/8, 1/4, 3/16, 1/8, etc.
 *
 * @param input_size     Original dimension (height or width)
 * @param output_size    Desired output dimension
 * @param out_num        Output: scale numerator
 * @param out_denom      Output: scale denominator
 */
void get_scale_factor(
    uint32_t input_size,
    uint32_t output_size,
    uint32_t *out_num,
    uint32_t *out_denom
) {
    // TurboJPEG scaling factors (sorted from largest to smallest)
    static const tjscalingfactor factors[] = {
        {2, 1}, {15, 8}, {7, 4}, {13, 8}, {3, 2}, {11, 8},
        {5, 4}, {9, 8}, {1, 1}, {7, 8}, {3, 4}, {5, 8},
        {1, 2}, {3, 8}, {1, 4}, {1, 8}
    };
    static const int num_factors = sizeof(factors) / sizeof(factors[0]);

    // Find best scaling factor (smallest that produces >= output_size)
    int best_idx = 8;  // Default: 1/1
    int best_size = input_size;

    for (int i = 0; i < num_factors; i++) {
        int scaled = TJSCALED(input_size, factors[i]);
        if (scaled >= (int)output_size && scaled < best_size) {
            best_size = scaled;
            best_idx = i;
        }
    }

    *out_num = factors[best_idx].num;
    *out_denom = factors[best_idx].denom;
}

// Python module definition (minimal - we use ctypes for actual bindings)
static PyMethodDef libslipstreamMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libslipstreammodule = {
    PyModuleDef_HEAD_INIT,
    "_libslipstream",
    "Fast JPEG decode with TurboJPEG + stb resize - accessed via ctypes",
    -1,
    libslipstreamMethods
};

PyMODINIT_FUNC PyInit__libslipstream(void) {
    return PyModule_Create(&libslipstreammodule);
}

}  // extern "C"
