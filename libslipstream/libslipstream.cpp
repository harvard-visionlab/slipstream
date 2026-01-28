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

// stb_image_resize2 for crop+resize (header-only, no dependencies)
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

extern "C" {

// Thread-local storage keys for TurboJPEG handles
static pthread_key_t key_tj_decompressor;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

/**
 * Initialize pthread keys for thread-local TurboJPEG handles.
 * Called once per process via pthread_once.
 */
static void make_keys() {
    pthread_key_create(&key_tj_decompressor, NULL);
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
