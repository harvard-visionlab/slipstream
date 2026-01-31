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
 * - resize_simple(): Resize whole image (no crop)
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

// Thread-local storage key for TurboJPEG decompressor handle
static pthread_key_t key_tj_decompressor;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

/**
 * Initialize pthread key for thread-local TurboJPEG handle.
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
 * Convert planar YUV420P image to packed RGB in a pre-allocated buffer.
 *
 * YUV420P layout (I420):
 *   Y plane: height * width bytes
 *   U plane: (height/2) * (width/2) bytes
 *   V plane: (height/2) * (width/2) bytes
 *
 * All three planes are stored contiguously in `input`:
 *   [Y data][U data][V data]
 *
 * BT.601 conversion (standard for JPEG/sRGB content):
 *   R = Y + 1.402 * (V - 128)
 *   G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
 *   B = Y + 1.772 * (U - 128)
 *
 * Uses fixed-point arithmetic (shift 16) for speed without floats.
 *
 * @param input       YUV420P data (Y + U + V planes contiguous)
 * @param input_size  Total size of YUV data
 * @param output      Pre-allocated RGB output buffer [height * width * 3]
 * @param height      Image height (must be even)
 * @param width       Image width (must be even)
 * @return 0 on success, -1 on error
 */
int yuv420p_to_rgb_buffer(
    unsigned char *input, uint64_t input_size,
    unsigned char *output, uint32_t height, uint32_t width
) {
    uint64_t y_size = (uint64_t)height * width;
    uint64_t uv_size = ((uint64_t)height / 2) * (width / 2);
    uint64_t expected_size = y_size + 2 * uv_size;

    if (input_size < expected_size) return -1;

    unsigned char *y_plane = input;
    unsigned char *u_plane = input + y_size;
    unsigned char *v_plane = input + y_size + uv_size;

    uint32_t uv_stride = width / 2;

    // Fixed-point coefficients (BT.601, shift 16)
    // R = Y + 1.402 * (V-128)         → coeff_rv = 91881
    // G = Y - 0.344136*(U-128) - 0.714136*(V-128) → coeff_gu = 22554, coeff_gv = 46802
    // B = Y + 1.772 * (U-128)         → coeff_bu = 116130
    const int32_t coeff_rv =  91881;
    const int32_t coeff_gu = -22554;
    const int32_t coeff_gv = -46802;
    const int32_t coeff_bu = 116130;

    for (uint32_t row = 0; row < height; row++) {
        uint32_t uv_row = row >> 1;
        unsigned char *y_row = y_plane + (uint64_t)row * width;
        unsigned char *u_row = u_plane + (uint64_t)uv_row * uv_stride;
        unsigned char *v_row = v_plane + (uint64_t)uv_row * uv_stride;
        unsigned char *out_row = output + (uint64_t)row * width * 3;

        for (uint32_t col = 0; col < width; col++) {
            int32_t y_val = (int32_t)y_row[col];
            int32_t u_val = (int32_t)u_row[col >> 1] - 128;
            int32_t v_val = (int32_t)v_row[col >> 1] - 128;

            int32_t r = y_val + ((coeff_rv * v_val + 32768) >> 16);
            int32_t g = y_val + ((coeff_gu * u_val + coeff_gv * v_val + 32768) >> 16);
            int32_t b = y_val + ((coeff_bu * u_val + 32768) >> 16);

            // Clamp to [0, 255]
            out_row[col * 3 + 0] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            out_row[col * 3 + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
            out_row[col * 3 + 2] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
        }
    }

    return 0;
}

// Python module definition (minimal - we use ctypes for actual bindings)
static PyMethodDef libslipstreamMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libslipstreammodule = {
    PyModuleDef_HEAD_INIT,
    "_libslipstream",
    "Fast JPEG/YUV decode with TurboJPEG + stb resize - accessed via ctypes",
    -1,
    libslipstreamMethods
};

PyMODINIT_FUNC PyInit__libslipstream(void) {
    return PyModule_Create(&libslipstreammodule);
}

}  // extern "C"
