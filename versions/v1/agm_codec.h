#ifndef AGM_CODEC_H
#define AGM_CODEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Unified Native .AGM Encoder & Decoder API
 *   - quality == 100 : AGM-L (Bit-Exact Lossless, "AGML" 9-byte header)
 *   - quality 1..99  : AGM-P (Perceptual Wavelet, "AGMP" 11-byte header)
 * ========================================================================= */
size_t agm_encode(
    const uint8_t *img, int w, int h, int channels,
    int quality, uint8_t *out_buf, size_t out_cap
);

int agm_decode(
    const uint8_t *in_buf, size_t in_len,
    uint8_t *out_img, int *out_w, int *out_h, int *out_ch
);

/* Direct Mode-Specific Native .AGM Functions */
size_t agm_encode_lossless(
    const uint8_t *img, int w, int h, int channels,
    uint8_t *out_buf, size_t out_cap
);

int agm_decode_lossless(
    const uint8_t *in_buf, size_t in_len,
    uint8_t *out_img
);

size_t agm_encode_lossy(
    const uint8_t *img, int w, int h, int channels,
    int quality, uint8_t *out_buf, size_t out_cap
);

int agm_decode_lossy(
    const uint8_t *in_buf, size_t in_len,
    uint8_t *out_img
);

/* Universal Standard-Format Scanline Pre-Conditioners (.PNG / .WEBP / .JPG) */
void agm_manifold_png_engine(
    const uint8_t *img, int w, int h, int channels,
    int quality, uint8_t *out_rec_img, uint8_t *out_png_scanlines
);

void agm_lattice_png_scanlines(
    const uint8_t *img, int w, int h, int channels,
    int delta_max, uint8_t *out_scanlines, uint8_t *out_rec_img
);

#ifdef __cplusplus
}
#endif

#endif /* AGM_CODEC_H */
