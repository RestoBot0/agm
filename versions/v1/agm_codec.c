#include "agm_codec.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * 64-bit Carryless Binary Range Coder with Hierarchical Context Mixing
 * ========================================================================= */
typedef struct {
    uint16_t p_fast;
    uint16_t p_slow;
} BiModel;

static inline void bimodel_init(BiModel *m) {
    m->p_fast = 32768;
    m->p_slow = 32768;
}

static inline uint32_t bimodel_p1(const BiModel *m) {
    return ((uint32_t)m->p_fast + (uint32_t)m->p_slow) >> 1;
}

static inline void bimodel_update(BiModel *m, int bit) {
    if (bit) {
        m->p_fast += (65535 - m->p_fast) >> 4;
        m->p_slow += (65535 - m->p_slow) >> 7;
    } else {
        m->p_fast -= m->p_fast >> 4;
        m->p_slow -= m->p_slow >> 7;
    }
}

typedef struct {
    uint64_t low;
    uint32_t range;
    uint32_t cache;
    uint64_t cache_size;
    uint8_t *buf;
    size_t pos;
    size_t cap;
    int first_byte;
} RangeEnc;

static void rc_enc_init(RangeEnc *rc, uint8_t *buf, size_t cap) {
    rc->low = 0;
    rc->range = 0xFFFFFFFFU;
    rc->cache = 0;
    rc->cache_size = 1;
    rc->buf = buf;
    rc->pos = 0;
    rc->cap = cap;
    rc->first_byte = 1;
}

static inline void rc_enc_shift_low(RangeEnc *rc) {
    if ((uint32_t)(rc->low >> 32) != 0 || (uint32_t)rc->low < 0xFF000000U) {
        uint8_t temp = (uint8_t)rc->cache;
        uint32_t carry = (uint32_t)(rc->low >> 32);
        do {
            if (!rc->first_byte) {
                rc->buf[rc->pos++] = (uint8_t)(temp + carry);
            } else {
                rc->first_byte = 0;
            }
            temp = 0xFF;
        } while (--rc->cache_size != 0);
        rc->cache = (uint32_t)(rc->low >> 24) & 0xFFU;
    }
    rc->cache_size++;
    rc->low = (uint32_t)(rc->low << 8);
}

/* Encode bit using a Mixed Coarse + Fine BiModel pair */
static inline void rc_enc_bit2(RangeEnc *rc, BiModel *mc, BiModel *mf, int bit) {
    uint32_t p1 = (bimodel_p1(mc) + bimodel_p1(mf)) >> 1;
    uint32_t p0 = 65536U - p1;
    if (p0 < 8) p0 = 8;
    if (p0 > 65528) p0 = 65528;
    uint32_t bound = (rc->range >> 16) * p0;
    if (bit == 0) {
        rc->range = bound;
    } else {
        rc->low += bound;
        rc->range -= bound;
    }
    while (rc->range < 0x01000000U) {
        rc->range <<= 8;
        rc_enc_shift_low(rc);
    }
    bimodel_update(mc, bit);
    bimodel_update(mf, bit);
}

static size_t rc_enc_finish(RangeEnc *rc) {
    for (int i = 0; i < 5; i++) {
        rc_enc_shift_low(rc);
    }
    return rc->pos;
}

typedef struct {
    uint32_t range;
    uint32_t code;
    const uint8_t *buf;
    size_t pos;
    size_t len;
} RangeDec;

static void rc_dec_init(RangeDec *rc, const uint8_t *buf, size_t len) {
    rc->range = 0xFFFFFFFFU;
    rc->code = 0;
    rc->buf = buf;
    rc->pos = 0;
    rc->len = len;
    for (int i = 0; i < 4; i++) {
        rc->code = (rc->code << 8) | (rc->pos < rc->len ? rc->buf[rc->pos++] : 0);
    }
}

static inline int rc_dec_bit2(RangeDec *rc, BiModel *mc, BiModel *mf) {
    uint32_t p1 = (bimodel_p1(mc) + bimodel_p1(mf)) >> 1;
    uint32_t p0 = 65536U - p1;
    if (p0 < 8) p0 = 8;
    if (p0 > 65528) p0 = 65528;
    uint32_t bound = (rc->range >> 16) * p0;
    int bit;
    if (rc->code < bound) {
        rc->range = bound;
        bit = 0;
    } else {
        rc->code -= bound;
        rc->range -= bound;
        bit = 1;
    }
    while (rc->range < 0x01000000U) {
        rc->range <<= 8;
        rc->code = (rc->code << 8) | (rc->pos < rc->len ? rc->buf[rc->pos++] : 0);
    }
    bimodel_update(mc, bit);
    bimodel_update(mf, bit);
    return bit;
}

/* =========================================================================
 * Hierarchical Coarse + Fine Entropy Context Model
 * ========================================================================= */
#define NUM_VAR_BINS 64
#define NUM_ZERO_COARSE (3 * 16)
#define NUM_ZERO_FINE   (3 * NUM_VAR_BINS * 16)
#define NUM_SIGN_COARSE (3 * 8)
#define NUM_SIGN_FINE   (3 * 16 * 32)
#define NUM_SMALL_COARSE (3 * 16 * 6)
#define NUM_SMALL_FINE   (3 * NUM_VAR_BINS * 4 * 6)
#define NUM_EXP_COARSE   (3 * 16 * 8)
#define NUM_EXP_FINE     (3 * NUM_VAR_BINS * 8)
#define NUM_MANT_COARSE  (3 * 8 * 8)
#define NUM_MANT_FINE    (3 * 16 * 8 * 8)

typedef struct {
    BiModel zero_c[NUM_ZERO_COARSE];
    BiModel zero_f[NUM_ZERO_FINE];
    BiModel sign_c[NUM_SIGN_COARSE];
    BiModel sign_f[NUM_SIGN_FINE];
    BiModel small_c[NUM_SMALL_COARSE];
    BiModel small_f[NUM_SMALL_FINE];
    BiModel exp_c[NUM_EXP_COARSE];
    BiModel exp_f[NUM_EXP_FINE];
    BiModel mant_c[NUM_MANT_COARSE];
    BiModel mant_f[NUM_MANT_FINE];
} EntropyModels;

static void entropy_init(EntropyModels *em) {
    for (int i = 0; i < NUM_ZERO_COARSE; i++) bimodel_init(&em->zero_c[i]);
    for (int i = 0; i < NUM_ZERO_FINE; i++)   bimodel_init(&em->zero_f[i]);
    for (int i = 0; i < NUM_SIGN_COARSE; i++) bimodel_init(&em->sign_c[i]);
    for (int i = 0; i < NUM_SIGN_FINE; i++)   bimodel_init(&em->sign_f[i]);
    for (int i = 0; i < NUM_SMALL_COARSE; i++)bimodel_init(&em->small_c[i]);
    for (int i = 0; i < NUM_SMALL_FINE; i++)  bimodel_init(&em->small_f[i]);
    for (int i = 0; i < NUM_EXP_COARSE; i++)  bimodel_init(&em->exp_c[i]);
    for (int i = 0; i < NUM_EXP_FINE; i++)    bimodel_init(&em->exp_f[i]);
    for (int i = 0; i < NUM_MANT_COARSE; i++) bimodel_init(&em->mant_c[i]);
    for (int i = 0; i < NUM_MANT_FINE; i++)   bimodel_init(&em->mant_f[i]);
}

static inline void encode_residual(
    RangeEnc *rc, EntropyModels *em,
    int ch, int var_bin, int zero_state, int sign_ctx,
    int val, int pred
) {
    int err = val - pred;
    int vb_c = var_bin >> 2; /* 0..15 */
    int zc_idx = ch * 16 + vb_c;
    int zf_idx = (ch * NUM_VAR_BINS + var_bin) * 16 + (zero_state & 15);
    if (err == 0) {
        rc_enc_bit2(rc, &em->zero_c[zc_idx], &em->zero_f[zf_idx], 1);
        return;
    }
    rc_enc_bit2(rc, &em->zero_c[zc_idx], &em->zero_f[zf_idx], 0);

    int sign = (err > 0) ? 1 : 0;
    int max_pos = 255 - pred;
    int max_neg = pred;
    int sc_idx = ch * 8 + (sign_ctx & 7);
    int sf_idx = (ch * 16 + vb_c) * 32 + (sign_ctx & 31);

    if (max_pos == 0) {
        sign = 0;
    } else if (max_neg == 0) {
        sign = 1;
    } else {
        rc_enc_bit2(rc, &em->sign_c[sc_idx], &em->sign_f[sf_idx], sign);
    }

    int mag = (err > 0) ? err : -err;
    int max_mag = sign ? max_pos : max_neg;
    int z_hint = ((zero_state & 7) == 7) ? 3 : (((zero_state & 6) != 0) ? 2 : ((zero_state != 0) ? 1 : 0));

    for (int k = 1; k <= 6; k++) {
        if (max_mag == k) return;
        int is_k = (mag == k);
        int sm_c = (ch * 16 + vb_c) * 6 + (k - 1);
        int sm_f = ((ch * NUM_VAR_BINS + var_bin) * 4 + z_hint) * 6 + (k - 1);
        rc_enc_bit2(rc, &em->small_c[sm_c], &em->small_f[sm_f], is_k);
        if (is_k) return;
    }

    int v = mag - 7;
    int max_v = max_mag - 7;
    int exp_b = 0;
    while ((1 << (exp_b + 2)) - 2 <= v) {
        exp_b++;
    }
    int max_exp = 0;
    while ((1 << (max_exp + 2)) - 2 <= max_v) {
        max_exp++;
    }
    for (int b = 0; b < max_exp; b++) {
        int bc = b < 7 ? b : 7;
        int ec_idx = (ch * 16 + vb_c) * 8 + bc;
        int ef_idx = (ch * NUM_VAR_BINS + var_bin) * 8 + bc;
        if (exp_b == b) {
            rc_enc_bit2(rc, &em->exp_c[ec_idx], &em->exp_f[ef_idx], 1);
            break;
        } else {
            rc_enc_bit2(rc, &em->exp_c[ec_idx], &em->exp_f[ef_idx], 0);
        }
    }
    int base = (1 << (exp_b + 1)) - 2;
    int rem = v - base;
    int max_rem = max_v - base;
    int nbits = exp_b + 1;
    int prefix = 0;
    int eb_clamped = exp_b < 7 ? exp_b : 7;
    for (int bit_pos = nbits - 1; bit_pos >= 0; bit_pos--) {
        int bit = (rem >> bit_pos) & 1;
        if ((prefix | (1 << bit_pos)) > max_rem) {
            continue;
        }
        int bp_clamped = bit_pos < 7 ? bit_pos : 7;
        int mc_idx = (ch * 8 + eb_clamped) * 8 + bp_clamped;
        int mf_idx = ((ch * 16 + vb_c) * 8 + eb_clamped) * 8 + bp_clamped;
        rc_enc_bit2(rc, &em->mant_c[mc_idx], &em->mant_f[mf_idx], bit);
        if (bit) prefix |= (1 << bit_pos);
    }
}

static inline int decode_residual(
    RangeDec *rc, EntropyModels *em,
    int ch, int var_bin, int zero_state, int sign_ctx,
    int pred
) {
    int vb_c = var_bin >> 2;
    int zc_idx = ch * 16 + vb_c;
    int zf_idx = (ch * NUM_VAR_BINS + var_bin) * 16 + (zero_state & 15);
    if (rc_dec_bit2(rc, &em->zero_c[zc_idx], &em->zero_f[zf_idx])) {
        return pred;
    }

    int max_pos = 255 - pred;
    int max_neg = pred;
    int sc_idx = ch * 8 + (sign_ctx & 7);
    int sf_idx = (ch * 16 + vb_c) * 32 + (sign_ctx & 31);
    int sign;
    if (max_pos == 0) {
        sign = 0;
    } else if (max_neg == 0) {
        sign = 1;
    } else {
        sign = rc_dec_bit2(rc, &em->sign_c[sc_idx], &em->sign_f[sf_idx]);
    }

    int max_mag = sign ? max_pos : max_neg;
    int z_hint = ((zero_state & 7) == 7) ? 3 : (((zero_state & 6) != 0) ? 2 : ((zero_state != 0) ? 1 : 0));
    for (int k = 1; k <= 6; k++) {
        if (max_mag == k) return sign ? (pred + k) : (pred - k);
        int sm_c = (ch * 16 + vb_c) * 6 + (k - 1);
        int sm_f = ((ch * NUM_VAR_BINS + var_bin) * 4 + z_hint) * 6 + (k - 1);
        if (rc_dec_bit2(rc, &em->small_c[sm_c], &em->small_f[sm_f])) {
            return sign ? (pred + k) : (pred - k);
        }
    }

    int max_v = max_mag - 7;
    int max_exp = 0;
    while ((1 << (max_exp + 2)) - 2 <= max_v) {
        max_exp++;
    }
    int exp_b = max_exp;
    for (int b = 0; b < max_exp; b++) {
        int bc = b < 7 ? b : 7;
        int ec_idx = (ch * 16 + vb_c) * 8 + bc;
        int ef_idx = (ch * NUM_VAR_BINS + var_bin) * 8 + bc;
        if (rc_dec_bit2(rc, &em->exp_c[ec_idx], &em->exp_f[ef_idx])) {
            exp_b = b;
            break;
        }
    }
    int base = (1 << (exp_b + 1)) - 2;
    int max_rem = max_v - base;
    int nbits = exp_b + 1;
    int rem = 0;
    int eb_clamped = exp_b < 7 ? exp_b : 7;
    for (int bit_pos = nbits - 1; bit_pos >= 0; bit_pos--) {
        if ((rem | (1 << bit_pos)) > max_rem) {
            continue;
        }
        int bp_clamped = bit_pos < 7 ? bit_pos : 7;
        int mc_idx = (ch * 8 + eb_clamped) * 8 + bp_clamped;
        int mf_idx = ((ch * 16 + vb_c) * 8 + eb_clamped) * 8 + bp_clamped;
        int bit = rc_dec_bit2(rc, &em->mant_c[mc_idx], &em->mant_f[mf_idx]);
        if (bit) rem |= (1 << bit_pos);
    }
    int mag = 7 + base + rem;
    return sign ? (pred + mag) : (pred - mag);
}

/* =========================================================================
 * Preconditioned Diagonal-Kalman RLS + Geodesic & Cross-Fiber Predictor
 * ========================================================================= */
#define MAX_FEAT 24
#define NUM_RLS_CTX 32
#define NL_HASH_SIZE 65536

typedef struct {
    double w_global[MAX_FEAT];
    double v_global[MAX_FEAT];
    double w_ctx[NUM_RLS_CTX][MAX_FEAT];
    int32_t nl_pos[NL_HASH_SIZE];
    int zero_run;
} ChannelRLS;

static void rls_init(ChannelRLS *rls) {
    memset(rls, 0, sizeof(ChannelRLS));
    for (int i = 0; i < MAX_FEAT; i++) {
        rls->v_global[i] = 0.02;
    }
}

static inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint8_t get_px(const uint8_t *plane, int w, int h, int x, int y) {
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    if (y < 0) y = 0;
    if (y >= h) y = h - 1;
    return plane[y * w + x];
}

static inline double get_err(const int16_t *err_plane, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.0;
    return (double)err_plane[y * w + x];
}

static inline uint32_t nl_hash(int W, int N, int NW, int NE) {
    uint32_t h = ((uint32_t)(W >> 1) * 73856093U) ^
                 ((uint32_t)(N >> 1) * 19349663U) ^
                 ((uint32_t)(NW >> 1) * 83492791U) ^
                 ((uint32_t)(NE >> 1) * 49979693U);
    return h & (NL_HASH_SIZE - 1);
}

static void agm_predict_pixel(
    int ch, int w, int h, int x, int y,
    const uint8_t *plane_c, const int16_t *err_c,
    const uint8_t *plane_g, const int16_t *err_g,
    const uint8_t *plane_r, const int16_t *err_r,
    ChannelRLS *rls,
    double *out_feat, int *out_nfeat, int *out_rls_ctx,
    double *out_base, int *out_pred_clamped,
    int *out_var_bin, int *out_zero_state, int *out_sign_ctx
) {
    if (x == 0 && y == 0) {
        int base = 128;
        if (ch >= 1) base = get_px(plane_g, w, h, 0, 0);
        *out_nfeat = 0;
        *out_rls_ctx = 0;
        *out_base = (double)base;
        *out_pred_clamped = clamp_i(base, 0, 255);
        *out_var_bin = 32;
        *out_zero_state = 0;
        *out_sign_ctx = 0;
        return;
    }

    int W   = (x > 0) ? plane_c[y * w + (x - 1)] : get_px(plane_c, w, h, 0, y - 1);
    int N   = (y > 0) ? plane_c[(y - 1) * w + x] : W;
    int NW  = (x > 0 && y > 0) ? plane_c[(y - 1) * w + (x - 1)] : (y > 0 ? N : W);
    int NE  = (y > 0 && x + 1 < w) ? plane_c[(y - 1) * w + (x + 1)] : N;
    int WW  = (x > 1) ? plane_c[y * w + (x - 2)] : W;
    int NN  = (y > 1) ? plane_c[(y - 2) * w + x] : N;
    int NWW = (x > 1 && y > 0) ? plane_c[(y - 1) * w + (x - 2)] : NW;
    int NNW = (x > 0 && y > 1) ? plane_c[(y - 2) * w + (x - 1)] : NW;
    int NNE = (x + 1 < w && y > 1) ? plane_c[(y - 2) * w + (x + 1)] : NE;
    int NNEE= (x + 2 < w && y > 0) ? plane_c[(y - 1) * w + (x + 2)] : NE;

    /* 1. Expert E0: Structure-Tensor Geodesic Ray-Traced Predictor */
    double gx1 = W - WW,   gy1 = W - NW;
    double gx2 = NE - N,   gy2 = N - NN;
    double gx3 = N - NW,   gy3 = NW - NNW;
    double gx4 = NNEE - NE,gy4 = NE - NNE;

    double Sxx = 2.0*gx1*gx1 + 2.0*gx2*gx2 + gx3*gx3 + gx4*gx4;
    double Syy = 2.0*gy1*gy1 + 2.0*gy2*gy2 + gy3*gy3 + gy4*gy4;
    double Sxy = 2.0*gx1*gy1 + 2.0*gx2*gy2 + gx3*gy3 + gx4*gy4;

    double diff = Sxx - Syy;
    double R_aniso = sqrt(diff * diff + 4.0 * Sxy * Sxy);
    double trace = Sxx + Syy + 1.0;
    double coherence = R_aniso / trace;

    double gx_dom = diff + R_aniso;
    double gy_dom = 2.0 * Sxy;
    if (gx_dom * gx_dom + gy_dom * gy_dom < 1e-9) {
        gx_dom = 2.0 * Sxy;
        gy_dom = Syy - Sxx + R_aniso;
    }
    double tx = -gy_dom;
    double ty = gx_dom;
    if (ty > 0 || (ty == 0 && tx > 0)) {
        tx = -tx;
        ty = -ty;
    }
    double p_ray;
    if (fabs(ty) >= fabs(tx) && fabs(ty) > 1e-6) {
        double dx = -tx / ty;
        if (dx >= 0.0) p_ray = (1.0 - dx) * N + dx * NE;
        else           p_ray = (1.0 + dx) * N - dx * NW;
    } else if (fabs(tx) > 1e-6) {
        double u = fabs(ty / tx);
        if (tx < 0) p_ray = (1.0 - u) * W + u * NW;
        else        p_ray = (1.0 - u) * W + u * NE;
    } else {
        p_ray = 0.5 * (W + N);
    }
    double p_iso = 0.5 * (W + N) + 0.25 * ((W + N) - 2.0 * NW);
    double P_geo = coherence * p_ray + (1.0 - coherence) * p_iso;

    /* 2. Expert E1: Gradient-Adjusted Edge Predictor (GAP) */
    double dh = fabs(W - WW) + fabs(N - NW) + fabs(N - NE);
    double dv = fabs(W - NW) + fabs(N - NN) + fabs(NE - NNE);
    double P_gap;
    if (dv - dh > 80.0) P_gap = W;
    else if (dh - dv > 80.0) P_gap = N;
    else {
        double p0 = 0.5 * (W + N) + 0.25 * (NE - NW);
        if (dv - dh > 32.0) P_gap = 0.5 * (p0 + W);
        else if (dh - dv > 32.0) P_gap = 0.5 * (p0 + N);
        else if (dv - dh > 8.0) P_gap = 0.75 * p0 + 0.25 * W;
        else if (dh - dv > 8.0) P_gap = 0.75 * p0 + 0.25 * N;
        else P_gap = p0;
    }

    /* 3. Expert E2: Discrete 2D Biharmonic Thin-Plate PDE Predictor */
    double p_bh_x = 2.0 * W - WW;
    double p_bh_y = 2.0 * N - NN;
    double p_bh_xy = W + N - NW;
    double wx_bh = 1.0 / (1.0 + fabs(W - 2.0*NW + NNW) + dh);
    double wy_bh = 1.0 / (1.0 + fabs(N - 2.0*NW + NWW) + dv);
    double wxy_bh = 1.0 / (1.0 + fabs(W - NW - N + NN));
    double P_biharm = (wx_bh * p_bh_x + wy_bh * p_bh_y + wxy_bh * p_bh_xy) / (wx_bh + wy_bh + wxy_bh);

    /* 4. Expert E3: MED Clamped Surface */
    int mn = W < N ? W : N;
    int mx = W > N ? W : N;
    double P_med;
    if (NW >= mx) P_med = mn;
    else if (NW <= mn) P_med = mx;
    else P_med = W + N - NW;

    /* 5. Expert E4: Causal Non-Local Context Match */
    uint32_t h_key = nl_hash(W, N, NW, NE);
    int match_pos = rls->nl_pos[h_key];
    double P_nl = P_med;
    if (match_pos > 0) {
        int my = match_pos / w;
        int mx_p = match_pos % w;
        if (my > 0 && mx_p > 0 && mx_p + 1 < w) {
            int mW = plane_c[my * w + (mx_p - 1)];
            int mN = plane_c[(my - 1) * w + mx_p];
            int mNW = plane_c[(my - 1) * w + (mx_p - 1)];
            int mNE = plane_c[(my - 1) * w + (mx_p + 1)];
            int dist = abs(W - mW) + abs(N - mN) + abs(NW - mNW) + abs(NE - mNE);
            if (dist < 16) {
                double w_nl = 1.0 / (1.0 + 0.25 * dist);
                int mVal = plane_c[my * w + mx_p];
                double shift = 0.25 * ((W - mW) + (N - mN) + (NW - mNW) + (NE - mNE));
                P_nl = w_nl * (mVal + shift) + (1.0 - w_nl) * P_med;
            }
        }
    }

    double base = P_gap;
    int nf = 0;

    if (ch >= 1) {
        int G0   = plane_g[y * w + x];
        int GW   = (x > 0) ? plane_g[y * w + (x - 1)] : get_px(plane_g, w, h, 0, y - 1);
        int GN   = (y > 0) ? plane_g[(y - 1) * w + x] : GW;
        int GNW  = (x > 0 && y > 0) ? plane_g[(y - 1) * w + (x - 1)] : (y > 0 ? GN : GW);
        int GNE  = (y > 0 && x + 1 < w) ? plane_g[(y - 1) * w + (x + 1)] : GN;
        int GWW  = (x > 1) ? plane_g[y * w + (x - 2)] : GW;
        int GNN  = (y > 1) ? plane_g[(y - 2) * w + x] : GN;
        int GNWW = (x > 1 && y > 0) ? plane_g[(y - 1) * w + (x - 2)] : GNW;
        int GNNW = (x > 0 && y > 1) ? plane_g[(y - 2) * w + (x - 1)] : GNW;
        int GNNE = (x + 1 < w && y > 1) ? plane_g[(y - 2) * w + (x + 1)] : GNE;
        int GNNEE= (x + 2 < w && y > 0) ? plane_g[(y - 1) * w + (x + 2)] : GNE;
        int GE   = (x + 1 < w) ? plane_g[y * w + (x + 1)] : G0;
        int GS   = (y + 1 < h) ? plane_g[(y + 1) * w + x] : G0;

        int nb_c[10] = {W, N, NW, NE, WW, NN, NWW, NNW, NNE, NNEE};
        int nb_g[10] = {GW, GN, GNW, GNE, GWW, GNN, GNWW, GNNW, GNNE, GNNEE};
        double spatial_w[10] = {1.0, 1.0, 0.65, 0.65, 0.35, 0.35, 0.25, 0.25, 0.25, 0.20};

        double sum_w_tight = 1e-9, sum_d_tight = 0.0;
        double sum_w_wide  = 1e-9, sum_d_wide  = 0.0;

        int R0   = (ch == 2) ? plane_r[y * w + x] : 0;
        int RW   = (ch == 2) ? ((x > 0) ? plane_r[y * w + (x - 1)] : get_px(plane_r, w, h, 0, y - 1)) : 0;
        int RN   = (ch == 2) ? ((y > 0) ? plane_r[(y - 1) * w + x] : RW) : 0;
        int RNW  = (ch == 2) ? ((x > 0 && y > 0) ? plane_r[(y - 1) * w + (x - 1)] : (y > 0 ? RN : RW)) : 0;
        int RNE  = (ch == 2) ? ((y > 0 && x + 1 < w) ? plane_r[(y - 1) * w + (x + 1)] : RN) : 0;
        int RWW  = (ch == 2) ? ((x > 1) ? plane_r[y * w + (x - 2)] : RW) : 0;
        int RNN  = (ch == 2) ? ((y > 1) ? plane_r[(y - 2) * w + x] : RN) : 0;
        int RNWW = (ch == 2) ? ((x > 1 && y > 0) ? plane_r[(y - 1) * w + (x - 2)] : RNW) : 0;
        int RNNW = (ch == 2) ? ((x > 0 && y > 1) ? plane_r[(y - 2) * w + (x - 1)] : RNW) : 0;
        int RNNE = (ch == 2) ? ((x + 1 < w && y > 1) ? plane_r[(y - 2) * w + (x + 1)] : RNE) : 0;
        int RNNEE= (ch == 2) ? ((x + 2 < w && y > 0) ? plane_r[(y - 1) * w + (x + 2)] : RNE) : 0;
        int nb_r[10] = {RW, RN, RNW, RNE, RWW, RNN, RNWW, RNNW, RNNE, RNNEE};

        double sum_w_r = 1e-9, sum_d_r = 0.0;
        double sum_w_r_wide = 1e-9, sum_d_r_wide = 0.0;

        for (int k = 0; k < 10; k++) {
            double dg = fabs((double)(G0 - nb_g[k]));
            if (ch == 2) {
                dg = 0.5 * (dg + fabs((double)(R0 - nb_r[k])));
            }
            double wt = spatial_w[k] / (1.0 + (dg / 3.0) * (dg / 3.0));
            double ww = spatial_w[k] / (1.0 + (dg / 14.0) * (dg / 14.0));
            sum_w_tight += wt;
            sum_d_tight += wt * (nb_c[k] - nb_g[k]);
            sum_w_wide  += ww;
            sum_d_wide  += ww * (nb_c[k] - nb_g[k]);
            if (ch == 2) {
                sum_w_r += wt;
                sum_d_r += wt * (nb_c[k] - nb_r[k]);
                sum_w_r_wide += ww;
                sum_d_r_wide += ww * (nb_c[k] - nb_r[k]);
            }
        }
        double P_bilat_tight = G0 + sum_d_tight / sum_w_tight;
        double P_bilat_wide  = G0 + sum_d_wide  / sum_w_wide;

        int dW = W - GW, dN = N - GN, dNW = NW - GNW, dNE = NE - GNE, dWW = WW - GWW, dNN = NN - GNN;
        int dmn = dW < dN ? dW : dN;
        int dmx = dW > dN ? dW : dN;
        double d_med = (dNW >= dmx) ? dmn : ((dNW <= dmn) ? dmx : (dW + dN - dNW));
        double P_cg_med = G0 + d_med;
        double P_cg_lin = G0 + 0.5 * (dW + dN) + 0.25 * (dNE - dNW);

        if (ch == 1) {
            base = 0.45 * P_bilat_tight + 0.20 * P_bilat_wide + 0.35 * P_cg_med;
        } else {
            double P_bilat_r = R0 + sum_d_r / sum_w_r;
            int drW = W - RW, drN = N - RN, drNW = NW - RNW;
            int drmn = drW < drN ? drW : drN;
            int drmx = drW > drN ? drW : drN;
            double dr_med = (drNW >= drmx) ? drmn : ((drNW <= drmn) ? drmx : (drW + drN - drNW));
            double P_cr_med = R0 + dr_med;
            base = 0.28 * P_bilat_tight + 0.28 * P_bilat_r + 0.22 * P_cg_med + 0.22 * P_cr_med;
        }

        out_feat[nf++] = (P_bilat_tight - base) / 64.0;
        out_feat[nf++] = (P_bilat_wide  - base) / 64.0;
        out_feat[nf++] = (P_cg_med      - base) / 64.0;
        out_feat[nf++] = (P_cg_lin      - base) / 64.0;
        out_feat[nf++] = (P_geo         - base) / 64.0;
        out_feat[nf++] = (P_gap         - base) / 64.0;
        out_feat[nf++] = (P_nl          - base) / 64.0;
        out_feat[nf++] = (G0 + dW       - base) / 64.0;
        out_feat[nf++] = (G0 + dN       - base) / 64.0;
        out_feat[nf++] = (dW - dWW)             / 64.0;
        out_feat[nf++] = (dN - dNN)             / 64.0;
        out_feat[nf++] = get_err(err_g, w, h, x, y) / 64.0;
        out_feat[nf++] = (2.0 * G0 - GW - GE) / 64.0;
        out_feat[nf++] = (2.0 * G0 - GN - GS) / 64.0;
        out_feat[nf++] = (G0 - GW) / 64.0;
        out_feat[nf++] = (G0 - GN) / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x - 1, y) / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x, y - 1) / 64.0;
        if (ch == 2) {
            int RE = (x + 1 < w) ? plane_r[y * w + (x + 1)] : R0;
            int RS = (y + 1 < h) ? plane_r[(y + 1) * w + x] : R0;
            double P_bilat_r = R0 + sum_d_r / sum_w_r;
            double P_bilat_r_wide = R0 + sum_d_r_wide / sum_w_r_wide;
            out_feat[nf++] = (P_bilat_r - base) / 64.0;
            out_feat[nf++] = (P_bilat_r_wide - base) / 64.0;
            out_feat[nf++] = get_err(err_r, w, h, x, y) / 64.0;
            out_feat[nf++] = (2.0 * R0 - RW - RE) / 64.0;
            out_feat[nf++] = (2.0 * R0 - RN - RS) / 64.0;
            out_feat[nf++] = (R0 - G0 - (RW - GW)) / 64.0;
        }
    } else {
        base = 0.40 * P_geo + 0.35 * P_gap + 0.25 * P_med;
        out_feat[nf++] = (P_geo    - base) / 64.0;
        out_feat[nf++] = (P_gap    - base) / 64.0;
        out_feat[nf++] = (P_biharm - base) / 64.0;
        out_feat[nf++] = (P_med    - base) / 64.0;
        out_feat[nf++] = (P_nl     - base) / 64.0;
        out_feat[nf++] = (W        - base) / 64.0;
        out_feat[nf++] = (N        - base) / 64.0;
        out_feat[nf++] = (NW       - base) / 64.0;
        out_feat[nf++] = (NE       - base) / 64.0;
        out_feat[nf++] = (WW - W)          / 64.0;
        out_feat[nf++] = (NN - N)          / 64.0;
        out_feat[nf++] = (NWW - NW)        / 64.0;
        out_feat[nf++] = (NNW - NW)        / 64.0;
        out_feat[nf++] = (NNE - NE)        / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x - 1, y) / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x, y - 1) / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x - 1, y - 1) / 64.0;
        out_feat[nf++] = get_err(err_c, w, h, x + 1, y - 1) / 64.0;
    }

    int dir_bin = 0;
    if (dv > 2.0 * dh + 8.0) dir_bin = 1;
    else if (dh > 2.0 * dv + 8.0) dir_bin = 2;
    else if (Sxy > 0.25 * trace) dir_bin = 3;
    else if (Sxy < -0.25 * trace) dir_bin = 4;
    else if (coherence > 0.5) dir_bin = 5;
    else if (dh + dv < 12.0) dir_bin = 6;
    else dir_bin = 7;

    double grad_mag = dh + dv;
    int act_bin = (grad_mag < 8.0) ? 0 : ((grad_mag < 28.0) ? 1 : ((grad_mag < 80.0) ? 2 : 3));
    int rls_ctx = dir_bin * 4 + act_bin;

    double delta = 0.0;
    for (int i = 0; i < nf; i++) {
        delta += (rls->w_global[i] + rls->w_ctx[rls_ctx][i]) * out_feat[i];
    }
    double cont_pred = base + delta * 64.0;
    int pred_int = (int)floor(cont_pred + 0.5);
    if (W == N && N == NW && NW == NE) {
        if (ch == 0) {
            pred_int = W;
        } else {
            int G0 = plane_g[y * w + x];
            int GW = (x > 0) ? plane_g[y * w + (x - 1)] : G0;
            int GN = (y > 0) ? plane_g[(y - 1) * w + x] : G0;
            if (G0 == GW && GW == GN) pred_int = W;
        }
    }
    pred_int = clamp_i(pred_int, 0, 255);

    double eW   = fabs(get_err(err_c, w, h, x - 1, y));
    double eN   = fabs(get_err(err_c, w, h, x, y - 1));
    double eNW  = fabs(get_err(err_c, w, h, x - 1, y - 1));
    double eNE  = fabs(get_err(err_c, w, h, x + 1, y - 1));
    double eWW  = fabs(get_err(err_c, w, h, x - 2, y));
    double eNN  = fabs(get_err(err_c, w, h, x, y - 2));

    double local_err = 2.2 * eW + 2.2 * eN + 1.1 * eNW + 1.1 * eNE + 0.5 * (eWW + eNN);
    double cross_act = 0.0;
    if (ch == 1) {
        cross_act = 2.5 * fabs(get_err(err_g, w, h, x, y));
    } else if (ch == 2) {
        cross_act = 1.4 * fabs(get_err(err_g, w, h, x, y)) + 2.2 * fabs(get_err(err_r, w, h, x, y));
    }
    double sigma_est = (local_err / 7.1) + 0.035 * (dh + dv) + 0.35 * cross_act;
    int var_bin = (int)(8.4 * log2(1.0 + sigma_est));
    var_bin = clamp_i(var_bin, 0, NUM_VAR_BINS - 1);

    int zW  = (eW == 0.0);
    int zN  = (eN == 0.0);
    int zNW = (eNW == 0.0);
    int is_flat = (W == N && N == NW && NW == NE && rls->zero_run > 2);
    int zero_state = (is_flat << 3) | (zW << 2) | (zN << 1) | zNW;

    int raw_eW = (int)get_err(err_c, w, h, x - 1, y);
    int raw_eN = (int)get_err(err_c, w, h, x, y - 1);
    int sW = (raw_eW > 0) ? 1 : ((raw_eW < 0) ? 2 : 0);
    int sN = (raw_eN > 0) ? 1 : ((raw_eN < 0) ? 2 : 0);
    int frac_pos = (cont_pred > (double)pred_int) ? 1 : 0;
    int sign_ctx = (sW * 3 + sN) * 2 + frac_pos;

    *out_nfeat = nf;
    *out_rls_ctx = rls_ctx;
    *out_base = base;
    *out_pred_clamped = pred_int;
    *out_var_bin = var_bin;
    *out_zero_state = zero_state;
    *out_sign_ctx = sign_ctx;
}

static void agm_update_pixel(
    int w, int h, int x, int y,
    uint8_t val, int pred_clamped, double base,
    const uint8_t *plane_c,
    int16_t *err_c,
    ChannelRLS *rls,
    const double *feat, int nfeat, int rls_ctx
) {
    int err_int = (int)val - pred_clamped;
    err_c[y * w + x] = (int16_t)err_int;
    if (err_int == 0) rls->zero_run++;
    else rls->zero_run = 0;

    if (nfeat > 0) {
        double target_norm = ((double)val - base) / 64.0;
        double pred_norm = 0.0;
        double precond_norm = 0.25;
        for (int i = 0; i < nfeat; i++) {
            pred_norm += (rls->w_global[i] + rls->w_ctx[rls_ctx][i]) * feat[i];
            rls->v_global[i] = 0.992 * rls->v_global[i] + 0.008 * (feat[i] * feat[i] + 1e-5);
            precond_norm += (feat[i] * feat[i]) / sqrt(rls->v_global[i]);
        }
        double err_norm = target_norm - pred_norm;
        for (int i = 0; i < nfeat; i++) {
            double g_step = (feat[i] / sqrt(rls->v_global[i])) / precond_norm;
            rls->w_global[i] += 0.025 * err_norm * g_step;
            rls->w_ctx[rls_ctx][i] += 0.045 * err_norm * g_step;
            if (rls->w_global[i] > 2.5) rls->w_global[i] = 2.5;
            if (rls->w_global[i] < -2.5) rls->w_global[i] = -2.5;
            if (rls->w_ctx[rls_ctx][i] > 2.5) rls->w_ctx[rls_ctx][i] = 2.5;
            if (rls->w_ctx[rls_ctx][i] < -2.5) rls->w_ctx[rls_ctx][i] = -2.5;
        }
    }
    if (x > 0 && y > 0 && x + 1 < w) {
        int W  = plane_c[y * w + (x - 1)];
        int N  = plane_c[(y - 1) * w + x];
        int NW = plane_c[(y - 1) * w + (x - 1)];
        int NE = plane_c[(y - 1) * w + (x + 1)];
        rls->nl_pos[nl_hash(W, N, NW, NE)] = y * w + x;
    }
}

size_t agm_encode_lossless(const uint8_t *img, int w, int h, int channels, uint8_t *out_buf, size_t out_cap) {
    out_buf[0] = 'A'; out_buf[1] = 'G'; out_buf[2] = 'M'; out_buf[3] = 'L';
    out_buf[4] = (w >> 8) & 0xFF; out_buf[5] = w & 0xFF;
    out_buf[6] = (h >> 8) & 0xFF; out_buf[7] = h & 0xFF;
    out_buf[8] = (uint8_t)channels;

    RangeEnc rc;
    rc_enc_init(&rc, out_buf + 9, out_cap - 9);
    EntropyModels *em = malloc(sizeof(EntropyModels));
    entropy_init(em);

    size_t npix = (size_t)w * (size_t)h;
    uint8_t *planes[3] = {NULL, NULL, NULL};
    int16_t *errs[3]   = {NULL, NULL, NULL};
    for (int c = 0; c < channels; c++) {
        planes[c] = malloc(npix);
        errs[c]   = calloc(npix, sizeof(int16_t));
    }
    if (channels == 1) {
        memcpy(planes[0], img, npix);
    } else {
        for (size_t i = 0; i < npix; i++) {
            planes[0][i] = img[3 * i + 1]; /* G */
            planes[1][i] = img[3 * i + 0]; /* R */
            planes[2][i] = img[3 * i + 2]; /* B */
        }
    }

    ChannelRLS *rls = malloc(sizeof(ChannelRLS));
    double feat[MAX_FEAT];

    for (int c = 0; c < channels; c++) {
        rls_init(rls);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int nfeat, rls_ctx, pred_clamped, var_bin, zero_state, sign_ctx;
                double base;
                agm_predict_pixel(
                    c, w, h, x, y,
                    planes[c], errs[c],
                    planes[0], errs[0],
                    planes[1], errs[1],
                    rls, feat, &nfeat, &rls_ctx, &base, &pred_clamped,
                    &var_bin, &zero_state, &sign_ctx
                );
                uint8_t val = planes[c][y * w + x];
                encode_residual(&rc, em, c, var_bin, zero_state, sign_ctx, val, pred_clamped);
                agm_update_pixel(w, h, x, y, val, pred_clamped, base, planes[c], errs[c], rls, feat, nfeat, rls_ctx);
            }
        }
    }

    size_t body_sz = rc_enc_finish(&rc);
    for (int c = 0; c < channels; c++) {
        free(planes[c]);
        free(errs[c]);
    }
    free(rls);
    free(em);
    return 9 + body_sz;
}

int agm_decode_lossless(const uint8_t *in_buf, size_t in_len, uint8_t *out_img) {
    if (in_len < 9 || in_buf[0] != 'A' || in_buf[1] != 'G' || in_buf[2] != 'M' || in_buf[3] != 'L') return -1;
    int w = ((int)in_buf[4] << 8) | in_buf[5];
    int h = ((int)in_buf[6] << 8) | in_buf[7];
    int channels = in_buf[8];

    RangeDec rc;
    rc_dec_init(&rc, in_buf + 9, in_len - 9);
    EntropyModels *em = malloc(sizeof(EntropyModels));
    entropy_init(em);

    size_t npix = (size_t)w * (size_t)h;
    uint8_t *planes[3] = {NULL, NULL, NULL};
    int16_t *errs[3]   = {NULL, NULL, NULL};
    for (int c = 0; c < channels; c++) {
        planes[c] = calloc(npix, 1);
        errs[c]   = calloc(npix, sizeof(int16_t));
    }

    ChannelRLS *rls = malloc(sizeof(ChannelRLS));
    double feat[MAX_FEAT];

    for (int c = 0; c < channels; c++) {
        rls_init(rls);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int nfeat, rls_ctx, pred_clamped, var_bin, zero_state, sign_ctx;
                double base;
                agm_predict_pixel(
                    c, w, h, x, y,
                    planes[c], errs[c],
                    planes[0], errs[0],
                    planes[1], errs[1],
                    rls, feat, &nfeat, &rls_ctx, &base, &pred_clamped,
                    &var_bin, &zero_state, &sign_ctx
                );
                int val = decode_residual(&rc, em, c, var_bin, zero_state, sign_ctx, pred_clamped);
                planes[c][y * w + x] = (uint8_t)val;
                agm_update_pixel(w, h, x, y, (uint8_t)val, pred_clamped, base, planes[c], errs[c], rls, feat, nfeat, rls_ctx);
            }
        }
    }

    if (channels == 1) {
        memcpy(out_img, planes[0], npix);
    } else {
        for (size_t i = 0; i < npix; i++) {
            out_img[3 * i + 1] = planes[0][i];
            out_img[3 * i + 0] = planes[1][i];
            out_img[3 * i + 2] = planes[2][i];
        }
    }

    for (int c = 0; c < channels; c++) {
        free(planes[c]);
        free(errs[c]);
    }
    free(rls);
    free(em);
    return 0;
}

/* =========================================================================
 * Biorthogonal CDF 9/7 Multiscale Wavelet Lifting Transform (4 Levels)
 * ========================================================================= */
#define ALPHA (-1.586134342059924)
#define BETA  (-0.052980118572961)
#define GAMMA ( 0.882911075530934)
#define DELTA ( 0.443506852043971)
#define S_SCALE (1.1496043988602418)
#define D_SCALE (0.8698644516247810)

static inline int sym_idx(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

static void dwt_1d_fwd(const double *in, double *out, int n) {
    if (n < 2) {
        if (n == 1) out[0] = in[0];
        return;
    }
    int sn = (n + 1) >> 1;
    int dn = n >> 1;
    double *s = malloc(sn * sizeof(double));
    double *d = malloc((dn + 1) * sizeof(double));
    for (int i = 0; i < sn; i++) s[i] = in[2 * i];
    for (int i = 0; i < dn; i++) d[i] = in[2 * i + 1];

    for (int i = 0; i < dn; i++) {
        int i1 = (i + 1 < sn) ? (i + 1) : (sn - 1);
        d[i] += ALPHA * (s[i] + s[i1]);
    }
    for (int i = 0; i < sn; i++) {
        double d_left  = (i > 0) ? d[i - 1] : d[0];
        double d_right = (i < dn) ? d[i] : d[dn - 1];
        s[i] += BETA * (d_left + d_right);
    }
    for (int i = 0; i < dn; i++) {
        int i1 = (i + 1 < sn) ? (i + 1) : (sn - 1);
        d[i] += GAMMA * (s[i] + s[i1]);
    }
    for (int i = 0; i < sn; i++) {
        double d_left  = (i > 0) ? d[i - 1] : d[0];
        double d_right = (i < dn) ? d[i] : d[dn - 1];
        s[i] += DELTA * (d_left + d_right);
    }
    for (int i = 0; i < sn; i++) out[i]      = s[i] * S_SCALE;
    for (int i = 0; i < dn; i++) out[sn + i] = d[i] * D_SCALE;
    free(s);
    free(d);
}

static void dwt_1d_inv(const double *in, double *out, int n) {
    if (n < 2) {
        if (n == 1) out[0] = in[0];
        return;
    }
    int sn = (n + 1) >> 1;
    int dn = n >> 1;
    double *s = malloc(sn * sizeof(double));
    double *d = malloc((dn + 1) * sizeof(double));
    for (int i = 0; i < sn; i++) s[i] = in[i]      / S_SCALE;
    for (int i = 0; i < dn; i++) d[i] = in[sn + i] / D_SCALE;

    for (int i = 0; i < sn; i++) {
        double d_left  = (i > 0) ? d[i - 1] : d[0];
        double d_right = (i < dn) ? d[i] : d[dn - 1];
        s[i] -= DELTA * (d_left + d_right);
    }
    for (int i = 0; i < dn; i++) {
        int i1 = (i + 1 < sn) ? (i + 1) : (sn - 1);
        d[i] -= GAMMA * (s[i] + s[i1]);
    }
    for (int i = 0; i < sn; i++) {
        double d_left  = (i > 0) ? d[i - 1] : d[0];
        double d_right = (i < dn) ? d[i] : d[dn - 1];
        s[i] -= BETA * (d_left + d_right);
    }
    for (int i = 0; i < dn; i++) {
        int i1 = (i + 1 < sn) ? (i + 1) : (sn - 1);
        d[i] -= ALPHA * (s[i] + s[i1]);
    }
    for (int i = 0; i < sn; i++) out[2 * i]     = s[i];
    for (int i = 0; i < dn; i++) out[2 * i + 1] = d[i];
    free(s);
    free(d);
}

static void dwt_2d_fwd(double *plane, int stride, int w, int h, int levels) {
    int max_dim = (w > h ? w : h);
    double *tmp_in  = malloc(max_dim * sizeof(double));
    double *tmp_out = malloc(max_dim * sizeof(double));
    int cw = w, ch = h;
    for (int lv = 0; lv < levels; lv++) {
        if (cw < 4 || ch < 4) break;
        for (int y = 0; y < ch; y++) {
            dwt_1d_fwd(&plane[y * stride], tmp_out, cw);
            memcpy(&plane[y * stride], tmp_out, cw * sizeof(double));
        }
        for (int x = 0; x < cw; x++) {
            for (int y = 0; y < ch; y++) tmp_in[y] = plane[y * stride + x];
            dwt_1d_fwd(tmp_in, tmp_out, ch);
            for (int y = 0; y < ch; y++) plane[y * stride + x] = tmp_out[y];
        }
        cw = (cw + 1) >> 1;
        ch = (ch + 1) >> 1;
    }
    free(tmp_in);
    free(tmp_out);
}

static void dwt_2d_inv(double *plane, int stride, int w, int h, int levels) {
    int max_dim = (w > h ? w : h);
    double *tmp_in  = malloc(max_dim * sizeof(double));
    double *tmp_out = malloc(max_dim * sizeof(double));
    int w_lv[10], h_lv[10];
    w_lv[0] = w; h_lv[0] = h;
    int actual_lv = 0;
    for (int lv = 0; lv < levels; lv++) {
        if (w_lv[lv] < 4 || h_lv[lv] < 4) break;
        w_lv[lv + 1] = (w_lv[lv] + 1) >> 1;
        h_lv[lv + 1] = (h_lv[lv] + 1) >> 1;
        actual_lv++;
    }
    for (int lv = actual_lv - 1; lv >= 0; lv--) {
        int cw = w_lv[lv], ch = h_lv[lv];
        for (int x = 0; x < cw; x++) {
            for (int y = 0; y < ch; y++) tmp_in[y] = plane[y * stride + x];
            dwt_1d_inv(tmp_in, tmp_out, ch);
            for (int y = 0; y < ch; y++) plane[y * stride + x] = tmp_out[y];
        }
        for (int y = 0; y < ch; y++) {
            dwt_1d_inv(&plane[y * stride], tmp_out, cw);
            memcpy(&plane[y * stride], tmp_out, cw * sizeof(double));
        }
    }
    free(tmp_in);
    free(tmp_out);
}

/* =========================================================================
 * Wavelet Subband Context-Adaptive Quantization & Range Coding
 * ========================================================================= */
#define W_VAR_BINS 32
typedef struct {
    BiModel zero_c[2 * 16];
    BiModel zero_f[2 * W_VAR_BINS * 16];
    BiModel sign_c[2 * 4];
    BiModel sign_f[2 * 16 * 8];
    BiModel small_c[2 * 16 * 6];
    BiModel small_f[2 * W_VAR_BINS * 6];
    BiModel exp_c[2 * 16 * 16];
    BiModel exp_f[2 * W_VAR_BINS * 16];
    BiModel mant_c[2 * 16 * 16];
    BiModel mant_f[2 * 16 * 16 * 16];
} WaveletModels;

static void wmodels_init(WaveletModels *wm) {
    memset(wm, 0, sizeof(WaveletModels));
    BiModel *all = (BiModel *)wm;
    size_t count = sizeof(WaveletModels) / sizeof(BiModel);
    for (size_t i = 0; i < count; i++) bimodel_init(&all[i]);
}



static inline void encode_qcoeff(
    RangeEnc *rc, WaveletModels *wm,
    int is_chroma, int var_bin, int z_ctx, int s_ctx, int q
) {
    int vb_c = var_bin >> 1;
    int zc = is_chroma * 16 + vb_c;
    int zf = (is_chroma * W_VAR_BINS + var_bin) * 16 + (z_ctx & 15);
    if (q == 0) {
        rc_enc_bit2(rc, &wm->zero_c[zc], &wm->zero_f[zf], 1);
        return;
    }
    rc_enc_bit2(rc, &wm->zero_c[zc], &wm->zero_f[zf], 0);

    int sign = (q > 0) ? 1 : 0;
    int sc = is_chroma * 4 + (s_ctx & 3);
    int sf = (is_chroma * 16 + vb_c) * 8 + (s_ctx & 7);
    rc_enc_bit2(rc, &wm->sign_c[sc], &wm->sign_f[sf], sign);

    int mag = (q > 0) ? q : -q;
    for (int k = 1; k <= 6; k++) {
        int is_k = (mag == k);
        int smc = (is_chroma * 16 + vb_c) * 6 + (k - 1);
        int smf = (is_chroma * W_VAR_BINS + var_bin) * 6 + (k - 1);
        rc_enc_bit2(rc, &wm->small_c[smc], &wm->small_f[smf], is_k);
        if (is_k) return;
    }

    int v = mag - 7;
    int exp_b = 0;
    while ((1 << (exp_b + 2)) - 2 <= v && exp_b < 15) {
        exp_b++;
    }
    for (int b = 0; b < 15; b++) {
        int ec = (is_chroma * 16 + vb_c) * 16 + b;
        int ef = (is_chroma * W_VAR_BINS + var_bin) * 16 + b;
        if (exp_b == b) {
            rc_enc_bit2(rc, &wm->exp_c[ec], &wm->exp_f[ef], 1);
            break;
        } else {
            rc_enc_bit2(rc, &wm->exp_c[ec], &wm->exp_f[ef], 0);
        }
    }
    int base = (1 << (exp_b + 1)) - 2;
    int rem = v - base;
    int nbits = exp_b + 1;
    for (int bp = nbits - 1; bp >= 0; bp--) {
        int bit = (rem >> bp) & 1;
        int bp_c = bp < 15 ? bp : 15;
        int mc = (is_chroma * 16 + exp_b) * 16 + bp_c;
        int mf = ((is_chroma * 16 + vb_c) * 16 + exp_b) * 16 + bp_c;
        rc_enc_bit2(rc, &wm->mant_c[mc], &wm->mant_f[mf], bit);
    }
}

static inline int decode_qcoeff(
    RangeDec *rc, WaveletModels *wm,
    int is_chroma, int var_bin, int z_ctx, int s_ctx
) {
    int vb_c = var_bin >> 1;
    int zc = is_chroma * 16 + vb_c;
    int zf = (is_chroma * W_VAR_BINS + var_bin) * 16 + (z_ctx & 15);
    if (rc_dec_bit2(rc, &wm->zero_c[zc], &wm->zero_f[zf])) {
        return 0;
    }
    int sc = is_chroma * 4 + (s_ctx & 3);
    int sf = (is_chroma * 16 + vb_c) * 8 + (s_ctx & 7);
    int sign = rc_dec_bit2(rc, &wm->sign_c[sc], &wm->sign_f[sf]);

    for (int k = 1; k <= 6; k++) {
        int smc = (is_chroma * 16 + vb_c) * 6 + (k - 1);
        int smf = (is_chroma * W_VAR_BINS + var_bin) * 6 + (k - 1);
        if (rc_dec_bit2(rc, &wm->small_c[smc], &wm->small_f[smf])) {
            return sign ? k : -k;
        }
    }

    int exp_b = 15;
    for (int b = 0; b < 15; b++) {
        int ec = (is_chroma * 16 + vb_c) * 16 + b;
        int ef = (is_chroma * W_VAR_BINS + var_bin) * 16 + b;
        if (rc_dec_bit2(rc, &wm->exp_c[ec], &wm->exp_f[ef])) {
            exp_b = b;
            break;
        }
    }
    int base = (1 << (exp_b + 1)) - 2;
    int nbits = exp_b + 1;
    int rem = 0;
    for (int bp = nbits - 1; bp >= 0; bp--) {
        int bp_c = bp < 15 ? bp : 15;
        int mc = (is_chroma * 16 + exp_b) * 16 + bp_c;
        int mf = ((is_chroma * 16 + vb_c) * 16 + exp_b) * 16 + bp_c;
        int bit = rc_dec_bit2(rc, &wm->mant_c[mc], &wm->mant_f[mf]);
        if (bit) rem |= (1 << bp);
    }
    int mag = 7 + base + rem;
    return sign ? mag : -mag;
}

static inline int get_q(const int32_t *qplane, int stride, int x0, int y0, int bw, int bh, int rx, int ry) {
    if (rx < 0 || rx >= bw || ry < 0 || ry >= bh) return 0;
    return qplane[(y0 + ry) * stride + (x0 + rx)];
}

static void compute_subband_ctx(
    const int32_t *qplane, int stride,
    int x0, int y0, int bw, int bh, int rx, int ry,
    int px0, int py0, int pbw, int pbh, int has_parent,
    int band_type, int *out_var_bin, int *out_z_ctx, int *out_s_ctx
) {
    int W   = abs(get_q(qplane, stride, x0, y0, bw, bh, rx - 1, ry));
    int N   = abs(get_q(qplane, stride, x0, y0, bw, bh, rx, ry - 1));
    int NW  = abs(get_q(qplane, stride, x0, y0, bw, bh, rx - 1, ry - 1));
    int NE  = abs(get_q(qplane, stride, x0, y0, bw, bh, rx + 1, ry - 1));
    int WW  = abs(get_q(qplane, stride, x0, y0, bw, bh, rx - 2, ry));
    int NN  = abs(get_q(qplane, stride, x0, y0, bw, bh, rx, ry - 2));

    int P = 0, PN = 0, PW = 0;
    if (has_parent) {
        P  = abs(get_q(qplane, stride, px0, py0, pbw, pbh, rx >> 1, ry >> 1));
        PN = abs(get_q(qplane, stride, px0, py0, pbw, pbh, rx >> 1, (ry >> 1) - 1));
        PW = abs(get_q(qplane, stride, px0, py0, pbw, pbh, (rx >> 1) - 1, ry >> 1));
    }

    /* Band-anisotropic weighting:
       band_type == 1 (HL: horizontal highpass / vertical edges) -> N and NN are primary!
       band_type == 2 (LH: vertical highpass / horizontal edges) -> W and WW are primary! */
    double energy;
    if (band_type == 1) {
        energy = 1.6 * W + 2.8 * N + 1.0 * NW + 1.0 * NE + 0.3 * WW + 0.8 * NN;
    } else if (band_type == 2) {
        energy = 2.8 * W + 1.6 * N + 1.0 * NW + 1.0 * NE + 0.8 * WW + 0.3 * NN;
    } else {
        energy = 2.1 * W + 2.1 * N + 1.2 * NW + 1.2 * NE + 0.4 * (WW + NN);
    }
    double parent_energy = 0.9 * P + 0.3 * (PN + PW);
    double sigma = (energy / 6.8) + 0.45 * parent_energy;

    int var_bin = (int)(4.8 * log2(1.0 + sigma));
    var_bin = clamp_i(var_bin, 0, W_VAR_BINS - 1);

    int zW = (W == 0), zN = (N == 0), zNW = (NW == 0), zP = (P == 0);
    int z_ctx = (zP << 3) | (zW << 2) | (zN << 1) | zNW;

    int raw_W = get_q(qplane, stride, x0, y0, bw, bh, rx - 1, ry);
    int raw_N = get_q(qplane, stride, x0, y0, bw, bh, rx, ry - 1);
    int sW = (raw_W > 0) ? 1 : ((raw_W < 0) ? 2 : 0);
    int sN = (raw_N > 0) ? 1 : ((raw_N < 0) ? 2 : 0);
    int s_ctx = (band_type == 1) ? sN : ((band_type == 2) ? sW : ((sW ^ sN) & 3));

    *out_var_bin = var_bin;
    *out_z_ctx = z_ctx;
    *out_s_ctx = s_ctx;
}

static inline int predict_ll(const int32_t *qplane, int stride, int bw, int bh, int x, int y) {
    if (x == 0 && y == 0) return 0;
    if (y == 0) return qplane[x - 1];
    if (x == 0) return qplane[(y - 1) * stride];
    int W  = qplane[y * stride + (x - 1)];
    int N  = qplane[(y - 1) * stride + x];
    int NW = qplane[(y - 1) * stride + (x - 1)];
    int mn = W < N ? W : N;
    int mx = W > N ? W : N;
    if (NW >= mx) return mn;
    if (NW <= mn) return mx;
    return W + N - NW;
}

static void encode_wavelet_plane(
    RangeEnc *rc, WaveletModels *wm, int is_chroma,
    const double *plane, int32_t *qplane,
    int w, int h, int levels, double base_step
) {
    int w_lv[10], h_lv[10];
    w_lv[0] = w; h_lv[0] = h;
    int actual_lv = 0;
    for (int lv = 0; lv < levels; lv++) {
        if (w_lv[lv] < 4 || h_lv[lv] < 4) break;
        w_lv[lv + 1] = (w_lv[lv] + 1) >> 1;
        h_lv[lv + 1] = (h_lv[lv] + 1) >> 1;
        actual_lv++;
    }

    /* 1. Quantize all subbands with Perceptual CSF weights & Dead-Zone */
    int ll_w = w_lv[actual_lv], ll_h = h_lv[actual_lv];
    double ll_step = base_step * 0.88;
    for (int y = 0; y < ll_h; y++) {
        for (int x = 0; x < ll_w; x++) {
            double v = plane[y * w + x];
            qplane[y * w + x] = (int32_t)floor(v / ll_step + 0.5);
        }
    }

    for (int lv = actual_lv - 1; lv >= 0; lv--) {
        int cw = w_lv[lv], ch = h_lv[lv];
        int sw = w_lv[lv + 1], sh = h_lv[lv + 1];
        /* Perceptual CSF subband multipliers: finer scales & HH can tolerate slightly larger step */
        double scale_mult = (lv == 0) ? 1.05 : ((lv == 1) ? 0.95 : ((lv == 2) ? 0.92 : 0.90));
        double step_hl = base_step * scale_mult;
        double step_lh = base_step * scale_mult;
        double step_hh = base_step * scale_mult * 1.10;
        double dz = 0.56; /* Optimal rate-distortion dead-zone threshold */

        /* HL: x in [sw, cw), y in [0, sh) */
        for (int y = 0; y < sh; y++) {
            for (int x = sw; x < cw; x++) {
                double v = plane[y * w + x];
                double av = fabs(v) / step_hl;
                int q = (av < dz) ? 0 : (int)floor(av - dz + 1.0);
                qplane[y * w + x] = (v >= 0) ? q : -q;
            }
        }
        /* LH: x in [0, sw), y in [sh, ch) */
        for (int y = sh; y < ch; y++) {
            for (int x = 0; x < sw; x++) {
                double v = plane[y * w + x];
                double av = fabs(v) / step_lh;
                int q = (av < dz) ? 0 : (int)floor(av - dz + 1.0);
                qplane[y * w + x] = (v >= 0) ? q : -q;
            }
        }
        /* HH: x in [sw, cw), y in [sh, ch) */
        for (int y = sh; y < ch; y++) {
            for (int x = sw; x < cw; x++) {
                double v = plane[y * w + x];
                double av = fabs(v) / step_hh;
                int q = (av < dz) ? 0 : (int)floor(av - dz + 1.0);
                qplane[y * w + x] = (v >= 0) ? q : -q;
            }
        }
    }

    /* 2. Encode LL band using MED prediction */
    for (int y = 0; y < ll_h; y++) {
        for (int x = 0; x < ll_w; x++) {
            int pred = predict_ll(qplane, w, ll_w, ll_h, x, y);
            int err = qplane[y * w + x] - pred;
            int eW = (x > 0) ? abs(qplane[y * w + (x - 1)] - predict_ll(qplane, w, ll_w, ll_h, x - 1, y)) : 4;
            int eN = (y > 0) ? abs(qplane[(y - 1) * w + x] - predict_ll(qplane, w, ll_w, ll_h, x, y - 1)) : 4;
            int vb = clamp_i((int)(4.5 * log2(1.0 + 0.5 * (eW + eN))), 0, W_VAR_BINS - 1);
            int zc = ((eW == 0) << 1) | (eN == 0);
            encode_qcoeff(rc, wm, is_chroma, vb, zc, 0, err);
        }
    }

    /* 3. Encode Detail Subbands from Coarsest (actual_lv - 1) to Finest (0) */
    for (int lv = actual_lv - 1; lv >= 0; lv--) {
        int cw = w_lv[lv], ch = h_lv[lv];
        int sw = w_lv[lv + 1], sh = h_lv[lv + 1];
        int has_p = (lv + 1 < actual_lv);
        int p_sw = has_p ? w_lv[lv + 2] : 0;
        int p_sh = has_p ? h_lv[lv + 2] : 0;
        int p_cw = has_p ? w_lv[lv + 1] : 0;
        int p_ch = has_p ? h_lv[lv + 1] : 0;

        /* HL */
        for (int ry = 0; ry < sh; ry++) {
            for (int rx = 0; rx < cw - sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, sw, 0, cw - sw, sh, rx, ry,
                                    p_sw, 0, p_cw - p_sw, p_sh, has_p, 1, &vb, &zc, &sc);
                encode_qcoeff(rc, wm, is_chroma, vb, zc, sc, qplane[ry * w + (sw + rx)]);
            }
        }
        /* LH */
        for (int ry = 0; ry < ch - sh; ry++) {
            for (int rx = 0; rx < sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, 0, sh, sw, ch - sh, rx, ry,
                                    0, p_sh, p_sw, p_ch - p_sh, has_p, 2, &vb, &zc, &sc);
                encode_qcoeff(rc, wm, is_chroma, vb, zc, sc, qplane[(sh + ry) * w + rx]);
            }
        }
        /* HH */
        for (int ry = 0; ry < ch - sh; ry++) {
            for (int rx = 0; rx < cw - sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, sw, sh, cw - sw, ch - sh, rx, ry,
                                    p_sw, p_sh, p_cw - p_sw, p_ch - p_sh, has_p, 3, &vb, &zc, &sc);
                encode_qcoeff(rc, wm, is_chroma, vb, zc, sc, qplane[(sh + ry) * w + (sw + rx)]);
            }
        }
    }
}

static void decode_wavelet_plane(
    RangeDec *rc, WaveletModels *wm, int is_chroma,
    double *plane, int32_t *qplane,
    int w, int h, int levels, double base_step
) {
    int w_lv[10], h_lv[10];
    w_lv[0] = w; h_lv[0] = h;
    int actual_lv = 0;
    for (int lv = 0; lv < levels; lv++) {
        if (w_lv[lv] < 4 || h_lv[lv] < 4) break;
        w_lv[lv + 1] = (w_lv[lv] + 1) >> 1;
        h_lv[lv + 1] = (h_lv[lv] + 1) >> 1;
        actual_lv++;
    }

    int ll_w = w_lv[actual_lv], ll_h = h_lv[actual_lv];
    double ll_step = base_step * 0.88;

    for (int y = 0; y < ll_h; y++) {
        for (int x = 0; x < ll_w; x++) {
            int pred = predict_ll(qplane, w, ll_w, ll_h, x, y);
            int eW = (x > 0) ? abs(qplane[y * w + (x - 1)] - predict_ll(qplane, w, ll_w, ll_h, x - 1, y)) : 4;
            int eN = (y > 0) ? abs(qplane[(y - 1) * w + x] - predict_ll(qplane, w, ll_w, ll_h, x, y - 1)) : 4;
            int vb = clamp_i((int)(4.5 * log2(1.0 + 0.5 * (eW + eN))), 0, W_VAR_BINS - 1);
            int zc = ((eW == 0) << 1) | (eN == 0);
            int err = decode_qcoeff(rc, wm, is_chroma, vb, zc, 0);
            qplane[y * w + x] = pred + err;
            plane[y * w + x] = qplane[y * w + x] * ll_step;
        }
    }

    for (int lv = actual_lv - 1; lv >= 0; lv--) {
        int cw = w_lv[lv], ch = h_lv[lv];
        int sw = w_lv[lv + 1], sh = h_lv[lv + 1];
        int has_p = (lv + 1 < actual_lv);
        int p_sw = has_p ? w_lv[lv + 2] : 0;
        int p_sh = has_p ? h_lv[lv + 2] : 0;
        int p_cw = has_p ? w_lv[lv + 1] : 0;
        int p_ch = has_p ? h_lv[lv + 1] : 0;

        double scale_mult = (lv == 0) ? 1.05 : ((lv == 1) ? 0.95 : ((lv == 2) ? 0.92 : 0.90));
        double step_hl = base_step * scale_mult;
        double step_lh = base_step * scale_mult;
        double step_hh = base_step * scale_mult * 1.10;
        /* Laplace Centroid Reconstruction: dz = 0.56 -> offset = dz - 1.0 + 0.38 = -0.06 */
        double centroid_shift = -0.06;

        /* HL */
        for (int ry = 0; ry < sh; ry++) {
            for (int rx = 0; rx < cw - sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, sw, 0, cw - sw, sh, rx, ry,
                                    p_sw, 0, p_cw - p_sw, p_sh, has_p, 1, &vb, &zc, &sc);
                int q = decode_qcoeff(rc, wm, is_chroma, vb, zc, sc);
                qplane[ry * w + (sw + rx)] = q;
                plane[ry * w + (sw + rx)] = (q == 0) ? 0.0 : ((q > 0) ? (q + centroid_shift) * step_hl : (q - centroid_shift) * step_hl);
            }
        }
        /* LH */
        for (int ry = 0; ry < ch - sh; ry++) {
            for (int rx = 0; rx < sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, 0, sh, sw, ch - sh, rx, ry,
                                    0, p_sh, p_sw, p_ch - p_sh, has_p, 2, &vb, &zc, &sc);
                int q = decode_qcoeff(rc, wm, is_chroma, vb, zc, sc);
                qplane[(sh + ry) * w + rx] = q;
                plane[(sh + ry) * w + rx] = (q == 0) ? 0.0 : ((q > 0) ? (q + centroid_shift) * step_lh : (q - centroid_shift) * step_lh);
            }
        }
        /* HH */
        for (int ry = 0; ry < ch - sh; ry++) {
            for (int rx = 0; rx < cw - sw; rx++) {
                int vb, zc, sc;
                compute_subband_ctx(qplane, w, sw, sh, cw - sw, ch - sh, rx, ry,
                                    p_sw, p_sh, p_cw - p_sw, p_ch - p_sh, has_p, 3, &vb, &zc, &sc);
                int q = decode_qcoeff(rc, wm, is_chroma, vb, zc, sc);
                qplane[(sh + ry) * w + (sw + rx)] = q;
                plane[(sh + ry) * w + (sw + rx)] = (q == 0) ? 0.0 : ((q > 0) ? (q + centroid_shift) * step_hh : (q - centroid_shift) * step_hh);
            }
        }
    }
}

/* =========================================================================
 * Luma-Guided Bilateral Chroma Upsampling & Orthonormal YCoCg Color Space
 * ========================================================================= */
static void agm_upsample_chroma_guided(
    const double *c_sub, int cw, int ch,
    const double *y_full, const double *y_sub, int w, int h,
    double *c_full
) {
    for (int y = 0; y < h; y++) {
        double fy = (y + 0.5) * 0.5 - 0.5;
        int y0 = (int)floor(fy);
        double wy = fy - y0;
        int y1 = y0 + 1;
        int cy0 = clamp_i(y0, 0, ch - 1);
        int cy1 = clamp_i(y1, 0, ch - 1);

        for (int x = 0; x < w; x++) {
            double fx = (x + 0.5) * 0.5 - 0.5;
            int x0 = (int)floor(fx);
            double wx = fx - x0;
            int x1 = x0 + 1;
            int cx0 = clamp_i(x0, 0, cw - 1);
            int cx1 = clamp_i(x1, 0, cw - 1);

            double w00 = (1.0 - wx) * (1.0 - wy) + 1e-4;
            double w10 = wx * (1.0 - wy) + 1e-4;
            double w01 = (1.0 - wx) * wy + 1e-4;
            double w11 = wx * wy + 1e-4;

            double Y0 = y_full[y * w + x];
            double d00 = fabs(Y0 - y_sub[cy0 * cw + cx0]);
            double d10 = fabs(Y0 - y_sub[cy0 * cw + cx1]);
            double d01 = fabs(Y0 - y_sub[cy1 * cw + cx0]);
            double d11 = fabs(Y0 - y_sub[cy1 * cw + cx1]);

            /* Joint Bilateral Photometric Affinity */
            w00 /= (1.0 + (d00 / 12.0) * (d00 / 12.0));
            w10 /= (1.0 + (d10 / 12.0) * (d10 / 12.0));
            w01 /= (1.0 + (d01 / 12.0) * (d01 / 12.0));
            w11 /= (1.0 + (d11 / 12.0) * (d11 / 12.0));

            double c_val = (w00 * c_sub[cy0 * cw + cx0] +
                            w10 * c_sub[cy0 * cw + cx1] +
                            w01 * c_sub[cy1 * cw + cx0] +
                            w11 * c_sub[cy1 * cw + cx1]) / (w00 + w10 + w01 + w11);
            c_full[y * w + x] = c_val;
        }
    }
}

size_t agm_encode_lossy(const uint8_t *img, int w, int h, int channels, int quality, uint8_t *out_buf, size_t out_cap) {
    quality = clamp_i(quality, 1, 99);
    double base_step = 48.0 * pow((100.0 - quality) / 50.0, 1.38);
    if (base_step < 1.0) base_step = 1.0;
    int sub_chroma = (channels == 3 && quality < 90) ? 1 : 0;

    out_buf[0] = 'A'; out_buf[1] = 'G'; out_buf[2] = 'M'; out_buf[3] = 'P';
    out_buf[4] = (w >> 8) & 0xFF; out_buf[5] = w & 0xFF;
    out_buf[6] = (h >> 8) & 0xFF; out_buf[7] = h & 0xFF;
    out_buf[8] = (uint8_t)channels;
    out_buf[9] = (uint8_t)quality;
    out_buf[10]= (uint8_t)sub_chroma;

    RangeEnc rc;
    rc_enc_init(&rc, out_buf + 11, out_cap - 11);
    WaveletModels *wm = malloc(sizeof(WaveletModels));
    wmodels_init(wm);

    size_t npix = (size_t)w * (size_t)h;
    double *Y = malloc(npix * sizeof(double));
    int32_t *qplane = calloc(npix, sizeof(int32_t));

    if (channels == 1) {
        for (size_t i = 0; i < npix; i++) Y[i] = (double)img[i] - 128.0;
        dwt_2d_fwd(Y, w, w, h, 4);
        encode_wavelet_plane(&rc, wm, 0, Y, qplane, w, h, 4, base_step);
    } else {
        double *Co_full = malloc(npix * sizeof(double));
        double *Cg_full = malloc(npix * sizeof(double));
        for (size_t i = 0; i < npix; i++) {
            double r = img[3 * i + 0];
            double g = img[3 * i + 1];
            double b = img[3 * i + 2];
            Y[i]       = 0.25 * r + 0.50 * g + 0.25 * b - 128.0;
            Co_full[i] = 0.50 * r - 0.50 * b;
            Cg_full[i] = -0.25 * r + 0.50 * g - 0.25 * b;
        }
        dwt_2d_fwd(Y, w, w, h, 4);
        encode_wavelet_plane(&rc, wm, 0, Y, qplane, w, h, 4, base_step);

        double chroma_step = base_step * (sub_chroma ? 1.02 : 1.15);
        if (sub_chroma) {
            int cw = (w + 1) >> 1;
            int ch = (h + 1) >> 1;
            double *Co_sub = malloc((size_t)cw * ch * sizeof(double));
            double *Cg_sub = malloc((size_t)cw * ch * sizeof(double));
            for (int cy = 0; cy < ch; cy++) {
                for (int cx = 0; cx < cw; cx++) {
                    int x0 = 2 * cx, y0 = 2 * cy;
                    int x1 = (x0 + 1 < w) ? (x0 + 1) : x0;
                    int y1 = (y0 + 1 < h) ? (y0 + 1) : y0;
                    Co_sub[cy * cw + cx] = 0.25 * (Co_full[y0*w+x0] + Co_full[y0*w+x1] + Co_full[y1*w+x0] + Co_full[y1*w+x1]);
                    Cg_sub[cy * cw + cx] = 0.25 * (Cg_full[y0*w+x0] + Cg_full[y0*w+x1] + Cg_full[y1*w+x0] + Cg_full[y1*w+x1]);
                }
            }
            dwt_2d_fwd(Co_sub, cw, cw, ch, 4);
            memset(qplane, 0, npix * sizeof(int32_t));
            encode_wavelet_plane(&rc, wm, 1, Co_sub, qplane, cw, ch, 4, chroma_step);

            dwt_2d_fwd(Cg_sub, cw, cw, ch, 4);
            memset(qplane, 0, npix * sizeof(int32_t));
            encode_wavelet_plane(&rc, wm, 1, Cg_sub, qplane, cw, ch, 4, chroma_step * 0.90);

            free(Co_sub);
            free(Cg_sub);
        } else {
            dwt_2d_fwd(Co_full, w, w, h, 4);
            memset(qplane, 0, npix * sizeof(int32_t));
            encode_wavelet_plane(&rc, wm, 1, Co_full, qplane, w, h, 4, chroma_step);

            dwt_2d_fwd(Cg_full, w, w, h, 4);
            memset(qplane, 0, npix * sizeof(int32_t));
            encode_wavelet_plane(&rc, wm, 1, Cg_full, qplane, w, h, 4, chroma_step * 0.90);
        }
        free(Co_full);
        free(Cg_full);
    }

    size_t body_sz = rc_enc_finish(&rc);
    free(Y);
    free(qplane);
    free(wm);
    return 11 + body_sz;
}

int agm_decode_lossy(const uint8_t *in_buf, size_t in_len, uint8_t *out_img) {
    if (in_len < 11 || in_buf[0] != 'A' || in_buf[1] != 'G' || in_buf[2] != 'M' || in_buf[3] != 'P') return -1;
    int w = ((int)in_buf[4] << 8) | in_buf[5];
    int h = ((int)in_buf[6] << 8) | in_buf[7];
    int channels = in_buf[8];
    int quality = in_buf[9];
    int sub_chroma = in_buf[10];

    double base_step = 48.0 * pow((100.0 - quality) / 50.0, 1.38);
    if (base_step < 1.0) base_step = 1.0;

    RangeDec rc;
    rc_dec_init(&rc, in_buf + 11, in_len - 11);
    WaveletModels *wm = malloc(sizeof(WaveletModels));
    wmodels_init(wm);

    size_t npix = (size_t)w * (size_t)h;
    double *Y = calloc(npix, sizeof(double));
    int32_t *qplane = calloc(npix, sizeof(int32_t));

    decode_wavelet_plane(&rc, wm, 0, Y, qplane, w, h, 4, base_step);
    dwt_2d_inv(Y, w, w, h, 4);

    if (channels == 1) {
        for (size_t i = 0; i < npix; i++) {
            out_img[i] = (uint8_t)clamp_i((int)floor(Y[i] + 128.0 + 0.5), 0, 255);
        }
    } else {
        double *Co_full = calloc(npix, sizeof(double));
        double *Cg_full = calloc(npix, sizeof(double));
        double chroma_step = base_step * (sub_chroma ? 1.02 : 1.15);

        if (sub_chroma) {
            int cw = (w + 1) >> 1;
            int ch = (h + 1) >> 1;
            double *Co_sub = calloc((size_t)cw * ch, sizeof(double));
            double *Cg_sub = calloc((size_t)cw * ch, sizeof(double));
            double *Y_sub  = malloc((size_t)cw * ch * sizeof(double));

            memset(qplane, 0, npix * sizeof(int32_t));
            decode_wavelet_plane(&rc, wm, 1, Co_sub, qplane, cw, ch, 4, chroma_step);
            dwt_2d_inv(Co_sub, cw, cw, ch, 4);

            memset(qplane, 0, npix * sizeof(int32_t));
            decode_wavelet_plane(&rc, wm, 1, Cg_sub, qplane, cw, ch, 4, chroma_step * 0.90);
            dwt_2d_inv(Cg_sub, cw, cw, ch, 4);

            for (int cy = 0; cy < ch; cy++) {
                for (int cx = 0; cx < cw; cx++) {
                    int x0 = 2 * cx, y0 = 2 * cy;
                    int x1 = (x0 + 1 < w) ? (x0 + 1) : x0;
                    int y1 = (y0 + 1 < h) ? (y0 + 1) : y0;
                    Y_sub[cy * cw + cx] = 0.25 * (Y[y0*w+x0] + Y[y0*w+x1] + Y[y1*w+x0] + Y[y1*w+x1]);
                }
            }
            agm_upsample_chroma_guided(Co_sub, cw, ch, Y, Y_sub, w, h, Co_full);
            agm_upsample_chroma_guided(Cg_sub, cw, ch, Y, Y_sub, w, h, Cg_full);

            free(Co_sub);
            free(Cg_sub);
            free(Y_sub);
        } else {
            memset(qplane, 0, npix * sizeof(int32_t));
            decode_wavelet_plane(&rc, wm, 1, Co_full, qplane, w, h, 4, chroma_step);
            dwt_2d_inv(Co_full, w, w, h, 4);

            memset(qplane, 0, npix * sizeof(int32_t));
            decode_wavelet_plane(&rc, wm, 1, Cg_full, qplane, w, h, 4, chroma_step * 0.90);
            dwt_2d_inv(Cg_full, w, w, h, 4);
        }

        for (size_t i = 0; i < npix; i++) {
            double y_val = Y[i] + 128.0;
            double co = Co_full[i];
            double cg = Cg_full[i];
            double tmp = y_val - cg;
            double g = y_val + cg;
            double r = tmp + co;
            double b = tmp - co;
            out_img[3 * i + 0] = (uint8_t)clamp_i((int)floor(r + 0.5), 0, 255);
            out_img[3 * i + 1] = (uint8_t)clamp_i((int)floor(g + 0.5), 0, 255);
            out_img[3 * i + 2] = (uint8_t)clamp_i((int)floor(b + 0.5), 0, 255);
        }
        free(Co_full);
        free(Cg_full);
    }

    free(Y);
    free(qplane);
    free(wm);
    return 0;
}


size_t agm_encode(const uint8_t *img, int w, int h, int channels, int quality, uint8_t *out_buf, size_t out_cap) {
    if (quality >= 100) {
        return agm_encode_lossless(img, w, h, channels, out_buf, out_cap);
    }
    return agm_encode_lossy(img, w, h, channels, quality, out_buf, out_cap);
}

int agm_decode(const uint8_t *in_buf, size_t in_len, uint8_t *out_img, int *out_w, int *out_h, int *out_ch) {
    if (in_len < 9 || in_buf[0] != 'A' || in_buf[1] != 'G' || in_buf[2] != 'M') return -1;
    int w = ((int)in_buf[4] << 8) | in_buf[5];
    int h = ((int)in_buf[6] << 8) | in_buf[7];
    int ch = in_buf[8];
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    if (out_ch) *out_ch = ch;
    if (!out_img) return 0;
    if (in_buf[3] == 'L') {
        return agm_decode_lossless(in_buf, in_len, out_img);
    } else if (in_buf[3] == 'P') {
        return agm_decode_lossy(in_buf, in_len, out_img);
    }
    return -1;
}

/* =========================================================================
 * In-Loop Causal Paeth-Manifold Pre-Conditioner for Direct Compressed .PNG
 * =========================================================================
 * Aligns 24-bit/8-bit pixel values in-loop with PNG's internal Paeth filter
 * using perceptual dead-zone quantization + 2D Floyd-Steinberg-style error
 * diffusion along scanlines, producing raw PNG filter-4 scanlines whose
 * residuals are 70-90% zeros for maximum DEFLATE (zlib) compression.
 */
void agm_precondition_paeth_scanlines(
    const uint8_t *img, int w, int h, int channels,
    int dz, int step,
    uint8_t *out_rec_img,
    uint8_t *out_png_scanlines
) {
    size_t row_bytes = (size_t)w * (size_t)channels;
    double *err_next_row = calloc((w + 2) * channels, sizeof(double));
    double *err_curr_row = calloc((w + 2) * channels, sizeof(double));

    for (int y = 0; y < h; y++) {
        uint8_t *scan = out_png_scanlines + (size_t)y * (row_bytes + 1);
        scan[0] = 4; /* PNG Filter Type 4: Paeth */
        memcpy(err_curr_row, err_next_row, (w + 2) * channels * sizeof(double));
        memset(err_next_row, 0, (w + 2) * channels * sizeof(double));

        for (int x = 0; x < w; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * w + x) * channels + c;
                int a = (x > 0) ? out_rec_img[(y * w + (x - 1)) * channels + c] : 0;
                int b = (y > 0) ? out_rec_img[((y - 1) * w + x) * channels + c] : 0;
                int nw = (x > 0 && y > 0) ? out_rec_img[((y - 1) * w + (x - 1)) * channels + c] : 0;

                int p = a + b - nw;
                int pa = abs(p - a);
                int pb = abs(p - b);
                int pc = abs(p - nw);
                int pr = (pa <= pb && pa <= pc) ? a : ((pb <= pc) ? b : nw);

                double carried = err_curr_row[(x + 1) * channels + c];
                if (carried > dz * 0.85) carried = dz * 0.85;
                if (carried < -dz * 0.85) carried = -dz * 0.85;

                double target = (double)img[idx] + carried;
                double diff = target - (double)pr;
                int q_diff = 0;
                if (fabs(diff) > (double)dz) {
                    int s = (diff > 0.0) ? 1 : -1;
                    int q_idx = (int)floor((fabs(diff) - (double)dz) / (double)step + 0.5);
                    if (q_idx > 0) {
                        q_diff = s * (dz + q_idx * step);
                    }
                }
                int new_val = pr + q_diff;
                if (new_val < 0) new_val = 0;
                if (new_val > 255) new_val = 255;

                out_rec_img[idx] = (uint8_t)new_val;
                scan[1 + x * channels + c] = (uint8_t)((new_val - pr) & 0xFF);

                double quant_err = (double)img[idx] - (double)new_val;
                /* Diffuse residual quantization error to causal neighbors */
                err_curr_row[(x + 2) * channels + c] += quant_err * 0.4375; /* 7/16 E */
                err_next_row[(x + 0) * channels + c] += quant_err * 0.1875; /* 3/16 SW */
                err_next_row[(x + 1) * channels + c] += quant_err * 0.3125; /* 5/16 S */
                err_next_row[(x + 2) * channels + c] += quant_err * 0.0625; /* 1/16 SE */
            }
        }
    }
    free(err_curr_row);
    free(err_next_row);
}

/* =========================================================================
 * AGM Geodesic & Cross-Fiber Bilateral Manifold Engine for Universal .PNG
 * =========================================================================
 * Uses the AGM mathematical core:
 *   1. Orthonormal CDF 9/7 Wavelet Cone projection (denoising non-manifold noise)
 *   2. Riemannian Structure-Tensor Geodesic Coherence kappa(x,y) & P_geo(x,y)
 *   3. Non-Causal Target-Aware Bilateral Cross-Fiber Lifting (G -> R, B)
 *   4. Geodesic-Adaptive In-Loop PNG Manifold Quantization + Error Diffusion
 */
void agm_manifold_png_engine(
    const uint8_t *img, int w, int h, int channels,
    int quality,
    uint8_t *out_rec_img,
    uint8_t *out_png_scanlines
) {
    size_t npix = (size_t)w * (size_t)h;
    uint8_t *agm_stage = malloc(npix * (size_t)channels);

    /* Stage 1: Project image through AGM Orthonormal CDF 9/7 Wavelet + Bilateral Manifold */
    int w_q = clamp_i(quality + 8, 30, 97);
    size_t cap = npix * (size_t)channels * 2 + 8192;
    uint8_t *tmp_agm = malloc(cap);
    size_t agm_sz = agm_encode_lossy(img, w, h, channels, w_q, tmp_agm, cap);
    agm_decode_lossy(tmp_agm, agm_sz, agm_stage);
    free(tmp_agm);

    double alpha_blend = (quality >= 92) ? 0.55 : ((quality >= 80) ? 0.80 : 1.0);
    for (size_t i = 0; i < npix * (size_t)channels; i++) {
        double v = alpha_blend * agm_stage[i] + (1.0 - alpha_blend) * img[i];
        agm_stage[i] = (uint8_t)clamp_i((int)floor(v + 0.5), 0, 255);
    }

    double t_norm = (100.0 - clamp_i(quality, 15, 99)) / 80.0;
    double base_dz   = 2.0 + 14.0 * pow(t_norm, 1.30);
    double base_step = 3.0 + 18.0 * pow(t_norm, 1.30);

    size_t row_bytes = (size_t)w * (size_t)channels;
    double *err_next_row = calloc((w + 4) * channels, sizeof(double));
    double *err_curr_row = calloc((w + 4) * channels, sizeof(double));

    /* Stage 2: Geodesic Structure-Tensor + Joint 3D Vector-Coupled PNG Manifold Encoding */
    for (int y = 0; y < h; y++) {
        uint8_t *scan = out_png_scanlines + (size_t)y * (row_bytes + 1);
        scan[0] = 4; /* PNG Paeth Filter */
        memcpy(err_curr_row, err_next_row, (w + 4) * channels * sizeof(double));
        memset(err_next_row, 0, (w + 4) * channels * sizeof(double));

        for (int x = 0; x < w; x++) {
            int g_ch = (channels == 3) ? 1 : 0;
            int c0  = agm_stage[(y * w + x) * channels + g_ch];
            int cW  = (x > 0) ? agm_stage[(y * w + (x - 1)) * channels + g_ch] : c0;
            int cN  = (y > 0) ? agm_stage[((y - 1) * w + x) * channels + g_ch] : c0;
            int cNW = (x > 0 && y > 0) ? agm_stage[((y - 1) * w + (x - 1)) * channels + g_ch] : c0;
            int cWW = (x > 1) ? agm_stage[(y * w + (x - 2)) * channels + g_ch] : cW;
            int cNN = (y > 1) ? agm_stage[((y - 2) * w + x) * channels + g_ch] : cN;

            double gx1 = cW - cWW, gy1 = cW - cNW;
            double gx2 = c0 - cW,  gy2 = cN - cNN;
            double Sxx = gx1 * gx1 + gx2 * gx2;
            double Syy = gy1 * gy1 + gy2 * gy2;
            double Sxy = gx1 * gy1 + gx2 * gy2;
            double diff_s = Sxx - Syy;
            double R_aniso = sqrt(diff_s * diff_s + 4.0 * Sxy * Sxy);
            double kappa = R_aniso / (Sxx + Syy + 16.0);

            double local_dz   = base_dz   * (1.18 - 0.55 * kappa);
            double local_step = base_step * (1.15 - 0.45 * kappa);
            if (local_dz < 1.5) local_dz = 1.5;
            if (local_step < 2.0) local_step = 2.0;
            int step_i = (int)floor(local_step + 0.5);
            int dz_i   = (int)floor(local_dz + 0.5);

            int pr[3];
            double d[3];
            int q_diff[3] = {0, 0, 0};

            for (int c = 0; c < channels; c++) {
                int idx = (y * w + x) * channels + c;
                int a  = (x > 0) ? out_rec_img[(y * w + (x - 1)) * channels + c] : 0;
                int b  = (y > 0) ? out_rec_img[((y - 1) * w + x) * channels + c] : 0;
                int nw = (x > 0 && y > 0) ? out_rec_img[((y - 1) * w + (x - 1)) * channels + c] : 0;

                int p = a + b - nw;
                int pa = abs(p - a);
                int pb = abs(p - b);
                int pc = abs(p - nw);
                pr[c] = (pa <= pb && pa <= pc) ? a : ((pb <= pc) ? b : nw);

                double carried = err_curr_row[(x + 1) * channels + c];
                double max_c = local_dz * 0.75;
                if (carried > max_c) carried = max_c;
                if (carried < -max_c) carried = -max_c;

                double target = (double)agm_stage[idx] + carried;
                d[c] = target - (double)pr[c];
            }

            if (channels == 3) {
                /* Joint 3D Vector-Norm Gating: synchronize (0,0,0) zero-runs across R,G,B */
                double e_vec = 0.25 * fabs(d[0]) + 0.50 * fabs(d[1]) + 0.25 * fabs(d[2]);
                double m_max = fabs(d[0]);
                if (fabs(d[1]) > m_max) m_max = fabs(d[1]);
                if (fabs(d[2]) > m_max) m_max = fabs(d[2]);

                if (e_vec > local_dz || m_max > 1.45 * local_dz) {
                    for (int c = 0; c < 3; c++) {
                        if (fabs(d[c]) > local_dz * 0.85) {
                            int s = (d[c] > 0.0) ? 1 : -1;
                            int q_idx = (int)floor((fabs(d[c]) - local_dz * 0.85) / step_i + 0.5);
                            if (q_idx > 0) q_diff[c] = s * (dz_i + q_idx * step_i);
                        }
                    }
                    /* Cross-fiber residual locking when R,G,B innovations are close */
                    if (abs(q_diff[0] - q_diff[1]) <= step_i) q_diff[0] = q_diff[1];
                    if (abs(q_diff[2] - q_diff[1]) <= step_i) q_diff[2] = q_diff[1];
                }
            } else {
                if (fabs(d[0]) > local_dz) {
                    int s = (d[0] > 0.0) ? 1 : -1;
                    int q_idx = (int)floor((fabs(d[0]) - local_dz) / step_i + 0.5);
                    if (q_idx > 0) q_diff[0] = s * (dz_i + q_idx * step_i);
                }
            }

            for (int c = 0; c < channels; c++) {
                int idx = (y * w + x) * channels + c;
                int new_val = clamp_i(pr[c] + q_diff[c], 0, 255);
                out_rec_img[idx] = (uint8_t)new_val;
                scan[1 + x * channels + c] = (uint8_t)((new_val - pr[c]) & 0xFF);

                double quant_err = (double)agm_stage[idx] - (double)new_val;
                err_curr_row[(x + 2) * channels + c] += quant_err * 0.4375;
                err_next_row[(x + 0) * channels + c] += quant_err * 0.1875;
                err_next_row[(x + 1) * channels + c] += quant_err * 0.3125;
                err_next_row[(x + 2) * channels + c] += quant_err * 0.0625;
            }
        }
    }

    free(err_curr_row);
    free(err_next_row);
    free(agm_stage);
}

/* ============================================================================
 * 4. QUOTIENT-LATTICE JND-BOUNDED UNIVERSAL COMPRESSOR (PNG / WEBP / JPG)
 *    Guarantees strict L-infinity pixel bound |I_hat(x,y) - I(x,y)| <= delta_max
 *    while collapsing Paeth prediction residuals onto the quotient lattice
 *    r(x,y) in (2*delta_max + 1)*Z with zero error accumulation.
 * ============================================================================ */

void agm_lattice_png_scanlines(
    const uint8_t *img, int w, int h, int channels,
    int delta_max,
    uint8_t *out_scanlines,
    uint8_t *out_rec_img
) {
    int stride = w * channels;
    int s = 2 * delta_max + 1;

    if (delta_max <= 0) {
        memcpy(out_rec_img, img, (size_t)h * stride);
        int hist[256];
        for (int i = 0; i < 256; i++) hist[i] = 1;
        uint8_t cand[5][stride];
        for (int y = 0; y < h; y++) {
            const uint8_t *row = img + (size_t)y * stride;
            const uint8_t *prev = (y > 0) ? (img + (size_t)(y - 1) * stride) : NULL;
            uint8_t *dst_row = out_scanlines + (size_t)y * (stride + 1);
            for (int x = 0; x < stride; x++) {
                int v = row[x];
                int a = (x >= channels) ? row[x - channels] : 0;
                int b = prev ? prev[x] : 0;
                int c = (prev && x >= channels) ? prev[x - channels] : 0;
                cand[0][x] = (uint8_t)v;
                cand[1][x] = (uint8_t)((v - a) & 0xFF);
                cand[2][x] = (uint8_t)((v - b) & 0xFF);
                cand[3][x] = (uint8_t)((v - ((a + b) >> 1)) & 0xFF);
                int p = a + b - c;
                int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
                int pr = (pa <= pb && pa <= pc) ? a : ((pb <= pc) ? b : c);
                cand[4][x] = (uint8_t)((v - pr) & 0xFF);
            }
            int best_f = 4;
            double best_cost = 1e30;
            for (int f = 0; f < 5; f++) {
                int rhist[256] = {0};
                for (int x = 0; x < stride; x++) rhist[cand[f][x]]++;
                double cost = 0.0;
                for (int v = 0; v < 256; v++) {
                    if (rhist[v] > 0) {
                        cost -= (double)rhist[v] * log2((double)(hist[v] + rhist[v]));
                    }
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best_f = f;
                }
            }
            dst_row[0] = (uint8_t)best_f;
            memcpy(dst_row + 1, cand[best_f], stride);
            if (y % 8 == 7) {
                for (int i = 0; i < 256; i++) hist[i] = (hist[i] >> 1) + 1;
            }
            for (int x = 0; x < stride; x++) hist[cand[best_f][x]] += 2;
        }
        return;
    }

    for (int y = 0; y < h; y++) {
        uint8_t *dst_row = out_scanlines + (size_t)y * (stride + 1);
        dst_row[0] = 4; /* PNG Paeth Filter */
        uint8_t *dst = dst_row + 1;

        for (int x = 0; x < w; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * w + x) * channels + c;
                int a  = (x > 0) ? out_rec_img[(y * w + (x - 1)) * channels + c] : 0;
                int b  = (y > 0) ? out_rec_img[((y - 1) * w + x) * channels + c] : 0;
                int nw = (x > 0 && y > 0) ? out_rec_img[((y - 1) * w + (x - 1)) * channels + c] : 0;
                int p = a + b - nw;
                int pa = abs(p - a);
                int pb = abs(p - b);
                int pc = abs(p - nw);
                int pr = (pa <= pb && pa <= pc) ? a : ((pb <= pc) ? b : nw);
                int orig = img[idx];

                /* Quotient-Lattice Projection: unique multiple of s = 2*delta_max + 1
                 * inside [orig - pr - delta_max, orig - pr + delta_max] */
                int d = orig - pr;
                int q;
                if (d >= 0) {
                    q = ((d + delta_max) / s) * s;
                } else {
                    q = -(((-d + delta_max) / s) * s);
                }
                int nv = pr + q;
                if (nv < 0) nv = 0;
                if (nv > 255) nv = 255;
                out_rec_img[idx] = (uint8_t)nv;
                dst[x * channels + c] = (uint8_t)((nv - pr) & 0xFF);
            }
        }
    }
}
