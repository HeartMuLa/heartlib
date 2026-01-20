#include <metal_stdlib>
using namespace metal;

// Must match `RoPEParams` in `ops.mm` (layout + types).
struct RoPEParams {
    uint d_model;
    uint rot_dim;
    uint half_rot;
    uint seq_len;
};

template <typename T>
inline void rope_impl(
    device const T* x,
    device const T* cos_t,
    device const T* sin_t,
    device T* out,
    constant RoPEParams& p,
    uint tid,
    uint tg_id
) {
    constexpr uint TG = 256;
    const uint vec = tg_id;
    const uint t = (p.seq_len > 0) ? (vec % p.seq_len) : 0;

    device const T* xr = x + vec * p.d_model;
    device T* yr = out + vec * p.d_model;

    device const T* c = cos_t + t * p.half_rot;
    device const T* s = sin_t + t * p.half_rot;

    for (uint i = tid; i < p.d_model; i += TG) {
        if (i < p.half_rot) {
            const float x1 = float(xr[i]);
            const float x2 = float(xr[i + p.half_rot]);
            const float cc = float(c[i]);
            const float ss = float(s[i]);
            yr[i] = T(x1 * cc - x2 * ss);
            yr[i + p.half_rot] = T(x1 * ss + x2 * cc);
        } else if (i >= p.rot_dim) {
            yr[i] = xr[i];
        }
    }
}

template <typename T>
inline void rope_bwd_impl(
    device const T* grad_y,
    device const T* cos_t,
    device const T* sin_t,
    device T* grad_x,
    constant RoPEParams& p,
    uint tid,
    uint tg_id
) {
    constexpr uint TG = 256;
    const uint vec = tg_id;
    const uint t = (p.seq_len > 0) ? (vec % p.seq_len) : 0;

    device const T* gr = grad_y + vec * p.d_model;
    device T* gx = grad_x + vec * p.d_model;

    device const T* c = cos_t + t * p.half_rot;
    device const T* s = sin_t + t * p.half_rot;

    for (uint i = tid; i < p.d_model; i += TG) {
        if (i < p.half_rot) {
            const float gy1 = float(gr[i]);
            const float gy2 = float(gr[i + p.half_rot]);
            const float cc = float(c[i]);
            const float ss = float(s[i]);
            gx[i] = T(gy1 * cc + gy2 * ss);
            gx[i + p.half_rot] = T(-gy1 * ss + gy2 * cc);
        } else if (i >= p.rot_dim) {
            gx[i] = gr[i];
        }
    }
}

kernel void rope_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* cos_t [[ buffer(1) ]],
    device const half* sin_t [[ buffer(2) ]],
    device half* out [[ buffer(3) ]],
    constant RoPEParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    rope_impl<half>(x, cos_t, sin_t, out, p, tid, tg_id);
}

kernel void rope_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* cos_t [[ buffer(1) ]],
    device const float* sin_t [[ buffer(2) ]],
    device float* out [[ buffer(3) ]],
    constant RoPEParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    rope_impl<float>(x, cos_t, sin_t, out, p, tid, tg_id);
}

kernel void rope_bwd_fp16(
    device const half* grad_y [[ buffer(0) ]],
    device const half* cos_t [[ buffer(1) ]],
    device const half* sin_t [[ buffer(2) ]],
    device half* grad_x [[ buffer(3) ]],
    constant RoPEParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    rope_bwd_impl<half>(grad_y, cos_t, sin_t, grad_x, p, tid, tg_id);
}

kernel void rope_bwd_fp32(
    device const float* grad_y [[ buffer(0) ]],
    device const float* cos_t [[ buffer(1) ]],
    device const float* sin_t [[ buffer(2) ]],
    device float* grad_x [[ buffer(3) ]],
    constant RoPEParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    rope_bwd_impl<float>(grad_y, cos_t, sin_t, grad_x, p, tid, tg_id);
}

