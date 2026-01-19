#include <metal_stdlib>
using namespace metal;

// Must match `RMSNormParams` in `ops.mm` (layout + types).
struct RMSNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

// Must match `RMSNormGradWParams` in `ops.mm`.
struct RMSNormGradWParams {
    uint d_model;
    uint rows;
    uint stride_row; // in elements
};

constant uint TG = 256;
constant uint SIMD = 32;
constant uint NSIMD = TG / SIMD; // 8

template <typename T, bool HAS_WEIGHT, bool HAS_BWD_INV>
inline void rmsnorm_fwd_impl(
    device const T* x,
    device const T* weight,
    device T* out,
    device T* inv_out,
    constant RMSNormParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tg_sum,
    threadgroup float* shared_inv
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device T* yr = out + row * p.stride_row;

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        sum += v * v;
    }

    const float sg_sum = simd_sum(sum);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        tg_sum[tid / SIMD] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            total += tg_sum[i];
        }
        const float mean = total / float(p.d_model);
        const float inv = rsqrt(mean + p.eps);
        *shared_inv = inv;
        if constexpr (HAS_BWD_INV) {
            inv_out[row] = T(inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = *shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        float val = v * inv;
        if constexpr (HAS_WEIGHT) {
            val *= float(weight[i]);
        }
        yr[i] = T(val);
    }
}

kernel void rmsnorm_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, true, false>(x, weight, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_fwd_inv_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    device half* inv_out [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, true, true>(x, weight, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    constant RMSNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, false, false>(x, nullptr, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fwd_inv_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    device half* inv_out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, false, true>(x, nullptr, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, true, false>(x, weight, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_fwd_inv_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    device float* inv_out [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, true, true>(x, weight, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    constant RMSNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, false, false>(x, nullptr, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fwd_inv_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    device float* inv_out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]
) {
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, false, true>(x, nullptr, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

