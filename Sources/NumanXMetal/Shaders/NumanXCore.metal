#include <metal_stdlib>
using namespace metal;

struct NXVectorUniforms {
    uint count;
    float alpha;
    float beta;
    uint reserved;
};

struct NXConeUniforms {
    uint contactCount;
    float rho;
    uint lambdaStride;
    uint velocityStride;
};

struct NXConeLawGPU {
    float friction;
    float normalCompliance;
    float tangentialCompliance;
    float restitutionVelocity;
    float stabilizationVelocity;
    float3 reserved;
};

inline float3 nx_lorentz_project(float3 q) {
    const float t = q.x;
    const float r = length(q.yz);
    if (r <= t) return q;
    if (r <= -t) return float3(0.0f);
    if (r == 0.0f) return float3(max(t, 0.0f), 0.0f, 0.0f);
    const float scale = 0.5f * (1.0f + t / r);
    return float3(0.5f * (r + t), scale * q.y, scale * q.z);
}

kernel void nx_axpby(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     constant NXVectorUniforms &u [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= u.count) return;
    y[gid] = fma(u.alpha, x[gid], u.beta * y[gid]);
}

kernel void nx_scale(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     constant NXVectorUniforms &u [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= u.count) return;
    y[gid] = u.alpha * x[gid];
}

kernel void nx_finite_guard(device const float *x [[buffer(0)]],
                            device atomic_uint *failure [[buffer(1)]],
                            constant NXVectorUniforms &u [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= u.count) return;
    if (!isfinite(x[gid])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void nx_dot_partials(device const float *a [[buffer(0)]],
                            device const float *b [[buffer(1)]],
                            device float *partials [[buffer(2)]],
                            constant NXVectorUniforms &u [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group [[threadgroup_position_in_grid]],
                            uint width [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    float value = gid < u.count ? a[gid] * b[gid] : 0.0f;
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = scratch[0];
}

kernel void nx_reduce_partials(device const float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant NXVectorUniforms &u [[buffer(2)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint gid [[thread_position_in_grid]],
                               uint group [[threadgroup_position_in_grid]],
                               uint width [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    scratch[tid] = gid < u.count ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[group] = scratch[0];
}

kernel void nx_coulomb_natural_residual(device const float3 *lambda [[buffer(0)]],
                                        device const float3 *velocity [[buffer(1)]],
                                        device const NXConeLawGPU *laws [[buffer(2)]],
                                        device float3 *residual [[buffer(3)]],
                                        device float2 *diagnostics [[buffer(4)]],
                                        constant NXConeUniforms &u [[buffer(5)]],
                                        uint gid [[thread_position_in_grid]]) {
    if (gid >= u.contactCount) return;
    const float3 l = lambda[gid];
    const float3 v = velocity[gid];
    const NXConeLawGPU law = laws[gid];
    const float mu = max(law.friction, 0.0f);
    const float3 corrected = float3(v.x + law.normalCompliance * l.x + law.restitutionVelocity + law.stabilizationVelocity,
                                    v.y + law.tangentialCompliance * l.y,
                                    v.z + law.tangentialCompliance * l.z);
    float3 lhat;
    float3 uhat;
    if (mu == 0.0f) {
        lhat = float3(l.x, 0.0f, 0.0f);
        uhat = float3(corrected.x, 0.0f, 0.0f);
    } else {
        lhat = float3(l.x, l.y / mu, l.z / mu);
        uhat = float3(corrected.x, mu * corrected.y, mu * corrected.z);
    }
    const float3 projected = nx_lorentz_project(lhat - u.rho * uhat);
    const float3 rhat = lhat - projected;
    residual[gid] = mu == 0.0f ? float3(rhat.x, 0.0f, 0.0f)
                               : float3(rhat.x, mu * rhat.y, mu * rhat.z);
    const float coneDistance = length(lhat - nx_lorentz_project(lhat));
    const float complementarity = fabs(l.x * corrected.x + l.y * corrected.y + l.z * corrected.z);
    diagnostics[gid] = float2(coneDistance, complementarity);
}
