// C-ABI entry for the FP8 flash-attention backward (hdim 128, causal, GQA; sbhd layout; TE-style per-tensor scaling).
#include <cstring>
#include "launch_bwd_fp8.hpp"

// 1 if qt/dot/kt must be produced by fa3_transpose_sbhd_to_bhds_p16 (FP8_PERM16 build), else 0
extern "C" int fa3_bwd_fp8_perm16() {
#ifdef FP8_PERM16
    return 1;
#else
    return 0;
#endif
}

extern "C" int fa3_bwd_fp8_hdim128(
    const void* q, const void* k, const void* v, const void* o, const void* dout,          // E4M3 [S, B, H, D] / [S, B, Hk, D]
    const void* qt, const void* dot, const void* kt,                                       // E4M3 [B, H, D, S] / [B, Hk, D, S]
    const float* lse,                                                                      // fp32 [B, H, S], natural log
    const float* desc_q, const float* desc_k, const float* desc_v, const float* desc_o, const float* desc_do,   // device scalars
    const float* scale_s, const float* scale_dp,                                           // device scalars
    const float* scale_dqkv, const float* inv_scale_dqkv,                                  // device scalars (nullable -> 1)
    float* amax_dp, float* amax_dqkv,                                                      // device scalars (nullable)
    int out_bf16,                                                                          // 1: dq/dk/dv bf16 (true values); 0: E4M3 scaled by scale_dqkv
    int grad_e5m2,                                                                         // 1: dO is E5M2 and dS is quantized to E5M2 (TE HYBRID); 0: E4M3
    float dq_prescale,                                                                     // fp16 dQ partials carry scale_dqkv*dq_prescale; use 4096/fp8_max(dqkv dtype)
    void* dq, void* dk, void* dv,
    float* dq_accum /*fp16 [B*H*Sqr*128]*/, float* dk_accum, float* dv_accum, float* softmax_d, float* lse_log2,
    int* dq_semaphore, int* dk_semaphore, int* dv_semaphore, int* tile_count_semaphore,
    int b, int s_q, int s_k, int h, int h_k, int d, float softmax_scale, int num_sm, cudaStream_t stream)
{
    Flash_bwd_params p;
    std::memset(&p, 0, sizeof(p));
    const int64_t hd = (int64_t)h * d, hkd = (int64_t)h_k * d;
    p.q_ptr = const_cast<void*>(q); p.q_row_stride = (int64_t)b * hd;  p.q_head_stride = d; p.q_batch_stride = hd;
    p.k_ptr = const_cast<void*>(k); p.k_row_stride = (int64_t)b * hkd; p.k_head_stride = d; p.k_batch_stride = hkd;
    p.v_ptr = const_cast<void*>(v); p.v_row_stride = (int64_t)b * hkd; p.v_head_stride = d; p.v_batch_stride = hkd;
    p.v_dim_stride = 1;
    p.o_ptr = const_cast<void*>(o); p.o_row_stride = (int64_t)b * hd; p.o_head_stride = d; p.o_batch_stride = hd;
    p.do_ptr = const_cast<void*>(dout); p.do_row_stride = (int64_t)b * hd; p.do_head_stride = d; p.do_batch_stride = hd;
    p.dq_ptr = dq; p.dq_row_stride = (int64_t)b * hd;  p.dq_head_stride = d; p.dq_batch_stride = hd;
    p.dk_ptr = dk; p.dk_row_stride = (int64_t)b * hkd; p.dk_head_stride = d; p.dk_batch_stride = hkd;
    p.dv_ptr = dv; p.dv_row_stride = (int64_t)b * hkd; p.dv_head_stride = d; p.dv_batch_stride = hkd;
    p.softmax_lse_ptr = const_cast<float*>(lse);
    p.softmax_lse_log2_ptr = lse_log2;
    p.dsoftmax_sum = softmax_d;
    p.dq_accum_ptr = dq_accum; p.dk_accum_ptr = dk_accum; p.dv_accum_ptr = dv_accum;
    p.dq_semaphore = dq_semaphore; p.dk_semaphore = dk_semaphore; p.dv_semaphore = dv_semaphore;
    p.tile_count_semaphore = tile_count_semaphore;
    p.b = b; p.b_k = b; p.h = h; p.h_k = h_k;
    p.seqlen_q = s_q; p.seqlen_k = s_k; p.total_q = b * s_q; p.total_k = b * s_k;
    p.d = d; p.d_rounded = 128; p.dv = d; p.dv_rounded = 128;
    const int kBlockM = 64, kBlockN = 128;
    p.seqlen_q_rounded = (s_q + kBlockM - 1) / kBlockM * kBlockM;
    p.seqlen_k_rounded = (s_k + kBlockN - 1) / kBlockN * kBlockN;
    p.scale_softmax = softmax_scale;
    p.softcap = 0.f;
    p.is_causal = true; p.is_local = false;
    p.window_size_left = -1; p.window_size_right = 0;
    p.attention_chunk = 0;
    p.is_e4m3 = true;
    p.deterministic = false;
    p.num_sm = num_sm;
    p.arch = 90;
    p.p_dropout = 1.f; p.rp_dropout = 1.f;
    const int64_t sqr = p.seqlen_q_rounded, skr = p.seqlen_k_rounded;
    cudaMemsetAsync(dk_accum, 0, (size_t)b * h_k * skr * 128 * sizeof(float), stream);
    cudaMemsetAsync(dv_accum, 0, (size_t)b * h_k * skr * 128 * sizeof(float), stream);
    Fp8BwdExtra x { qt, dot, kt, desc_q, desc_k, desc_v, desc_o, desc_do, scale_s, scale_dp, scale_dqkv, amax_dp, amax_dqkv, inv_scale_dqkv, dq_prescale };
    if (grad_e5m2) {
        if (out_bf16) run_flash_bwd_fp8<kBlockM, kBlockN, true, cutlass::bfloat16_t, cutlass::float_e5m2_t>(p, x, stream);
        else          run_flash_bwd_fp8<kBlockM, kBlockN, true, cutlass::float_e4m3_t, cutlass::float_e5m2_t>(p, x, stream);
    } else {
        if (out_bf16) run_flash_bwd_fp8<kBlockM, kBlockN, true, cutlass::bfloat16_t, cutlass::float_e4m3_t>(p, x, stream);
        else          run_flash_bwd_fp8<kBlockM, kBlockN, true, cutlass::float_e4m3_t, cutlass::float_e4m3_t>(p, x, stream);
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

#ifdef DIAG_TIMING
// copies out (and zeroes) the per-WG phase cycle counters: out[wg*17 + phase], 2 x 17 entries
extern "C" int fa3_bwd_fp8_timing(unsigned long long* out) {
    cudaDeviceSynchronize();
    cudaError_t e = cudaMemcpyFromSymbol(out, flash::g_fp8_timing, sizeof(unsigned long long) * 34);
    if (e != cudaSuccess) return (int)e;
    unsigned long long z[34] = {};
    e = cudaMemcpyToSymbol(flash::g_fp8_timing, z, sizeof(z));
    return e == cudaSuccess ? 0 : (int)e;
}
#endif
