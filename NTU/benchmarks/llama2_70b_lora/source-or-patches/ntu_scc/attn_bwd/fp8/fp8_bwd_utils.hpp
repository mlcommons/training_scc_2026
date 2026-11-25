// Helpers for the FP8 flash-attention backward (Hopper).
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

// Build-time defaults. FP8_PERM16 (perm16 operand order; see transpose_fp8.cu) is on: -8.3% kernel time (2.53 vs 2.76 ms, 10-08).
// Opt-outs/opt-ins: -DFP8_NO_PERM16, -DFP8_LSE_LATE (-1.5% alone, slower with PERM16), -DFP8_EARLY_DV / -DFP8_PIPE_S (both slower),
// -DFP8_DQ_FP32 (fp32 dQ partials, same accuracy, +0.24 ms), -DDIAG_TIMING (per-phase clock64 counters).
#if !defined(FP8_NO_PERM16) && !defined(FP8_PERM16)
#define FP8_PERM16 1
#endif

namespace flash {

using namespace cute;

// Store one warpgroup's dS^T accumulator tile (E4M3, in the wgmma C-fragment layout: rows = n (kv), cols = m (q)) into
// smem as dS[m][n] with n contiguous (the K-major A operand of the dQ GEMM dQ = dS . K), using stmatrix.trans on
// 16-bit units that pack two consecutive n for one m.
//
// C fragment of a 64(n) x 64(m) tile per warpgroup, layout ((2,2,8), 1, 1):  element (v0, v1, v2) of lane l in warp w is
//   n = 16 w + l/4 + 8 v1,   m = 8 v2 + 2 (l%4) + v0.
// For each v2 (8 m-columns) and warp (16 n-rows) we build an 8x8 b16 matrix whose row p holds b16 pairs (n=2p, n=2p+1)
// for the 8 m's; lane t = 4p+q supplies b16 columns 2q, 2q+1 (m0 = 8v2+2q, m0+1). stmatrix.trans writes row m
// (= source column) with the 16 n's contiguous.
template <int kBlockM, int kBlockN, typename FragC, typename SmemTensor>
CUTLASS_DEVICE void store_dSt_transposed_fp8(FragC const& rdS, SmemTensor&& sdS, int thread_idx, int n_offset_wg) {
    static_assert(sizeof(typename FragC::value_type) == 1);
    static_assert(decltype(size<0, 0>(rdS))::value == 2 && decltype(size<0, 1>(rdS))::value == 2);
    constexpr int kV2 = decltype(size<0, 2>(rdS))::value;      // number of 8-column groups along m
    static_assert(kV2 % 4 == 0);
    int const lane = thread_idx % 32, warp_in_wg = (thread_idx % 128) / 32;
    int const q = lane % 4, p = lane / 4;
    int const n_base = n_offset_wg + 16 * warp_in_wg;
    // my 4 bytes per v2: [v1=0: m0, m1 | v1=1: m0, m1]
    uint8_t const* raw = reinterpret_cast<uint8_t const*>(rdS.data());
    // source lanes for rows 2p and 2p+1 of this warp's 16-row block, and which half (v1) to take
    int const src0 = 4 * ((2 * p) % 8) + q, src1 = 4 * ((2 * p + 1) % 8) + q;
    bool const hi = p >= 4;
    #pragma unroll
    for (int j = 0; j < kV2 / 4; ++j) {
        uint32_t regs[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int const v2 = 4 * j + i;
            // pack (v0, v1) -> byte v0 + 2 v1 ; fragment element index ((v0, v1, v2), 0, 0) is at v0 + 2 v1 + 4 v2 in memory
            uint32_t packed = uint32_t(raw[4 * v2 + 0]) | (uint32_t(raw[4 * v2 + 1]) << 8)
                            | (uint32_t(raw[4 * v2 + 2]) << 16) | (uint32_t(raw[4 * v2 + 3]) << 24);
            uint32_t const w0 = __shfl_sync(0xffffffffu, packed, src0);
            uint32_t const w1 = __shfl_sync(0xffffffffu, packed, src1);
            uint32_t const a = hi ? (w0 >> 16) : (w0 & 0xffffu);   // (dS[2p][m0], dS[2p][m1])
            uint32_t const b = hi ? (w1 >> 16) : (w1 & 0xffffu);   // (dS[2p+1][m0], dS[2p+1][m1])
            regs[i] = __byte_perm(a, b, 0x5140);                  // b16 col 2q = (n=2p, 2p+1) @ m0 ; col 2q+1 @ m1
        }
        // row addresses: lanes 8i..8i+7 give row R = lane%8 of matrix i -> m = 8 (4j + i) + R, 16 n starting at n_base
        int const i_m = lane / 8, R = lane % 8;
        int const m = 8 * (4 * j + i_m) + R;
        uint32_t const addr = cute::cast_smem_ptr_to_uint(&sdS(m, n_base));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                     :: "r"(addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
    }
}

// dS^T store for the perm16 n-order (see transpose_fp8.cu): smem dS[m][pos] with pos = 4q' + i <-> n = 2q' + (i&1) + 8(i>>1) within
// each 16-n group, matching the perm16 order of K^T so the dQ GEMM contracts correctly.
// Lane (p = l/4, q = l%4) holds, per v2, bytes [v1=0: m0, m0+1 | v1=1: m0, m0+1] for n = p and n = p + 8, m0 = 8 v2 + 2 q. The word
// for a given m at positions 4q'..4q'+3 (q' = p/2) takes bytes from lanes p = 2q' (even positions) and p = 2q'+1 (odd): one
// shuffle (lane ^ 4) and one byte permute; even-p lanes store the word for m0, odd-p lanes the word for m0 + 1.
template <int kBlockM, int kBlockN, typename FragC, typename SmemTensor>
CUTLASS_DEVICE void store_dSt_perm16_fp8(FragC const& rdS, SmemTensor&& sdS, int thread_idx, int n_offset_wg) {
    static_assert(sizeof(typename FragC::value_type) == 1);
    static_assert(decltype(size<0, 0>(rdS))::value == 2 && decltype(size<0, 1>(rdS))::value == 2);
    constexpr int kV2 = decltype(size<0, 2>(rdS))::value;
    int const lane = thread_idx % 32, warp_in_wg = (thread_idx % 128) / 32;
    int const q = lane % 4, p = lane / 4;
    bool const odd = p & 1;
    int const col = n_offset_wg + 16 * warp_in_wg + 4 * (p / 2);
    uint32_t const sel = odd ? 0x3715u : 0x6240u;
    uint32_t const* words = reinterpret_cast<uint32_t const*>(rdS.data());
    #pragma unroll
    for (int v2 = 0; v2 < kV2; ++v2) {
        uint32_t const mine = words[v2];
        uint32_t const partner = __shfl_xor_sync(0xffffffffu, mine, 4);
        uint32_t const w = __byte_perm(mine, partner, sel);
        int const m = 8 * v2 + 2 * q + (odd ? 1 : 0);
        *reinterpret_cast<uint32_t*>(&sdS(m, col)) = w;
    }
}

// In-lane replacement for FA3's permute_Aregs_fp8 when the B operand carries the perm16 k-order. After convert_layout_acc_Aregs
// each 32-bit word holds [row p: k=2q, 2q+1 | row p+8: k=2q, 2q+1] (rows mixed); the next word the same for k+8. Splitting the
// rows gives a0 = [p: 2q, 2q+1, 2q+8, 2q+9], a1 = [p+8: ...]: exactly the perm16 positions 4q..4q+3. Two byte permutes per 8
// bytes, no shuffles.
template <typename Fragment>
CUTLASS_DEVICE void unmix_Aregs_fp8_perm16(Fragment& frag) {
    static_assert(decltype(size<0, 0>(frag))::value == 4);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(decltype(stride<0, 1>(frag))::value == 4);
    static_assert(sizeof(typename Fragment::value_type) == 1);
    Tensor frag_64b = recast<uint2>(frag);
    #pragma unroll
    for (int i = 0; i < size(frag_64b); ++i) {
        uint32_t const x = frag_64b[i].x, y = frag_64b[i].y;
        frag_64b[i].x = __byte_perm(x, y, 0x5410);
        frag_64b[i].y = __byte_perm(x, y, 0x7632);
    }
}

// Block-level amax -> global atomicMax on the float bit pattern (non-negative floats order like ints).
CUTLASS_DEVICE void atomic_max_amax(float* amax_ptr, float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); }
    if ((threadIdx.x % 32) == 0 && amax_ptr != nullptr) {
        atomicMax(reinterpret_cast<int*>(amax_ptr), __float_as_int(v));
    }
}

// cp.reduce.async.bulk add on fp16: halves the dQ partial-sum traffic (the partials are pre-scaled to the E4M3 output range)
struct SM90_BULK_REDUCE_ADD_F16 {
    CUTLASS_DEVICE static void copy(void const* smem_ptr, void* gmem_ptr, int32_t store_bytes, uint64_t cache_hint) {
        uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.noftz.f16 [%0], [%1], %2, %3;\n"
                     :: "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint) : "memory");
    }
};

#ifdef DIAG_TIMING
// per-WG phase cycle accumulators (16 phases + m_block count), summed over all CTAs/calls; read via fa3_bwd_fp8_timing()
__device__ unsigned long long g_fp8_timing[2 * 17];
#endif

}  // namespace flash
