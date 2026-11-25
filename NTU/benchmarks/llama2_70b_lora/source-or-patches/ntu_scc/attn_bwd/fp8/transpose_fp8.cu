// Byte-matrix transposes for the FP8 attention backward: [S, B, H, D] -> [B, H, D, S]  (per (b, h): [S][D] -> [D][S]).
// Tiled through shared memory, 16-byte vector loads and stores; bandwidth-bound (~2x33 MB per call for Q and dO).
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int TS = 64;    // rows of S per tile
constexpr int TD = 128;   // D (full head dim)

// kPerm16: within each 16-group of s, destination position 4q+i holds s = 2q + (i&1) + 8(i>>1). This is the k-order in which
// a wgmma C fragment (converted to FP8) already sits in registers, so Q^T/dO^T/K^T written this way let the kernel skip FA3's
// permute_Aregs_fp8 shuffles for P^T/dS^T and store dS^T with one shuffle per word; the contraction is unchanged as long as both
// GEMM operands use the same order.
__device__ __forceinline__ int perm16_src(int j) { return 2 * (j >> 2) + (j & 1) + 8 * ((j >> 1) & 1); }

// grid: (S/TS, H, B), block: 256 threads
template <bool kPerm16>
__global__ void __launch_bounds__(256) transpose_sbhd_to_bhds_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                                                     int S, int B, int H, int D) {
    __shared__ __align__(16) uint8_t tile[TS][TD + 16];   // +16 pad: keeps the column gathers off a single bank set
    const int s0 = blockIdx.x * TS, h = blockIdx.y, b = blockIdx.z;
    const int t = threadIdx.x;
    // load 64 rows x 128 bytes: 8192 B = 512 x 16 B -> 2 vectors per thread (row = v / 8, col chunk = v % 8)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = t + i * 256, r = v >> 3, c = (v & 7) * 16;
        const uint8_t* p = src + (((int64_t)(s0 + r) * B + b) * H + h) * D + c;
        *reinterpret_cast<uint4*>(&tile[r][c]) = *reinterpret_cast<const uint4*>(p);
    }
    __syncthreads();
    // store 128 d-rows x 64 bytes (s contiguous): 8192 B = 512 x 16 B -> 2 per thread (d = v / 4, s chunk = (v % 4) * 16)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = t + i * 256, d = v >> 2, sc = (v & 3) * 16;
        uint8_t bytes[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) bytes[j] = tile[sc + (kPerm16 ? perm16_src(j) : j)][d];
        uint8_t* p = dst + (((int64_t)b * H + h) * D + d) * S + s0 + sc;
        *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(bytes);
    }
}

}  // namespace

template <bool kPerm16>
static int launch_transpose(const void* src, void* dst, int S, int B, int H, int D, cudaStream_t stream) {
    if (D != TD || S % TS != 0) return -1;
    dim3 grid(S / TS, H, B);
    transpose_sbhd_to_bhds_kernel<kPerm16><<<grid, 256, 0, stream>>>(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), S, B, H, D);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
extern "C" int fa3_transpose_sbhd_to_bhds(const void* src, void* dst, int S, int B, int H, int D, cudaStream_t stream) {
    return launch_transpose<false>(src, dst, S, B, H, D, stream);
}
extern "C" int fa3_transpose_sbhd_to_bhds_p16(const void* src, void* dst, int S, int B, int H, int D, cudaStream_t stream) {
    return launch_transpose<true>(src, dst, S, B, H, D, stream);
}
