// fp32 accumulator tile (in FA3's dQaccum / dKVaccum partition order) -> scale -> E4M3 output + amax.
// Read-back of the accumulator follows FlashAttnBwdPostprocessConvertdQ exactly (same TiledMma partition).
#pragma once
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/arch/barrier.h"
#include "seqlen.h"
#include "utils.h"
#include "fp8_bwd_utils.hpp"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class ElementAccum, class OutElement, int kNThreads, class TiledMma, bool swapAB>
class ConvertAccumToFp8 {
public:
    using TileShape_MK = TileShape_MK_;
    using Element = OutElement;
    using ArchTag = cutlass::arch::Sm90;
    static constexpr int kOutBytes = sizeof(OutElement);
    static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    static_assert(kNThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumWG = kNThreads / cutlass::NumThreadsPerWarpGroup;
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumWG>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape<_4>>{}));
    static constexpr int SmemdQaccumSize = size(TileShape_MK{});
    using SmemLayoutdQaccumFlat = Layout<Shape<Int<SmemdQaccumSize>>>;
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim / NumWG>, Int<NumWG>>>;

    struct SharedStorage : cute::aligned_struct<128> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>> smem_dqacc;
        cute::array_aligned<uint8_t, kBlockM * kHeadDim * kOutBytes> smem_out;     // plain [kBlockM][kHeadDim] output elements
        alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_dQaccum;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    using ShapedQ = cute::Shape<int32_t, int32_t, int32_t, int32_t>;   // (seqlen, d, head, batch)
    using StridedQ = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_rounded * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    struct Arguments {
        ElementAccum const* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        Element* ptr_dQ;
        ShapedQ const shape_dQ;
        StridedQ const stride_dQ;
        float const scale_host;       // output = acc * scale_host * (*scale_ptr or 1)
        float const* scale_ptr;
        float const* amax_descale_ptr;// amax reported = max|output| * (*amax_descale_ptr or 1)  (to report the UNSCALED amax)
        float* amax_ptr;
    };
    using Params = Arguments;
    static Params to_underlying_arguments(Arguments const& args) { return args; }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        Tensor sdQaccum = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sdQaccum_flat = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumFlat{});
        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;
        int const seqlen = get<0>(params.shape_dQ);

        Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum)(_, bidh, bidb);
        Tensor gdQaccum = local_tile(mdQaccum, Shape<Int<kBlockM * kHeadDim>>{}, make_coord(m_block));  // (M * K)
        static constexpr uint32_t TmaTransactionBytesdQaccum = static_cast<uint32_t>(size(SmemLayoutdQaccumFlat{}) * cute::sizeof_bits_v<ElementAccum> / 8);
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
        if (thread_idx == 0) {
            shared_storage.barrier_dQaccum.init(1 /*numThreads*/);
            shared_storage.barrier_dQaccum.arrive_and_expect_tx(TmaTransactionBytesdQaccum);
            copy(bulk_copy.with(*reinterpret_cast<uint64_t*>(&shared_storage.barrier_dQaccum)), gdQaccum, sdQaccum_flat);
        }
        __syncthreads();
        shared_storage.barrier_dQaccum.wait(0);

        R2STiledCopydQaccum s2r_tiled_copy_dQaccum;
        auto s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum);
        TiledMma tiled_mma_dQ;
        Tensor taccdQrdQaccum_f32 = partition_fragment_C(tiled_mma_dQ, select<!swapAB ? 0 : 1, !swapAB ? 1 : 0>(TileShape_MK{}));
        Tensor taccdQrdQaccum = make_fragment_like<ElementAccum>(taccdQrdQaccum_f32);
        CUTE_STATIC_ASSERT_V(size(taccdQrdQaccum) == size(tdQsdQaccum));
        Tensor tdQrdQaccum = s2r_thr_copy_dQaccum.retile_D(taccdQrdQaccum);
        cute::copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum);

        float const scale = params.scale_host * (params.scale_ptr != nullptr ? *params.scale_ptr : 1.f);
        // coordinates of each accumulator element in the (row, col) tile (or (col, row) if swapAB)
        auto thread_mma = tiled_mma_dQ.get_thread_slice(thread_idx);
        Tensor cTile = make_identity_tensor(select<!swapAB ? 0 : 1, !swapAB ? 1 : 0>(TileShape_MK{}));
        Tensor taccdQcdQ = thread_mma.partition_C(cTile);
        CUTE_STATIC_ASSERT_V(size(taccdQcdQ) == size(taccdQrdQaccum));
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < size(taccdQrdQaccum); ++i) {
            float const v = float(taccdQrdQaccum(i)) * scale;
            amax = fmaxf(amax, fabsf(v));
            int const r = !swapAB ? get<0>(taccdQcdQ(i)) : get<1>(taccdQcdQ(i));
            int const c = !swapAB ? get<1>(taccdQcdQ(i)) : get<0>(taccdQcdQ(i));
            reinterpret_cast<Element*>(shared_storage.smem_out.data())[r * kHeadDim + c] = Element(v);
        }
        if (params.amax_ptr != nullptr) {
            flash::atomic_max_amax(params.amax_ptr, amax * (params.amax_descale_ptr != nullptr ? *params.amax_descale_ptr : 1.f));
        }
        __syncthreads();
        // coalesced copy-out: 16 bytes per thread per step
        Tensor mdQ = make_tensor(make_gmem_ptr(params.ptr_dQ), params.shape_dQ, params.stride_dQ)(_, _, bidh, bidb);
        constexpr int kPerChunk = 16 / kOutBytes;                  // elements per 16-byte chunk
        constexpr int kChunks = kBlockM * kHeadDim / kPerChunk;
        for (int ch = thread_idx; ch < kChunks; ch += kNThreads) {
            int const r = ch / (kHeadDim / kPerChunk), c0 = (ch % (kHeadDim / kPerChunk)) * kPerChunk;
            int const row = m_block * kBlockM + r;
            if (row < seqlen) {
                uint4 const val = *reinterpret_cast<uint4 const*>(&reinterpret_cast<Element*>(shared_storage.smem_out.data())[r * kHeadDim + c0]);
                *reinterpret_cast<uint4*>(&mdQ(row, c0)) = val;
            }
        }
    }
};

}  // namespace flash
