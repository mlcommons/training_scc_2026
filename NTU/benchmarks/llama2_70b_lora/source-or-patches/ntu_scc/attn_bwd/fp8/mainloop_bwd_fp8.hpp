/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "softmax.h"
#include "utils.h"
#include "copy_sm90_bulk_reduce.hpp"
#include "fp8_bwd_utils.hpp"

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dS, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementG_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool Deterministic,
        bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
        bool Mma_dP_is_RS=false>
struct CollectiveMainloopBwdSm90Fp8 {
    static_assert(cute::is_same_v<Element_, cutlass::float_e4m3_t>, "FP8 E4M3 mainloop");

    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_dS = Stages_dS;
    static_assert(kStages >= kStages_dO);
    static_assert(Stages_dS == 1 || Stages_dS == kStages);
    static_assert(!Mma_dP_is_RS || SdP_swapAB_);  // If Mma_dP_is_RS, we need SdP_SwapAB
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;          // q, k, v, o, P: E4M3
    using ElementG = ElementG_;        // dO, dS: E4M3 or E5M2 (TE HYBRID recipe uses E5M2 for gradients)
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
#ifdef FP8_DQ_FP32
    using ElementdQacc = float;             // dQ partial sums reduced in fp32 (accuracy reference; +~0.16 ms)
#else
    using ElementdQacc = cutlass::half_t;   // dQ partial sums reduced in fp16 (pre-scaled to the output range)
#endif

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * 2;

    static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups && AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
    // static constexpr GMMA::Major PdS_Major = GMMA::Major::MN;
    static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarpGroups / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TiledMmadPRS = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));
    // dP^T = V . dO^T : A = V (E4M3), B = dO (ElementG)
    using TiledMmadPG = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, ElementG, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(      // dK: dS^T (ElementG, regs) . Q^T (E4M3)
        cute::GMMA::rs_op_selector<ElementG, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdKV{}));
    using TiledMmadV = decltype(cute::make_tiled_mma(       // dV: P^T (E4M3, regs) . dO^T (ElementG)
        cute::GMMA::rs_op_selector<Element, ElementG, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    using TiledMmadQ = decltype(cute::make_tiled_mma(       // dQ: dS (ElementG, smem) . K^T (E4M3)
        cute::GMMA::ss_op_selector<ElementG, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdQ{}));

    // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
    // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
    // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for the layout, only the K dimension
    // changes the layout.
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dKV_Mma
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
                                       Int<kBlockM / AtomLayoutMSdP>,
                                       Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
        std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Need stride to be multiple of 32, otherwise we get error (misaligned address) when doing TMA if e.g. kBlockM=80
    // We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
    // it's still a valid smem address.
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    using SmemLayoutLSEMma = std::conditional_t<
        SdP_swapAB,
        cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
        cute::Layout<cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>
    >;

    // FP8: wgmma needs K-major smem for both operands, so Q^T, dO^T, K^T are real (pre-transposed in gmem, TMA-loaded) tiles.
    using SmemLayoutAtomQt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                      Int<kHeadDim>, Int<kBlockM>>());
    using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages>{})));
    using SmemLayoutdOt = decltype(tile_to_shape(SmemLayoutAtomQt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dO>{})));
    using SmemLayoutAtomKt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                      Int<kHeadDim>, Int<kBlockN>>());
    using SmemLayoutKt = decltype(tile_to_shape(SmemLayoutAtomKt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
    using SmemLayoutPdSt = SmemLayoutPdS;  // unused in FP8 (P^T / dS^T feed RS GEMMs from registers)

    // Thread layout, 256 or 384 threads per row
    // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each WG separately.
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementdQacc>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;

    static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along N mode for this M load, if any

    using TMA_dO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<ElementG const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdO{}),
        TileShape_MNK{},
        ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV

    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV
    // Transposed operands: gmem (d, seqlen, head, batch) with seqlen contiguous
    using ShapeT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideT = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using TMA_Qt = decltype(cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}),
        SmemLayoutQt{}(_, _, _0{})));
    using TMA_dOt = decltype(cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<ElementG const*>(nullptr)), ShapeT{}, StrideT{}),
        SmemLayoutdOt{}(_, _, _0{})));
    using TMA_Kt = decltype(cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}),
        SmemLayoutKt{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(2 * size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);  // Q + Q^T (dO + dO^T: same bits)
    static_assert(cutlass::sizeof_bits_v<Element> == cutlass::sizeof_bits_v<ElementG>);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(2 * size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);  // K + K^T
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);

    // These are tuned for speed. They don't affect correctness.
    // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
    // this helps quite a bit to not have to do causal masking for most of the iterations.
    // For hdim 192, separating masking iterations results in register spills.
    static constexpr bool SeparateMaskingIterations = true;   // FP8: elementwise work is the bottleneck, so mask only the diagonal blocks
    // Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share them and then
    // shuffle to get the value whenever we need? This can reduce register pressure when SdP_swapAB, where each
    // thread needs to keep statistics for (kBlockM / 4) rows. If !SdP_swapAB, each thread only needs to keep
    // statistic for 2 rows.
    static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool dQacc_use_TMA = true;      // FP8 bwd: fp16 partials via bulk reduce-add (the register-atomics path is slower)
    // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
    // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
    static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
    static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
    static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
    static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS ? SmemAlignmentQKVdO : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
    static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

    // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't line up w smem_k and smem_v due to alignment?
    using SmemdQacc_t = cute::array_aligned<ElementdQacc, cute::cosize_v<SmemLayoutdQaccum>, 128>;
    using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKt>, SmemAlignmentQKVdO> smem_kt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQt>, SmemAlignmentQKVdO> smem_qt;
        cute::array_aligned<ElementG, cute::cosize_v<SmemLayoutdOt>, SmemAlignmentQKVdO> smem_dot;
        SmemdQacc_t smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<ElementG, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        SmemP_t smem_p;
        cute::array_aligned<ElementG, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    };

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        ElementG const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        Element const* const ptr_Qt;   // (d, seqlen_q, h, b), seqlen contiguous
        ElementG const* const ptr_dOt;
        Element const* const ptr_Kt;   // (d, seqlen_k, h_k, b)
        float const* const desc_q;     // TE per-tensor scale_inv / scale device scalars
        float const* const desc_k;
        float const* const desc_v;
        float const* const desc_o;
        float const* const desc_do;
        float const* const scale_s;    // P quantization scale (descale_s = 1/scale_s in-kernel)
        float const* const scale_dp;   // dS quantization scale
        float const* const scale_dq;   // dQ output scale (nullable -> 1)
        float const dq_prescale;       // host: the fp16 partials carry softmax_scale*descale_dP*descale_k*(*scale_dq)*dq_prescale
        float* const amax_dP;
        ElementdQacc* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right, attention_chunk;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_V;
        ShapeQKV const shape_dO;
        ElementdQacc* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum stride_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q;
        TMA_dO tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_Qt tma_load_Qt;
        TMA_dOt tma_load_dOt;
        TMA_Kt tma_load_Kt;
        float const* const desc_q; float const* const desc_k; float const* const desc_v; float const* const desc_o; float const* const desc_do;
        float const* const scale_s; float const* const scale_dp; float const* const scale_dq;
        float const dq_prescale;
        float* const amax_dP;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale, softmax_scale_log2;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_dO, args.stride_dO);
        TMA_dO tma_load_dO = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        auto shape_Qt = make_shape(get<1>(args.shape_Q), get<0>(args.shape_Q), get<2>(args.shape_Q), get<3>(args.shape_Q));
        auto stride_Qt = make_stride(int64_t(get<0>(args.shape_Q)), _1{}, int64_t(get<0>(args.shape_Q)) * get<1>(args.shape_Q),
                                     int64_t(get<0>(args.shape_Q)) * get<1>(args.shape_Q) * get<2>(args.shape_Q));
        Tensor mQt = make_tensor(make_gmem_ptr(args.ptr_Qt), shape_Qt, stride_Qt);
        TMA_Qt tma_load_Qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, SmemLayoutQt{}(_, _, _0{}));
        Tensor mdOt = make_tensor(make_gmem_ptr(args.ptr_dOt), shape_Qt, stride_Qt);
        TMA_dOt tma_load_dOt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mdOt, SmemLayoutdOt{}(_, _, _0{}));
        auto shape_Kt = make_shape(get<1>(args.shape_K), get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K));
        auto stride_Kt = make_stride(int64_t(get<0>(args.shape_K)), _1{}, int64_t(get<0>(args.shape_K)) * get<1>(args.shape_K),
                                     int64_t(get<0>(args.shape_K)) * get<1>(args.shape_K) * get<2>(args.shape_K));
        Tensor mKt = make_tensor(make_gmem_ptr(args.ptr_Kt), shape_Kt, stride_Kt);
        TMA_Kt tma_load_Kt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mKt, SmemLayoutKt{});
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        // In the backward, we need to multiply by
        // (1 - tanh^2) * softmax_scale / softcap_val * softcap_val = (1 - tanh^2) * softmax_scale.
        // Instead we multiply by (1 - tanh^2) and multiply dK and dV by params.softmax_scale
        // (the original softmax_scale) at the end.
        return {args.shape_Q, args.shape_K, 
                args.shape_V, args.shape_dO,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
                tma_load_Qt, tma_load_dOt, tma_load_Kt, args.desc_q, args.desc_k, args.desc_v, args.desc_o, args.desc_do,
                args.scale_s, args.scale_dp, args.scale_dq, args.dq_prescale, args.amax_dP,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.num_batch, args.dq_semaphore,
                args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Qt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dOt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Kt.get_tma_descriptor());
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline_dO pipeline_do,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord
         ) {

        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb,
            params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Loading Q, K can cause illegal memory access.
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) {
                scheduler_prefetch();
                return;
            }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_kt.data()), SmemLayoutKt{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mQt = params.tma_load_Qt.get_tma_tensor(make_shape(get<1>(params.shape_Q), get<0>(params.shape_Q), get<2>(params.shape_Q), get<3>(params.shape_Q)))(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdOt = params.tma_load_dOt.get_tma_tensor(make_shape(get<1>(params.shape_dO), get<0>(params.shape_dO), get<2>(params.shape_dO), get<3>(params.shape_dO)))(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mKt = params.tma_load_Kt.get_tma_tensor(make_shape(get<1>(params.shape_K), get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K)))(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, !is_varlen_q ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        // transposed tiles: (d, seqlen) tiles of (kHeadDim, kBlockM) along seqlen; K^T: (kHeadDim, kBlockN) at n_block
        Tensor gQt = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_q), mQt), Shape<Int<kHeadDim>, Int<kBlockM>>{}, make_coord(_0{}, _));  // (K, M, _)
        Tensor gdOt = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_q), mdOt), Shape<Int<kHeadDim>, Int<kBlockM>>{}, make_coord(_0{}, _));
        Tensor gKt = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k), mKt), Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_coord(_0{}, n_block));  // (K, N)
        auto [tQtgQt, tQtsQt] = tma_partition(params.tma_load_Qt, _0{}, Layout<_1>{}, group_modes<0, 2>(sQt), group_modes<0, 2>(gQt));  // (TMA, k), (TMA, PIPE)
        auto [tdOtgdOt, tdOtsdOt] = tma_partition(params.tma_load_dOt, _0{}, Layout<_1>{}, group_modes<0, 2>(sdOt), group_modes<0, 2>(gdOt));
        Tensor sKt_x = make_tensor(sKt.data(), make_layout(sKt.layout(), Layout<_1>{}));
        Tensor gKt_x = make_tensor(gKt.data(), make_layout(gKt.layout(), Layout<_1>{}));
        auto [tKtgKt, tKtsKt] = tma_partition(params.tma_load_Kt, _0{}, Layout<_1>{}, group_modes<0, 2>(sKt_x), group_modes<0, 2>(gKt_x));

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        // auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, block_rank_in_cluster, Layout<ClusterShape>{},
        //                                   group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));  // (TMA, k), (TMA, PIPE)
        // auto [tdOgdO, tdOsdO] = tma_partition(params.tma_load_dO, block_rank_in_cluster, Layout<ClusterShape>{},
        //                                   group_modes<0, 2>(sdO), group_modes<0, 2>(gdO));  // (TMA, k), (TMA, PIPE)
        auto block_tma_Q = params.tma_load_Q.get_slice(cluster_local_block_id.y);
        auto block_tma_dO = params.tma_load_dO.get_slice(cluster_local_block_id.y);
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

        uint16_t mcast_mask_qdo = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
            }
        }

        int m_block = m_block_min;

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(params.tma_load_Qt.with(*pipeline_q.producer_get_barrier(smem_pipe_write), 0, TMA::CacheHintSm90::EVICT_LAST),
                 tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write.index()));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                 gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
        }

        // // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM.
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);
            copy(params.tma_load_Kt.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKtgKt, tKtsKt);

            #pragma unroll (kHeadDim < 256 ? 2 : 1)
            for (; m_block < m_block_max - 1; ++m_block) {
                // If Q and dO have the same number of stages, we can use the same pipeline state variable
                // to reduce registers
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(params.tma_load_dOt.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), 0, TMA::CacheHintSm90::EVICT_LAST),
                     tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_do_cur.index()));
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                     gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(params.tma_load_Qt.with(*pipeline_q.producer_get_barrier(smem_pipe_write), 0, TMA::CacheHintSm90::EVICT_LAST),
                     tQtgQt(_, m_block + 1), tQtsQt(_, smem_pipe_write.index()));
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                     gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(params.tma_load_dOt.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), 0, TMA::CacheHintSm90::EVICT_LAST),
                 tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_do_cur.index()));
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                 gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            ++smem_pipe_write;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write) {
        static_assert(Q_dO_same_stages, "Q and dO must have the same number of stages");
        // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will increment smem_pipe_write
        PipelineState smem_pipe_write_do = smem_pipe_write;
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do) {
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord
             ) {
        if constexpr (!dQacc_use_TMA) { return; }
#ifdef DIAG_NO_DQ
        return;
#endif
        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Exit early
        // Though if local and deterministic, still need to increment dq semaphore
        if constexpr ((Is_causal || Is_local || Varlen) && !(Is_local && Deterministic)) {
            if (m_block_max <= m_block_min) { return; }
        }

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        static constexpr int dQ_TMA_num_bytes = CUTE_STATIC_V(size<0>(sdQ)) * sizeof(ElementdQacc);

        bool const is_varlen = Varlen && params.cu_seqlens_q;
        Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum = cute::flat_divide(gdQaccum_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)

        int const num_batch = params.num_batch;
        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        bool const lane_predicate = cute::elect_one_sync();
        int m_block = m_block_min;
        constexpr int kBlockM = get<0>(TileShape_MNK{});
        constexpr int kBlockN = get<1>(TileShape_MNK{});
        int n_block_global_max = cute::ceil_div(seqlen_info.seqlen_k, kBlockN);
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            if constexpr (Deterministic) {
                if constexpr(Is_causal) {
                    int n_block_max_for_m_block = std::min(n_block_global_max, cute::ceil_div((m_block + 1) * kBlockM + seqlen_info.seqlen_k - seqlen_info.seqlen_q, kBlockN));
                    Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block_max_for_m_block - 1 - n_block);
                } else {
                    Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
                }
            }
            #pragma unroll
            for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warpgroup_idx /*id*/);  // sdQ full, to be written to gmem
                if (lane_predicate) {
                    if constexpr (std::is_same_v<ElementdQacc, float>) {
                        SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()), raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()), dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                    } else {
                        flash::SM90_BULK_REDUCE_ADD_F16::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()), raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()), dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                    }
                    tma_store_arrive();
                }
            }
            // Note, the for_each() function is required here to ensure `warpgroup_idx` is of type Int<x>.
            for_each(make_int_sequence<NumMmaWarpGroups>{}, [&] (auto warpgroup_idx) {
                if (lane_predicate) { tma_store_wait<NumMmaWarpGroups - 1 - CUTE_STATIC_V(warpgroup_idx)>(); }
                cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + warpgroup_idx /*id*/);  // sdQ empty, ready to be written to
            });
            if constexpr (Deterministic) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
        }
        if constexpr (Is_local && Deterministic) {
            int const m_block_global_max = cute::ceil_div(seqlen_info.seqlen_q, kBlockM);
            #pragma unroll 2
            for (; m_block < m_block_global_max; ++m_block) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // We're not currently using this bc we're not using persistent scheduler
        // // Tell producer (warp 0) that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if constexpr (dQacc_use_TMA) {
            if (warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);  // sdQ empty, ready to be written to
            }
        }
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline_dO pipeline_do,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Exit early
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) { return false; }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_kt.data()), SmemLayoutKt{});
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
        // FP8 scales (per tensor; TE delayed scaling)
        float const descale_q = *params.desc_q, descale_k = *params.desc_k, descale_v = *params.desc_v;
        float const descale_o = *params.desc_o, descale_do = *params.desc_do;
        float const scale_s = *params.scale_s, scale_dp = *params.scale_dp;
        float const descale_s = 1.f / scale_s, descale_dp = 1.f / scale_dp;
#ifdef DIAG_PRINT_SCALES
        if (blockIdx.x == 0 && blockIdx.y == 0 && thread_idx == 0) {
            printf("[ptrs] desc_q %p desc_k %p scale_s %p scale_dp %p scale_dq %p amax_dP %p\n", params.desc_q, params.desc_k, params.scale_s, params.scale_dp, params.scale_dq, params.amax_dP);
            printf("[scales] dq %g dk %g dv %g do_ %g ddo %g scale_s %g scale_dp %g scale_dq %g lse_log2[0] %g dpsum[0] %g\n",
                   descale_q, descale_k, descale_v, descale_o, descale_do, scale_s, scale_dp,
                   params.scale_dq ? *params.scale_dq : -1.f, params.ptr_LSE_log2[0], params.ptr_dPsum[0]);
        }
#endif
        float const log2_scale_s = log2f(scale_s);
        float const c_dp = descale_do * descale_v * scale_dp / scale_s;      // dP term, with P pre-scaled by scale_s and dS by scale_dP
        float const c_dsum = descale_o * descale_do * scale_dp / scale_s;
        float const c_qk = params.softmax_scale_log2 * descale_q * descale_k;
        float amax_dp_thread = 0.f;
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumMmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadPG tiled_mma_dP;
        TiledMmadKV tiled_mma_dKV;   // dK
        TiledMmadV tiled_mma_dV;
        TiledMmadQ tiled_mma_dQ;

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));


        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);
        // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQaccum); printf("\n"); }

        // Allocate "fragments/descriptors"
        // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
        // because some partition_fragment_A/B don't compile.
        // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
        Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dV, sdOt);
        Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
        Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
        Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS); print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP); printf("\n"); print(tdSsdS); printf("\n"); }

        // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N, PIPE), we only take the col indices
        // or row indices, depending on whether SdP_swapAB.
        Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),  // (2, MMA_M, PIPE)
            group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _)));  // (2, V, MMA_N, PIPE)
        Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),
            group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _)));
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma); printf("\n"); print(tLSEsLSE); printf("\n"); }
        // If we want to split the stats among the 8 threads that share the same rows.
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int bidh = get<1>(block_coord);
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;

        // For the case where we do atomicAdd directly to gdQaccum instead of using TMA
        bool const is_varlen = Varlen && params.cu_seqlens_q;
        float const dq_scale = params.softmax_scale * descale_dp * descale_k * (params.scale_dq != nullptr ? *params.scale_dq : 1.f) * params.dq_prescale;
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum); printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum); printf("\n"); print(tdQgdQaccum); printf("\n"); }

        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );

        int m_block = m_block_min;

        clear(tdKrdK);
        clear(tdVrdV);
        // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        if constexpr (Mma_dP_is_RS) {
            using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
            Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
            cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
        }

#ifdef DIAG_TIMING
        bool const is_timer = (thread_idx % cutlass::NumThreadsPerWarpGroup) == 0;
        long long tacc[16] = {}; long long tprev = clock64(); int t_mblocks = 0;
#define TP(i) do { if (is_timer) { long long _now = clock64(); tacc[i] += _now - tprev; tprev = _now; } } while (0)
#else
#define TP(i) do {} while (0)
#endif
#if defined(FP8_EARLY_DV) && defined(FP8_PIPE_S)
#error "FP8_EARLY_DV and FP8_PIPE_S are exclusive"
#endif
#ifdef FP8_PIPE_S
        // Software pipelining of S = Q K^T: S(m+1) is issued in the middle of step m, right after P(m) has been converted out of
        // the accumulator, so the dS^T store / fragment permutes of step m run under an MMA instead of with the tensor pipe idle.
        Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
        int const m_block_last = m_block_max;
        auto issue_S = [&](auto const& pipe_state) {
            consumer_wait(pipeline_q, pipe_state);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, pipe_state.index()), tSrK, tSrS);
        };
        issue_S(smem_pipe_read);   // prologue: S(m_block_min)
#endif
        auto bwd_step = [&](int m_block, auto mask_fn) {
#ifndef FP8_PIPE_S
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            TP(15);   // loop overhead / outside
            consumer_wait(pipeline_q, smem_pipe_read);
            TP(0);    // wait Q (TMA)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
#else
            TP(15);   // loop overhead / outside (S(m) already in flight)
#endif
            Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            auto load_lse = [&] {
                if constexpr (!ShuffleLSE) {
                    cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
                } else {
                    #pragma unroll
                    for (int i = 0; i < kStatsPerThread; ++i) {
                        // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
                        tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read.index());
                    }
                }
            };
#ifndef FP8_LSE_LATE
            load_lse();
#endif
            Tensor tdPrdP = partition_fragment_C(tiled_mma_dP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            TP(1);    // S issue + LSE copy
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            TP(2);    // wait dO (TMA)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
#ifdef FP8_LSE_LATE
            // The LSE smem loads (and the lse - log2 scale_s subtractions) otherwise get scheduled between the dO wait and the dP
            // issue, where their latency delays the dP GEMM (ncu: 4.6% of stall samples on the first FADD). Pin them after it.
            asm volatile("" ::: "memory");
            load_lse();
#endif
            warpgroup_wait<1>();
            TP(3);    // dP issue + wait S
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            // dtanh needs to happen before masking, otherwise we get 1 - (-inf)^2 = NaN in the dtanh
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
            mask_fn(tSrS, m_block);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = [&] {
                    if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                    else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
#ifdef DIAG_NO_EXP
                    scores(mi, ni) = scores(mi, ni) * c_qk - (lse_scaled - log2_scale_s);
#else
                    scores(mi, ni) = exp2f(scores(mi, ni) * c_qk - (lse_scaled - log2_scale_s));   // = P * scale_s
#endif
                }
            }

            TP(4);    // mask + exp2
            Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffledPsum) {
                cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
                }
            }

#ifdef FP8_EARLY_DV
            // P^T -> E4M3 RS fragments and the dV GEMM go out before waiting for dP so the dS elementwise chain runs under the
            // dV MMA. Measured +3.5% SLOWER (2.84 vs 2.75 ms, 10-08) -> opt-in only.
            Tensor tdVrP_acc = make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMmadV>(tSrS.layout()));
            Tensor tdVrP = make_tensor_like<Element>(tdVrP_acc);
            flash::convert_type_out(tdVrP_acc, tdVrP);
#ifndef FP8_PERM16
            flash::permute_Aregs_fp8(tdVrP);
#else
            flash::unmix_Aregs_fp8_perm16(tdVrP);
#endif
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
            warpgroup_wait<1>();   // dP done (dV may still run)
#else
            warpgroup_wait<0>();
#endif
            TP(5);    // dPsum copy + wait dP
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur = [&] {
                    if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                    else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * c_dp - dP_sum_cur * c_dsum);   // = dS * scale_dP
                    amax_dp_thread = fmaxf(amax_dp_thread, fabsf(dS(mi, ni)));
                }
            }

            // FP8: P^T * scale_s and dS^T * scale_dP -> E4M3, laid out as RS A-operand fragments (permute_Aregs_fp8).
            // dS^T is also stored transposed to smem (as dS, K-major over n) for the dQ GEMM.
            Tensor rdS_c = make_tensor_like<ElementG>(tdPrdP);             // C layout, for the transposed smem store
            flash::convert_type_out(tdPrdP, rdS_c);
            TP(6);    // dS compute + amax + convert
#ifdef FP8_PIPE_S
            // P^T -> E4M3 now (the last reader of tSrS), then S(m+1) goes into tSrS
            Tensor tdVrP_acc = make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMmadV>(tSrS.layout()));
            Tensor tdVrP = make_tensor_like<Element>(tdVrP_acc);
            flash::convert_type_out(tdVrP_acc, tdVrP);
            if (m_block + 1 < m_block_last) {
                auto smem_pipe_read_next = smem_pipe_read;
                ++smem_pipe_read_next;
                issue_S(smem_pipe_read_next);
            }
            TP(0);    // (pipe) P convert + wait Q(m+1) + S(m+1) issue
#endif
            // Need to sync to make sure dS has already been used in the previous iteration before writing new values
            if constexpr (kStages_dS == 1) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
            }
            TP(7);    // PdS barrier #1 (dS buffer free)
#ifdef FP8_PERM16
            flash::store_dSt_perm16_fp8<kBlockM, kBlockN>(rdS_c, sdS(_, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())),
                                                          thread_idx, warp_group_idx * (kBlockN / NumMmaWarpGroups));
#else
            flash::store_dSt_transposed_fp8<kBlockM, kBlockN>(rdS_c, sdS(_, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())),
                                                               thread_idx, warp_group_idx * (kBlockN / NumMmaWarpGroups));
#endif
#if !defined(FP8_EARLY_DV) && !defined(FP8_PIPE_S)
            Tensor tdVrP_acc = make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMmadV>(tSrS.layout()));
            Tensor tdVrP = make_tensor_like<Element>(tdVrP_acc);
            flash::convert_type_out(tdVrP_acc, tdVrP);
#endif
#ifndef FP8_EARLY_DV
#ifndef FP8_PERM16
            flash::permute_Aregs_fp8(tdVrP);
#else
            flash::unmix_Aregs_fp8_perm16(tdVrP);
#endif
#endif
#ifndef FP8_DUP_DS_CVT
            // dS^T RS fragments: the same E5M2 bytes as rdS_c (the Aregs layout is a reinterpretation), permuted in place after
            // the transposed store above has consumed them in C order.
            Tensor tdKrdS = make_tensor(rdS_c.data(), convert_layout_acc_Aregs<TiledMmadKV>(rdS_c.layout()));
#else
            Tensor tdKrdS_acc = make_tensor(tdPrdP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
            Tensor tdKrdS = make_tensor_like<ElementG>(tdKrdS_acc);
            flash::convert_type_out(tdKrdS_acc, tdKrdS);
#endif
#ifndef FP8_PERM16
            flash::permute_Aregs_fp8(tdKrdS);
#else
            flash::unmix_Aregs_fp8_perm16(tdKrdS);
#endif
            TP(8);    // dS^T store + P/dS RS-fragment conversion/permute

            if constexpr (!Slice_dQKV_Mma) {
                // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
                static_assert(Mma_dKV_is_RS, "FP8 bwd: dK/dV GEMMs take P^T/dS^T from registers");
#ifndef FP8_EARLY_DV
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
#endif
#ifndef DIAG_NO_DQ
                // SMEM fence to make sure sdS is written before it's read by WGMMA
                cutlass::arch::fence_view_async_shared();
#ifndef DIAG_NO_PDS_SYNC
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
#else
                __syncwarp();
#endif
                TP(9);    // dV issue + fence + PdS barrier #2 (both dS halves stored)
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
#else
                warpgroup_wait<0>();
#endif
                TP(10);   // dQ issue + wait dV
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                TP(11);   // dK issue + wait dQ
#ifndef DIAG_NO_DQ
                if constexpr (dQacc_use_TMA) {
                    int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                    TP(12);   // dQEmpty barrier (previous bulk reduce has read sdQ)
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ); ++i) { tdQrdQ(i) *= dq_scale; }
                    Tensor rdQ16 = make_tensor_like<ElementdQacc>(tdQrdQ);
                    if constexpr (std::is_same_v<ElementdQacc, float>) { cute::copy(tdQrdQ, rdQ16); } else { flash::convert_type_out(tdQrdQ, rdQ16); }
                    Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(rdQ16);
                    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                    TP(13);   // dQ scale/convert/smem write
                }
#endif

            } else {
                static_assert(!Slice_dQKV_Mma, "not supported in the FP8 bwd");
            }

            warpgroup_wait<0>();
            TP(14);   // wait dK
#ifdef DIAG_TIMING
            if (is_timer) ++t_mblocks;
#endif
            pipeline_q.consumer_release(smem_pipe_read);   // release Q
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
        };

        // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
        // this helps quite a bit to not have to do causal masking for most of the iterations.
        if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            int const m_block_masking_max = ((n_block + 1) * kBlockN - 1 + seqlen_q - seqlen_k - params.window_size_right) / kBlockM + 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block_max_before_local_mask = !Is_local || !SeparateMaskingIterations
            ? m_block_max
            : std::min(m_block_max, (n_block * kBlockN + seqlen_q - seqlen_k + params.window_size_left) / kBlockM);

        auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal && !SeparateMaskingIterations, Is_local && !SeparateMaskingIterations>(tSrS, m_block, n_block); };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max_before_local_mask; ++m_block) {
            bwd_step(m_block, mask_fn);
        }

        if constexpr (Is_local && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.softmax_scale * descale_dp * descale_q; }
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= descale_s * descale_do; }
        flash::atomic_max_amax(params.amax_dP, amax_dp_thread / scale_dp);
#ifdef DIAG_TIMING
        if (is_timer) {
            int const wg = thread_idx / cutlass::NumThreadsPerWarpGroup;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { atomicAdd(reinterpret_cast<unsigned long long*>(&flash::g_fp8_timing[wg * 17 + i]), (unsigned long long)tacc[i]); }
            atomicAdd(reinterpret_cast<unsigned long long*>(&flash::g_fp8_timing[wg * 17 + 16]), (unsigned long long)t_mblocks);
        }
#endif
#undef TP

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        ++work_idx;
        return true;
    }

};

} // namespace flash
