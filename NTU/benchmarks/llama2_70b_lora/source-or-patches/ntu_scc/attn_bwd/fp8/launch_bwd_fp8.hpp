// Launch of the FP8 flash-attention backward (hdim 128, causal, GQA), derived from FA3's run_flash_bwd.
#pragma once
#include "cute/tensor.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cuda_check.h"
#include "static_switch.h"
#include "flash.h"
#include "preprocess_fp8.hpp"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_fp8.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"
#include "postprocess_fp8.hpp"

using namespace cute;

struct Fp8BwdExtra {
    void const *qt, *dot, *kt;           // pre-transposed E4M3: (d, seqlen, head, batch), seqlen contiguous
    float const *desc_q, *desc_k, *desc_v, *desc_o, *desc_do;   // TE scale_inv device scalars
    float const *scale_s, *scale_dp;                             // TE quantizer scales (device)
    float const *scale_dqkv;             // dQ/dK/dV output scale (device, nullable -> 1). Also pre-scales the fp16 dQ partials.
    float *amax_dp, *amax_dqkv;          // device scalars (nullable): amax of dS, and max over dQ/dK/dV (unscaled)
    float const *inv_scale_dqkv;         // device scalar 1/scale_dqkv (nullable -> 1); needed because the dQ partials carry scale_dqkv
    float dq_prescale;                   // host: extra factor on the fp16 dQ partials (choose so partials stay in fp16 range: ~4096/fp8_max)
};

// OutElement: cutlass::float_e4m3_t (outputs quantized with scale_dqkv) or cutlass::bfloat16_t (true values)
template <int kBlockM, int kBlockN, bool Is_causal, class OutElement, class ElementG>
void run_flash_bwd_fp8(Flash_bwd_params &params, Fp8BwdExtra const& x, cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    constexpr bool kOutFp8 = cute::is_same_v<OutElement, cutlass::float_e4m3_t>;
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    constexpr int kHeadDim = 128;
    constexpr bool Is_local = false, Has_softcap = false, Varlen = false, Deterministic = false, GQA = true;
#ifndef FP8_STAGES
#define FP8_STAGES 2
#endif
    constexpr int Stages_dO = FP8_STAGES, Stages_dS = FP8_STAGES;
    constexpr bool SdP_swapAB = true, dKV_swapAB = false, dQ_swapAB = false;
    constexpr int NumMmaWarpGroups = 2, AtomLayoutMSdP = 1, AtomLayoutNdKV = 2, AtomLayoutMdQ = 1;

    int const seqlen_q = params.seqlen_q, seqlen_k = params.seqlen_k;
    int const seqlen_q_rounded = params.seqlen_q_rounded, seqlen_k_rounded = params.seqlen_k_rounded;
    int const batch_q = params.b, batch_k = params.b;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocessFp8<TileShape_MK, Element, ElementG, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;   // clears the fp16 dQaccum (viewed as half as many floats)
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O
        static_cast<ElementG const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {seqlen_q_rounded, params.h, batch_q},  // shape_dPsum
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, params.h * params.seqlen_q},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded / PreprocessKernel::kdQDiv, params.h, batch_q},  // shape_dQaccum in fp32 units (fp16 buffer: half as many floats)
        {_1{}, seqlen_q_rounded * params.d_rounded / PreprocessKernel::kdQDiv, params.d_rounded / PreprocessKernel::kdQDiv * seqlen_q_rounded * params.h},
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q,
        params.seqused_q
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    CHECK_CUTLASS(cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false));

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90Fp8<FP8_STAGES, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementG, ElementAccum, ArchTag,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, false>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Deterministic>;
    using Scheduler = std::conditional_t<
        Is_causal,
        flash::SingleTileBwdLPTScheduler<Varlen, kBlockN, false /*SPT*/>,
        flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kBlockN>
    >;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, batch_k},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, batch_k},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_V
        static_cast<ElementG const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<Element const*>(x.qt), static_cast<ElementG const*>(x.dot), static_cast<Element const*>(x.kt),
        x.desc_q, x.desc_k, x.desc_v, x.desc_o, x.desc_do, x.scale_s, x.scale_dp, x.scale_dqkv, x.dq_prescale, x.amax_dp,
        static_cast<typename AttnKernel::CollectiveMainloop::ElementdQacc*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * params.seqlen_q_rounded * params.h}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_dPsum
        params.scale_softmax,
        params.window_size_left, params.window_size_right, 0 /*attention_chunk*/,
        params.softcap,
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k,
        params.seqused_q, params.seqused_k
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementAccum*>(params.dk_accum_ptr),
        typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dKaccum
        typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, params.h_k * params.d_rounded * params.seqlen_k_rounded},
        static_cast<ElementAccum*>(params.dv_accum_ptr),
        typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k},  // shape_dVaccum
        typename CollectiveEpilogue::StridedKV {_1{}, params.dv_rounded * seqlen_k_rounded, params.h_k * params.dv_rounded * params.seqlen_k_rounded},
        params.b,
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        params.cu_seqlens_k,
        params.seqused_k,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false));

    // dQ: fp32 accum -> E4M3   (dQ = softmax_scale * descale_dP * descale_k * acc; the two descales come in via scale_dq_host)
    using PostdQ = flash::ConvertAccumToFp8<TileShape_MK, typename AttnKernel::CollectiveMainloop::ElementdQacc, OutElement, AttnKernel::CollectiveMainloop::NumMmaThreads,
                                            typename AttnKernel::CollectiveMainloop::TiledMmadQ, AttnKernel::CollectiveMainloop::dQ_swapAB>;
    typename PostdQ::Arguments post_dq_args {
        static_cast<typename AttnKernel::CollectiveMainloop::ElementdQacc const*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * params.seqlen_q_rounded * params.h},
        static_cast<OutElement*>(params.dq_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},
        1.f / x.dq_prescale, kOutFp8 ? nullptr : x.inv_scale_dqkv, kOutFp8 ? x.inv_scale_dqkv : nullptr, x.amax_dqkv
    };
    {
        int smem = PostdQ::SharedStorageSize;
        if (smem >= 48 * 1024) { CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostdQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); }
        CHECK_CUTLASS(cutlass::kernel_launch<PostdQ>(dim3(num_m_block, params.h, params.b), PostdQ::MaxThreadsPerBlock, smem, stream, PostdQ::to_underlying_arguments(post_dq_args), false));
    }
    // dK, dV (GQA fp32 accum, already scaled in the mainloop) -> E4M3
    using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
    using PostdKV = flash::ConvertAccumToFp8<TileShape_NK, ElementAccum, OutElement, AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
                                             typename AttnKernel::CollectiveMainloop::TiledMmadKV, AttnKernel::CollectiveMainloop::dKV_swapAB>;
    int num_n_block = cute::ceil_div(params.seqlen_k, kBlockN);
    for (int which = 0; which < 2; ++which) {
        typename PostdKV::Arguments a {
            static_cast<ElementAccum const*>(which == 0 ? params.dk_accum_ptr : params.dv_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},
            {_1{}, seqlen_k_rounded * params.d_rounded, params.d_rounded * params.seqlen_k_rounded * params.h_k},
            static_cast<OutElement*>(which == 0 ? params.dk_ptr : params.dv_ptr),
            {seqlen_k, params.d, params.h_k, batch_k},
            {which == 0 ? params.dk_row_stride : params.dv_row_stride, _1{}, which == 0 ? params.dk_head_stride : params.dv_head_stride,
             which == 0 ? params.dk_batch_stride : params.dv_batch_stride},
            1.f, kOutFp8 ? x.scale_dqkv : nullptr, nullptr, x.amax_dqkv
        };
        int smem = PostdKV::SharedStorageSize;
        if (smem >= 48 * 1024) { CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostdKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); }
        CHECK_CUTLASS(cutlass::kernel_launch<PostdKV>(dim3(num_n_block, params.h_k, params.b), PostdKV::MaxThreadsPerBlock, smem, stream, PostdKV::to_underlying_arguments(a), false));
    }
}
