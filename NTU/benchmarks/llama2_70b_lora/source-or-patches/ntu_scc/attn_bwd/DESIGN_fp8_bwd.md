# Hopper FP8 flash-attention backward for llama2-70B LoRA (design, 2026-10-08)

## Why
nsys "Tensor Active" counter in training (400 W): cuDNN FP8 attention bwd 36%, fwd 50%; dense GEMMs 85-93%.
Attention bwd = 3.36 ms/layer = 12% of the step. cuDNN has exactly one FP8 bwd engine (eng6 knob 26), TE has no
engine knob. Target: >=55% tensor-busy -> ~-1.2 ms/layer -> ~-4% step. Go/no-go after the bf16 FA3 baseline
measurement + an FP8 prototype (>=45% busy on the isolated shape, else stop).

## Shape (per GPU, TP=2)
b=1, s_q=s_k=8192, h_q=32, h_kv=4 (GQA 8:1), d=128, causal, layout sbhd, inputs q/k/v/o/dO E4M3 with per-tensor
descales, stats = LSE (natural log) fp32 [b,h,s]. Outputs dQ/dK/dV E4M3 with per-tensor scales + amax (TE delayed
scaling), plus amax_dP. P and dS quantized to E4M3 with scale_s / scale_dP (same quantization points as cuDNN).

## FA3 bf16 bwd structure (hopper/, config for hdim128 causal sm90)
kBlockM=64 (q rows per iteration), kBlockN=128 (kv rows per CTA, resident), 1 producer WG (warp 0: TMA loads of Q/dO/LSE/dPsum
per m_block + K/V once; warp 1: TMA bulk-reduce-add of dQaccum fp32 to gmem), 2 MMA WGs (256 threads, 240 regs).
SdP_swapAB=true: S^T = K Q^T computed so the accumulator holds S^T/P^T/dS^T (kBlockN x kBlockM) per WG (each WG owns 64 kv rows?
no: AtomLayoutMSdP=1 -> both WGs split kBlockM? see TileShapeAtomSdP = <kBlockN, kBlockM/1, d> with AtomLayout <2,1,1>: the 2 WGs split
the kBlockN=128 rows, 64 each). GEMMs per iteration:
  1. S^T   = K   . Q^T   (SS: A=K smem K-major, B=Q smem K-major)        128x64x128
  2. dP^T  = V   . dO^T  (SS)                                            128x64x128
  3. dV   += P^T . dO    (RS: A = P^T from registers, B = dO^T view, MN-major smem)   128x128x64
  4. dQ    = dS  . K     (SS: A = dS smem (written by STSM.T), B = K^T view MN-major)  64x128x128  -> fp32 smem -> bulk-reduce-add to gmem
  5. dK   += dS^T . Q    (RS: A = dS^T registers, B = Q^T view MN-major)             128x128x64
Pre-kernel: dPsum = rowsum(dO o O), LSE*log2e, zero dQaccum. Post: dQaccum*scale -> bf16 dQ; GQA: dK/dV accum fp32 (atomics
across the 8 q heads) -> convert kernel.

## What FP8 changes (wgmma FP8 needs BOTH operands K-major in smem; A may come from registers)
  GEMM 1,2: fine (K,Q,V,dO tiles are K-major over d).
  GEMM 3: B = dO as [N=d, K=kBlockM] -> needs dO^T (d x 64) materialized K-major. A = P^T regs: FP8 RS needs the accumulator->A
          register permutation (utils.h permute_Aregs_fp8, from FA3's FP8 fwd) and FP8 conversion with scale_s.
  GEMM 4: A = dS [64 x 128] K-major over kBlockN: registers hold dS^T, so the store is a transposed 8-bit store
          (STSM.T works on 16-bit; use FA3-fwd's LDSM.T + byte_perm + STSM trick, or write dS^T K-major and swap GEMM roles:
          compute dQ^T = K^T . dS^T instead: A = K^T? no, A must be K-major over kBlockN too -> K^T materialized once per CTA
          (128x128, cheap, K resident). Then dQ^T (d x 64) accumulates in regs, same as now but transposed -> dQaccum layout
          transposed (fine, postprocess adapts).  => choose: dQ^T = K^T(A, smem, K-major over n) . dS^T(B, smem, K-major over n).
          dS^T written from the accumulator with plain STSM (non-transposed) as E4M3: 128x64 tile, K-major over kBlockN? The accumulator
          tile is (kBlockN rows x kBlockM cols) per WG; "K-major over kBlockN" for B[K=kBlockN, N=kBlockM] means contiguous along
          kBlockN = along the accumulator's ROW index -> that IS a transposed store. Alternative: make dS^T the A operand of
          dQ^T? A[M=d? no]. Resolve in prototype: either the LDSM.T/byte-perm trick (proven in FA3 FP8 fwd for V) or an
          extra smem transpose pass by the producer WG.
  GEMM 5: B = Q as [N=d, K=kBlockM] -> needs Q^T (d x 64) K-major. A = dS^T regs (permute_Aregs_fp8 + convert with scale_dP).
  Transposes per m_block iteration: Q^T, dO^T (64x128 FP8 = 8 KB each). Done by the producer warpgroup's idle warps 2-3
  (after TMA lands, before releasing the stage to consumers): adds one pipeline hop but keeps MMA WGs free. K^T once per CTA.
  Scales: dV = descale_s*descale_dO * acc; dK = descale_dP*descale_q * acc * softmax_scale; dQ = descale_dP*descale_k * acc * softmax_scale.
  amax: |dS| per thread -> warp reduce -> atomicMax(amax_dP) per CTA; dQ/dK/dV amax in the convert kernels. Output E4M3 via scale_*.
  Preprocess: O, dO are E4M3 -> dequantize with descale_o, descale_dO when forming dPsum.
  P quantization: P in [0,1]; scale_s from TE (amax history). dS: scale_dP.
## Numerics parity with cuDNN/TE
Same FP8 quantization points (inputs, P, dS, outputs E4M3 per-tensor); fp32 accumulation; softmax recompute in fp32 from LSE.
## Integration
Add-on hook on transformer_engine.pytorch.attention.dot_product_attention.backends.FusedAttnFunc.backward (NTU_ATTN_BWD=1): call
our kernel with the same (q,k,v,o,dO fp8, stats, quantizers) and return dq/dk/dv Float8Tensors + amax updates as TE expects.
## Milestones
M1 bf16 FA3 bwd standalone at our shape: time + tensor-busy (go/no-go input).            [files: fa3_bwd_bf16.cu, bench_fa3_bwd.py]
M2 FP8 prototype: Q^T/dO^T/K^T transposes + FP8 GEMMs 1,2,3,5 (dQ via fp32 path first); correctness vs fp32 ref.  go if >=45% busy.
M3 dS store + dQ GEMM in FP8; scales/amax; GQA convert kernels -> E4M3 outputs.
M4 TE hook, 2-GPU equivalence vs cuDNN (same quantization), 16-GPU A/B, full convergence run.
