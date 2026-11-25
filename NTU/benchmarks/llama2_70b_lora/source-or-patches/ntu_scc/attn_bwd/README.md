# Hand-written Hopper FP8 flash-attention backward (`NTU_ATTN_BWD=1`)

Replaces cuDNN's FP8 attention backward (the only compute kernel in the step with real headroom: 36% tensor-pipe
utilization vs 85-93% for the dense GEMMs) with a kernel derived from FlashAttention-3's bf16 backward, for the
training shape: sbhd, causal, head dim 128, GQA, FP8 delayed scaling (HYBRID: E4M3 activations, E5M2 gradients).

| | cuDNN FP8 bwd | this kernel |
|---|---|---|
| isolated, 1 GPU @ 400 W (incl. pre/post kernels) | 3.05-3.5 ms | 2.73 ms |
| tensor pipes busy (main kernel) | 36% | 41% |
| dq/dk/dv error vs fp32, E5M2 grads | 5.95 / 5.94 / 3.71 e-2 | 5.3 / 5.3 / 2.6 e-2 |
| inside TE's DotProductAttention (module backward) | 3.15 ms | 2.90 ms |
| **16-GPU training step** (job 3578 vs 3511) | 2.368 s | **2.278 s (-3.8%)** |

## How it works
- `fp8/mainloop_bwd_fp8.hpp` is a patched copy of FA3's `mainloop_bwd_sm90_tma_gmma_ws.hpp`. FP8 wgmma needs both operands
  K-major in smem, so the caller passes pre-transposed E4M3/E5M2 copies of Q, dO and K (`fp8/transpose_fp8.cu`, 36 us per
  34 MB); every operand is then a plain TMA load. P^T and dS^T feed the dK/dV GEMMs from registers (`permute_Aregs_fp8`);
  dS is the only in-kernel transpose (register shuffles + `stmatrix.trans`, `fp8_bwd_utils.hpp`).
- P is quantized to E4M3 with the S quantizer's scale, dS to the dP quantizer's dtype/scale (E5M2 under HYBRID), exactly
  where cuDNN quantizes. All scale multiplies are folded into the softmax constants; the causal mask runs only on diagonal blocks.
- dQ partial sums are reduced in fp16 (`cp.reduce.async.bulk.add.noftz.f16`, half the traffic of fp32), pre-scaled to a 4096
  target so partials fit fp16. dK/dV accumulate in fp32 (GQA atomics). `fp8/postprocess_fp8.hpp` converts to bf16 (what TE
  would dequantize to anyway, so TE's dequantize kernels disappear) and writes amax_dP / amax_dQKV into TE's quantizers.
- `../addons/ntu_attn_bwd.py` hooks `transformer_engine...backends.fused_attn_bwd`; anything outside the supported shape falls
  back to TE. Enabled by `NTU_ATTN_BWD=1` through the same site-packages autoload as the other add-ons.

## Build
```bash
A=/work/benchmark/MLPerf/attnkernel; mkdir -p $A && cd $A
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git   # tested at 94e22c9
cp -r <repo>/ntu_scc/attn_bwd/fp8 $A/ && cd $A/fp8 && ./build.sh        # nvcc 13.2 inside the training container, ~25 s
```
Never rebuild `libfa3_bwd_fp8.so` while a training job is queued or running (it is dlopen'ed at the first backward);
build with `OUT=lib_next.so`, test, then copy.

## Tests
- `bench_fa3_bwd_fp8.py` (1 GPU): correctness vs an fp32 reference using the same FP8 inputs, cuDNN comparison, sustained timing.
  `GRAD_E5M2=1 OUT_BF16=1` is the training configuration.
- `tests/test_attn_hook.py` (1 GPU): TE DotProductAttention, cuDNN path vs hooked path vs fp32.
- `bench_fa3_bwd.py` / `fa3_bwd_bf16.cu`: the FA3 bf16 baseline (73% tensor-busy) that motivated the port.
- `nsys_kernel_metrics.py`: tensor-pipe utilization per kernel from an `nsys --gpu-metrics-devices` export.
- `tests/attn_cudnn_autotune.py`: shows cuDNN offers exactly one FP8 backward engine for this shape.

## What was tried and did not help (isolated kernel)
3 TMA stages (0), removing exp2 (0), removing the cross-warpgroup barrier (0), red.global atomics for dQ (slower).
Removing the whole dQ path is -28%: its GEMM plus the partial-sum traffic are the remaining cost.
