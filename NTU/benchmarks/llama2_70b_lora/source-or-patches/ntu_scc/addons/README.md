# NTU-SCC kernel add-ons (no benchmark-source change)

All of these load through the same site-packages hook as the DeepGEMM dgrad
(`zz_dg_dgrad.pth` -> `dg_dgrad_autoload.py`). Each one is inert unless its
environment variable is `1`. A meta-path hook waits until the target module has
finished importing, then re-binds functions/methods in it. `train.py` and the
NeMo / Megatron / TE sources inside the container stay stock.

| flag | file | what it replaces | numerics |
|---|---|---|---|
| `NTU_FUSED_LORA` | `ntu_lora_fused.py` | LoRA output path: `addmm(base, h, B^T, alpha=scale)`; linear_proj has no 64 MB hidden->sequence all-to-all | proj grads differ by summation order (3e-3 rel); qkv bit-identical |
| `NTU_LORA_QKV_LOCAL` | `ntu_lora_fused.py` | linear_qkv LoRA on the local sequence shard, so TE all-gathers the LN output in FP8 instead of bf16 | same math; dropout masks independent per TP rank |
| `NTU_FP8_DY_AG` | `ntu_te_comm.py` | te.ops ReduceScatter backward (MLP fc2 dy): quantize, then FP8 all-gather, instead of a bf16 all-gather then quantize | bit-identical |
| `NTU_DSWIGLU_DG` | `ntu_dg_quant.py` | Triton dSwiGLU writing DeepGEMM's E4M3 1x128 operand directly (replaces TE dswiglu to E5M2 + requant) | more accurate (FC1 dx 2.3% vs 5.8% rel err) |
| `NTU_DG_GRADQ` | `ntu_dg_quant.py` | the same one-pass E4M3 1x128 quantization for the qkv / proj / fc2 dgrad operands | more accurate |
| `NTU_ROPE` | `ntu_rope.py` | Triton fused QKV RoPE fwd/bwd | bit-identical |
| `FP8W_T_NOFWD` | `dg_dgrad.py` | don't build frozen FP8 weight transposes in the forward (ONDEMAND drops them anyway) | bit-identical |

## Why the LoRA qkv input gradient is absent in both stock and add-on

Megatron-Bridge wraps `linear_qkv` with `return_layernorm_output_gathered=True`.
In TE 2.14, `_LayerNormLinear.backward` only adds `grad_outputs[1]` (the
gradient of the returned LN output) when the output is NOT gathered
(`layernorm_linear.py:960`). So in the shipped stack the LoRA-qkv branch never
contributes to dx. `kernels/test_ln_grad.py` demonstrates this. The add-ons
keep that behaviour: they skip computing the discarded gradient rather than
"fixing" it.

## Tests
`tests/`: `run_tests.sh <ntasks> <script>` runs one in the training container on an NVLink pair.

## Results (16x H200 NVL, 400 W/GPU, seed 1234, full runs to the 0.925 target)

| job | configuration | step | TTT | eval @3072 |
|---|---|---|---|---|
| 3465 | DeepGEMM add-on only, **USE_TE_OPS unset** (launcher bug) | 2.599 s | 18.65 min | |
| 3479 | + NTU_FUSED_LORA (no TE ops) | 2.536 s | 18.21 min | 0.9241 |
| 3502 | + QKV_LOCAL, ROPE (no TE ops, so the te.ops add-ons were inert) | 2.50 s | 17.95 min | 0.9248 |
| **3511** | **all add-ons + USE_TE_OPS=1** | **2.365 s** | **17.02 min** | **0.9239** |

Each row is n=1. Whether a run stops at 3072 or 3456 samples is a seed coin flip.

## What does NOT pay off here, and why

- **The step is energy-bound.** All 16 GPUs average 377-405 W with `sw_power_cap` active, so time ~ energy / 400 W.
  Removing low-power phases (communication, waits) saves much less time than their duration: in 3479 -> 3502, about
  1/3 of the kernel time removed showed up as step time.
- Copy-engine all-gather / reduce-scatter (symmetric memory): only −11% / −5% vs NCCL on an idle pair, and NCCL with
  64 CTAs gets most of that. Not built.
- Async-TP (`torch` symmetric-memory fused GEMM + collective): ≈0 on FC1/FC2 at 400 W.
- Triton dropout + LoRA-A: limited by Philox RNG throughput, ~0.6% at best (`ntu_lora_dropout_a.py` is not shipped).
