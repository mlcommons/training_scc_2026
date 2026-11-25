# Patch set: llama2_70b_lora

Every change our team made to the software stack, expressed as patches against
the exact upstream commit each component was taken from. What each optimization
does, and the measured gain, is in [`h200-tuning/optimization.md`](../h200-tuning/optimization.md); every
section there links to the hunk in these patches that implements it.

## Patches

| # | patch | upstream | base | files | lines |
|---|---|---|---|---|---|
| 01 | [`01-llama2_70b_lora-nemo.patch`](01-llama2_70b_lora-nemo.patch) | mlcommons/training_results_v6.0, `NVIDIA/benchmarks/llama2_70b_lora/implementations/` | `eabf23a0` | 32 | +9145 / -117 |
| 02 | [`02-megatron-bridge.patch`](02-megatron-bridge.patch) | NVIDIA-NeMo/Megatron-Bridge | tag `26.09-beta.rc1` (`634ec709`) | 4 | +49 / -10 |
| 03 | [`03-transformer-engine/`](03-transformer-engine/) (9 patches, apply in file order) | NVIDIA/TransformerEngine | tag `v2.20.2` | 24 | +1775 / -250 |
| 04 | [`04-flash-attention.patch`](04-flash-attention.patch) | Dao-AILab/flash-attention | `e9515d5d` | 21 | +1896 / -274 |
| 05 | [`05-deepgemm.patch`](05-deepgemm.patch) | deepseek-ai/DeepGEMM | `057ca596` (Public release 26/09/30) | 7 | +1811 / -7 |
| 06 | [`06-nccl.patch`](06-nccl.patch) | NVIDIA/nccl | tag `v2.32.3-1` | 5 | +47 / -47 |

- **01**: the benchmark code (`train.py`, `utils.py`, callbacks) plus our new
  modules: the FA3 fp8 attention adapter, the CP all-to-all TMA transport and
  its wide layout copies, the TP collective transports, the DeepGEMM router,
  LoRA local/fused adapters, the fused post-RS norm, the fused RoPE and FP8
  quantize of the attention input written straight into the CP a2a send layout (its backward also writes the
  E5M2 qkv output gradient), the E5M2 cast of the attention output gradient into the CP a2a send layout, the fused residual add + RMSNorm + FP8 cast (and its backward add + FP8
  cast, and its backward RMSNorm folded into the residual-grad add + FP8 cast), the LoRA dropout folded into the LoRA-A GEMMs, the FP8 attention output fed straight into
  linear_proj, the rowwise-only qkv input cast with the pre-zeroed fc1 amax slot, the TP1 fused qkv eval
  path, SwiGLU fusion, and
  the calibrated FP8 checkpoint load with a resident FP8 transpose of the qkv and proj weights for
  their dgrad, and the uneven context parallel shards with their warmup calibration and per-rank
  resident fc2 transposes. It also adds
  the standalone SwiGLU kernel, `swiglu_fusion/swiglu.cu`, next to the `nemo`
  directory.
- **02**: eval-shape userbuffer sizing, attention-mask memoization, the
  early return of the optimizer success and grad-norm checks when the
  model-parallel group holds one rank, and the removal of the device sync
  after each evaluation iteration.
- **03**: the TransformerEngine series. It covers the fused post-RS norm seam,
  the userbuffer chunk provider and per-phase token adoption, the
  userbuffer receive count for SM-driven ring sends, the LoRA
  pre-reduce-scatter join and qkv userbuffer all gather, the CP head-chunk
  pipeline, the LoRA join fold, the so-live resolution, the transient weight
  transpose and the transport switch. The files are numbered in a working
  apply order.
- **04**: the FP8 (e4m3) sm90 backward kernel, PDL chaining, and FP8 O
  quantization in the forward epilogue.
- **05**: the per-tensor-scaled sm90 FP8 GEMM kernels and the host entry.
- **06**: enables the symmetric-memory TMA kernels of NCCL (ReduceScatter and
  AllReduce TMA load paths) for sm90, which upstream compiles only for
  compute capability 10.0 and newer. It is kept for a retest of the NCCL
  SymK item in `h200-tuning/docs/todo_speed.md`; the production recipe uses
  the NCCL from the venv wheel and does not build this patch.

Megatron-LM (the `3rdparty/Megatron-LM` submodule of Megatron-Bridge, pin
`6a366090`) has no changes, so it has no patch.

## Apply

```bash
patches/apply_all.sh /path/to/empty/dir
```

The script clones each upstream at its base commit, initialises the
submodules the builds need (TransformerEngine `3rdparty/{cutlass,googletest,nccl-extensions}`,
DeepGEMM `third-party/{cutlass,deep_jit}`, flash-attention `csrc/cutlass`) and applies with
`git apply`. Manual equivalent: `git apply <patch>` at the root of each
upstream checkout. The 01 paths are relative to the training_results_v6.0
root.

## Build notes

All kernels target sm90a.

- **DeepGEMM**: `pip install --no-build-isolation -e .`, then link
  `third-party/cutlass/include/{cutlass,cute}` into `deep_gemm/include/`
  (as DeepGEMM's own `develop.sh` does).
- **TransformerEngine**: build with `NVTE_FRAMEWORK=pytorch`,
  `NVTE_CUDA_ARCHS=90a` and `NVTE_WITH_USERBUFFERS=1`, and set
  `NVTE_UB_DEEPGEMM_ROOT` to the patched DeepGEMM checkout so the userbuffer
  chunk provider compiles against its headers.
- **flash-attention**: in `hopper/`, `pip install --no-build-isolation -e .`
  with `FLASHATTENTION_ENABLE_BWD_FP8=TRUE`.
- **SwiGLU kernel** (after TE, because it includes the TE common headers):
  in `implementations/swiglu_fusion/`, run
  `mkdir -p build && nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_90a -I <TE>/transformer_engine/common swiglu.cu -o build/libswiglu_fusion.so`.
  `swiglu_fusion.py` loads the library from that `build/` directory.
- **Megatron-LM**: `pip install --no-build-isolation -e .` in
  `3rdparty/Megatron-LM` of the Megatron-Bridge checkout
  (`git submodule update --init 3rdparty/Megatron-LM`).
