# RACKlette (ETH Zurich): llama2_70b_lora with tensor-parallel communication overlap

Base: [mlcommons/training_results_v6.0](https://github.com/mlcommons/training_results_v6.0) at `eabf23a`,
`Dell/benchmarks/llama2_70b_lora/implementations/nemo` (NeMo / Megatron-Bridge).

## Contents

| File | What it is |
|---|---|
| `source-or-patches/0001-...patch` | Passes `TP_COMM_OVERLAP` and the overlap config to the model (7 lines in `train.py`) |
| `source-or-patches/0002-...patch` | Keeps the LoRA gradients correct when the overlap is on (11 lines in `train.py`) |
| `source-or-patches/te-bulk-wgrad-repro.sh` | Reproduces the TE bulk wgrad bug with TE's own layer test (Change 2, item 1) |
| `config.sh` | Our system: 2 nodes x 8 H200 NVL, TP=4, SP on, DP=4, GBS 8, FP8, CUDA graphs, overlap on with CUDA IPC and atomic GEMM off |
| `run_and_time.sh` | Unchanged copy from the base implementation |

Both patches are also on the branch `scc26-tp-comm-overlap-v2` of
[techboy-coder/training_results_v6.0](https://github.com/techboy-coder/training_results_v6.0/tree/scc26-tp-comm-overlap-v2).

To apply, from a clone of training_results_v6.0 (`git am` needs `user.name` and `user.email` set; `git apply`
with the same `--directory` works too):

```
git clone https://github.com/mlcommons/training_results_v6.0 && cd training_results_v6.0 && git checkout eabf23a
P=<this repository>/RACKlette/benchmarks/llama2_70b_lora
git am --directory=Dell/benchmarks/llama2_70b_lora/implementations/nemo $P/source-or-patches/0*.patch
cp $P/config.sh Dell/benchmarks/llama2_70b_lora/implementations/nemo/
```

Without `--directory`, git looks for `train.py` at the repository root. Then launch as in the Dell README
(`source config.sh` and `sbatch run.sub`).

Our runs used bare-metal Python venvs, not the implementation's container (`nvcr.io/nvidia/pytorch:26.04-py3`):
TE 2.14 (`f031cf87`), torch 2.12.1 (cu130), CUDA 13.2 on A100 and 13.1 on GH200.

## Change 1: pass `TP_COMM_OVERLAP` to the model

`conf/megatron_gpt_peft_tuning_config.yaml` reads `TP_COMM_OVERLAP` into `model.ub_tp_comm_overlap` and loads
`conf/tp_overlap/<GPU_ARCH>100tp<TP>mbs<MBS>.yaml` into `model.ub_tp_comm_overlap_cfg`, but `train.py` never passes
either value to `GPTModelProvider`. With `TP_COMM_OVERLAP=True` the run is the same as without it, and Transformer
Engine's `initialize_ub` is never called. The first patch passes both values. With `TP_COMM_OVERLAP` unset nothing
changes.

## Change 2: correct LoRA gradients with the overlap on

Once the overlap reached Transformer Engine (patch 0001 only), the training loss after one optimizer step no longer
matched the run without overlap: it was off by 5.5e-3 to 7.0e-3, while two runs with the overlap off (patch 0001
only; both patches) differed from it by 1.6e-5 and 5.3e-5. The second patch handles two things: a TE regression that hits frozen
weights such as the LoRA base weights, and a loss deviation with the ring_exchange reduce-scatter that we measured
but could not trace. The second part is a precaution and does nothing for our config.

1. Bulk wgrad overlap with frozen weights. For a frozen weight, TE skips the wgrad GEMM. With bulk wgrad overlap,
   `LayerNormLinear` runs the reduce-scatter of the dgrad inside that GEMM, so the input gradient is never reduced.
   TE raises no error. Here this hits `linear_qkv`; with Megatron-Core's default layer spec it would
   also hit `linear_fc1`, but this implementation runs the MLP through TE's op fuser (`USE_TE_OPS=1`), which keeps
   bulk wgrad off for frozen weights. This is a TE regression: TE PR 702 (`b0f65354`, 2024-03-05) turned bulk wgrad
   off for frozen weights in both `layernorm_linear.py` and `layernorm_mlp.py`, and the TE 2.0 code drop (`544dd14b`,
   2025-02-07) removed the check from `layernorm_linear.py`. `LayerNormMLP` still has it (TE 2.14:
   `ctx.ub_bulk_wgrad = ctx.fc1_weight_requires_grad and ctx.ub_bulk_wgrad`). `LayerNormLinear` has no such check,
   both in TE 2.14 (`f031cf87`) and on TE main as of 2026-11-08 (`c1c2d4d2`). We have not reported it to TE yet.
   `source-or-patches/te-bulk-wgrad-repro.sh` shows it with TE's own
   `tests/pytorch/distributed/run_layer_with_overlap.py` plus an option to freeze all weights: on 4x A100 the input
   gradient of a frozen `LayerNormLinear` comes out at 1/4 of the reference (1/TP), while trainable weights and a
   frozen `LayerNormMLP` match. The patch sets `tp_comm_bulk_wgrad=False`.
2. Ring-exchange reduce-scatter in the forward of `linear_proj` / `linear_fc2`. With `method: ring_exchange` for
   both `proj_fprop` and `fc2_fprop` (`b100tp4mbs1`) and bulk wgrad already off, the loss after one optimizer step was
   still 3.0e-3 off (3.1e-3 with CUDA graphs off; 8.1e-4 with ring_exchange for `proj_fprop` only). With
   `h100tp4mbs1` (pipeline for both) it matched. The shipped `b100tp2mbs1`, `b100tp4mbs1` and `h100tp2mbs1` yamls
   use ring_exchange there. For those the patch keeps the overlap for all other GEMMs and uses the plain
   reduce-scatter for these two (`tp_comm_overlap_rs=False`). `h100tp4mbs1` (our config) and `h100tp8mbs1` use
   pipeline, so this part does nothing for them. We have not found the cause. In TE's layer test (single modules and
   4 stacked layers, frozen and trainable, BF16), ring_exchange and pipeline are as close to an FP32 reference as
   plain NCCL. That test has no CUDA graphs and no full model, so it may not reach the code path that differs.

With the overlap off, the patches change nothing: Megatron-Core reads `tp_comm_bulk_wgrad` and `tp_comm_overlap_rs`
only when `tp_comm_overlap` is set.

## Atomic GEMM and multicast

`h100tp4mbs1` turns atomic GEMM on for the `proj_fprop` and `fc2_fprop` reduce-scatter by default
(`NVTE_UB_ATOMIC_GEMM_RS_PROJ/FC2`, default 1). TE 2.14 supports atomic GEMM only when built with CUDA 12.2 to 12.x
(`cublaslt_gemm.cu`: "Atomic GEMM requires CUDA >=12.2.0 and <13.0.0"). The implementation's container and all our
runs use CUDA 13. With FP8 and atomic GEMM on, the run fails at the first step, so `config.sh` sets both to 0.

Userbuffers use CUDA multicast by default, and `initialize_ub` raises an error when the GPUs do not support it unless
`UB_SKIPMC=1` is set. Without `UB_SKIPMC=1`, `initialize_ub` stopped with that error on our A100 PCIe nodes and on
GH200. H200 NVL nodes have no NVSwitch either, so `config.sh` uses `UB_SKIPMC=1` (CUDA IPC). We have not tested
multicast on H200 NVL.

`GPU_ARCH` defaults to `b`, which selects the Blackwell yamls (`b100tp<TP>mbs<MBS>.yaml`), so on Hopper set
`GPU_ARCH=h` (as in `config.sh`).

## Tests so far

4x A100 80GB PCIe (CUDA IPC with `UB_SKIPMC=1`), BF16, TP=4 with SP, full activation recompute, seed 1234. Train
loss at step 2, i.e. after one optimizer step (the step 1 loss is before any update):

| Code | Overlap config | Loss at step 2 | Difference to base |
|---|---|---|---|
| base | off | 0.4713242 | |
| both patches | off | 0.4712713 | 5.3e-5 (step 1 loss bit-identical) |
| patch 1 only | h100tp4mbs1 | 0.4768726 | 5.5e-3 |
| patch 1 only | b100tp4mbs1 | 0.4783207 | 7.0e-3 |
| patch 1 + `tp_comm_bulk_wgrad=False` | b100tp4mbs1 | 0.4743286 | 3.0e-3 |
| both patches | h100tp4mbs1 | 0.4713831 | 5.9e-5 |
| both patches | b100tp4mbs1 | 0.4713475 | 2.3e-5 |
| both patches (second run) | b100tp4mbs1 | 0.4712431 | 8.1e-5 |

All `h100tp4mbs1` runs had atomic GEMM off (`NVTE_UB_ATOMIC_GEMM_RS_PROJ=0`, `NVTE_UB_ATOMIC_GEMM_RS_FC2=0`). The other
bisection runs are summarized in Change 2, item 2.

Step time on Hopper: 1 node with 4x GH200 96 GB (CSCS Alps, Daint), TP=4, SP on, DP=1, MBS 1, GBS 8, FP8 (BF16
weights), CUDA graphs, no activation recompute, 60 steps, seed 1234, `GPU_ARCH=h` (`h100tp4mbs1`), `UB_SKIPMC=1`.
Both runs on the same node; median step time over the 30 logged steps:

| Overlap | Step time (range) | Train loss at step 60 |
|---|---|---|
| off | 8.151 s (8.102 to 8.180) | 0.17176 |
| on, atomic GEMM off | 7.829 s (7.806 to 7.873) | 0.17135 |

Step time is 4.0% lower with the overlap on. With the default atomic GEMM the run stops at the first step with the
CUDA 13 error above.

## Open: FP8 loss difference on GH200

With the overlap on, the FP8 training loss on GH200 differs from the runs without it. A second run without overlap,
on another node, differs from the first by 4e-4 on average (at most 1.6e-3) over the 30 logged steps. The run with
overlap is lower than both at every logged step. Against their mean the gap is largest early (7.5e-3 at step 4),
below 3.5e-3 from step 14, and 2.7e-4 to 1.0e-3 from step 26 to 60. In BF16 on A100 the patched overlap matched within
1e-4, but those runs also differ in hardware and recompute, so we cannot say which part causes the gap. FP8 is our
main suspect: `h100tp4mbs1` sets `fp8_buf: 1` for `proj_fprop`, which gives that reduce-scatter an FP8 buffer in TE.

Not tested yet: CUDA multicast, our 2-node H200 NVL system, and a full run to the 0.925 target with the overlap on.
Our competition runs use the overlap only if that full run converges and passes the RCP check; otherwise we run with
`TP_COMM_OVERLAP=False`.
