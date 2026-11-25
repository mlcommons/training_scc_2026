# MLPerf Training llama2_70b_lora - Supernova

Closed division. Llama 2 70B LoRA fine-tune on SCROLLS GovReport (8k context), reference
`training/llama2_70b_lora`; target validation loss 0.925.

## System

2 nodes x 8 NVIDIA H200 NVL (16 GPUs), InfiniBand between the nodes. Parallelism TP1 x CP2 x DP8,
global batch 8 (micro batch 1, one microbatch per GPU).

## Files

| file | purpose |
|---|---|
| `run_and_time.sh` | one timed run: `SEED=<seed> NODELIST=<node0>,<node1> ./run_and_time.sh` |
| `config.sh` | every environment variable of the run (parallelism, FP8 recipe, hyperparameters, optimizations) |
| `source-or-patches/patches/` | the patch set against each pinned upstream commit; `apply_all.sh <empty dir>` rebuilds the patched trees |
| `source-or-patches/h200-tuning/` | the launcher (`run_rcp_ref8.sh` -> `run_rcp_het.sh`), setup scripts and `optimization.md` |
| `source-or-patches/COMMIT.txt` | the exact repository commit |

## Reproduce

1. **Build** (each node, shared filesystem): `source-or-patches/h200-tuning/setup/setup_sc26.sh` builds the venv,
   Megatron Bridge, TransformerEngine, FlashAttention 3, DeepGEMM and the SwiGLU library for sm90a
   (`JOBS` = build parallelism).
2. **Data and model**: `source-or-patches/reference/scripts/download_data.sh` and the model conversion of the
   reference README; `setup/stage_checkpoint_ram.py` stages the converted checkpoint.
3. **Run** from the first node, which must be one of `NODELIST`:
   ```
   for s in $(seq 12552 12561); do SEED=$s NODELIST=node0,node1 ./run_and_time.sh; done
   ```
   Every run drops the page cache first and writes `results/<run>/node0.log` (the MLPerf log; the submitted
   `result_N.txt` files are these logs verbatim).
4. **CP-shard calibration**: the first run on a new machine or power setting measures the per-GPU split of the
   context-parallel shards before training, caches it in `cp_shard_cache/`, exits, and the launcher starts the
   run again on the cache (the calibration log is kept in `results/<run>-cal`). This happens before `run_start`
   and only once; every later run reads the cache. `CP_SHARD_WEIGHTS=` (empty) runs equal shards instead.

## GPU settings

Set on all 16 GPUs for every run:

| setting | value | command |
|---|---|---|
| power limit (TDP) | 450 W | `nvidia-smi -pl 450` |
| vboost | 2 | `nvidia-smi boost-slider --vboost 2` |
| SM clock lock | 1200 MHz | `nvidia-smi -lgc 1200` |

## Hyperparameters

AdamW, LR 5e-4, linear warmup 32 steps, cosine decay to 0 over 800 steps, gradient clip 0.3, LoRA rank 16,
alpha 32. Evaluation every 384 samples after the first three intervals (first at 1536 samples), on the full
validation set.

## Optimizations

FP8 delayed scaling with FlashAttention 3 FP8 attention, DeepGEMM, fused SwiGLU and RoPE kernels, a copy-engine
context-parallel all-to-all, full-iteration CUDA graphs, and per-GPU balanced context-parallel shards. Each one is
described, with its measurement, in `source-or-patches/h200-tuning/optimization.md` and implemented in
`source-or-patches/patches/`.
