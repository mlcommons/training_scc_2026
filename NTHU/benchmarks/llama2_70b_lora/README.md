# MLPerf Training v6.0 — Llama 2 70B LoRA (closed division)

SCC26 student cluster submission: **14 × NVIDIA GH200 96GB, one GPU per node**,
pure data parallelism (TP = PP = CP = 1), GBS = 14.

System description: `../../systems/student_system.json`.
Results: `../../results/student_system/llama2_70b_lora/result_0.txt` … `result_9.txt`.

## What was run

NVIDIA's MLPerf v5.0 `llama2_70b_lora` implementation (NeMo / Megatron-Core in the
`nvcr.io/nvidia/pytorch:25.04-py3` container, run with Apptainer), plus the patches in
`source-or-patches/`. The configuration of all ten runs is in `config.sh`; one timed
run is `run_and_time.sh`.

Two things are worth pointing out, because they are what make a 70B model trainable on
a 96 GB GPU without tensor or context parallelism:

1. **The frozen base weights are stored and computed in FP8.** 69 G parameters of bf16
   would be ~138 GB. `FP8=True` comes unmodified from NVIDIA's own `config_common.sh`,
   so this is an exercised path, not something invented here; the bf16 checkpoint is
   restored into FP8 parameters. A run's own log records
   `lowest_numerical_precision_in_linear: fp8`.
2. **Saved activations are offloaded to Grace host memory instead of being recomputed.**
   With `CP=1` the per-GPU sequence is the full 8192 tokens, and the activations do not
   fit beside the weights. Rather than pay full recompute, 79 of the 80 layers' saved
   tensors are moved over NVLink-C2C (900 GB/s) into the node's 480 GB of LPDDR5X
   (`source-or-patches/cpu_offload_patch.py`). Measured peak HBM is ~76.9 GiB, against
   ~79.6 GiB for full recompute on the same code, and it is faster.

Because LoRA freezes the base weights there is no optimizer state to shard and no base
gradient to reduce, so the only cross-node traffic per step is the LoRA adapter gradient
all-reduce — tens of MB at `lora_rank=16` over `qkv_proj,o_proj`. That is what makes
one-GPU-per-node data parallelism the right shape here, and why `TP`/`CP` groups larger
than 1 (which would cross InfiniBand) were not used.

## Reproducing

```bash
# 1. dataset and starting checkpoint (SCC26 copies)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/scc-2026-scrolls-gov-report-8k.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/scc-2026-llama-2-70b-fused-qkv-mlperf.uri

# 2. container and patched implementation
scripts/build_container.sh
nvidia/setup.sh            # clones NVIDIA v5.0 and applies source-or-patches/

# 3. one run
SEED=1234 ./run_and_time.sh

# 4. all ten, as submitted (one Slurm dependency chain, one output dir per run)
CERT=1 TEAM=<team> SEEDS='1234 2468 4680 5678 5791 3579 6802 7913 8024 9135' \
  scripts/run_competition.sh
```

## Convergence and compliance

Target: evaluation loss ≤ 0.925, first reached at an evaluation that is a multiple of
392 samples, with the first 3 evaluations skipped (`FLOOR(0.125 × 14 + 2)`).

The ten runs converge at 3528–4312 samples. Checked with the MLPerf logging package at
ruleset 6.0.0:

```bash
python -m mlperf_logging.rcp_checker      ../../results/student_system/llama2_70b_lora \
                                          --rcp_usage training --rcp_version 6.0.0 --verbose
python -m mlperf_logging.package_checker  <team> training 6.0.0 --werror
bash sources/logging/scripts/verify_for_v6.0_training.sh <team>
```

Rank 0 alone writes the MLPerf log, and every run has its own output directory: on NFS,
`O_APPEND` from two clients is not atomic, and a second writer silently overwrites the
first one's tail.

## Power

SCC26 allows 10 kW for the whole cluster, measured at 1-second intervals, during the
runs and between them. These runs hold the Grace+Hopper module cap at 580 W per node
and the chassis fans at a fixed 30%, with the CPU governor at `schedutil` and a 2.0 GHz
ceiling. The power settings are not part of the MLPerf configuration, but they are part
of how these times were produced, so they are recorded in `config.sh`.
