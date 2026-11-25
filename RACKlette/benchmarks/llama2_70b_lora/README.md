# RACKlette (ETH Zurich): llama2_70b_lora, unmodified vendor code

Code: [mlcommons/training_results_v6.0](https://github.com/mlcommons/training_results_v6.0) at `eabf23a`,
`Dell/benchmarks/llama2_70b_lora/implementations/nemo` (NeMo / Megatron-Bridge), with no changes. Our checkout has no
modified tracked files. No patches are applied: our launcher can apply a patch to a per-run copy of the code, but only
when `PATCH` is set, and it was not set for these runs (no `PIPE_PATCH` line in the logs).

TP communication overlap is off (`TP_COMM_OVERLAP=False`). Our overlap patches (PR #6) were not used for these runs.

## Contents

| File | What it is |
|---|---|
| `config.sh` | Our settings: 2 nodes x 8 H200 NVL, TP=4, SP on, DP=4, GBS 8, FP8, CUDA graphs, overlap off |
| `run_and_time.sh` | Unchanged copy from the base implementation |
| `source-or-patches/NO-PATCHES.txt` | States that the benchmark code is unmodified |
| `source-or-patches/launcher/` | The scripts that launched these runs (see below) |

## How the runs were launched

Bare metal, no container. One Slurm job held both GPU nodes for all 10 runs. `mlperf-loop.sh` ran the 10 trainings
back to back, each with its own seed (10 distinct seeds, drawn before the job started). Each training is one call of
`run-cert.sbatch`, which:

1. sources `pipeline.env` (paths, pinned source commits, software versions), then the vendor `config_common.sh`, then
   applies the values in `config.sh` here (the log of every run prints them in its `PIPE_CONFIG` line);
2. drops the model and dataset from the page cache on both nodes with `clear_cache.py` (`posix_fadvise`, no root) and
   logs the `cache_clear` event;
3. starts 8 ranks per node with `srun`; each rank runs `task.sh`, which sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and
   `LOCAL_WORLD_SIZE` from Slurm and runs `python -u train.py`. That is the same command `run_and_time.sh` runs when
   `LOCAL_WORLD_SIZE` > 1.

`run-cert.sbatch` is our pipeline's `run.sbatch` with four changes so that 10 runs fit in one job: a separate log
directory per run, a different `MASTER_PORT` per run, and a hang guard that stops only the current run, not the job,
and is stopped when the run ends.

## Software

Python venv built with uv, Python 3.12.15:

- PyTorch 2.12.1 (cu130), CUDA 13.2, driver 615.71.09
- Transformer Engine 2.14.0 (`f031cf87`), built from source against CUDA 13.2. The build uses cudnn-frontend v1.25.0
  headers instead of the bundled v1.21.0: v1.21.0 raises an error when both `libcudart.so.12` and `libcudart.so.13`
  get loaded (our nodes' library cache also lists CUDA 12), and v1.25.0 is the first version that honours
  `CUDNN_FRONTEND_CUDART_LIB_NAME`. It is a header-only build dependency; no benchmark code changes.
- Megatron-Bridge `68d8bcc` (tag 26.04-alpha.rc1, as in the vendor Dockerfile), Megatron-LM `9978968`
- cuDNN 9.21.1.3, cuBLAS 13.4.1.3, NCCL 2.30.4, Open MPI 5.0.6
- Model `michal1270/llama2_70b_megatron_bridge` at `2ae1be8`, GovReport dataset

## System settings during the runs

- 2 nodes (`sbrinz`, `sbrinz1`), 8 H200 NVL each, two NVLink groups of 4 GPUs per node, no NVSwitch
  (so `NCCL_NVLS_ENABLE=0`). TP=4 keeps each tensor-parallel group inside one NVLink group.
- GPU power limit 442 W on all 16 GPUs. The job sets it once, before the first run's `run_start` (the timed part),
  reads it back on every GPU (the readback line is in `cert-1-mlperf.run`), and does not change it for the rest of
  the 10 runs.
- GPU fans (zone 1) at 50 % on both nodes.
- The two CPU-only nodes (`sbrinz4`, `sbrinz5`) were suspended for the whole job.

The MLPerf log fields `submission_org` and `submission_platform` hold placeholder values: the logging library reads
them from `MLPERF_SUBMISSION_ORG` and `MLPERF_SUBMISSION_PLATFORM`, which our launcher does not set. No checker uses
them.
