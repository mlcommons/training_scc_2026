# MLPerf Training Llama 2 70B LoRA — student_system (SegFAUlt)

Student Cluster Competition 2026 (SC26) submission package, MLPerf Training v6.0 rules (`mlperf-logging`, ruleset 6.0.0).

## System

2 nodes, **12 GPUs** NVIDIA H200 NVL 141GB on the team SegFAUlt cluster: all 8 GPUs of deuterium + GPUs 0-3 (one 4-GPU NVLink island) of tritium (Giga Computing G493-ZB2):
2x AMD EPYC 9965 (192 cores) and 768 GB DDR5 per node, two 4-way NVLink bridges per node, 2x NVIDIA ConnectX-7 InfiniBand 200 Gb/s per node,
NFS over RDMA from tritium, air-cooled, 10 kW power cap for the whole cluster. tritium's GPUs 4-7 are allocated but idle (hardware errors under load). Metadata: `systems/student_system.json` (full MLPerf v6.0 system description, plus the fields of the SCC
template).

Rank placement: `deuterium:8,tritium:4` (host:ranks, `RANK_LAYOUT` in `run_helma.sub`, `srun --distribution=arbitrary`); on a node running fewer ranks than it has GPUs, only its first GPUs are visible to CUDA and NCCL.

## Recipe (allowed knobs only)

| Knob | Value |
| --- | --- |
| Global batch size | 9 (3 micro-batches of 1 per data-parallel rank, DP=3) |
| Learning rate | cosine schedule, peak 0.0005, at most 800 steps |
| Parallelism | tensor parallel 4 + sequence parallel inside each 4-GPU NVLink island, data parallel 3 across islands |
| Precision | FP8 (Transformer Engine, delayed scaling), FP8 attention; CUDA graphs |
| Fixed | Llama 2 70B frozen base weights, LoRA r=16 alpha=32, dropout, weight decay, loss, Scrolls GovReport data |

Implementation: NVIDIA NeMo / Megatron-Bridge from NVIDIA's MLPerf Training v6.0 submission
(`source-or-patches/nemo/`, unchanged), run with Apptainer instead of Pyxis/Enroot
(`source-or-patches/helma/run_helma.sub`). Target: validation loss <= 0.925, evaluated every 387 samples
(ceil(384 / GBS) = 43 steps); the first 3 evaluations are skipped (first check at 1548 samples).

GBS 9 has no reference convergence point of its own: `rcp_checker` interpolates the reference between GBS 8 and GBS 16 (as for the GBS 5, 9 and 10 submissions of MLPerf Training v6.0).

## GPU settings per job

Set by the launcher at the start of each job on every GPU of its nodes and logged in the job's `logs/power_limit.txt`
(`GPU_POWER_LIMIT`, `GPU_CLOCK_LOCK`); `GPU_RESET=1` resets the GPUs first (`tools/node_guard.sh`).

| Slurm job | trials in this package | GPU settings | GPU reset at start |
|---|---|---|---|
| 67 | 1 | 550 W, SM clock free | no |
| 137 | 1 | 550 W, SM clock locked at 1300 MHz | yes |
| 143 | 3 | 550 W, SM clock free | yes |

## Files

| Path | Content |
| --- | --- |
| `run_and_time.sh` | per-trial script run in the container (NVIDIA's, unchanged; copy of `source-or-patches/nemo/run_and_time.sh`) |
| `config.sh` | `config_scc_segfault_ngpu12.sh`: current configuration of this system (loads `config_HelmaH100_common.sh`, then NVIDIA `config_common.sh`) |
| `source-or-patches/nemo/` | code executed in the container (snapshot taken by the job), with `llama-zeph.def` (Apptainer recipe) |
| `source-or-patches/helma/` | Slurm/Apptainer launcher and its tools (rank placement, CPU/NUMA binding, page-cache eviction, NCCL pre-test, node guard) |
| `source-or-patches/configs/` | system configuration, launcher defaults, NVIDIA `config_common.sh`, `scc.env` |
| `source-or-patches/configs/job_<id>/` | exact configuration files and command-line overrides of each Slurm job used |
| `source-or-patches/scripts/` | `submit.sh`, packaging and verification tools |

## Steps to launch training

```
apptainer build nvidia-zeph.sif source-or-patches/nemo/llama-zeph.def   # NGC PyTorch 26.04 + NeMo, clean build context
# (runs used an image built earlier with the same recipe, its build-context data stripped without rebuilding:
#  source-or-patches/scripts/tools/make_slim_container.sh; same binaries, /workspace/ft-llm = source-or-patches/nemo)
# dataset and checkpoint: source-or-patches/nemo/README.md and nemo/scripts/ (train.npy, validation.npy, checkpoint)
# train.npy / validation.npy re-created inside the container: source-or-patches/helma/tools/prepare_dataset.sh
./submit.sh segfault_ngpu12 --dry-run                            # resolved config, memory estimate, no GPU
./submit.sh segfault_ngpu12                                      # one Slurm job, 10 consecutive trials, random seeds
./submit.sh segfault_ngpu12 NEXP=<n>                             # additional trials (overrides: see configs/job_<id>/)
tools/build_scc_package.py segfault_ngpu12 --jobs <job dirs> && tools/scc_verify.sh <team_dir>
```

## Results

| # | Slurm job | seed | status | samples to converge | final eval loss | time-to-train (min) | GPU settings |
|---|---|---|---|---|---|---|---|
| 0 | 67 | 12746 | converged | 3096 | 0.9214 | 21.35 | 550 W, SM clock free |
| 1 | 137 | 22737 | converged | 3096 | 0.9233 | 21.70 | 550 W, SM clock locked at 1300 MHz |
| 2 | 143 | 3274 | converged | 3096 | 0.9245 | 21.67 | 550 W, SM clock free |
| 3 | 143 | 3275 | converged | 3096 | 0.9229 | 21.80 | 550 W, SM clock free |
| 4 | 143 | 3276 | converged | 3483 | 0.9240 | 24.21 | 550 W, SM clock free |

**Partial submission: 5 trials** (MLPerf asks for 10; the SCC rules accept partial submissions). Olympic mean time-to-train of these 5 trials (fastest and slowest removed, mean of the 3 middle runs): **21.72 min**.
