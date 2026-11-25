# Llama 2 70B LoRA (MLPerf Training v6.0, Closed) - Texas A&M, Gig 'Em Bytes

SC26 Student Cluster Competition submission. Single GPU node of 8x NVIDIA H200 NVL.

## System

- Node: Dell PowerEdge XE7745, 2x AMD EPYC 9575F (64C each, SMT off), 1.5 TB DDR5-6400
- Accelerators: 8x NVIDIA H200 NVL (141 GB HBM3e), two NVLink groups of 4
- OS: Rocky Linux 9, kernel 5.14.0-687.59.1.el9_8; driver 580.196.07
- Scheduler: Flux 0.81.0
- Full details: `systems/student_system.json`

## Software

- Container: `pytorch_26.04-py3.sif` (NVIDIA NGC PyTorch 26.04)
- PyTorch 2.12.0a0, CUDA 13.2, cuDNN 9.21, NCCL 2.29.7
- Transformer Engine 2.14.0, Triton 3.6.0, FlashAttention 2.7.4.post1
- Megatron-Bridge commit `68d8bcc94cb5a25a36cf91413cb376103b2fa9dc` (loaded from source)
- Megatron-LM commit `997896883b04dee1259d92889e7344be528eefa8` (Megatron-Core 0.18.0)
- Training code and logging callbacks: `source-or-patches/train_grace/nemo/`

## Configuration

Set in `config.sh`:

| Setting | Value |
|---|---|
| Learning rate | 4e-4 |
| Warmup steps | 11 |
| Global batch size | 8 |
| Max (LR decay) steps | 550 |
| Parallelism | TP4, DP2, mini batch size 4 |
| LoRA rank / alpha | 16 / 32 |
| Gradient clip norm | 0.3 |
| Weight decay | 1e-4 |
| Precision | FP8 (Transformer Engine hybrid format, delayed recipe) on a BF16 base |

Target: validation loss <= 0.925 (logged as `eval_accuracy`).
Evaluation runs every 384 samples (48 steps at GBS 8) after the initial skipped checks.

## Data and checkpoint

Both come from the SCC 2026 MLCommons downloads (packed 8192-token GovReport dataset
and the fused-QKV Llama 2 70B checkpoint). Download commands are in the SCC guide, section 7.3.

## How to run

```bash
# one run (what run_and_time.sh does)
SEED=<seed> bash run_and_time.sh

# the 10 scored runs (one Flux job)
flux batch -N1 -n1 -g8 --exclusive -q gpu -t 540 source-or-patches/run_scored_10_rand.sh
```

`run_and_time.sh` is a copy of `source-or-patches/run_lora70b_triton_fix.sh`.
Each of the 10 runs draws a fresh seed derived from the system clock (nanosecond time
mixed with a random value), so all seeds are distinct. Each run is checked for
`run_stop` success, `cache_clear`, `eval_accuracy <= 0.925` and a clean train exit
before its log is kept as `result_N.txt`.

## Local fixes

- Triton needs a libcuda that matches the container's CUDA 13.2 on the host driver
  (CUDA 13.0). The run script symlinks `libcuda.so.1` into a `cuda_compat` directory and
  sets `TRITON_LIBCUDA_PATH` to it. It deliberately does not put that directory on
  `LD_LIBRARY_PATH`.
- Extra Python packages are provided through `PYTHONPATH` (`pylibs_h200`).
  Packages installed there: annotated_doc 0.0.5, antlr4-python3-runtime 4.9.3, anyio 4.15.1,
  certifi 2026.10.19, click 8.5.0, filelock 4.2.4, fsspec 2026.9.0, h11 0.16.0, hf_xet 1.7.0,
  httpcore 1.0.9, httpx 0.28.1, huggingface_hub 1.33.0, hydra-core 1.3.2, idna 3.20,
  markdown-it-py 4.2.0, mdurl 0.1.2, mlperf_common 0.3, mlperf_logging 4.1.47, numpy 2.5.4,
  omegaconf 2.3.0, packaging 26.3, pandas 3.0.6, pygments 2.21.0, python-dateutil 2.9.0.post0,
  pyyaml 6.0.3, regex 2026.9.29, rich 15.0.0, safetensors 0.8.0, scipy 1.18.1,
  shellingham 1.5.4, six 1.17.0, tokenizers 0.22.2, tqdm 4.70.1, transformers 5.5.4,
  typer 0.27.3, typing_extensions 4.16.0.
- Megatron-Bridge and Megatron-LM are used unmodified at the commits listed above
  (clean git working trees).
- `source-or-patches/train_grace/nemo/` is the NVIDIA NeMo / Megatron-Bridge Llama 2 70B LoRA
  implementation with these local changes (each is commented in the code):
  - `utils.py`: Flux/torchrun compatibility. torchrun exports `RANK`/`LOCAL_RANK` where the
    original code reads `SLURM_PROCID`/`SLURM_LOCALID`, so the Slurm names are set from them.
  - `utils.py`: the validation loss is reduced to the two-element `[sum_loss, count]` vector
    that the Megatron-Bridge evaluation expects (no change in behaviour at `VAL_MBS=1`, which
    is what the runs use).
  - `train.py`: optional per-rank CPU pinning (`AFFINITY=1`), because Flux binds only the
    parent task; each rank is pinned to the cores of the NUMA node of its GPU.
  - `train.py`: the NVFP4 callback import is lazy; FP4 is not used in these runs.
  - `config_common.sh`: `FP8_PARAM_GATHER=0`.
- Runs are launched with torchrun inside Apptainer under Flux (`run_and_time.sh`), not with
  the Slurm/Pyxis `run.sub` shipped with the original implementation.

## Results

Seeds and TTT (run_stop - run_start) per run:

Runs 0-4 were submitted at hardware certification (cert1-submission1). Runs 5-9 come
from the same Flux job, with unchanged system and hyper-parameters; this directory
holds all ten.

| Run | Seed | TTT (s) | Stop step | Final val. loss |
|---|---|---|---|---|
| 0 | 763539718 | 1984.6 | 384 | 0.92402 |
| 1 | 801320609 | 1984.7 | 384 | 0.92389 |
| 2 | 54871835 | 1988.3 | 384 | 0.92372 |
| 3 | 1019020567 | 1987.0 | 384 | 0.92444 |
| 4 | 1289223584 | 1985.9 | 384 | 0.92463 |
| 5 | 1339199950 | 1987.0 | 384 | 0.92311 |
| 6 | 1082460127 | 2246.5 | 432 | 0.92210 |
| 7 | 1558731373 | 2249.3 | 432 | 0.92182 |
| 8 | 1876230927 | 1984.0 | 384 | 0.92297 |
| 9 | 879747156 | 1981.9 | 384 | 0.92369 |

Scoring uses the mean of the middle 8 runs (Olympic trimming).
Result logs are in `results/student_system/llama2_70b_lora/`.
