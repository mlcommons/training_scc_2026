# `llama2_70b_lora` — SCC26 submission code (MLPerf Training v6.0, closed)

Code-only companion to the `GigEmBytes` SCC26 entry for the Llama-2-70B LoRA
benchmark. It ships the launcher changes that made the NVIDIA NeMo recipe run on
a single 8-GPU node, plus two small vendor changes carried by this tree.

**Naming:** the enclosing folder is `GigEmBytes/benchmarks/llama2_70b_lora/`
(team Gig'Em Bytes, folder slug `GigEmBytes`). Use the same spelling in the
PR title and commit message.

This PR contains no result logs, no `systems/` JSON, no container image, and no
vendored NeMo source.

## What is here

| path | purpose |
|---|---|
| `config.sh` | Scored hyperparameters, parallelism, precision/communication environment, eval cadence. |
| `run_and_time.sh` | Single-node 8-GPU entry point: Triton/libcuda fix, `torchrun`, `cache_clear`, real exit status. Runs exactly one seed per invocation. |
| `source-or-patches/README` | Patch inventory and the vendor files checked unchanged. |
| `source-or-patches/train.py.patch` | Lazy NVFP4 callback import. |
| `source-or-patches/utils.py.patch` | torchrun → Slurm rank-variable normalization. |
| `source-or-patches/run_lora70b_triton_fix.sh` | The scored Flux/Singularity launcher, site paths replaced by placeholders. |

## Configuration and why

One 8-GPU node. A 70B model in bf16 is about 138 GB before activations — more
than a single 141 GB GPU can hold — so TP=1 is impossible and the model must be
split at least four ways.

| knob | value |
|---|---|
| `TP` / `PP` / `CP` / `SP` | 4 / 1 / 1 / 0 |
| `DP` | 2 |
| `GBS` | 8 |
| `MINIBS` / `MBS` | 4 / 1 |
| precision | FP8 (NGC 26.04 PyTorch, Megatron-Bridge 26.04-alpha.rc1) |
| `LR` / `WARMUP_STEPS` | 4e-4 / 11 |
| `MAX_STEPS` | 550 (cosine horizon; runs stop at the first eval that meets the target) |

TP=4 leaves roughly 35 GB of bf16 base weights per GPU, enough for the
8k-sequence activations. DP=2 supplies the data parallelism; `MINIBS` is
`GBS / DP = 4` because the vendor YAML defines `GBS = MINIBS * DP` (CP=1).

`LR` is an unconstrained knob. The scored runs use 4e-4, the learning rate
recorded in the v6.0 RCP data for this benchmark (`opt_base_learning_rate:
0.0004`). Hotter LRs (5e-4, 6e-4, 7e-4) converged faster than the reference
points and are rejected by the RCP rules; 4e-4 reproduces the reference
convergence instead of gaming the target. `WARMUP_STEPS=11` is 2% of 550,
matching the reference warmup shape.

**Checkpoint.** The starting checkpoint is the converted Llama-2-70B checkpoint
in torch_dist format at TP=2, PP=1. The runs use TP=4; Megatron's checkpoint
loader reshards it from TP=2 to TP=4 at load time, so no separate conversion is
needed. Point `CKPT` at the checkpoint as downloaded and keep `LOAD_CKPT=True`.

**Scored runs.** One seed per invocation: `run_and_time.sh` takes `SEED` from
the environment, runs that single seed, and exits with the training return code.
A scheduler-side driver looped seeds 1–10 sequentially, one allocation per
invocation, each writing its own `$LOGDIR/train_seed<SEED>_<timestamp>.log`.
`run_and_time.sh` never submits jobs itself.

## 1. Triton / libcuda fix

### Symptom

Inside the NGC 26.04 SIF, the bridge's startup `torch.compile(bias_swiglu)`
aborts, and every rank prints:

```
torch._inductor.exc.InductorError: AssertionError: libcuda.so cannot found!
Possible files are located at ['/usr/local/cuda/compat/lib/libcuda.so.1'].
Please create a symlink of libcuda.so to any of the files.
```

The traceback ends in `triton/backends/nvidia/driver.py:libcuda_dirs()`. The
SIF's `ldconfig` entry points at `/usr/local/cuda/compat/lib/libcuda.so.1`,
which does not exist in the image. `TORCH_COMPILE_DISABLE=1` only hides the
startup compile; TE's cross-entropy loss still launches a Triton kernel on the
first training step and hits the same assertion.

### Fix

`--nv` injects the real driver at `/.singularity.d/libs/libcuda.so.1`. The
launcher links it into a writable directory and points Triton at it:

```bash
for d in /.singularity.d/libs /usr/local/nvidia/lib64; do
  if [ -e "$d/libcuda.so.1" ]; then
    mkdir -p "$LOGDIR/cuda_compat"
    ln -sf "$d/libcuda.so.1" "$LOGDIR/cuda_compat/libcuda.so.1"
    ln -sf "$d/libcuda.so.1" "$LOGDIR/cuda_compat/libcuda.so"
    export TRITON_LIBCUDA_PATH=$LOGDIR/cuda_compat
    break
  fi
done
```

`TRITON_LIBCUDA_PATH` is the only override needed. **Do not add `cuda_compat`
to `LD_LIBRARY_PATH`.** That shadows the SIF compat `libcuda` with the raw host
driver and segfaults `cuInit`: the job dies with `Fatal Python error:
Segmentation fault` in `torch.cuda.is_available` and exits 139.

## 2. Precision and communication flags

The values below come from NVIDIA's `config_common.sh` and are reproduced
verbatim in `config.sh` so the benchmark folder is self-contained:

| group | flags |
|---|---|
| FP8 | `FP8=True`, `FP8_AMAX_ALGO=max`, `FP8_REDUCE_AMAX=True`, `FP8_AMAX_HISTORY=4`, `FP8_DPA=1`, `NVTE_FP8_DPA_BWD=1` |
| LoRA A2A | `LORA_A2A=1` |
| TE / fused ops | `USE_TE_OPS=1`, `FUSED_SINGLE_QKV_ROPE=1`, `NEMO_LORA_USE_THUNDER_DROPOUT=1`, `NVTE_NORM_FWD_USE_CUDNN=1`, `NVTE_NORM_BWD_USE_CUDNN=1` |
| TP/NCCL | `MC_TP_OVERLAP_AG=True`, `MC_TP_OVERLAP_RS=True`, `MC_TP_OVERLAP_RS_DGRAD=True`, `NCCL_MIN_P2P_NCHANNELS=32`, `NCCL_MIN_CTAS=32`, `NCCL_NCHANNELS_PER_NET_PEER=32`, `NCCL_NVLS_ENABLE=1` |

One correction to the usual description: the YAML master switch
`TP_COMM_OVERLAP` was **not** enabled in these runs — the logged training config
shows `tp_comm_overlap=False`, despite the `MC_TP_OVERLAP_*` variables being
exported by the vendor recipe. Set `TP_COMM_OVERLAP=True` to opt in;
`conf/tp_overlap/h100tp4mbs1.yaml` is the matching config for this shape.

## 3. Eval cadence

The v6.0 schedule is kept exactly:

- `VAL_CHECK_INTERVAL=384`; the YAML computes
  `val_check_interval = ceil(384 / GBS) = 48` steps at GBS=8.
- `skip_evals = floor(0.125 * GBS + 2) = 3`; `train.py` sets
  `start_eval_at_iter = (skip_evals + 1) * val_check_interval = 192`.
- The first logged eval is therefore step **192**, then every **48 steps**:
  192, 240, 288, …

`config.sh` (sourced by `run_and_time.sh`) exports `SKIP_EVALS` only when the
caller supplied a non-empty value and unsets it otherwise. Scored runs leave it
unset so the YAML formula applies. An empty `SKIP_EVALS` would make the YAML's
`oc.decode` fail, so the launcher removes it. `SKIP_EVALS=1` is only for short
screens (first eval at step 96) and is not the scored schedule.

## 4. How to run

Prerequisites: one node with 8 GPUs, the GovReport data, the converted
Llama-2-70B checkpoint, and a PyTorch NGC container that provides
Megatron-Bridge 26.04-alpha.rc1
(`68d8bcc94cb5a25a36cf91413cb376103b2fa9dc`, Megatron-LM
`997896883b04dee1259d92889e7344be528eefa8`), `mlperf_logging` and
`mlperf_common`.

Apply the two vendor patches, then run one seed:

```bash
PATCHES=/path/to/benchmark/folder/source-or-patches
cd /path/to/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo
patch -p1 < "$PATCHES/train.py.patch"
patch -p1 < "$PATCHES/utils.py.patch"

cd /path/to/benchmark/folder
IMPL_DIR=/path/to/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo \
DATADIR=/path/to/gov_report \
CKPT=/path/to/model_checkpoint \
LOGDIR=$PWD/logs \
SEED=1 ./run_and_time.sh
```

To let the script enter the SIF, also set
`CONTAINER=/path/to/pytorch_26.04-py3.sif`. It binds `DATADIR`, `CKPT`,
`LOGDIR`, the benchmark folder and `IMPL_DIR` into the container. Without a
SIF, run the script inside the container yourself.

Outputs: `$LOGDIR/train_seed<SEED>_<timestamp>.log` with the `:::MLLOG` lines,
starting with `cache_clear`. The script exits with the training return code.
Run one seed per invocation. On a scheduler, request one task holding all eight
GPUs (e.g. Flux `-n 1 -g 8`; `-n 8 -g 8` requests 64).

## 5. Vendor patches

| file | change |
|---|---|
| `train.py` | Moves `from callback_nvfp4 import NVFP4Callback` inside the `if cfg.model.fp4:` branch. The module needs TE NVFP4 symbols that are absent from the fallback container, even when `FP4=False`. |
| `utils.py` | Sets `SLURM_PROCID`/`SLURM_LOCALID` from torchrun's `RANK`/`LOCAL_RANK` (only if not already set). The rest of the file already reads the Slurm names for rank-0 logging, device selection and OOM profiling. |

`config_common.sh`, `conf/`, the callbacks, `run.sub` and the vendor
`run_and_time.sh` are byte-identical to the v6.0 release. The scored launcher's
Flux metadata, provenance and power-logging blocks are site-local; the optional
`power_logger.py` is not included in this code PR.
