#!/bin/bash
# Llama-2-70B LoRA: single-node, 8-GPU launcher for the NVIDIA v6.0 NeMo
# implementation. Runs train.py with the configuration in config.sh.
#
# Runs exactly one seed per invocation: SEED comes from the environment, and a
# scheduler-side driver loops the seeds (one allocation per invocation).
#
# Does not submit a job and needs no scheduler. Run it inside the PyTorch NGC
# container, from an allocation that owns all 8 GPUs:
#
#   IMPL_DIR=/path/to/nvidia/nemo DATADIR=/path/to/gov_report \
#   CKPT=/path/to/model_checkpoint LOGDIR=$PWD/logs SEED=1 ./run_and_time.sh
#
# If CONTAINER points at a SIF and singularity is available, the script
# re-execs itself inside the container with the data, checkpoint, log and
# implementation directories bind-mounted.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL_DIR=${IMPL_DIR:-$HERE}
source "$HERE/config.sh"

: "${DATADIR:?set DATADIR to the gov_report directory}"
: "${CKPT:?set CKPT to the converted Llama-2-70B checkpoint}"
: "${LOGDIR:?set LOGDIR to a writable output directory}"
[ -d "$DATADIR" ] || { echo "FATAL: DATADIR=$DATADIR is not a directory" >&2; exit 2; }
[ -d "$CKPT" ] || { echo "FATAL: CKPT=$CKPT is not a directory" >&2; exit 2; }
[ -d "$IMPL_DIR" ] || { echo "FATAL: IMPL_DIR=$IMPL_DIR is not a directory" >&2; exit 2; }
mkdir -p "$LOGDIR"
DATADIR=$(cd "$DATADIR" && pwd)
CKPT=$(cd "$CKPT" && pwd)
LOGDIR=$(cd "$LOGDIR" && pwd)
IMPL_DIR=$(cd "$IMPL_DIR" && pwd)

if [ ! -f "$IMPL_DIR/train.py" ]; then
  echo "FATAL: no train.py in IMPL_DIR=$IMPL_DIR" >&2
  echo "       point IMPL_DIR at NVIDIA's llama2_70b_lora NeMo implementation" >&2
  exit 2
fi
if [ "$DP" -lt 1 ] || [ $((TP * PP * CP * DP)) -ne "$NPROC" ] \
   || [ $((GBS % DP)) -ne 0 ] || [ $((MINIBS * DP)) -ne "$GBS" ]; then
  echo "FATAL: inconsistent batch geometry: NPROC=$NPROC TP=$TP PP=$PP CP=$CP DP=$DP GBS=$GBS MINIBS=$MINIBS" >&2
  exit 2
fi

# Optional: run inside the SIF. Inside the container SINGULARITY_NAME/APPTAINER_NAME
# is set, so this block does not recurse.
if [ -z "${SINGULARITY_NAME:-}${APPTAINER_NAME:-}" ] && [ -f "${CONTAINER:-/nonexistent}" ] \
   && command -v singularity >/dev/null 2>&1; then
  exec singularity exec --nv \
    --bind "$DATADIR:$DATADIR" \
    --bind "$CKPT:$CKPT" \
    --bind "$LOGDIR:$LOGDIR" \
    --bind "$HERE:$HERE" \
    --bind "$IMPL_DIR:$IMPL_DIR" \
    "$CONTAINER" bash "$0" "$@"
fi

command -v torchrun >/dev/null 2>&1 || {
  echo "FATAL: torchrun not found; run inside the PyTorch NGC container" >&2
  exit 2
}

export DATA_ROOT=$DATADIR CKPT_ROOT=$CKPT
export DGXNNODES=1 DGXNGPU=$NPROC
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}   # single node; avoids DNS issues
export MASTER_PORT=${MASTER_PORT:-29511}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

# Triton resolves libcuda through ldconfig first. The NGC 26.04 SIF has a stale
# ldconfig entry, so the TE cross-entropy kernel aborts with:
#   AssertionError: libcuda.so cannot found!
# --nv injects the real driver under /.singularity.d/libs; link it into a
# writable directory and point TRITON_LIBCUDA_PATH there.
# Do NOT add that directory to LD_LIBRARY_PATH: it shadows the SIF compat
# libcuda with the raw host driver and segfaults cuInit (exit 139).
if [ -z "${TRITON_LIBCUDA_PATH:-}" ]; then
  for d in /.singularity.d/libs /usr/local/nvidia/lib64; do
    if [ -e "$d/libcuda.so.1" ]; then
      mkdir -p "$LOGDIR/cuda_compat"
      ln -sf "$d/libcuda.so.1" "$LOGDIR/cuda_compat/libcuda.so.1"
      ln -sf "$d/libcuda.so.1" "$LOGDIR/cuda_compat/libcuda.so"
      export TRITON_LIBCUDA_PATH=$LOGDIR/cuda_compat
      break
    fi
  done
fi

RUNLOG=$LOGDIR/train_seed${SEED}_$(date +%Y%m%d_%H%M%S).log
cd "$IMPL_DIR"
echo "CFG LR=$LR WU=$WARMUP_STEPS GBS=$GBS STEPS=$MAX_STEPS SEED=$SEED TP=$TP DP=$DP MINIBS=$MINIBS"
echo "IMPL=$IMPL_DIR DATA=$DATADIR CKPT=$CKPT LOG=$RUNLOG"

# cache_clear must precede run_start; emit it once before torchrun.
python -c "from mlperf_logging import mllog; mllog.get_mllogger().event(key=mllog.constants.CACHE_CLEAR, value=True)" 2>&1 | tee "$RUNLOG"

set +e
torchrun --nnodes=1 --nproc_per_node="$NPROC" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" train.py \
  2>&1 | tee -a "$RUNLOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "TRAIN_EXIT:$TRAIN_RC"
echo "LOG:$RUNLOG"
exit "$TRAIN_RC"
