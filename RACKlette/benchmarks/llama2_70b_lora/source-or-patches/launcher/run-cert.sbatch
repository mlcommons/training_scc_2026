#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
# run.sbatch: one MLPerf llama2_70b_lora run on N nodes (N comes from sbatch -N). Submit through submit.sh.
# Every knob is an env var. Unset knobs fall back to: our defaults below, then the vendor config_common.sh.
set -eo pipefail
: "${PIPE_DIR:?submit via submit.sh (PIPE_DIR unset)}"
source "$PIPE_DIR/pipeline.env"
# /home variant: drop every /scratch entry inherited from the submitting shell (Kez bins/libs/modules).
_noscratch() { tr ":" "\n" <<<"$1" | grep -v "^/scratch" | paste -sd: -; }
export PATH=$(_noscratch "$PATH") LD_LIBRARY_PATH=$(_noscratch "${LD_LIBRARY_PATH:-}") MODULEPATH=$(_noscratch "${MODULEPATH:-}")
unset KEZ_ENV KEZ_WORKDIR

# ---- knobs: remember what the caller set, load vendor defaults, then apply ours
KNOBS="GBS TP CP PP SP MBS MINIBS VAL_MBS LR MAX_STEPS WARMUP_STEPS SEED MCORE_CUDA_GRAPH CUDA_GRAPH_SCOPE \
FP8 FP8_DPA FP8_ACT FP8_PARAM_GATHER TP_COMM_OVERLAP NCCL_NVLS_ENABLE NUM_WORKERS GRADIENT_ACCUMULATION_FUSION \
PARALLEL_LOAD CP_EVAL CP_COMM_TYPE VAL_CHECK_INTERVAL SKIP_EVALS USE_TE_OPS FUSED_SINGLE_QKV_ROPE ALLOC_CONF LOG_EVERY_N_STEPS"
for k in $KNOBS; do eval "_U_$k=\${$k-__unset__}"; done
source "$FTLLM/config_common.sh"
declare -A DEF=([GBS]=8 [TP]=4 [CP]=1 [PP]=1 [SP]=1 [MBS]=1 [VAL_MBS]=1 [LR]=0.00045 [MAX_STEPS]=672
  [WARMUP_STEPS]=0 [SEED]=$(( (RANDOM << 15 | RANDOM) )) [MCORE_CUDA_GRAPH]=1 [CUDA_GRAPH_SCOPE]=full_iteration
  [FP8_PARAM_GATHER]=0 [NCCL_NVLS_ENABLE]=0 [NUM_WORKERS]=4 [GRADIENT_ACCUMULATION_FUSION]=0 [PARALLEL_LOAD]=1
  [TP_COMM_OVERLAP]=False)
for k in $KNOBS; do
  eval "u=\$_U_$k"
  if [ "$u" != __unset__ ]; then export "$k=$u"; elif [ -n "${DEF[$k]+x}" ]; then export "$k=${DEF[$k]}"; fi
done
# config_common.sh overwrites vars outside KNOBS (FP8_REDUCE_AMAX, FP8_AMAX_*, NCCL_MIN_*, CUDA_DEVICE_MAX_CONNECTIONS, ...).
# Re-apply every KEY=VAL from the grid line (EXTRA_ENV_KEYS, ';'-separated) so grid settings always win.
IFS=';' read -ra _kvs <<<"${EXTRA_ENV_KEYS:-}"
for _kv in "${_kvs[@]}"; do [[ "$_kv" == *=* ]] && export "$_kv"; done
NGPU=$(( SLURM_NNODES * ${GPN:-8} )); DP=$(( NGPU / (TP * PP * CP) ))
[ -n "${MINIBS:-}" ] || export MINIBS=$(( GBS / DP ))
[ $(( MINIBS * DP )) -eq "$GBS" ] || { echo "PIPE_EVENT bad_config GBS=$GBS DP=$DP MINIBS=$MINIBS"; exit 2; }
export DGXNNODES=$SLURM_NNODES DGXNGPU=${GPN:-8}   # GPN: GPUs per node (default 8; 4 = one NVLink island for scaling runs)

# ---- environment
SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
export DATA_ROOT=$DATA_DIR/gov_report CKPT_ROOT=$DATA_DIR/model
export LOGDIR=$ROOT/runs/$SLURM_JOB_ID-r${RUN_IDX:?}; mkdir -p "$LOGDIR/data_index/train" "$LOGDIR/data_index/val"
export PYTHONPATH=$SRC/Megatron-Bridge/src:$SRC/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=$ROOT/cache/hf HF_HUB_OFFLINE=1
export NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/targets/x86_64-linux/include   # TE's NVRTC needs CUDA headers on every node
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13                  # sbrinz's ldconfig also lists cuda-12
# vendor + Nico's working container do NOT set this; expandable segments (cuMem/VMM allocations) is the prime
# suspect for the intermittent 2-node CUDA-graph capture deadlock (jobs 7727, 7773). Default: vendor behaviour (unset).
unset PYTORCH_CUDA_ALLOC_CONF   # never inherit it from the submitting shell (sbatch --export=ALL)
[ -n "${ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 + 1000 * RUN_IDX ))
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# CUDA_HOME/lib64 last: TE's NVRTC (13.2.78 from CUDA_HOME) dlopens libnvrtc-builtins.so.13.2 by name; without
# this, sbrinz finds a mismatched 13.2.86 copy via ldconfig and sbrinz1 finds none (job 7771 died on that).
export LD_LIBRARY_PATH=$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib64
export PATH=$VENV/bin:$PATH

echo "PIPE_CONFIG job=$SLURM_JOB_ID label=${LABEL:-none} nodes=$SLURM_JOB_NODELIST env=$ENV_NAME GBS=$GBS DP=$DP PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset} NCCL_DEBUG=$NCCL_DEBUG $(for k in $KNOBS; do [ -n "${!k+x}" ] && printf '%s=%s ' $k "${!k}"; done)"
echo "PIPE_EXTRA ${EXTRA_ENV_KEYS:-}"
ONE_PER_NODE=(-N "$SLURM_NNODES" -n "$SLURM_NNODES" --ntasks-per-node=1)
srun "${ONE_PER_NODE[@]}" bash -c 'echo "PIPE_NODE $(hostname) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) gpus=$(nvidia-smi -L | wc -l)"'

# ---- hang guard: kill the job if the log is silent for HANG_MIN minutes (dumps rank stacks first).
# STOP_AFTER_CAPTURE=1 ends the job as soon as the CUDA graph capture completed (diagnostic runs).
OUT=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')
HANG_MIN=${HANG_MIN:-12}
(
  last=$(date +%s); size=0
  while sleep 30; do
    s=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
    [ "$s" != "$size" ] && { size=$s; last=$(date +%s); }
    if [ "${STOP_AFTER_CAPTURE:-0}" = 1 ] && grep -q "CUDA graph capture done for training" "$OUT"; then
      echo "PIPE_EVENT capture_done_stop"; sleep 10; pkill -TERM -P $$ -x srun; exit 0
    fi
    if [ $(( $(date +%s) - last )) -gt $(( HANG_MIN * 60 )) ]; then
      echo "PIPE_EVENT hang_guard no_output_for=${HANG_MIN}min; full native stacks of every rank -> $LOGDIR/stacks/"
      mkdir -p "$LOGDIR/stacks"
      srun --overlap "${ONE_PER_NODE[@]}" bash -c 'for p in $(pgrep -u $USER -f "[t]rain.py"); do timeout 90 uvx py-spy dump --native --pid $p > '"$LOGDIR"'/stacks/$(hostname)-$p.txt 2>&1 & done; wait' || true
      for f in "$LOGDIR"/stacks/*.txt; do echo "  $(basename $f): $(grep -A1 MainThread "$f" | tail -1 | tr -s " ")"; done
      pkill -TERM -P $$ -x srun; exit 0
    fi
  done
) & HANG_PID=$!; trap 'kill $HANG_PID 2>/dev/null' EXIT

# ---- optional GPU clock/temperature log (CLOCKLOG=1; off by default: the NVML monitor on sbrinz records
# per-GPU power/clocks/temps cluster-wide, so no extra nvidia-smi processes run next to the benchmark)
if [ "${CLOCKLOG:-0}" = 1 ]; then
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  ssh -o BatchMode=yes "$n" "nvidia-smi --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader -l 10" > "$LOGDIR/clocks-$n.csv" 2>/dev/null &
done
fi

# ---- clear page cache for model+data on every node (no root needed), log the MLPerf cache_clear event once
srun "${ONE_PER_NODE[@]}" "$VENV/bin/python" "$PIPE_DIR/clear_cache.py" "$CKPT_ROOT" "$DATA_ROOT"
"$VENV/bin/python" "$PIPE_DIR/clear_cache.py" --log-event

# ---- optional code patch (PATCH=<name> -> patches/<name>.patch) applied to a PER-JOB COPY of the vendor code;
# the pristine vendor checkout is never modified. Default (PATCH unset) = unmodified vendor code.
if [ -n "${PATCH:-}" ]; then
  rsync -a --exclude outputs --exclude __pycache__ "$FTLLM/" "$LOGDIR/ft-llm/"
  ( cd "$LOGDIR/ft-llm" && patch -p1 --forward < "$PIPE_DIR/patches/$PATCH.patch" ) || { echo "PIPE_EVENT patch_failed $PATCH"; exit 3; }
  export FTLLM=$LOGDIR/ft-llm
  echo "PIPE_PATCH $PATCH applied -> $FTLLM (sha256 $(sha256sum "$PIPE_DIR/patches/$PATCH.patch" | cut -c1-16))"
fi

# ---- launch: 8 ranks per node; one dead rank kills all (no zombie ranks holding nodes)
cd "$FTLLM"
srun --kill-on-bad-exit=1 --cpu-bind=none --ntasks-per-node=${GPN:-8} "$PIPE_DIR/task.sh"
echo "PIPE_EVENT finished rc=$?"
