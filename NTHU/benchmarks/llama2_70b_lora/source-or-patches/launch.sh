#!/bin/bash
# Run NVIDIA's v5.0 llama2_70b_lora implementation under Apptainer on this cluster.
#
#   nvidia/launch.sh [config]          # default config_GH200_8x1x1xtp1pp1cp1.sh
#   SEED=1234 NNODES=8 nvidia/launch.sh
#
# NVIDIA's own run.sub drives Pyxis/enroot, which this cluster does not have.
# This does the same job with sbatch + srun + `apptainer exec`, and binds the
# same four paths run.sub mounts: /data (npy dataset), /ckpt (NeMo checkpoint),
# /results, and the implementation at /workspace/ft-llm.
#
# Networking: despite NCCL_IB_DISABLE=1 below, NCCL runs RDMA. The image's
# HPC-X net plugin (/opt/hpcx/nccl_rdma_sharp_plugin, "NCCL RDMA Plugin v9")
# ignores that variable and uses mlx5_0 and mlx5_1 through the container's inbox
# libibverbs, which does open these HCAs (probe job 176). Measured busbw,
# 8 nodes: 36-38 GB/s as configured here, 22 GB/s with NCCL_IB_HCA=mlx5_0. The
# 89 MB LoRA gradient all-reduce takes 4.2 ms of a 4.18 s step.
# NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME only carry bootstrap and gloo traffic.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck source=/dev/null
[ -f "$ROOT/configs/site.env" ] && . "$ROOT/configs/site.env"
BUILD_DIR="${BUILD_DIR:-$(cat "$ROOT/.build_dir")}"
: "${DATA_ROOT:=$WORK_ROOT}"   # site.env.example leaves DATA_ROOT commented out; it then defaults to WORK_ROOT
# site.env defaults EXCLUDE_NODES to head; EXCLUDE_NODES=none runs on every node.
[ "${EXCLUDE_NODES:-}" = none ] && EXCLUDE_NODES=

CONFIG="${1:-config_GH200_8x1x1xtp1pp1cp1.sh}"
SIF="${SIF:-$BUILD_DIR/containers/llama2_70b_lora_ngc25.04.sif}"
IMPL="${IMPL:-$WORK_ROOT/nvidia_impl}"
DATADIR="${DATADIR:-$DATA_ROOT/nvidia_npy}"
CKPT="${CKPT:-$DATA_ROOT/nemo_torch_dist}"
NNODES="${NNODES:-8}"
SEED="${SEED:-$RANDOM}"
TIME="${TIME:-02:00:00}"
RUN_DIR="${RUN_DIR:-$WORK_ROOT/runs/nvidia-$(date +%Y%m%d-%H%M%S)-seed$SEED}"

for p in "$SIF" "$IMPL/train.py" "$DATADIR/train.npy" "$CKPT/weights" "$IMPL/$CONFIG"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done
mkdir -p "$RUN_DIR"
[ -n "${OVERRIDES:-}" ] && printf '%s\n' "$OVERRIDES" > "$RUN_DIR/overrides.sh"
echo "run dir: $RUN_DIR   config: $CONFIG   seed: $SEED   nodes: $NNODES"

# DEPENDENCY=afterany:<jobid> queues this run behind another job instead of
# running it now; scripts/run_competition.sh chains all twelve jobs with it.
jid=$(sbatch --job-name=nv-lora --partition="$PARTITION" --nodes="$NNODES" --ntasks-per-node=1 \
  --gres=gpu:1 --cpus-per-task=48 --time="$TIME" ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} \
  ${NODELIST:+--nodelist=$NODELIST} ${DEPENDENCY:+--dependency=$DEPENDENCY} \
  --output="$RUN_DIR/slurm-%j.out" --wrap "
set -euo pipefail
# Job-scoped machine settings, undone by one trap at exit (also on scancel).
# The --wrap script runs under /bin/sh (dash), so POSIX syntax only here:
#   FAN_PROFILE=MLPerf  CMC fan profile on all 7 chassis via SC26-System's
#                       cmc_fans (MLPerf = every fan at 100%); back to 'default'.
#   FAN_SPEED=N         the site's fan_speed tool: fixed duty N% on all 7
#                       chassis (verified 2026-09-19: 45 -> 7350/6000 RPM,
#                       100 -> ~15.5k/13k); back to FAN_RESTORE (default 50).
#                       The job then waits 10 s so the fans are actually at that
#                       duty before the trainer starts -- before run_start, so it
#                       costs wall clock and not TTT.
#   POWER_CAP=W         module (Grace+Hopper+memory) power limit on the job's
#                       nodes via the site's nv_power; back to 1000 W.
#   PRECOOL=S           fans at PRECOOL_FAN% (default 100) for S seconds before
#                       anything else, the same pre-cool /software/hpl/tools/run.sbatch
#                       does. It happens before the trainer starts, so it is outside
#                       MLPerf's timed region (which opens at run_start) and well
#                       inside the Closed-division 30-minute init allowance. The
#                       FAN_SPEED line below then sets the run's own duty. A kill
#                       during the sleep is covered by the dependent restore job,
#                       which is queued whatever happens.
CLEANUP=''
trap 'eval \"\$CLEANUP\"; exit' EXIT TERM INT
${PRECOOL:+fan_speed ${PRECOOL_FAN:-100} > /dev/null 2>&1 && echo 'pre-cool: fans ${PRECOOL_FAN:-100}% for ${PRECOOL}s' || echo 'pre-cool: fan_speed failed'; sleep $PRECOOL;}
${FAN_PROFILE:+FAN_TOOL=\"${FAN_TOOL:-$HOME/SC26-System/tools/cmc_fans.py}\"; export CMC_ENV=\"${CMC_ENV:-$HOME/SC26-System/ansible/files/cmc.env}\"; python3 \$FAN_TOOL set $FAN_PROFILE ${FAN_CMCS:+--hosts $FAN_CMCS} || echo 'fan profile switch failed'; CLEANUP=\"\$CLEANUP python3 \$FAN_TOOL set default ${FAN_CMCS:+--hosts $FAN_CMCS} || true; \"}
${FAN_SPEED:+fan_speed $FAN_SPEED > /dev/null && echo 'fans: $FAN_SPEED%' || echo 'fan_speed failed'; sleep 10; CLEANUP=\"\$CLEANUP fan_speed ${FAN_RESTORE:-50} > /dev/null; \"}
${POWER_CAP:+nv_power $POWER_CAP --hosts \$(scontrol show hostnames \$SLURM_JOB_NODELIST | paste -sd, -) > /dev/null 2>&1 && echo 'power cap: ${POWER_CAP}W module' || echo 'nv_power failed'; CLEANUP=\"\$CLEANUP nv_power ${POWER_RESTORE:-1000} --hosts \$(scontrol show hostnames \$SLURM_JOB_NODELIST | paste -sd, -) > /dev/null 2>&1; \"}
# POWER_CAP_MAP="node=W,node=W,..." overrides the cap of single nodes after POWER_CAP has set them all
# (per-node balancing: sync training runs at the pace of the slowest GPU).
${POWER_CAP_MAP:+for kv in \$(echo '$POWER_CAP_MAP' | tr ',' ' '); do n=\$(echo \$kv | cut -d= -f1); w=\$(echo \$kv | cut -d= -f2); nv_power \$w --hosts \$n > /dev/null 2>&1 || echo \"nv_power failed on \$n\"; done; echo 'per-node caps: $POWER_CAP_MAP'}
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export SEED=$SEED
export NCCL_IB_DISABLE=1
${NCCL_SOCKET_IFNAME:+export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME}
export DATA_ROOT=/data CKPT_ROOT=/ckpt
# Python packages layered over the image without a rebuild (pip --target into
# \$BUILD_DIR/containers/pyextra); empty unless something needed fixing.
export APPTAINERENV_PREPEND_PATH=
export APPTAINERENV_PYTHONPATH=/pyextra
${EXTRA_ENV:-}
# NFS clients cache negative lookups: job 167 died because work3 could not see a
# .sif created hours earlier. Listing the parent directories refreshes them.
srun ls '$(dirname "$SIF")' '$IMPL' '$DATADIR' '$CKPT' > /dev/null
# Hydra output dirs, one per rank, created here so that no two ranks race on
# mkdir over NFS (jobs 311 and 315 died on FileExistsError in that race).
for i in \$(seq 0 \$((SLURM_NTASKS - 1))); do mkdir -p '$RUN_DIR'/hydra/rank\$i; done
# MLPerf wants the page cache flushed before a timed run; that needs root here.
# The outcome is logged as cache_clear rather than asserted.
if srun bash -c 'sync && sudo -n /sbin/sysctl -q vm.drop_caches=3' 2>/dev/null; then
  export CACHE_CLEARED=1
else
  echo 'cache_clear: could not drop page caches (no passwordless sudo)'
  export CACHE_CLEARED=0
fi
export CONFIG_FILENAME=$CONFIG
export MLPERF_SUBMISSION_ORG=\${MLPERF_SUBMISSION_ORG:-NTHU}
export MLPERF_SUBMISSION_PLATFORM=\${MLPERF_SUBMISSION_PLATFORM:-GH200}
# GPUMON=1 logs power/clocks/throttling per node (nvidia/gpumon.sh) into the run dir.
${GPUMON:+srun --overlap --ntasks=$NNODES --ntasks-per-node=1 --cpus-per-task=1 bash '$IMPL/gpumon.sh' '$RUN_DIR' ${GPUMON_INTERVAL:-5} &}
srun --kill-on-bad-exit=1 apptainer exec --nv \
  --bind '$DATADIR:/data:ro,$CKPT:/ckpt:ro,$RUN_DIR:/results,$IMPL:/workspace/ft-llm,$BUILD_DIR/containers/pyextra:/pyextra:ro' \
  --pwd /workspace/ft-llm '$SIF' bash -c '
    source ./$CONFIG
    export DGXNNODES=$NNODES
    # Per-run overrides written by the caller into the run directory, sourced
    # after the config so they win (MAX_STEPS for a smoke test, say).
    [ -f /results/overrides.sh ] && source /results/overrides.sh
    export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
    # IPoIB interface names differ between nodes (work9: ibP4s4f0).
    if [ -z \"\${NCCL_SOCKET_IFNAME:-}\" ]; then
      export NCCL_SOCKET_IFNAME=\$(python ib_ifname.py ${IB_SUBNET:-10.18.18.})
      export GLOO_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME
    fi
    # PROFILE_CMD prefixes the trainer, e.g. an nsys invocation (see nvidia/torch_profile.py).
    # Hydra creates outputs/<date>/<time> under the shared implementation dir;
    # 14 ranks doing that in the same second on NFS raced at midnight (job 311,
    # FileExistsError on the new date dir). One directory per rank in the run dir.
    exec \${PROFILE_CMD:-} python -u train.py hydra.run.dir=/results/hydra/rank\$SLURM_PROCID
  '
" | tee /dev/stderr | awk '/Submitted batch job/{print $4}')

# Restoring fans and power caps from inside the job is unreliable on scancel
# (job 274: the batch script is SIGKILLed after 30 s, before 7 CMCs and 14
# nodes are done). A dependent job runs the restore however the main job ends.
if [ -n "${jid:-}" ] && [ -n "${FAN_PROFILE:-}${FAN_SPEED:-}${POWER_CAP:-}" ]; then
  sbatch --job-name=nv-lora-restore --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --time=00:15:00 --dependency=afterany:$jid --output="$RUN_DIR/restore-%j.out" --wrap "
${FAN_PROFILE:+export CMC_ENV=\"${CMC_ENV:-$HOME/SC26-System/ansible/files/cmc.env}\"; python3 \"${FAN_TOOL:-$HOME/SC26-System/tools/cmc_fans.py}\" set default ${FAN_CMCS:+--hosts $FAN_CMCS}}
${FAN_SPEED:+fan_speed ${FAN_RESTORE:-50} > /dev/null 2>&1 && echo 'fans back to ${FAN_RESTORE:-50}%' || echo 'fan_speed restore failed'}
${POWER_CAP:+nv_power ${POWER_RESTORE:-1000} --hosts \$(scontrol show hostnames \$(sacct -j $jid -X -n -o NodeList%200 | tr -d ' ') | paste -sd, -) > /dev/null 2>&1 && echo 'cap back to ${POWER_RESTORE:-1000}W' || echo 'nv_power restore failed'}
" > /dev/null && echo "restore job queued after $jid"
fi
