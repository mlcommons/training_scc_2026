#!/bin/bash
set -e
RUN_NAME=${1:-trial}
if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && [ -z "${LD_LIBRARY_PATH:-}" ]; then
    echo "run_rcp: running under sudo without -E (LD_LIBRARY_PATH empty); use sudo -E" >&2
fi
H=${MLPERF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source $H/h200-tuning/config_H200NVL4.sh
if [ -n "${NODELIST:-}" ]; then NODES=(${NODELIST//,/ })
elif [ -n "${PBS_NODEFILE:-}" ]; then mapfile -t NODES < <(awk '!s[$0]++' "$PBS_NODEFILE")
elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else NODES=($(hostname)); fi
NNODES=${#NODES[@]}
if [ "${NODES[0]%%.*}" != "$(hostname -s)" ]; then
    echo "run_rcp: first node ${NODES[0]} is not this host $(hostname -s)" >&2; exit 1
fi
if [ -n "${GPN_LIST:-}" ]; then IFS=, read -ra GPNS <<< "$GPN_LIST"
else GPNS=(); for _n in "${NODES[@]}"; do GPNS+=("${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"); done; fi
[ ${#GPNS[@]} -eq $NNODES ] || { echo "run_rcp: GPN_LIST needs $NNODES entries" >&2; exit 1; }
GPN=${GPNS[0]}; GPN_TOT=0; for _g in "${GPNS[@]}"; do GPN_TOT=$((GPN_TOT + _g)); done
[ "$GPN" -ge 1 ] || { echo "run_rcp: no GPUs detected" >&2; exit 1; }
if [ -n "${EXPECT_GBS:-}" ]; then
    DPW=$(( GPN_TOT / ${TP:?} / ${CP:?} ))
    GBS=$(( ${MBS:?} * ${MINIBS:?} * DPW ))
    if [ $(( GPN_TOT % (TP * CP) )) -ne 0 ] || [ "$GBS" -ne "$EXPECT_GBS" ]; then
        echo "run_rcp: ${NNODES}x${GPN} GPUs at TP=$TP CP=$CP MBS=$MBS MINIBS=$MINIBS gives GBS $GBS, expected $EXPECT_GBS" >&2; exit 1
    fi
fi
BOOT_IFACE=${BOOT_IFACE:-enp73s0f0np0}
if [ -z "${NCCL_SOCKET_IFNAME:-}" ] && ip -o -4 addr show dev "$BOOT_IFACE" 2>/dev/null | grep -q inet; then IFACE=$BOOT_IFACE
else IFACE=${NCCL_SOCKET_IFNAME:-$(ip -o route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')}; fi
[ -n "$IFACE" ] || { echo "run_rcp: no default-route interface; set NCCL_SOCKET_IFNAME" >&2; exit 1; }
MASTER_ADDR=${MASTER_ADDR:-$(ip -o -4 addr show dev "$IFACE" | awk '{sub(/\/.*/, "", $4); print $4; exit}')}
[ -n "$MASTER_ADDR" ] || { echo "run_rcp: no IPv4 address on $IFACE; set MASTER_ADDR" >&2; exit 1; }
export GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE
if [ -z "${NCCL_IB_HCA:-}" ]; then
    HCAS=$(for d in /sys/class/infiniband/*; do
        [ "$(cut -c1 $d/ports/1/state 2>/dev/null)" = 4 ] && [ "$(cat $d/ports/1/link_layer 2>/dev/null)" = InfiniBand ] && basename $d
    done | paste -sd,)
    [ -z "$HCAS" ] || export NCCL_IB_HCA=$HCAS
fi
SHARP_SO=${HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR:+$HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR/lib/libnccl-net.so}
export USE_SHARP=${USE_SHARP:-0}
if [ "$USE_SHARP" = 1 ]; then
    [ -f "${SHARP_SO:-/nonexistent}" ] || {
        echo "run_rcp: USE_SHARP=1 but no HPC-X SHARP plugin (HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR unset or $SHARP_SO missing)" >&2; exit 1; }
    export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-$SHARP_SO} NCCL_COLLNET_ENABLE=1
fi
export SHARP_ALLOW_SM_PORT=1 SHARP_COLL_ENABLE_SAT=${SHARP_COLL_ENABLE_SAT:-0}
export DGXSYSTEM=H200NVL${NNODES}x${GPN} DGXNGPU=$((GPN_TOT / NNODES)) DGXNNODES=$NNODES
export MINIBS=${MINIBS:-4}
export DATA_ROOT=${DATA_ROOT:-$H/data} CKPT_ROOT=${CKPT_ROOT:-$H/model/llama2_70b_megatron_bridge} LOAD_CKPT=${LOAD_CKPT:-True}
export TORCHINDUCTOR_CACHE_DIR=$H/cache/inductor TRITON_CACHE_DIR=$H/cache/triton DG_JIT_CACHE_DIR=$H/cache/deep_jit
ln -sfn $H/deepgemm/third-party/cutlass/include/cutlass $H/deepgemm/deep_gemm/include/cutlass
ln -sfn $H/deepgemm/third-party/cutlass/include/cute $H/deepgemm/deep_gemm/include/cute
export TMPDIR=/tmp/mlperf-$USER TMP=/tmp/mlperf-$USER TEMP=/tmp/mlperf-$USER
mkdir -p $TMPDIR
RSH=${RSH:-ssh -n -o BatchMode=yes}
for n in "${NODES[@]:1}"; do $RSH $n "mkdir -p $TMPDIR; rm -rf $TMPDIR/pyc-* $TMPDIR/pymp-*"; done
rm -rf $TMPDIR/pyc-* $TMPDIR/pymp-* 2>/dev/null
TMO=${TMO:-10800}
OUTDIR=$H/results/$RUN_NAME; mkdir -p $OUTDIR $H/cache
export ERRDIR=$OUTDIR/err
mkdir -p "$ERRDIR"
export LGC=${LGC-1200}
if [ -n "$LGC" ] && [ "$LGC" != "0" ]; then
    for n in "${NODES[@]}"; do
        if [ "${n%%.*}" = "$(hostname -s)" ]; then bash $H/h200-tuning/lgc.sh "$LGC"
        else ssh -o BatchMode=yes $n "bash $H/h200-tuning/lgc.sh $LGC"; fi 2>&1 | tee -a $OUTDIR/node0.log
    done
fi

PORT=${MASTER_PORT:-$((29500 + RANDOM % 400))}
V=${TRAIN_VENV:-$H/venv-train}
export TRAIN_VENV=$V
if [ -x "$V/bin/python" ] && [ -d "$H/deepgemm/deep_gemm/include" ]; then
  DG_HDR=deep_gemm/impls/sm90_fp8_gemm_per_tensor_legacy.cuh
  if [ -f "$H/deepgemm/deep_gemm/include/$DG_HDR" ]; then
    DGPKG=$("$V/bin/python" -c 'import importlib.util,os
s=importlib.util.find_spec("deep_gemm")
print(os.path.dirname(os.path.abspath(s.origin)) if s and s.origin else "")' 2>/dev/null)
    if [ -n "$DGPKG" ] && [ -f "$DGPKG/include/$DG_HDR" ] && \
       ! cmp -s "$H/deepgemm/deep_gemm/include/$DG_HDR" "$DGPKG/include/$DG_HDR"; then
      echo "!!! WARNING: installed DeepGEMM JIT headers are STALE vs the tree."
      echo "!!!   tree:      $H/deepgemm/deep_gemm/include/$DG_HDR"
      echo "!!!   installed: $DGPKG/include/$DG_HDR"
      echo "!!! The JIT compiles against the installed copy, so every DeepGEMM"
      echo "!!! shape fails to compile and silently falls back to cuBLAS, with"
      echo "!!! long per-step retries. Reinstall deepgemm/ into the venv before"
      echo "!!! timing anything (h200-tuning/setup/setup_sc26.sh)."
    fi
  fi
fi
export CC=gcc PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1   # Python stacks on SIGSEGV
CR=${CODE_ROOT:-$H}
export PYTHONPATH="$H/site-overlay:$CR/Megatron-Bridge/src:$CR/Megatron-Bridge/3rdparty/Megatron-LM:$CR/work-nemo${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
export PYTHONPYCACHEPREFIX=/tmp/mlperf-$USER/pyc-$RUN_NAME
rm -rf $CR/work-nemo/__pycache__
env | grep -E "^(TP|CP|SP|MBS|MINIBS|MAX_STEPS|MCORE_CUDA_GRAPH|CUDA_GRAPH|FP8|NCCL_|CUDA_DEVICE|DEEPGEMM_|DG_JIT|NSYS|USE_|LORA|NVTE_|VAL_|SKIP_EVALS|WARMUP|UB_|EVAL|DL_|DGX|TP_COMM|LR|LR_DECAY|SEED|TARGET)" | sort > $OUTDIR/env.txt
echo "${NODES[*]}" > $OUTDIR/nodes.txt
ENVFILE=$OUTDIR/env.sh
export -p | grep -v -E " (BASH_FUNC|SSH_|DISPLAY|TERM|SHLVL|PWD|OLDPWD|_)=" > $ENVFILE
L0F() { echo "$V/bin/torchrun --nnodes $NNODES --nproc-per-node ${GPNS[$1]} --master-addr $MASTER_ADDR --master-port $PORT"; }
EP="--no-python $H/h200-tuning/rank_exec_cp.sh train.py"
echo "=== run_rcp: $RUN_NAME NODES=${NODES[*]} GPN=${GPNS[*]} MASTER=$MASTER_ADDR IFACE=$IFACE HCA=${NCCL_IB_HCA:-auto} CODE_ROOT=$CR TP=$TP CP=$CP MINIBS=$MINIBS MAX_STEPS=$MAX_STEPS SEED=$SEED TMO=${TMO}s ==="
: > $OUTDIR/node0.log
if [ "${CACHE_CLEAR:-0}" = 1 ]; then
    if [ "$(id -u)" = 0 ]; then DROPCACHE_CMD=${DROPCACHE_CMD:-PATH=$PATH:/usr/sbin:/sbin sysctl vm.drop_caches=3}
    else DROPCACHE_CMD=${DROPCACHE_CMD:-sudo -n sysctl vm.drop_caches=3}; fi
    CC_SH="h=\$(hostname); sync; b=\$(awk '/^Cached:/ {print \$2}' /proc/meminfo); $DROPCACHE_CMD >/dev/null || exit 1; echo \"\$h cache_cleared \${b}kB to \$(awk '/^Cached:/ {print \$2}' /proc/meminfo)kB\""
    for n in "${NODES[@]}"; do
        if [ "${n%%.*}" = "$(hostname -s)" ]; then bash -c "$CC_SH"; else ssh -o BatchMode=yes $n bash -s <<< "$CC_SH"; fi >> $OUTDIR/node0.log ||
            { echo "run_rcp: cache clear failed on $n (DROPCACHE_CMD='$DROPCACHE_CMD')" >&2; exit 1; }
    done
    (cd $CR/work-nemo && $V/bin/python -c "
from callback_logging import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)" >> $OUTDIR/node0.log 2>&1) ||
        { echo "run_rcp: cache_clear event call failed" >&2; exit 1; }
fi
cd $CR/work-nemo
set +e
for i in $(seq 1 $((NNODES - 1))); do
    $RSH ${NODES[$i]} "source $ENVFILE; export DGXNGPU=$((GPN_TOT / NNODES)); cd $CR/work-nemo && timeout -k 10 $TMO $(L0F $i) --node-rank $i $EP" > $OUTDIR/node$i.log 2>&1 &
done
timeout -k 10 $TMO $(L0F 0) --node-rank 0 $EP >> $OUTDIR/node0.log 2>&1 &
TPID=$!
STALL_S=${STALL_S:-300}; POW_MIN_W=${POW_MIN_W:-120}; POW_LOW_S=${POW_LOW_S:-5}
WPAT="$(basename $V)/bin/python -u train.py"
SMI="nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 1000"
$SMI > $OUTDIR/power.node0.txt 2>/dev/null & PW0=$!
PWS="$PW0"
for i in $(seq 1 $((NNODES - 1))); do
    $RSH ${NODES[$i]} "exec $SMI" > $OUTDIR/power.node$i.txt 2>/dev/null & PWS="$PWS $!"
done
trap 'kill $PWS 2>/dev/null; rm -rf "$PYTHONPYCACHEPREFIX" $TMPDIR/pymp-* 2>/dev/null' EXIT INT TERM
_trip() {
    echo "=== $1 -> SIGUSR1 stack dumps, then kill ===" | tee -a $OUTDIR/node0.log
    RPAT="$(basename $V)/bin/python -u trai[n].py"
    pkill -USR1 -f "$WPAT"
    for n in "${NODES[@]:1}"; do $RSH $n "pkill -USR1 -f '$RPAT'"; done
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader >> $OUTDIR/stall_gpu.txt
    for n in "${NODES[@]:1}"; do $RSH $n "nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader" >> $OUTDIR/stall_gpu.txt; done
    sleep 15
    pkill -KILL -f "$WPAT"
    for n in "${NODES[@]:1}"; do $RSH $n "pkill -KILL -f '$RPAT'"; done
}
_minpow() {  # min power over the last sample of every GPU of every node
    for f in $OUTDIR/power.node*.txt; do tail -n $GPN $f 2>/dev/null; done |
        awk -F', *' 'NF>=2 && $2+0==$2 {if (m=="" || $2<m) m=$2} END{print (m=="" ? 9999 : m)}'
}
last=-1; idle=0; low=0
while kill -0 $TPID 2>/dev/null; do
    sleep 1
    sz=$(stat -c %s $OUTDIR/node0.log 2>/dev/null || echo 0)
    if [ "$sz" = "$last" ]; then idle=$((idle + 1)); else idle=0; last=$sz; fi
    if [ $idle -ge $STALL_S ]; then _trip "LOG STALL: node0.log idle ${idle}s"; break; fi
    if grep -q '"key": "run_start"' $OUTDIR/node0.log 2>/dev/null && \
       ! grep -q '"key": "run_stop"' $OUTDIR/node0.log 2>/dev/null; then
        mp=$(_minpow)
        if awk "BEGIN{exit !($mp < $POW_MIN_W)}"; then low=$((low + 1)); else low=0; fi
        if [ $low -gt $POW_LOW_S ]; then
            _trip "POWER STALL: a GPU < ${POW_MIN_W}W for ${low}s (min ${mp}W)"; break
        fi
    else
        low=0
    fi
done
kill $PWS 2>/dev/null
wait $TPID
RC=$?
wait
echo "=== done: $RUN_NAME rc=$RC (124 = wall-clock timeout) ===" | tee -a $OUTDIR/node0.log
if grep -aq "cp_shards: calibrated, split cached; relaunch on the cache" $OUTDIR/node0.log; then
    if [ -n "${CP_SHARD_RELAUNCHED:-}" ]; then
        echo "run_rcp: calibrated again on the relaunch; the cache was not picked up" >&2; exit 1
    fi
    rm -rf ${OUTDIR}-cal; mv $OUTDIR ${OUTDIR}-cal
    echo "=== $RUN_NAME: calibration done (log in ${OUTDIR}-cal), relaunching on the cache ==="
    CP_SHARD_RELAUNCHED=1 CP_SHARD_RECALIBRATE=0 exec bash "$H/h200-tuning/$(basename "${BASH_SOURCE[0]}")" "$RUN_NAME"
fi
if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R $SUDO_USER: $OUTDIR $H/cache $OUTDIR.launch.log 2>/dev/null || true
fi
