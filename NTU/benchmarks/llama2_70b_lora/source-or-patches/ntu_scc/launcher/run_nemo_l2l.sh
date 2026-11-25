#!/bin/bash
# Launch the NVIDIA/Fujitsu NeMo llama2_70b_lora benchmark on one ntu-scc node.
# Config = Fujitsu's config_PGCDI_1x8x4xtp4pp1cp1.sh verbatim except LR 5e-4 -> 4e-4.
# Assets and image are read from the node's LOCAL /scratch (NFS is 1 GbE).
#
# Usage: NODE=ntu-scc02 [NEXP=1] bash tools/run_nemo_l2l.sh
set -euo pipefail
NODE=${NODE:?set NODE=ntu-scc01|ntu-scc02}
IMPL=/work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b
LOCAL=/scratch/benchmark/MLPerf
TAG=${TAG:-nemo_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}

# enroot/pyxis MUST use node-local storage. The default is ~/.local/share/enroot,
# which here is NFS home SHARED BY BOTH NODES -> extraction fails with "Stale file
# handle", and run.sub's `enroot remove -f $(enroot list)` cleanup on one node
# deletes the other node's container mid-extraction.
export ENROOT_DATA_PATH=/scratch/enroot-data
export ENROOT_CACHE_PATH=/scratch/enroot-cache
export ENROOT_TEMP_PATH=/scratch/enroot-tmp
export ENROOT_RUNTIME_PATH=/scratch/enroot-runtime
# XDG_RUNTIME_DIR (/run/user/$UID) is created by systemd-logind and DELETED when the
# user's last login session ends. enroot's default runtime path is
# ${XDG_RUNTIME_DIR}/enroot, so pyxis dies with "mkdir: cannot create directory
# '/run/user/10001': Permission denied" on a node where we have no live session.
# Point it at local /scratch, which always exists and is group-writable.
export XDG_RUNTIME_DIR=/scratch

# CU12=1 selects the CUDA 12.9 image (NGC 25.06-py3 base) instead of the CUDA 13 one.
# atomic_gemm in TE requires CUDA >=12.2 and <13.0, so the Hopper tp_overlap config
# (pipeline + atomic_gemm) can only work on the CUDA 12 build.
if [ "${CU12:-0}" = "1" ]; then
  export CONT=/scratch/containers/l2l-cu12d.sqsh   # c = modelopt 0.46.1 + datasets/GitPython/... (missing from the 25.06 base)
else
  export CONT=/scratch/containers/l2l-nemo.sqsh
fi
export DATADIR=$LOCAL/data/dataset/nemo_npy
export MODEL=$LOCAL/data/models/llama2_70b_megatron_bridge
export CHECKPOINT=$MODEL
export LOGDIR=/work/benchmark/MLPerf/runs/$TAG
export NEXP=${NEXP:-1}
export MLPERF_RULESET=6.0.0
export MLPERF_CLUSTER_NAME=ntu-scc
export MLPERF_SCALE=single-node
export SEED_BASE=${SEED_BASE:-1234}
export CHECK_COMPLIANCE=1
export NCCL_TEST=0          # no RDMA fabric here; single node
# Cache clearing (mandatory cache_clear record for a compliant log) needs root. We have no passwordless sudo; the site systool
# /mnt/opt/systools/drop-caches (setuid launcher, source in ~/systools/drop-caches) does it when installed. Auto-enable then.
if [ -z "${CLEAR_CACHES:-}" ]; then
  if [ -u /mnt/opt/systools/drop-caches ] && [ -x /mnt/opt/systools/drop-caches ]; then
    export CLEAR_CACHES=1 DROPCACHE_CMD=/mnt/opt/systools/drop-caches
  else
    export CLEAR_CACHES=0   # log will lack cache_clear -> not submittable
  fi
fi
export CLEAR_CACHES DROPCACHE_CMD=${DROPCACHE_CMD:-"sudo /sbin/sysctl vm.drop_caches=3"}
# Submission metadata (mlperf_common reads these; defaults are *_PLACEHOLDER strings). PLATFORM gets a "<nodes>x" prefix.
export MLPERF_SUBMISSION_ORG=${MLPERF_SUBMISSION_ORG:-NTU}
export MLPERF_SUBMISSION_PLATFORM=${MLPERF_SUBMISSION_PLATFORM:-G494-ZB4_8xH200NVL}
# profiling passthrough: NVTX_FLAG wraps python in nsys (run_and_time.sh:61),
# PROFILE/BREAKDOWN/PROFILE_* drive cfg.nsys_profile (conf/megatron_gpt_peft_tuning_config.yaml:118)
export NVTX_FLAG=${NVTX_FLAG:-0}
export PROFILE=${PROFILE:-False}
export BREAKDOWN=${BREAKDOWN:-False}
export PROFILE_START_STEP=${PROFILE_START_STEP:-4}
export PROFILE_END_STEP=${PROFILE_END_STEP:-8}
export PROFILE_RANKS=${PROFILE_RANKS:-0}
export REUSE_CONTAINER=${REUSE_CONTAINER:-1}
export DATESTAMP=$(date +%y%m%d%H%M%S%N)

mkdir -p "$LOGDIR"
cd "$IMPL"
source configs/${CFG:-config_NTUSCC_1x8x4xtp4pp1cp1.sh}   # sets DGXNGPU/DGXNNODES/TP/MINIBS/LR/FP8/...

# The configs `export VAR=...` UNCONDITIONALLY, so anything passed in the caller's
# environment is silently overwritten for the vars they set (SP, TP, PP,
# TP_COMM_OVERLAP, FUSED_SINGLE_QKV_ROPE, USE_TE_OPTS). FORCE_ENV applies overrides
# AFTER the source so they actually take effect:  FORCE_ENV="SP=0 TP_COMM_OVERLAP=False"
# Accepts space- or semicolon-separated pairs; use ';' when passing FORCE_ENV
# through EXTRA, since `env $EXTRA` itself word-splits on whitespace.
# EVAL_MBS_PATCH=1 mounts a utils.py whose eval loss_func reduces over the
# micro-batch, so eval can run with VAL_MBS>1 (the shipped one only works at 1).
if [ "${EVAL_MBS_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/utils_evalmbs.py:/workspace/ft-llm/utils.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi

# --- startup-time fixes (2026-10-01; measured with STARTUP_PROF=1, see results/STARTUP_2026-10-01.md) ---
# slurm.conf has MpiDefault=pmix: every srun step pays ~9.5 s of PMIx server setup (12+ steps per job).
# Nothing here uses MPI/PMIx (slurm2pytorch + MASTER_ADDR env rendezvous), so default it off.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-none}
# Fully parallel checkpoint load: each DP rank reads 1/DP of its shard, then exchanges it (46 s -> 11.5 s).
export PARALLEL_LOAD=${PARALLEL_LOAD:-True}
# callback_warmup sets eval_iters=WARMUP_VALIDATION_STEPS AND loops WARMUP_VALIDATION_STEPS times, so the
# config's 5 runs 25 eval iterations (45 s, more than a real eval). 1 = one iteration (18 s, mostly first-call setup).
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS_OVERRIDE:-1}
# Skip run.sub's informational prolog steps (ibv_devinfo, nvidia-smi topo/vbios/clique): ~20 s, nothing consumes them.
export SKIP_SYS_DIAG=${SKIP_SYS_DIAG:-1}
# Container bashes source /etc/shinit_v2 (nvidia-smi + CUDA init every time, ~8 s per bash layer); run.sub skips it.
export SKIP_SHINIT=${SKIP_SHINIT:-1}
# run.sub re-queries the system JSON + git commit per experiment (~16 s each); once per job is enough.
export CACHE_SYSINFO=${CACHE_SYSINFO:-1}
# Train warmup steps (config: 5). Override with WARMUP_TRAIN_STEPS_OVERRIDE=<n>.
if [ -n "${WARMUP_TRAIN_STEPS_OVERRIDE:-}" ]; then export WARMUP_TRAIN_STEPS=$WARMUP_TRAIN_STEPS_OVERRIDE; fi

if [ -n "${FORCE_ENV:-}" ]; then
  for _kv in $(printf '%s' "$FORCE_ENV" | tr ';' ' '); do
    export "${_kv?}"
    echo "[FORCE_ENV] ${_kv}"
  done
fi

# Mixed driver versions across nodes break the shared (NFS-home) pyxis rootfs: enroot rewrites the
# soname links per start, last node wins -> 'Failed to open libnvidia-ml.so.1'. Keep ALL nodes on the
# same driver (580.173.02 as of 2026-09-30).
# NCCL bootstrap: each node otherwise picks its own IPoIB interface; scc01 picks ibp7s0f0
# (192.168.107.x), which scc03 (6 HCAs, no .107 subnet) cannot reach. ibp101s0f0 exists on all.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibp101s0f0}
# Gloo (torch CPU collectives) resolves the node's own hostname; scc03's /etc/hosts maps it to
# 127.0.1.1 (Ubuntu default), so peers get 'Connection refused, remote=[127.0.1.1]'. Pin it too.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ibp101s0f0}

# --- overrides that must come AFTER sourcing the config ---
# VBOOST needs `sudo nvidia-smi boost-slider`, and we have no passwordless sudo.
# Set VBOOST_VALUE_OVERRIDE=1 only after applying it manually on the node.
export VBOOST_VALUE=${VBOOST_VALUE_OVERRIDE:-0}
# run.sub's mountcheck can't resolve expected-mounts.csv inside the container;
# tools/stage_nemo_assets.sh already verified all 15 files by exact byte size.
export VERIFY_MOUNTS=0
# Their Dockerfile sets this as an image ENV; our enroot build put it in
# /etc/bash.bashrc, which a direct `python` launch never sources ->
# ModuleNotFoundError: No module named 'megatron.bridge'. Export it here so
# sbatch -> srun -> pyxis propagates it into the container.
export PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:/workspace/ft-llm
# ^ /workspace/ft-llm: run.sub's cache_clear logging step (`python -c "from callback_logging import mllogger"`, run.sub:294) has
#   no --container-workdir, so with our mounted layout the import fails without it (job 4474). train.py is unaffected (its own dir
#   is already sys.path[0]).

# Optional NUMA binding (NUMA_BIND=1). run_and_time.sh calls ${BINDCMD:-} but the
# shipped tree never sets it, so ranks are unbound; profiling showed GPU4-7
# (NUMA 1) arriving late at the global all-reduce, costing ranks 0-3 ~0.7 s/step.
# TP_OVERLAP_PATCH=1 mounts a train.py that actually wires tp_comm_overlap through
# (the shipped one never does - see results/PROFILE_REPORT_2026-09-17.md section 2).
# PP_PATCH=1 mounts a callback_logging.py that tolerates PP>1 (the shipped one indexes
# loss_dict["lm loss"] unconditionally, which only exists on the last pipeline stage).
# CU12 needs a guarded NVFP4 import (TE 2.4.0 in the 25.06 base has no NVFP4).
if [ "${CU12:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/callback_nvfp4_cu12.py:/workspace/ft-llm/callback_nvfp4.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
  # TE 2.4.0 (25.06 base) has no transformer_engine.pytorch.attention.rope.
  # apply_fused_qkv_rotary_pos_emb, so Mcore sets HAVE_FUSED_QKV_ROPE=False and
  # asserts if the config still asks for it. config_common.sh exports it =1
  # unconditionally, so this override must come AFTER the config is sourced.
  export FUSED_SINGLE_QKV_ROPE=0
fi

# NO_BULK=1 makes train_tpoverlap.py pass tp_comm_bulk_dgrad/_wgrad=False, so TE
# never calls bulk_overlap() -- which needs CUDA multicast that H200 NVL lacks.
# Read by the patch via os.getenv, so it needs TP_OVERLAP_PATCH=1 to have effect.
export NO_BULK=${NO_BULK:-0}

# MEM_PROFILE=1 mounts a callback_warmup.py that prints a per-dtype parameter memory
# census after model construction, before warmup. Diagnostic only.
if [ "${MEM_PROFILE:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/callback_warmup_mem.py:/workspace/ft-llm/callback_warmup.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi

# SEL_BLOCK=1 mounts a patched mcore transformer_layer.py that scopes SELECTIVE
# recompute to the first SELECTIVE_NUM_LAYERS layers (stock Megatron applies selective
# to all layers and forbids combining it with recompute_num_layers).
if [ "${SEL_BLOCK:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/mcore/transformer_layer.py:/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi

if [ "${PP_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/callback_logging_pp.py:/workspace/ft-llm/callback_logging.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi
# AMAXRED_PATCH=1 mounts a train.py that passes tp_only_amax_red from TP_ONLY_AMAX_RED (the shipped
# train.py never sets it, and FP8_REDUCE_AMAX is only read by the FP4 callback -> inert on H200).
# Cannot be combined with TP_OVERLAP_PATCH (both replace train.py).
# GPUMETRICS_PATCH=1: nsys also samples each rank's own GPU (SM clock, power, tensor activity) at
# GPU_METRICS_HZ (default 10 kHz). One sampler per GPU, keyed by SLURM_LOCALID. Needs PROFILE/NVTX_FLAG.
if [ "${GPUMETRICS_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/run_and_time_gpumetrics.sh:/workspace/ft-llm/run_and_time.sh:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi
# FASTACC_PATCH=1 mounts a train.py that sets TE's _2X_ACC_DGRAD=False when FAST_ACC_DGRAD=1 (fast FP8 accumulation
# for the dgrad GEMMs; +10% GEMM throughput at 300 W in gemmlib). Cannot be combined with other train.py patches.
if [ "${FASTACC_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/train_fastacc.py:/workspace/ft-llm/train.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi
# DG_DGRAD_PATCH=1 mounts a train.py that, with DG_DGRAD=1, routes TE's FP8 dgrad GEMMs to DeepGEMM (patches/dg_dgrad.py,
# gemmlib/pylib_dg). Also mounts gemmlib + patches into the container. Cannot be combined with other train.py patches.
if [ "${DG_DGRAD_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/train_dgdgrad.py:/workspace/ft-llm/train.py:ro,/work/benchmark/MLPerf/gemmlib:/work/benchmark/MLPerf/gemmlib,/work/benchmark/MLPerf/patches:/work/benchmark/MLPerf/patches:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/work/benchmark/MLPerf/gemmlib/triton_cache}
fi
# DG_DGRAD_ADDON=1: the same DeepGEMM dgrad + FP8W_T_ONDEMAND, but WITHOUT touching the benchmark source (stock train.py).
# A .pth in the container's site-packages (patches/zz_dg_dgrad.pth -> patches/dg_dgrad_autoload.py) calls dg_dgrad.apply()
# as soon as transformer_engine.pytorch is imported. Still needs DG_DGRAD=1 at runtime. Do not combine with DG_DGRAD_PATCH.
if [ "${DG_DGRAD_ADDON:-0}" = "1" ]; then
  [ "${DG_DGRAD_PATCH:-0}" = "1" ] && { echo "DG_DGRAD_ADDON and DG_DGRAD_PATCH are alternatives; pick one"; exit 1; }
  _dp=/usr/local/lib/python3.12/dist-packages; _p=/work/benchmark/MLPerf/patches
  EXTRA_MOUNTS="$_p/dg_dgrad_autoload.py:$_dp/dg_dgrad_autoload.py:ro,$_p/zz_dg_dgrad.pth:$_dp/zz_dg_dgrad.pth:ro,/work/benchmark/MLPerf/gemmlib:/work/benchmark/MLPerf/gemmlib,/work/benchmark/MLPerf/attnkernel:/work/benchmark/MLPerf/attnkernel:ro,$_p:$_p:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/work/benchmark/MLPerf/gemmlib/triton_cache}
fi
if [ "${AMAXRED_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/train_amaxred.py:/workspace/ft-llm/train.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi
if [ "${TP_OVERLAP_PATCH:-0}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/patches/train_tpoverlap.py:/workspace/ft-llm/train.py:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS
fi
# COMPILE_CACHE=1 (default): persistent TorchInductor cache. The container's /tmp is empty every job, so
# Megatron's _warmup_jit_function recompiled its fused kernels from scratch each run (~10 s).
if [ "${COMPILE_CACHE:-1}" = "1" ]; then
  EXTRA_MOUNTS="/work/benchmark/MLPerf/cache:/compile_cache${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/compile_cache/inductor}
fi
# STARTUP_PROF=1: timestamps on every run.sub trace line (PS4) + a Python stack sampler in the train.py processes of
# STARTUP_PROF_RANKS (tools/startup_prof, loaded via a .pth in dist-packages). Output: $LOGDIR/startup_prof/.
if [ "${STARTUP_PROF:-0}" = "1" ]; then
  _sp=/work/benchmark/MLPerf/tools/startup_prof; _dp=/usr/local/lib/python3.12/dist-packages
  EXTRA_MOUNTS="$_sp/startup_prof.py:$_dp/startup_prof.py:ro,$_sp/zz_startup_prof.pth:$_dp/zz_startup_prof.pth:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
  export EXTRA_MOUNTS STARTUP_PROF STARTUP_PROF_RANKS=${STARTUP_PROF_RANKS:-0,1,8} PS4='+[$EPOCHREALTIME] '
fi
if [ "${NUMA_BIND:-0}" = "1" ]; then
  export BINDCMD=/numa_bind.sh
  export EXTRA_MOUNTS="/work/benchmark/MLPerf/tools/numa_bind.sh:/numa_bind.sh:ro${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"   # append, never replace
fi
echo "=== launching on $NODE ==="
echo "  DGXSYSTEM=$DGXSYSTEM  TP=$TP  MINIBS=$MINIBS  LR=$LR  MAX_STEPS=$MAX_STEPS  FP8=$FP8"
echo "  LOGDIR=$LOGDIR"
sbatch --parsable -p gpu -w "$NODE" -N "$DGXNNODES" ${SBATCH_EXTRA:-} \
       --time="${WALLTIME:-360}" -J l2lnemo run.sub | tee "$LOGDIR/jobid.txt"
