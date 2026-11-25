#!/bin/bash
# The best TTT configuration: 16.49 min (job 3583, 2026-10-08; 17.02 min in job 3511 on 2026-10-07, 18.06 min in job 2908 on 2026-10-02).
# 16x H200 NVL (ntu-scc01 + ntu-scc02), TP=2 + SP, MBS 1, GBS 8, no activation recompute (FP8 weight transposes on
# demand), DeepGEMM for the FP8 dgrad GEMMs, the NTU kernel add-ons, 400 W per GPU.
# PREREQUISITE: every GPU capped at 400 W BEFORE launching (powercap/setcap.sh; never change caps during startup).
# usage: [SEED_BASE=1234] [NEXP=1] [TAG=name] bash ntu_scc/launcher/run_best_18min.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export TAG=${TAG:-best18_$(date +%H%M%S)}
export NODE=${NODE:-ntu-scc01,ntu-scc02}
export CFG=config_NTUSCC16TP2_2x8x1xtp2pp1cp1.sh            # TP=2, SP, MBS 1, LR 5e-4, MAX_STEPS 800, USE_TE_OPS=1, FP8
# DeepGEMM for TE's FP8 dgrad GEMMs (ntu_scc/deepgemm_dgrad). Default: the site-packages add-on, which leaves the
# benchmark's train.py STOCK. DG_DGRAD_PATCH=1 DG_DGRAD_ADDON=0 selects the old route (train_dgdgrad.py over train.py).
export DG_DGRAD=1 DG_DGRAD_ADDON=${DG_DGRAD_ADDON:-1} DG_DGRAD_PATCH=${DG_DGRAD_PATCH:-0}
export FP8W_T_ONDEMAND=1                                    # FP8 weight transposes built on demand (frees ~32 GiB/GPU)
# NTU kernel add-ons (ntu_scc/addons, loaded by the same site-packages hook; benchmark source stays stock).
# Job 3511 (2026-10-07): all on + USE_TE_OPS=1 -> 17.02 min, step 2.365 s; + NTU_ATTN_BWD=1 (job 3583, 2026-10-08) -> 16.49 min, step 2.278 s. Set any to 0 to drop it.
export NTU_FUSED_LORA=${NTU_FUSED_LORA:-1} NTU_LORA_QKV_LOCAL=${NTU_LORA_QKV_LOCAL:-1} NTU_FP8_DY_AG=${NTU_FP8_DY_AG:-1}
export NTU_DSWIGLU_DG=${NTU_DSWIGLU_DG:-1} NTU_DG_GRADQ=${NTU_DG_GRADQ:-1} NTU_ROPE=${NTU_ROPE:-1} FP8W_T_NOFWD=${FP8W_T_NOFWD:-1}
# Hand-written FP8 attention backward (ntu_scc/attn_bwd, needs the built libfa3_bwd_fp8.so). Step -3.8% (2.278 vs 2.371 s).
# Convergence: seed 1234 stopped at 3456 (job 3579) but seed 1235 at 3072 with LOWER evals than its baseline (job 3583 vs 3580)
# -> seed noise, not a kernel effect. Job 3583: 16.49 min, the best so far.
export NTU_ATTN_BWD=${NTU_ATTN_BWD:-1}
# TE operation-based MLP (te.ops fuser). The NVIDIA config sets USE_TE_OPS=1 but configs/config_common.sh carries the
# Fujitsu typo USE_TE_OPTS, so it must be set here. Job 2908 (18.06 min) had it; runs without it are ~3% slower.
export USE_TE_OPS=${USE_TE_OPS:-1}
export FORCE_ENV="RECOMPUTE_GRANULARITY=null;RECOMPUTE_METHOD=null;RECOMPUTE_NUM_LAYERS=null${FORCE_ENV_EXTRA:+;$FORCE_ENV_EXTRA}"   # recompute off (+ FORCE_ENV_EXTRA, e.g. "NCCL_MIN_CTAS=64;NCCL_MAX_CTAS=64")
export WARMUP_TRAIN_STEPS_OVERRIDE=${WARMUP_TRAIN_STEPS_OVERRIDE:-2}   # startup: 2 warm-up steps instead of 5
# MCORE_CUDA_GRAPH=1 needs >= 4 warm-up steps: FullCudaGraphWrapper keeps the step counter and the captured graph in class state
# shared by the warm-up wrapper (captures after 1 step) and the training wrapper (captures after 3); with 2 warm-up steps the
# counter hits 3 at training step 1 and the second capture asserts (jobs 4848, 4850). With >= 4 training only replays.
[ "${MCORE_CUDA_GRAPH:-0}" = "1" ] && [ "$WARMUP_TRAIN_STEPS_OVERRIDE" -lt 4 ] && export WARMUP_TRAIN_STEPS_OVERRIDE=4
export SEED_BASE=${SEED_BASE:-1234} NEXP=${NEXP:-1}
export WALLTIME=${WALLTIME:-310} WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-300}
# Everything else (PARALLEL_LOAD, WARMUP_VALIDATION_STEPS=1, SLURM_MPI_TYPE=none, SKIP_SHINIT, SKIP_SYS_DIAG,
# CACHE_SYSINFO, COMPILE_CACHE, NCCL/GLOO_SOCKET_IFNAME) is a default of the launcher.
exec bash "${LAUNCHER:-$HERE/run_nemo_l2l.sh}"   # identical to tools/run_nemo_l2l.sh on the cluster
