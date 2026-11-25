#!/bin/bash
# config.sh - the exact environment of the submitted runs: the system config plus the launcher settings of
# source-or-patches/ntu_scc/launcher/run_best_18min.sh (kernel add-ons on, recompute off, cache clear). Node state:
# GPU cap 390 W x 8, SM lock 345-1050 MHz, CPUs performance / 2.2 GHz, BMC fans 40 % (source-or-patches/ntu_scc/launcher/run_final.sh).
source "$(dirname "${BASH_SOURCE[0]}")/configs/config_NTUSCC16TP2_2x8x1xtp2pp1cp1.sh"
export TAG=${TAG:-best18_$(date +%H%M%S)}
export NODE=${NODE:-ntu-scc01,ntu-scc02}
export CFG=config_NTUSCC16TP2_2x8x1xtp2pp1cp1.sh            # TP=2, SP, MBS 1, LR 5e-4, MAX_STEPS 800, USE_TE_OPS=1, FP8
export DG_DGRAD=1 DG_DGRAD_ADDON=${DG_DGRAD_ADDON:-1} DG_DGRAD_PATCH=${DG_DGRAD_PATCH:-0}
export FP8W_T_ONDEMAND=1                                    # FP8 weight transposes built on demand (frees ~32 GiB/GPU)
export NTU_FUSED_LORA=${NTU_FUSED_LORA:-1} NTU_LORA_QKV_LOCAL=${NTU_LORA_QKV_LOCAL:-1} NTU_FP8_DY_AG=${NTU_FP8_DY_AG:-1}
export NTU_DSWIGLU_DG=${NTU_DSWIGLU_DG:-1} NTU_DG_GRADQ=${NTU_DG_GRADQ:-1} NTU_ROPE=${NTU_ROPE:-1} FP8W_T_NOFWD=${FP8W_T_NOFWD:-1}
export NTU_ATTN_BWD=${NTU_ATTN_BWD:-1}
export USE_TE_OPS=${USE_TE_OPS:-1}
export FORCE_ENV="RECOMPUTE_GRANULARITY=null;RECOMPUTE_METHOD=null;RECOMPUTE_NUM_LAYERS=null${FORCE_ENV_EXTRA:+;$FORCE_ENV_EXTRA}"   # recompute off (+ FORCE_ENV_EXTRA, e.g. "NCCL_MIN_CTAS=64;NCCL_MAX_CTAS=64")
export WARMUP_TRAIN_STEPS_OVERRIDE=${WARMUP_TRAIN_STEPS_OVERRIDE:-2}   # startup: 2 warm-up steps instead of 5
export SEED_BASE=${SEED_BASE:-1234} NEXP=${NEXP:-1}
export WALLTIME=${WALLTIME:-310} WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-300}
