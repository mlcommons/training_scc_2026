#!/bin/bash
# RACKlette, SC26: 2 nodes x 8 H200 NVL (two NVLink groups of 4 per node), TP=4 SP=1 DP=4, GBS 8.
# Same settings as our best run (14.45 min time to train, Oct 2026, overlap off, unpatched code), plus the overlap.
# Copy next to config_common.sh in Dell/benchmarks/llama2_70b_lora/implementations/nemo
# (mlcommons/training_results_v6.0) after applying source-or-patches/.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=672  # cosine horizon, 5376 samples
export LR=0.00045
export MINIBS=2
export TP=4
export SP=1
export CP=1
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=4
export GRADIENT_ACCUMULATION_FUSION=0  # slower in our 1-node A/B
export PARALLEL_LOAD=1
unset PYTORCH_CUDA_ALLOC_CONF  # expandable_segments deadlocked 2-node CUDA graph capture

# TP comm overlap (needs source-or-patches/). GPU_ARCH=h selects conf/tp_overlap/h100tp4mbs1.yaml;
# without it the Hydra config defaults to the b100 (Blackwell) yaml.
# Set TP_COMM_OVERLAP=False for our 14.45 min setting if the 2-node full run with the overlap fails.
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-True}
export GPU_ARCH=h
export UB_SKIPMC=1  # no NVSwitch, so no CUDA multicast: userbuffers over CUDA IPC
export NVTE_UB_ATOMIC_GEMM_RS_PROJ=0 NVTE_UB_ATOMIC_GEMM_RS_FC2=0  # TE supports atomic GEMM only when built with CUDA 12

# system parameters
export DGXNNODES=2
export DGXNGPU=8
export WALLTIME_RUNANDTIME=30
export NEXP=1  # random SEED per job; the seed check proposed in mlcommons/logging PR 479 flags consecutive seeds
export WALLTIME=UNLIMITED
# Both need root in run.sub. With CLEAR_CACHES=0 run.sub logs no cache_clear, which the compliance checker
# requires; our launcher drops the model and data from the page cache without root and logs it.
export VBOOST_VALUE=0 CLEAR_CACHES=0
export NCCL_NVLS_ENABLE=0  # no NVSwitch on H200 NVL
