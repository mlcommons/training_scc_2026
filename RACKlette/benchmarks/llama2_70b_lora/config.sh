#!/bin/bash
# RACKlette, SC26: 2 nodes x 8 H200 NVL (two NVLink groups of 4 per node), TP=4 SP=1 DP=4, GBS 8.
# Unmodified vendor code (no patches), TP communication overlap off.
# Values are the ones our launcher logged for every run of this set (PIPE_CONFIG line in each log).
# Copy next to config_common.sh in Dell/benchmarks/llama2_70b_lora/implementations/nemo
# (mlcommons/training_results_v6.0 at eabf23a).

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export GBS=8
export MBS=1
export VAL_MBS=1
export MINIBS=2       # GBS / DP
export MAX_STEPS=672  # cosine horizon, 5376 samples
export WARMUP_STEPS=0
export LR=0.00045
export TP=4
export SP=1
export CP=1
export PP=1
export MCORE_CUDA_GRAPH=1
export CUDA_GRAPH_SCOPE=full_iteration
export FP8_PARAM_GATHER=0
export NUM_WORKERS=4
export GRADIENT_ACCUMULATION_FUSION=0
export PARALLEL_LOAD=1
export TP_COMM_OVERLAP=False
unset PYTORCH_CUDA_ALLOC_CONF  # expandable_segments deadlocked 2-node CUDA graph capture

# system parameters
export DGXNNODES=2
export DGXNGPU=8
export WALLTIME_RUNANDTIME=30
export NEXP=1  # one training per launch; SEED is set per run (10 distinct random seeds for this set)
export WALLTIME=UNLIMITED
# Both need root in run.sub. With CLEAR_CACHES=0 run.sub logs no cache_clear, which the compliance checker
# requires; our launcher drops the model and data from the page cache without root and logs it.
export VBOOST_VALUE=0 CLEAR_CACHES=0
export NCCL_NVLS_ENABLE=0  # no NVSwitch on H200 NVL
