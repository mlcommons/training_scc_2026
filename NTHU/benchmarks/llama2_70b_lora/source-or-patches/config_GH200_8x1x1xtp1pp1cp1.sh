#!/bin/bash
# NVIDIA v5.0 llama2_70b_lora on 8 nodes x 1 GH200, TP=PP=CP=1, GBS=8.
#
# Starts from NVIDIA's own 8-GPU TP=1 config (config_GB200_2x4x1xtp1pp1cp1.sh)
# and takes the hyperparameters Krai used on GH200 in v6.0, where 10 of 10 runs
# converged in 2688-3072 samples.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters -- Krai's GH200 values (config_gh200.sh, v6.0)
export MAX_STEPS=800
export LR=0.00055
export MINIBS=1
export TP=1
export SP=0
export CP=1
export FP8_ACT=1

# Memory at TP=1 on 96GB; see nvidia/patch_nvidia_impl.py. These are the
# patched-in knobs, set explicitly here so a config reader sees them.
export DISABLE_TRANSPOSE_CACHE=1
export ACG=full
export ACM=uniform
export ACL=1
# As Krai's config_gh200.sh.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

# CUDA graphs off to start with: capture under full recompute is untested here.
export LAYER_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

# No NVLink switch on this cluster.
export NCCL_NVLS_ENABLE=0

# system parameters
export DGXNNODES=8
export DGXNGPU=1
