#!/bin/bash
# As config_GH200_8x1x1xtp1pp1cp1.sh, with the saved activations kept in Grace
# memory (NVLink-C2C) instead of being recomputed. See nvidia/cpu_offload_patch.py.
#
# Megatron refuses recompute together with cpu_offloading, so ACG=none here.
# CPU_OFFLOADING_NUM_LAYERS must be below num_layers (80); the last layer's
# activations stay on the GPU.

source $(dirname ${BASH_SOURCE[0]})/config_GH200_8x1x1xtp1pp1cp1.sh

export ACG=none
# Cosine length. The base config's 800 was replaced by 600 after a 15-seed arm
# (LR 5.5e-4, 14 nodes, 575 W): mean 3842 samples to converge against 3920
# for 800 steps, 33% of runs at 3528, no throttling; rule verdict ADOPT.
export MAX_STEPS=${MAX_STEPS:-600}
export CPU_OFFLOADING=True
export CPU_OFFLOADING_NUM_LAYERS=${CPU_OFFLOADING_NUM_LAYERS:-79}
# Untagged saved tensors of at least this size are offloaded too. 0 (only what
# TE and the LoRA adapter tag) grew 1.56 GiB per layer in job 234: Megatron's
# fused SwiGLU and friends save untagged tensors.
export OFFLOAD_UNTAGGED_MIN_MB=${OFFLOAD_UNTAGGED_MIN_MB:-8}

# Attention backward. 1 (FP8 flash backward) is what every measured run of this
# repo used, including the ten-seed rehearsal and the 15-seed LR arm. It is set
# explicitly because config_common.sh exports 1, so an "unless set" default of 0
# here never took effect (found in the 2026-09-21 Nsight trace). 0 selects the
# bf16 backward: job 251 measured it 3.7% faster per step, but the convergence
# statistics and the RCP check were not repeated with it, so pass it through
# OVERRIDES only as an experiment.
export NVTE_FP8_DPA_BWD=1
