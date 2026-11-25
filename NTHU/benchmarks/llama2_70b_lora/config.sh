#!/bin/bash
# MLPerf Training v6.0 -- llama2_70b_lora, closed division
# SCC26 student cluster: 14 x NVIDIA GH200 96GB, one GPU per node.
#
# This is the configuration of every one of the ten submitted runs. The authoritative
# copies of the two files it mirrors are in source-or-patches/:
#   config_GH200_8x1x1xtp1pp1cp1.sh         base config (hyperparameters, memory knobs)
#   config_GH200_8x1x1xtp1pp1cp1_offload.sh what the runs actually load (activation offload)
# and the launcher that applies them is source-or-patches/launch.sh.

# ---- what the rules fix (CLOSED division) --------------------------------------
# lora_rank 16, lora_alpha 32, lora_dropout 0.1, opt_gradient_clip_norm 0.3,
# opt_weight_decay 1e-4, sequence_length 8192, LoRA on qkv_proj and o_proj only.
# Evaluation every CEIL(384/GBS)*GBS = 392 samples, skipping the first
# FLOOR(0.125*14 + 2) = 3 evaluations. Target: eval loss <= 0.925.

# ---- scale and parallelism ------------------------------------------------------
export DGXNNODES=14            # every node, head included
export DGXNGPU=1               # one GH200 per node
export TP=1                    # tensor parallel
export PP=1                    # pipeline parallel
export CP=1                    # context parallel
export SP=0                    # sequence parallel
export MINIBS=1                # micro-batch per GPU -> GBS = 14 x 1 x 1 = 14
export VAL_CHECK_INTERVAL=392  # evaluate every 392 samples (28 steps)

# ---- unconstrained hyperparameters ---------------------------------------------
export LR=0.00055              # opt_base_learning_rate
export MAX_STEPS=600           # opt_learning_rate_training_steps (cosine length)
                               # Warmup ratio is 0, as in the reference.

# ---- precision ------------------------------------------------------------------
# "Any type of precision is okay." The frozen 69B base weights are stored and
# computed in FP8 (FP8=True, inherited from NVIDIA's config_common.sh); 138 GB of
# bf16 weights would not fit in 96 GB. The attention backward stays FP8.
export FP8=True
export FP8_ACT=1
export NVTE_FP8_DPA_BWD=1

# ---- memory: activations to Grace instead of recompute --------------------------
# At CP=1 on 96 GB the activations do not fit, so they are offloaded over
# NVLink-C2C to the 480 GB of Grace LPDDR5X rather than recomputed.
# See source-or-patches/cpu_offload_patch.py. Megatron refuses recompute together
# with cpu_offloading, hence ACG=none.
export ACG=none
export CPU_OFFLOADING=True
export CPU_OFFLOADING_NUM_LAYERS=79    # of 80; the last layer stays on the GPU
export OFFLOAD_UNTAGGED_MIN_MB=8
export DISABLE_TRANSPOSE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

# ---- cluster / fabric -----------------------------------------------------------
export NCCL_NVLS_ENABLE=0      # no NVLink switch between nodes
export NCCL_IB_HCA=mlx5_0      # mlx5_1 has no link on these nodes
export LAYER_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

# ---- power and thermals (SCC26: 10 kW for the whole cluster) --------------------
# Not MLPerf settings, but they are part of how these numbers were produced:
# Grace+Hopper module cap 580 W per node, chassis fans fixed at 30%, CPU governor
# schedutil with a 2.0 GHz ceiling, and a 100 s fan pre-cool before each run
# (before the trainer starts, so off the clock).
