#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=True
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1
export NEMO_LORA_USE_THUNDER_DROPOUT=1

export POSSIBLE_USER_WARNINGS=0
# export NVTE_FLASH_ATTN=0
# export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Disable caching NCCL communication buffer
export NCCL_NVLS_ENABLE=0 # Disable NVL SHARP, which don't use
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8=True
export FP8_AMAX_ALGO=max
#export FP8_REDUCE_AMAX=False
#export FP8_AMAX_HISTORY=32
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=4
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export HF_HUB_OFFLINE=1
export TORCH_NCCL_HIGH_PRIORITY=1

export HYDRA_FULL_ERROR=1

export PP=1
export SP=1

export USE_TE_OPTS=1
export FUSED_SINGLE_QKV_ROPE=1

export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

export WARMUP=True
export WARMUP_TRAIN_STEPS=5
export WARMUP_VALIDATION_STEPS=5

# other
export MBS=1
export VAL_CHECK_INTERVAL=384
export LOAD_CKPT=True
