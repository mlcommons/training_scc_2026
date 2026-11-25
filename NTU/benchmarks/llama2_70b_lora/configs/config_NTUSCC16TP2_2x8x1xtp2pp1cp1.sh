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

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# overwrite common parameters
export TP_COMM_OVERLAP=True
export UB_SKIPMC=1

# remove specific environment variables
#unset NCCL_MIN_P2P_NCHANNELS
#unset NCCL_MIN_CTAS
#unset NCCL_NCHANNELS_PER_NET_PEER
#unset MC_TP_OVERLAP_AG
#unset MC_TP_OVERLAP_RS
#unset MC_TP_OVERLAP_RS_DGRAD

# hyperparameters
export MAX_STEPS=800
export LR=0.0005                    # Fujitsu/NVIDIA-tuned LR
export MINIBS=1
export TP=2
export CP=1
#export TP_COMM_OVERLAP=1
export VBOOST_VALUE=1
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export SKIP_EVALS=3

# system parameters

# TP=2 does not fit without freeing activation memory: weights are 64.3 GiB/GPU and
# activations ~68.8 GiB, totalling 133.1 GiB on a 139.8 GiB card. Block recompute of
# the first N layers frees ~0.86 GiB/layer; measured boundary is N=14 fits, N=13 OOMs.
# N=16 is used rather than the 3.894 s N=14 because that 0.4% is inside the 0.27%
# noise floor, and these numbers come from 12-step runs with NO eval phases -- a full
# convergence run evaluates 5-6 times and needs the extra ~0.9 GiB of headroom.
#   N=80 (uniform) 5.145 s | N=24 4.040 | N=16 3.910 | N=14 3.894 | N=13 OOM
# vs TP=4 on the same 16 GPUs: 4.156 s.
export RECOMPUTE_GRANULARITY=full
export RECOMPUTE_METHOD=block
export RECOMPUTE_NUM_LAYERS=16

export DGXNNODES=2
export WALLTIME_RUNANDTIME=300
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
