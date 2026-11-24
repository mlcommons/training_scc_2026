#!/bin/bash
# Scored configuration: Llama-2-70B LoRA, MLPerf Training v6.0 (closed).
# Source this file; run_and_time.sh does it for you.
#
# Profile: one 8-GPU NVLink node (8x H200 NVL, 141 GB/GPU here). TP=4/DP=2 is
# memory-forced: 70B in bf16 is ~138 GB before activations, so TP=1 cannot fit
# on one GPU. TP=4 also fits a 2x4 (two 4-GPU trays) allocation.

# ---- paths (placeholders; override in the environment) ----
: "${DATADIR:=/path/to/gov_report}"                  # train.npy + validation.npy
: "${CKPT:=/path/to/model_checkpoint}"               # TP=2/PP=1 distcp; loaded with TP=4 reshard
: "${LOGDIR:=/path/to/logs}"
: "${CONTAINER:=/path/to/pytorch_26.04-py3.sif}"     # optional; see run_and_time.sh

# ---- parallelism: 8 GPUs, one node ----
export NPROC=${NPROC:-8}
export TP=${TP:-4}
export PP=1 CP=1 SP=0
export DP=$((NPROC / (TP*PP*CP)))
export GBS=${GBS:-8}
export MINIBS=${MINIBS:-$((GBS / DP))}               # YAML: GBS = MINIBS * DP (CP=1)
export MBS=1 VAL_MBS=1

# ---- hyperparameters ----
# LR is an unconstrained knob. 4e-4 is the value recorded in the v6.0 RCP
# data for this benchmark; hotter points (5e-4, 6e-4, 7e-4) converged faster
# than the reference and are rejected by the RCP rules.
export LR=${LR:-0.0004}
export WARMUP_STEPS=${WARMUP_STEPS:-11}              # 11/550 = 0.02 warmup ratio
export MAX_STEPS=${MAX_STEPS:-550}                   # cosine horizon; runs stop at target
export SEED=${SEED:-1}
export NUM_WORKERS=${NUM_WORKERS:-4}
export GPU_ARCH=${GPU_ARCH:-h100}                    # H200 is Hopper: conf/tp_overlap/h100*

# ---- precision / communication environment ----
# Values copied verbatim from NVIDIA's v6.0 config_common.sh so this folder is
# self-contained. Changing them changes the recipe, not just the launcher.
export FP8=True FP4=False
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=4
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=True
export LORA_A2A=1
export NEMO_LORA_USE_THUNDER_DROPOUT=1
export USE_TE_OPS=1
export FUSED_SINGLE_QKV_ROPE=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# Vendor startup warmup callback (distinct from the 11-step LR warmup).
export WARMUP=True WARMUP_TRAIN_STEPS=5 WARMUP_VALIDATION_STEPS=5
export LOAD_CKPT=True
export VAL_CHECK_INTERVAL=384                         # 384/GBS = 48 steps between evals

# 26.04 SIF: the bridge's startup torch.compile(bias_swiglu) warmup cannot find
# libcuda. Disabling it is safe (it is not used by the 70B GEMMs); set 0 to
# re-enable once TRITON_LIBCUDA_PATH resolves libcuda.
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}

# MLPerf identity; train.py logs these as submission_org / submission_platform.
# Set MLPERF_SUBMISSION_ORG to the submitting organization and keep the
# platform consistent with the separate systems artifact.
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-GigEmBytes}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-H200_NVL_8x}"
export DGXSYSTEM=${DGXSYSTEM:-H200_NVL_8x}

# ---- eval cadence ----
# Scored runs must NOT set SKIP_EVALS. The vendor YAML then computes
# skip_evals = floor(0.125*GBS + 2) = 3 for GBS=8, so the first logged eval is
# step 192, then every 48 steps. SKIP_EVALS=1 is only for short screens; an
# empty value is unset below because the YAML uses oc.decode on it.
if [ -n "${SKIP_EVALS:-}" ]; then
  export SKIP_EVALS
else
  unset SKIP_EVALS   # empty would make the YAML's oc.decode fail
fi
