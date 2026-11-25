# ---------- tunable ----------

# hyperparameters. RCP at GBS 8: a mean below 2960.6 samples fails, below 3178.7 is scaled up to it.
export LR=0.0004             # opt_base_learning_rate
export WARMUP_RATIO=0.0      # opt_learning_rate_warmup_ratio
export MAX_STEPS=1024        # opt_learning_rate_training_steps
export MINIBS=2              # global batch size = MINIBS x 8 / (TP x PP x CP): 8, 16, 32 or 128 (the sizes with RCPs)
export MBS=1                 # micro batch size; gradient accumulation steps = MINIBS / MBS

# parallelism
export TP=1
export PP=1
export CP=2                  # context parallel: each 8192-token sequence split over 2 GPUs, halves activation memory
export SP=False
export LORA_A2A=1
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

# precision
export FP8=True
export FP8_DPA=0             # 1 fails in this container
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

# activation checkpointing
export ACG=null              # recompute off (as AMD's MI325X config): CP=2 frees the memory
export ACM=null
export ACL=null              # with CP=1: layers recomputed, 0 to 80 (21 = minimum that fits)

# warm-up before the clock starts
export WARMUP=True

# communication
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32

# kernels
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_USE_RMSNORM_TRITON=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export USE_TE_SWIGLU=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export OMP_NUM_THREADS=1

# ---------- fixed by the rules ----------
export VAL_CHECK_INTERVAL=384                    # evaluate every 384 samples
export SKIP_EVALS=$((MINIBS*8/(TP*PP)/CP/8+2))   # skip FLOOR(0.125 x global batch size + 2) evaluations
export LIMIT_VAL_BATCHES=1.0
export LIMIT_TRAIN_BATCHES=1.0
export GRADIENT_CLIP_VAL=0.3
export LORA_DROPOUT=0.1
export LORA_ALPHA=32
export LORA_DIM=16
export WEIGHT_DECAY=0.0001
export MAX_SEQLEN=8192
export MIN_LR=0.0
export ANSWER_ONLY_LOSS=True
export LOAD_CKPT=True

# ---------- not for tuning: cn01, the log labels, the error output ----------
export DGXNGPU=8
export DGXNNODES=1
export MLPERF_SUBMISSION_ORG=BriCS
export MLPERF_SUBMISSION_PLATFORM=MI300X
export HSA_ENABLE_IPC_MODE_LEGACY=0      # needed by cn01's amdgpu driver
export NVIDIA_PYTORCH_VERSION=25.04      # needed by NeMo
export HYDRA_FULL_ERROR=1
export POSSIBLE_USER_WARNINGS=0

# ---------- machine settings (tunable), run once per job as root in a privileged container ----------
# AMD's runtime_tunables.sh without sudo: the container is privileged, so /proc/sys and /sys are the host's.
# scripts/mlperf.sbatch flushes the page cache before each run.
machine_settings() {
  rocm_agent_enumerator
  for f in /sys/devices/system/cpu/cpu*/cpuidle/state2/disable; do [ -e $f ] && echo 1 > $f; done           # cpupower idle-set -d 2
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -e $f ] && echo performance > $f; done   # cpupower frequency-set -g performance
  echo 0 > /proc/sys/kernel/nmi_watchdog
  echo 0 > /proc/sys/kernel/numa_balancing
  echo 0 > /proc/sys/kernel/randomize_va_space
  echo always > /sys/kernel/mm/transparent_hugepage/enabled
  echo always > /sys/kernel/mm/transparent_hugepage/defrag
  # amd-smi set -g all -o 750    # GPU power cap, watts per GPU: the same for every run of a result set
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null), numa_balancing $(cat /proc/sys/kernel/numa_balancing), THP $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
}
