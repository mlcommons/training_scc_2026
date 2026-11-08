#!/bin/bash
set -u
echo "=== CPU check ==="
echo "nproc (honors OMP_NUM_THREADS): $(nproc)"
echo "nproc --all: $(nproc --all)"
echo "OMP_NUM_THREADS unset: $(env -u OMP_NUM_THREADS nproc)"
grep Cpus_allowed_list /proc/self/status
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
export USE_ROCM_AITER_ROPE_BACKEND=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DATADIR=$PWD/data
export LOGDIR=$PWD/results
export HF_HOME=/projects/scc/mlperf/mi300x/huggingface
export MLLOG_TRAIN_LOSS_LOG_FREQ=1
export CONT=rocm/amd-mlperf:llama2_70b_training_6.0


cd /workspace/code
source config_MI300X_1x8x1.sh

: "${NEXP:=10}"
: "${LOGDIR:=/results}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.0.0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
_logfile_base="${LOGDIR}/${DATESTAMP}"
mkdir -p "${LOGDIR}"

for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    export SEED=$( [ -n "${SEED_LIST:-}" ] && echo $SEED_LIST | cut -d" " -f${_experiment_index} || echo $RANDOM )
    echo "launching experiment using SEED=${SEED} /workspace/code/run_and_time.sh"
    bash /workspace/code/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log" \
    || echo "WARNING: trial ${_experiment_index} exited non-zero"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
    python3 -m mlperf_logging.compliance_checker --usage training \
         --ruleset "${MLPERF_RULESET}" \
         --log_output "${LOGDIR}/compliance_${DATESTAMP}.out" \
         "${_logfile_base}_${_experiment_index}.log" \
    || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi
done
