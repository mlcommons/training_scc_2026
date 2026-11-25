#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00          # the partition default is 4 hours and ten runs need about 6
# One batch of MLPerf Training runs: Llama 2 70B LoRA on 8x MI300X in AMD's v5.1 container.
# Each run is implementation/run_and_time.sh in the container. result_<i>.txt keeps its :::MLLOG lines, log.txt everything.
# Containers run privileged, with AMD's run_with_docker.sh flags; compiled kernels are kept in a Docker volume on the node.
# ./run.sh <config> [runs] submits it from the repository folder as:
#   sbatch --job-name=mlperf_<config> --output=<results folder>/log.txt scripts/mlperf.sbatch <results folder> <runs>
# Slurm runs the copy taken at submission: a change here applies to jobs submitted after it.
CONT=rocm/amd-mlperf@sha256:568de7713c8af69f853e3d087b2377805d77527ed71fdea3f95f7666046c8519   # tag llama2_70b_training_5.1
TEAM=Isambards   # the submission folder name in the checks
SPARES=4         # spare seeds for runs that do not converge
START_LIMIT=600  # seconds a run may take to log run_start. A normal start takes about 270
RUN_LIMIT=3000   # seconds a run may take in total. A run of 3840 samples takes about 2400

set -euo pipefail

OUT=$1
RUNS=$2
cd "${SLURM_SUBMIT_DIR:-.}"
now_ms() { local t=${EPOCHREALTIME/[.,]/}; echo $(( t / 1000 )); }   # milliseconds; date +%s%3N is wrong on Ubuntu 26.04 (Rust coreutils)
JOB_START=$(now_ms)
echo "=== commit $(git rev-parse --short HEAD) ($(git diff --quiet HEAD && echo clean || echo changed files))"
SOURCE=$OUT/source
mkdir -p "$SOURCE"
cp -a implementation patches configs scripts systems run.sh README.md "$SOURCE/"
source $OUT/config.sh
# Apply the reviewed patches named in PATCHES to this job's copy of AMD's code
for p in ${PATCHES:-}; do git apply --directory="$SOURCE/implementation" "patches/$p.patch"; done
GBS=$(( MINIBS * 8 / (TP * PP) / CP ))

docker image inspect $CONT >/dev/null 2>&1 || { echo "=== container (13 GB)"; docker pull -q $CONT >/dev/null; }
# Compiled kernels kept between runs and jobs, per container image: Triton's (TE's Triton kernels), Inductor's
# (torch.compile) and ROCm's code-object cache (hipRTC). Each run starts a new container, so without this every run
# compiles them again. Compiled code only, no data: the page-cache flush is unchanged.
# In a Docker volume, on the node's own disk: in cache/ on BeeGFS, where the 8 ranks write the same kernel files at the
# same time, a rank could not read a kernel file the cache listed (FileNotFoundError .../_rmsnorm_bwd_dg_reduce_triton.hsaco,
# run 1022-1553). docker volume rm <volume> starts from nothing.
CACHE=mlperf_cache_$(docker image inspect -f '{{.Id}}' $CONT | cut -d: -f2 | cut -c1-12)
docker volume create $CACHE >/dev/null
echo "=== compile cache: volume $CACHE, $(docker volume inspect -f '{{.Mountpoint}}' $CACHE)"
# What needs root runs in a privileged container with AMD's run_with_docker.sh flags, without sudo, as agreed with
# cn01's admin: the machine settings and the page-cache flush write the host's /proc/sys and /sys from it.
as_root() {
  docker run --rm --privileged --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --group-add video \
    -v "$PWD/$OUT:/results/artifacts" $CONT bash -c "$1"
}
echo "=== machine settings"
as_root "$(declare -f machine_settings); machine_settings"
docker rm -f mlperf_run >/dev/null 2>&1 || true   # left by a cancelled job
# The container is not a process of this job: scancel or the time limit would leave it running on the GPUs. Remove it
# when Slurm ends the job. The run goes in the background (wait) so that the trap runs as soon as Slurm sends TERM.
stop_container() { docker rm -f mlperf_run >/dev/null 2>&1 || true; }
trap 'stop_container; exit 143' TERM INT
# The certification rules ask for Chicago timestamps just before and after the benchmark
echo "$(TZ=America/Chicago date)" > $OUT/mlperf.tstamps
trap 'stop_container; echo "$(TZ=America/Chicago date)" >> $OUT/mlperf.tstamps' EXIT

# SEEDS in the config replaces the random seeds
seeds=(${SEEDS:-$(shuf -i 1-32767 -n $((RUNS + SPARES)))})
i=0
attempt=0
bad=0
row=0
# Run until RUNS runs have converged
while [ $i -lt $RUNS ]; do
  # More random seeds once the list is used up
  seed=${seeds[$attempt]:-$(shuf -i 1-32767 -n 1)}
  attempt=$((attempt + 1))
  echo "=== run $((i + 1)) of $RUNS, attempt $attempt, seed $seed, global batch size $GBS"
  # MLPerf: "You must flush the cache or restart the system prior to benchmarking."
  as_root 'c() { grep ^Cached: /proc/meminfo | tr -s " " | cut -d" " -f2; }; b=$(c); sync && echo 3 > /proc/sys/vm/drop_caches && echo "page cache flushed: $((b / 1048576)) -> $(( $(c) / 1048576 )) GiB"' \
    || { echo "ERROR: the page-cache flush failed"; exit 1; }
  LAUNCH=$(now_ms)
  status=0
  # inside the container: AMD's code from implementation/, the config and the seed, as AMD's run_with_docker.sh
  ( docker run --rm --init --name mlperf_run --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined \
    --privileged --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --group-add video \
    -v "$PWD/data:/data" -v "$PWD/data/model:/ckpt" -v "$PWD/$OUT:/results/artifacts" \
    -v "$PWD/$SOURCE/implementation:/workspace/code:ro" -v "$CACHE:/cache" \
    -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e AMD_COMGR_CACHE_DIR=/cache/comgr \
    $CONT bash -c "
$(cat $OUT/config.sh)
export SEED=$seed
bash run_and_time.sh" | tee $OUT/result_$i.txt ) &
  run=$!
  # Stop a run that stalls before training or runs longer than any converging run
  ( sleep $START_LIMIT
    grep -q '"key": "run_start"' $OUT/result_$i.txt || { echo "ERROR: no run_start after $START_LIMIT s"; docker rm -f mlperf_run >/dev/null; }
    sleep $((RUN_LIMIT - START_LIMIT))
    echo "ERROR: still running after $RUN_LIMIT s"; docker rm -f mlperf_run >/dev/null ) &
  watch=$!
  wait $run || status=$?   # the run's exit status (pipefail); returns at once when Slurm sends TERM
  kill $watch 2>/dev/null || true
  sed -i -n 's/^.*:::MLLOG/:::MLLOG/p' $OUT/result_$i.txt   # the result file keeps only the :::MLLOG lines, as AMD's
  as_root "chown -R $(id -u):$(id -g) /results/artifacts"   # files written in the container belong to root
  if [ $status -eq 0 ] && grep -q '"key": "run_stop".*"status": "success"' $OUT/result_$i.txt; then
    echo "result_$i $LAUNCH $(now_ms)" >> $OUT/times.txt
    i=$((i + 1))
    row=0
    continue
  fi
  # A run that did not converge keeps its log under another name and the next seed takes its place
  bad=$((bad + 1))
  row=$((row + 1))
  if grep -q '"key": "run_start"' $OUT/result_$i.txt; then kind=failed; else kind=not_started; fi
  mv $OUT/result_$i.txt $OUT/${kind}_$bad.txt
  echo "ERROR: attempt $attempt $kind with exit status $status. $bad set aside so far"
  [ $row -lt 3 ] || { echo "ERROR: three attempts in a row did not converge. Stopping the job"; exit 1; }
done
echo "job $JOB_START $(now_ms)" >> $OUT/times.txt

if [ "$RUNS" = 10 ]; then
  PY=$PWD/.venv/bin/python
  # Exit status 2 means a partial set that is packaged for the organisers but cannot pass the checks
  full=0
  PKG=$($PY scripts/package_results.py "$OUT" --team "$TEAM") || full=$?
  [ $full -eq 0 ] || [ $full -eq 2 ] || exit 1
  if [ $full -eq 2 ]; then echo "Partial submission: $PKG"; echo; scripts/results.sh $OUT; exit 1; fi
  LOG=$PWD/$OUT/package_checker.log
  REPO_LOG=$PWD/$OUT/repo_checker.log
  : > "$LOG"
  : > "$REPO_LOG"
  (
    cd "$(dirname "$PKG")"
    "$PY" -m mlperf_logging.package_checker "$TEAM" training 6.0.0 --werror --log_output "$LOG"
    grep -Fq "PACKAGE CHECKER FOUND NO ERRORS, SUCCESS !" "$LOG" || { echo "Package checker failed: $LOG"; exit 1; }
    SCALE="$TEAM/results/student_system/llama2_70b_lora/scaling.json"
    if [ ! -e "$SCALE" ]; then printf '%s\n' '{"scaling_factor": 1.0}' > "$SCALE"; fi
    "$PY" -m mlperf_logging.result_summarizer "$TEAM" training 6.0.0
    "$PY" -m mlperf_logging.repo_checker "$TEAM" training 6.0.0 --log_output "$REPO_LOG"
    grep -Fq "ALL CHECKS PASSED." "$REPO_LOG" || { echo "Repository checker failed: $REPO_LOG"; exit 1; }
  )
  cp "$(dirname "$PKG")/summary.json" "$OUT/"
  cp "$PKG/results/student_system/llama2_70b_lora/scaling.json" "$OUT/"
  echo "Submission: $PKG"
fi

echo
scripts/results.sh $OUT
