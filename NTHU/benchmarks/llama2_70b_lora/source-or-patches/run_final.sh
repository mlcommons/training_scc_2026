#!/bin/bash
# One timed run of the final configuration, start to finish.
#
#   SEED=1234 nvidia/run_final.sh
#   SEED=1234 JOB=<id> RUN_DIR=<dir> nvidia/run_final.sh   # only wait for and process a running job
#   SEED=1234 SUBMIT_ONLY=1 DEPENDENCY=afterany:<id> nvidia/run_final.sh   # queue it and return
#
# SUBMIT_ONLY=1 submits and exits, writing the job id to $RUN_DIR/jobid; the caller is
# then responsible for extracting the result once the job has run -- which is what
# scripts/run_competition.sh's collector job does, via nvidia/extract_result.sh.
#
# 14 nodes (every node, head included), module power cap 580 W, fans 30%, GBS=14,
# evaluation every 392 samples, the offload config with LR 5.5e-4 / 600-step cosine.
# Waits for the job, extracts the MLPerf log to $RUN_DIR/result_seed$SEED.txt and
# runs the compliance checker on it. FAN_SPEED and POWER_CAP use the site's fan_speed
# and nv_power tools and are restored by a dependent job after the run; set them empty
# on a cluster without those tools.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
. "$ROOT/configs/site.env"
SEED="${SEED:?set SEED}"
RUN_DIR="${RUN_DIR:-$WORK_ROOT/runs/final/seed$SEED}"
if [ -n "${JOB:-}" ]; then jid=$JOB          # attach to a job that is already running
else
  # sbatch's "Submitted batch job N" reaches launch.sh's stderr, hence 2>&1.
  out=$(NNODES=14 EXCLUDE_NODES=none SEED=$SEED TIME=00:50:00 GPUMON=1 \
        POWER_CAP="${POWER_CAP-580}" FAN_SPEED="${FAN_SPEED-30}" RUN_DIR="$RUN_DIR" \
        PRECOOL="${PRECOOL:-}" PRECOOL_FAN="${PRECOOL_FAN:-50}" \
        FAN_RESTORE="${FAN_RESTORE:-50}" POWER_RESTORE="${POWER_RESTORE:-1000}" \
        DEPENDENCY="${DEPENDENCY:-}" \
        OVERRIDES="export VAL_CHECK_INTERVAL=392 OFFLOAD_TRACE=1 ${EXTRA_OVR:-}" \
        "$HERE/launch.sh" config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1) \
    || { echo "$out"; echo "launch.sh failed" >&2; exit 1; }
  # Without the || above, set -e aborts on the failed assignment before $out is ever
  # printed, and the caller gets an empty log instead of launch.sh's reason.
  echo "$out"
  jid=$(awk '/Submitted batch job/{print $4}' <<<"$out" | head -1)
fi
[ -n "$jid" ] || { echo "submit failed" >&2; exit 1; }
if [ "${SUBMIT_ONLY:-}" = 1 ]; then echo "$jid" > "$RUN_DIR/jobid"; echo "queued as $jid"; exit 0; fi
while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 30; done
rj=$(squeue -h -n nv-lora-restore -o %i | head -1)
while [ -n "$rj" ] && squeue -h -j "$rj" 2>/dev/null | grep -q .; do sleep 10; done
exec "$HERE/extract_result.sh" "$RUN_DIR" "$RUN_DIR/result_seed$SEED.txt"
