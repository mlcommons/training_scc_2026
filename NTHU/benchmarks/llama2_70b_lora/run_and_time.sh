#!/bin/bash
# MLPerf Training v6.0 -- llama2_70b_lora: one timed run.
#
#   SEED=1234 ./run_and_time.sh
#
# The timed region is the trainer's own run_start -> run_stop, logged by rank 0 into
# the MLPerf log; TTT is the difference. Nothing here pauses the clock.
#
# This is a thin wrapper: the submitted runs were produced by scripts/run_competition.sh,
# which queues all ten of them as one Slurm dependency chain so that no two runs ever
# share a node or an output directory. One run of that chain is exactly this:
#
#   nvidia/run_final.sh  -- 14 nodes, module cap 580 W, fans 30%, 100 s pre-cool,
#                           then nvidia/launch.sh with the config below
#   nvidia/launch.sh     -- sbatch + srun + apptainer exec, applies the patches in
#                           source-or-patches/ to NVIDIA's v5.0 implementation
#
# Both are in source-or-patches/, together with the configs and every patch.
set -euo pipefail
: "${SEED:?set SEED -- the ten submitted runs used 1234 2468 4680 5678 5791 3579 6802 7913 8024 9135}"
REPO="${REPO:-$HOME/SC26-MLPerf}"           # this repository
RUN_DIR="${RUN_DIR:-$PWD/run_seed$SEED}"

mkdir -p "$RUN_DIR"
SEED="$SEED" RUN_DIR="$RUN_DIR" POWER_CAP=600 FAN_SPEED=30 PRECOOL=100 PRECOOL_FAN=100 \
  "$REPO/nvidia/run_final.sh"

# run_final.sh waits for the job, then extracts the MLPerf log of the run into
# $RUN_DIR/result_seed$SEED.txt and runs the 6.0.0 compliance checker on it.
