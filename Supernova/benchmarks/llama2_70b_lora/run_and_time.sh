#!/bin/bash
# One timed llama2_70b_lora run: GBS 8, TP1 x CP2 x DP8 on 2 nodes x 8 GPUs (NODELIST=node0,node1, node0 = this
# host). SEED selects the run (default 12552). Env spec: config.sh next to this file. Launcher, setup and
# patches: source-or-patches/. Result log: results/rcp-s<SEED>/node0.log under the repository root.
set -e
D=$(cd "$(dirname "$0")" && pwd)
SEED=${SEED:-12552}
export SUBMISSION_CONFIG=$D/config.sh
exec bash "$D/source-or-patches/h200-tuning/run_rcp_ref8.sh" "$SEED" "$SEED"
