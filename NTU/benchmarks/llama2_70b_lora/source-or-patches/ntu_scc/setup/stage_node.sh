#!/bin/bash
# stage_node.sh <node-ibhost>  -- copy the llama2 LoRA local assets (container, dataset, model) from this node's
# /scratch to the same paths on another node, over IB, 6 files in parallel (16 trips sshd MaxStartups). Idempotent (rsync).
set -u
D=${1:?target, e.g. ntu-scc03-ib5}
M=/scratch/benchmark/MLPerf/data/models/llama2_70b_megatron_bridge
t0=$(date +%s)
ssh $D "mkdir -p /scratch/containers /scratch/benchmark/MLPerf/data/dataset $M /scratch/enroot-data /scratch/enroot-cache /scratch/enroot-tmp /scratch/enroot-runtime"
rsync -a /scratch/containers/l2l-nemo.sqsh $D:/scratch/containers/ &
rsync -a /scratch/benchmark/MLPerf/data/dataset/nemo_npy $D:/scratch/benchmark/MLPerf/data/dataset/ &
( cd $M && find . -type f | xargs -P 6 -I{} rsync -aR {} $D:$M/ ) &
wait
echo "STAGE DONE in $(( $(date +%s) - t0 )) s"
