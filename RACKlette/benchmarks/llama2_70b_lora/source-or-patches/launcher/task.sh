#!/bin/bash
# task.sh: per-rank launcher (Slurm task ids -> torch.distributed env)
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$(( SLURM_NTASKS / SLURM_NNODES ))
cd "$FTLLM"
exec "$VENV/bin/python" -u train.py
