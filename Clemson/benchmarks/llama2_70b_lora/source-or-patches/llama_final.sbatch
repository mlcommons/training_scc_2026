#!/bin/bash
#SBATCH -J llama-final
#SBATCH -p debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH -t 05:00:00
#SBATCH -o /home/cjgeo/training/logs/llama-final-%j.out
/share/software/comp_scripts/peak_power.sh start
export MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun --label bash /home/cjgeo/training/llama2_70b_lora/run_llama_final.sh
echo "job $SLURM_JOB_ID seed $SEED $(TZ=America/Chicago date)" >> /home/cjgeo/training/logs/peak_power
/share/software/comp_scripts/peak_power.sh stop >> /home/cjgeo/training/logs/peak_power
