#!/bin/bash
# SCC26 — MLPerf Training Llama2-70B LoRA, configuration 12 GPU du 16/11 : deuterium (8 x H200 NVL) + GPU 0-3 de
# tritium (son îlot NVLink 0-3). L'îlot 4-7 de tritium est écarté : il provoque des Xid 95 (erreur mémoire non contenue,
# FBHUB) au chargement du checkpoint (journal brain/wiki/journals/2026-11-16.md). Les deux nœuds sont alloués en entier
# (personne d'autre sur tritium 4-7), mais seuls 12 rangs tournent : RANK_LAYOUT (helma_runs/run_helma.sub).
#
# Découpage : 3 îlots NVLink de 4 GPU = TP4 + SP dans chaque îlot, DP3 entre les îlots (PCIe dans deuterium, InfiniBand
# vers tritium). Le GBS 8 est impossible à 12 GPU (DP3) : GBS 9 = MINIBS 3 x DP3, avec LR 0.0005 et 800 steps max,
# comme la config officielle NVIDIA MLPerf v6.0 à GBS 9 (nemo/config_GB300_18x4x1xtp1pp1cp8.sh). RCP : interpolé par
# rcp_checker entre GBS 8 (3187 samples) et 16 (4051), ~3300 samples. Évaluation : tous les ceil(384/9) = 43 steps
# (387 samples), 3 sautées -> 1re à 1548 samples (règle « CEIL(384/GBS) * GBS »).
_scc_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${_scc_dir}/scc.env"
: "${RESULTS_ROOT:=${_scc_dir}/segfault_ngpu12/runs}"
: "${SEED_BASE:=${RANDOM}}"     # essais du job : SEED_BASE, SEED_BASE+1, ... (graines aléatoires et distinctes)
: "${NEXP:=10}"                 # 10 essais exigés (score olympique)
: "${SBATCH_EXCLUDE:=}"
: "${STAGE_TO_TMPDIR:=0}"
: "${BLOCKED_GPU_DRIVERS:=610.57.04,615.71.09}"
: "${SEG_PARTITION:=gpu}"
: "${SEG_CPUS_PER_TASK:=48}"    # 384 CPU / 8 rangs (tritium : 4 rangs x 48)
: "${MODEL:=/fs/largenfs/hpc-workspace/data/stephan-llama2_70B/segfault_ngpu16/model}"
: "${NCCL_TEST:=0}"             # le pré-test (tools/nccl_check.py) suppose autant de rangs par nœud
export RESULTS_ROOT SEED_BASE NEXP SBATCH_EXCLUDE STAGE_TO_TMPDIR BLOCKED_GPU_DRIVERS MODEL NCCL_TEST
source "${ZEPH_DIR}/helma_runs/configs/config_HelmaH100_common.sh"
export SBATCH_SWITCHES=none

export RUN_ID=SEG12
export RUN_NAME=segfault_h200nvl_ngpu12
export MLPERF_SYSTEM=student_system
export MLPERF_SUBMISSION_ORG="${TEAM_NAME}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SYSTEM}"
export DGXSYSTEM=SegFAUlt_H200NVL_8+4x3xtp4pp1cp1
export DGXNNODES=2
# Vu par NeMo pour le calcul du batch (GBS = MINIBS x DGXNNODES x DGXNGPU / (TP x PP x CP)) : 2 x 6 = 12 GPU au total.
# Le placement réel (8 + 4) vient de RANK_LAYOUT ; l'allocation Slurm reste de 8 GPU par nœud (SBATCH_EXTRA).
export DGXNGPU=6
export RANK_LAYOUT="deuterium:8,tritium:4"
export TP=4 PP=1 CP=1 SP=1
export MINIBS=3                 # GBS = 3 x DP3 = 9 (3 micro-batchs par step)
export LR=0.0005                # LR de NVIDIA à GBS 9 (au lieu de 0.00055 à GBS 8)
export GPU_MEM_GIB=131
export SBATCH_EXTRA="--partition=${SEG_PARTITION} --nodelist=deuterium,tritium --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=${SEG_CPUS_PER_TASK}"
# Limite de puissance posée par le lanceur au début du job sur tous les GPU de ses nœuds (run_helma.sub) : 550 W,
# choisie le 16/11 (450 W bridait : sw_power_cap actif sur les 8 GPU, 5,02 s/step). Le comité autorise une autre limite pour HPL et HPL-MxP (confirmé le 16/11).
: "${GPU_POWER_LIMIT:=550}"
export GPU_POWER_LIMIT
# Fréquence SM verrouillée à 1300 MHz sur tous les GPU pendant le job (run_helma.sub : -lgc au début, -rgc à la fin),
# choisie le 16/11 : horloge identique sur les 12 GPU, puissance plus basse et plus stable.
: "${GPU_CLOCK_LOCK:=1300}"
export GPU_CLOCK_LOCK
export SCC_SYSTEM_PROFILE=profil_systeme.json

export WALLTIME_RUNANDTIME=70   # limite d'un essai : 800 steps x ~3,8 s + 18 évaluations + init ≈ 60 min
: "${TRIAL_BUDGET:=35}"         # essai attendu : ~3 min d'init + ~25 min de TTT
export WALLTIME=$(( 15 + NEXP * TRIAL_BUDGET ))
