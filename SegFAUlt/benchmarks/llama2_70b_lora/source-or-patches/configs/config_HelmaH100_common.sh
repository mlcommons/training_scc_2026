#!/bin/bash
# Réglages communs à tous les runs Helma H100 (partition h100 : 4x H100 94 Go / nœud, pas de NVSwitch).
# Ordre de chargement : nemo/config_common.sh (valeurs NVIDIA) -> ce fichier (adaptations Helma)
# -> config_HelmaH100_Rxx_*.sh (ce qui distingue le run). Les VAR=valeur passées à submit.sh
# sont appliquées en dernier. Justification de chaque valeur : docs/02_adaptation_helma.md

_cfg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HELMA_DIR=$(cd "${_cfg_dir}/.." && pwd)
export ZEPH_DIR=$(cd "${HELMA_DIR}/.." && pwd)
export NEMO_DIR="${ZEPH_DIR}/nemo"

source "${NEMO_DIR}/config_common.sh"

# ---------------------------------------------------------------- chemins (surchargeables)
: "${CONT:=${NEMO_DIR}/nvidia-zeph.sif}"          # seul conteneur NeMo/Megatron-Bridge (PyTorch 26.04, CUDA 13.2)
: "${DATADIR:=${ZEPH_DIR}/data/gov_report}"       # train.npy / validation.npy (tools/prepare_dataset.sh)
: "${MODEL:=${NEMO_DIR}/scripts/models}"          # checkpoint Megatron (iter_0000000, 2 x 69 Go)
: "${RESULTS_ROOT:=${HELMA_DIR}/results}"
export CONT DATADIR MODEL RESULTS_ROOT

# ---------------------------------------------------------------- matériel Helma
export DGXNGPU=4
export GPU_ARCH=h                     # conf/tp_overlap/h100tp<TP>mbs1.yaml au lieu de b100... (Blackwell)
export NCCL_NVLS_ENABLE=0             # NVLS exige un NVSwitch, absent des nœuds h100
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # index CUDA == index nvidia-smi (requis par tools/bind_helma.sh)
# NCCL 2.30.4 (conteneur) : fonctions récentes désactivées. Avec leurs valeurs par défaut, R01 (job 910158)
# a planté dès la 1re collective (« CUDA error: unspecified launch failure »), et le job 910208 a reproduit
# le crash. NCCL attribuait au communicateur GIN_IB_GDAKI (le GPU pilote directement la carte IB), des
# noyaux « symétriques » et 32 canaux CollNet/SHARP. Les processus plantés sont restés impossibles à tuer
# et Slurm a mis les nœuds hors service (« Kill task failed »). NE PAS RÉACTIVER sans l'accord des admins
# NHR@FAU. Aucune de ces fonctions n'apporte de gain ici (pas de NVSwitch, SHARP non vérifié, ~90 Mo/step
# sur l'IB en TP4) : on revient au transport IB interne de NCCL, validé sur Helma (91 Go/s). docs/05 § 10.
# Job 910321 (OK) : GIN reste « Assigned » même avec ces réglages, et seul NCCL_NET_PLUGIN=none est confirmé
# lu par NCCL -> déclencheur le plus probable : le plugin réseau HPC-X (nccl_rdma_sharp_plugin). docs/05 § 13.
export NCCL_GIN_ENABLE=0
export NCCL_SYM_NOWIN_ENABLE=0
export NCCL_WIN_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_PLUGIN=none
export MLPERF_SUBMISSION_ORG="NHR@FAU"
export MLPERF_SUBMISSION_PLATFORM="HelmaH100"

# ---------------------------------------------------------------- recette Hopper 4 GPU/nœud
# Valeurs reprises de la soumission MLPerf v6.0 de Krai sur Isambard-AI (GH200 96 Go, 4 GPU/nœud,
# TP4/SP/FP8) : 10/10 runs convergés à 1, 2, 4, 8 et 16 nœuds avec LR=0.00055 et MAX_STEPS=800.
export LR=0.00055
export MAX_STEPS=800
export FP8_ACT=1                      # entrée de SwiGLU stockée en FP8 (≈ -9 Go d'activations/GPU en TP4)
export NUM_WORKERS=2
export MCORE_CUDA_GRAPH=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

# ---------------------------------------------------------------- lanceur (tools/, run_helma.sub)
: "${SEED_BASE:=42}"                  # graine fixe : rend les runs comparables entre eux (run i -> SEED_BASE+i-1)
: "${NEXP:=1}"                        # nombre d'essais enchaînés dans le même job
: "${NCCL_TEST:=1}"                   # pré-test NCCL + affinité (tools/nccl_check.py), arrêt si échec
: "${VERIFY_MOUNTS:=1}"               # empreintes MLPerf des données et du checkpoint
: "${TELEMETRY:=1}"                   # nvidia-smi toutes les TELEMETRY_INTERVAL s -> telemetry/*.csv
: "${TELEMETRY_INTERVAL:=5}"
: "${CLEAR_CACHES:=1}"                # éviction du cache de pages (fadvise), pas de sudo sur Helma
: "${CHECK_COMPLIANCE:=1}"
: "${STAGE_TO_TMPDIR:=0}"             # 1 = copie données + checkpoint sur le NVMe local ($TMPDIR)
: "${HELMA_BIND:=1}"                  # liaison CPU/NUMA de chaque rang à son GPU
: "${SBATCH_SWITCHES:=1@01:00:00}"    # jobs multi-nœuds : même switch leaf (attente max 1 h)
export SEED_BASE NEXP NCCL_TEST VERIFY_MOUNTS TELEMETRY TELEMETRY_INTERVAL CLEAR_CACHES \
       CHECK_COMPLIANCE STAGE_TO_TMPDIR HELMA_BIND SBATCH_SWITCHES
export MLPERF_RULESET=6.0.0
export BINDCMD="/helma_tools/bind_helma.sh"   # préfixe utilisé par nemo/run_and_time.sh (chemin conteneur)

# Durée Slurm demandée (minutes) : prologue + (staging) + NEXP x (run + compliance).
helma_compute_walltime() {
    local staging=0
    [[ "${STAGE_TO_TMPDIR}" == 1 ]] && staging=15
    export WALLTIME=$(( 15 + staging + NEXP * (WALLTIME_RUNANDTIME + 5) ))
}
