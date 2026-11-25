#!/bin/bash
# Exécute une commande dans le conteneur NeMo avec les montages du benchmark.
# Lancé par srun (une instance par tâche) depuis run_helma.sub.
#
#   /data                <- données (Lustre, ou copie locale si STAGE_TO_TMPDIR=1), lecture seule
#   /ckpt                <- checkpoint Megatron, lecture seule
#   /results             <- dossier du job (logs MLPerf, nsys, sorties Hydra)
#   /workspace/ft-llm    <- instantané du code nemo/ pris au début du job (remplace la copie de l'image)
#   /helma_tools         <- ce dossier tools/, lecture seule
# $HOME du conteneur = dossier vide sur le NVMe local : le ~/.bashrc de l'hôte (ws_find, HF_TOKEN)
# n'est pas relu dans le conteneur.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/node_env.sh"

mkdir -p "${NODE_TMP}/home" "${NODE_TMP}/cache"
export TRITON_CACHE_DIR="${NODE_TMP}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${NODE_TMP}/cache/inductor"
export MPLCONFIGDIR="${NODE_TMP}/cache/matplotlib"
export HF_HOME="${NODE_TMP}/cache/hf"
export CUDA_CACHE_PATH="${NODE_TMP}/cache/cuda"
export XDG_CACHE_HOME="${NODE_TMP}/cache/xdg"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Placement inégal (RANK_LAYOUT, voir run_helma.sub) : le nombre de rangs de CE nœud vient de RANK_LAYOUT (hôte:n).
# slurm2pytorch le prendrait dans SLURM_NTASKS_PER_NODE (celui du job, 8, faux sur tritium) ; on ne garde visibles
# que les n premiers GPU du nœud, pour que ni CUDA ni NCCL ne touchent aux autres (îlot 4-7 de tritium).
if [[ -n "${RANK_LAYOUT:-}" && -n "${SLURM_STEP_ID:-}" ]]; then
    _n=$(tr ',' '\n' <<< "${RANK_LAYOUT}" | awk -F: -v h="$(hostname -s)" '$1 == h {print $2}')
    if [[ -n "${_n}" && "${SLURM_STEP_NUM_TASKS:-1}" -gt "${SLURM_STEP_NUM_NODES:-1}" ]]; then
        export LOCAL_WORLD_SIZE="${_n}"
        export CUDA_VISIBLE_DEVICES=$(cut -d, -f1-"${_n}" <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( _n - 1 )))}")
    fi
fi
# Pas de fichier core : un rang qui plante réserve ~100 Go d'espace virtuel et écrirait dans le code (Lustre).
ulimit -c 0

exec apptainer exec --nv \
    --home "${NODE_TMP}/home" \
    -B "${DATA_SRC}:/data:ro" \
    -B "${CKPT_SRC}:/ckpt:ro" \
    -B "${RUN_DIR:?}:/results" \
    -B "${CODE_DIR:?}:/workspace/ft-llm" \
    -B "${HELMA_DIR:?}/tools:/helma_tools:ro" \
    -B "${NODE_TMP}" \
    --pwd /workspace/ft-llm \
    "${CONT:?}" "$@"
