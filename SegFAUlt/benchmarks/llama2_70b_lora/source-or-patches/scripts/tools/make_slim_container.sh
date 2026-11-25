#!/bin/bash
# Construit le conteneur des essais, ${ZEPH_DIR}/nemo/nvidia-zeph.sif (~24 Go de logiciels), à partir de l'image
# zeph/nvidia-imp/zepg-nvidia.sif (117 Go), SANS la reconstruire depuis Internet : logiciels identiques au bit près.
#
# Pourquoi : zepg-nvidia.sif a été construite avec llama-zeph.def (= llama2.def) depuis zeph/nvidia-imp/, dont le
# dossier contenait alors model/ (checkpoint, 129 Go), data/ et scripts/data/ (parquet). « %files . /workspace/ft-llm »
# les a copiés dans l'image. Ils n'y servent à rien : le lanceur monte le checkpoint sur /ckpt, les données sur /data
# et l'instantané du code nemo/ sur /workspace/ft-llm (helma_runs/tools/apptainer_launch.sh).
# Ce script :
#   1. extrait l'image sans /workspace/ft-llm (unsquashfs à l'offset de la partition squashfs du SIF) ;
#   2. y remet /workspace/ft-llm = code nemo/ exact (code NVIDIA v6.0 + llama-zeph.def), comme le ferait
#      llama-zeph.def construit depuis un contexte propre ;
#   3. reconstruit un SIF (apptainer build --fakeroot), contrôle versions et imports, puis nettoie.
# Seule différence hors /workspace/ft-llm : ~150 fichiers système non possédés par root (groupe shadow, /var/mail,
# rdma) le deviennent, et les xattrs sont abandonnés. Sans effet sur l'entraînement (exécuté en utilisateur).
#
# Usage : tools/make_slim_container.sh                 (frontal ou nœud, sans GPU ; ~15 min, ~50 Go temporaires)
# Variables : SRC_SIF, OUT_SIF, WORK, NPROC (cœurs utilisés, 8 par défaut : rester discret si un benchmark tourne),
#             FORCE=1 pour remplacer un OUT_SIF existant.
set -euo pipefail

SCC_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "${SCC_DIR}/scc.env"
: "${SRC_SIF:=$(cd "${SCC_DIR}/.." && pwd)/zeph/nvidia-imp/zepg-nvidia.sif}"
: "${OUT_SIF:=${ZEPH_DIR}/nemo/nvidia-zeph.sif}"
: "${WORK:=${ZEPH_DIR}/.slim-build}"
: "${NPROC:=8}"
: "${FORCE:=0}"

[[ -f "${SRC_SIF}" ]] || { echo "ERREUR : image source ${SRC_SIF} introuvable" >&2; exit 1; }
[[ ! -e "${OUT_SIF}" || "${FORCE}" == 1 ]] || { echo "ERREUR : ${OUT_SIF} existe déjà (FORCE=1 pour remplacer)" >&2; exit 1; }
[[ ! -e "${WORK}" ]] || { echo "ERREUR : ${WORK} existe (construction précédente interrompue ?) : le supprimer d'abord" >&2; exit 1; }
[[ "${OUT_SIF}" == *.sif ]] || { echo "ERREUR : OUT_SIF doit finir par .sif" >&2; exit 1; }

cpus="0-$(( NPROC - 1 ))"
low() { nice -n 19 ionice -c 3 taskset -c "${cpus}" "$@"; }   # mksquashfs/unsquashfs : autant de fils que de cœurs permis
tmp_sif="${OUT_SIF%.sif}.tmp.sif"
mkdir -p "${WORK}"
cleanup() { chmod -R u+rwX "${WORK}" 2>/dev/null || true; rm -rf "${WORK}" "${tmp_sif}"; }
trap cleanup EXIT

echo "Source : ${SRC_SIF} ($(du -h --apparent-size "${SRC_SIF}" | cut -f1))"
echo "Sortie : ${OUT_SIF}"
echo "Travail : ${WORK} ; ${NPROC} cœurs, priorité basse"

offset=$(apptainer sif list "${SRC_SIF}" | awk -F'|' '/Squashfs/ {split($4, a, "-"); gsub(/ /, "", a[1]); print a[1]; exit}')
[[ "${offset}" =~ ^[0-9]+$ ]] || { echo "ERREUR : partition squashfs introuvable dans ${SRC_SIF}" >&2; exit 1; }

echo "=== 1/4 extraction sans /workspace/ft-llm (offset ${offset}) $(date +%T)"
printf 'workspace/ft-llm\n' > "${WORK}/exclude.txt"
# Code de retour 2 = avertissements non bloquants d'un utilisateur non root (propriétaires non restaurés).
rc=0
low unsquashfs -o "${offset}" -d "${WORK}/rootfs" -no-xattrs -no-progress -processors "${NPROC}" \
    -exclude-file "${WORK}/exclude.txt" "${SRC_SIF}" > "${WORK}/unsquashfs.log" 2>&1 || rc=$?
tail -3 "${WORK}/unsquashfs.log"
[[ ${rc} -le 2 && -x "${WORK}/rootfs/usr/bin/python3" ]] || { echo "ERREUR : extraction (code ${rc})" >&2; exit 1; }

echo "=== 2/4 /workspace/ft-llm = code nemo/ (${ZEPH_DIR}/nemo) $(date +%T)"
mkdir -p "${WORK}/rootfs/workspace/ft-llm"
rsync -a --exclude '*.sif' --exclude 'scripts/models' --exclude 'scripts/dataset' --exclude 'outputs' \
    --exclude '__pycache__' --exclude '.*' "${ZEPH_DIR}/nemo/" "${WORK}/rootfs/workspace/ft-llm/"
du -sh "${WORK}/rootfs" "${WORK}/rootfs/workspace/ft-llm"

echo "=== 3/4 SIF (apptainer build --fakeroot) $(date +%T)"
rm -f "${tmp_sif}"
low apptainer build --fakeroot "${tmp_sif}" "${WORK}/rootfs" 2>&1 | grep -v '^INFO: *Creating SIF' | tail -5

echo "=== 4/4 contrôle $(date +%T)"
home=$(mktemp -d)
apptainer exec --home "${home}" "${tmp_sif}" bash -c '
    echo "NGC ${NVIDIA_PYTORCH_VERSION} | PyTorch ${PYTORCH_VERSION} | CUDA ${CUDA_VERSION} | NCCL ${NCCL_VERSION} | TE ${TRANSFORMER_ENGINE_VERSION}"
    echo "Megatron-Bridge $(cat /MBRIDGE_COMMIT_HASH.env) | Megatron-LM $(cat /MCORE_COMMIT_HASH.env)"
    python3 -c "import torch, transformer_engine, megatron.bridge, mlperf_logging, mlperf_common; print(\"imports OK\")"
    ls /workspace/ft-llm | tr "\n" " "; echo' 2>&1 | grep -v -e 'ws_find' -e 'cache_position'
rm -rf "${home}"
mv "${tmp_sif}" "${OUT_SIF}"
echo "OK : ${OUT_SIF} ($(du -h "${OUT_SIF}" | cut -f1)) $(date +%T)"
echo "Image d'origine conservée : ${SRC_SIF} (à supprimer à la main une fois les essais validés)"
