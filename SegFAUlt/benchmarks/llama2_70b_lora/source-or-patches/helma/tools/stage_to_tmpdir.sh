#!/bin/bash
# Copie les données (~200 Mo) et le checkpoint (~138 Go) sur le NVMe local du nœud ($TMPDIR, 15 To).
# Lancé une fois par nœud (srun --ntasks-per-node=1) quand STAGE_TO_TMPDIR=1. Les deux gros
# fichiers .distcp sont copiés en parallèle. $TMPDIR est vidé par Slurm à la fin du job.
set -euo pipefail
export STAGE_TO_TMPDIR=1
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/node_env.sh"

host=$(hostname -s)
mkdir -p "${DATA_SRC}" "${CKPT_SRC}"
df -h "${TMPDIR:-/tmp}" | sed "s/^/[${host}] /"

t0=$(date +%s)
cp "${DATADIR}/train.npy" "${DATADIR}/validation.npy" "${DATA_SRC}/"

# Checkpoint sans les caches Hugging Face (.cache) : petits fichiers d'abord, puis .distcp en parallèle.
(cd "${MODEL}" && find . -path ./.cache -prune -o -type f ! -name '*.distcp' -print) | while read -r f; do
    mkdir -p "${CKPT_SRC}/$(dirname "${f}")"
    cp "${MODEL}/${f}" "${CKPT_SRC}/${f}"
done
pids=()
while read -r f; do
    mkdir -p "${CKPT_SRC}/$(dirname "${f}")"
    cp "${MODEL}/${f}" "${CKPT_SRC}/${f}" &
    pids+=($!)
done < <(cd "${MODEL}" && find . -path ./.cache -prune -o -type f -name '*.distcp' -print)
for p in "${pids[@]}"; do wait "${p}"; done
t1=$(date +%s)

bytes=$(du -sb "${STAGE_DIR}" | cut -f1)
secs=$(( t1 - t0 > 0 ? t1 - t0 : 1 ))
echo "[${host}] STAGE_DONE seconds=${secs} bytes=${bytes} GBps=$(awk -v b="${bytes}" -v s="${secs}" 'BEGIN{printf "%.2f", b/s/1e9}')"

# Contrôle : mêmes tailles que la source pour chaque fichier copié.
(cd "${CKPT_SRC}" && find . -type f -printf '%P %s\n') | while read -r rel size; do
    src_size=$(stat -c %s "${MODEL}/${rel}")
    [[ "${size}" == "${src_size}" ]] || { echo "[${host}] ERREUR taille ${rel}: ${size} != ${src_size}" >&2; exit 1; }
done
echo "[${host}] STAGE_VERIFIED"
