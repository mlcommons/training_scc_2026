#!/bin/bash
# Convertit le dataset GovReport (.parquet déjà téléchargés) en train.npy / validation.npy,
# le format que lit nemo/train.py (PackedSequenceSpecs). Réutilise la fonction de
# transformation de NVIDIA (nemo/scripts/download_dataset.py::DatasetProcessor.transform_row)
# et le même enchaînement pandas.apply + np.save que convert_split(), sans rien retélécharger.
#
# Usage : tools/prepare_dataset.sh            (depuis n'importe où, frontal ou nœud de calcul)
# Variables optionnelles : CONT, PARQUET_DIR, DATADIR
set -euo pipefail

HELMA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ZEPH_DIR=$(cd "${HELMA_DIR}/.." && pwd)

: "${CONT:=${ZEPH_DIR}/nemo/nvidia-zeph.sif}"
: "${PARQUET_DIR:=${ZEPH_DIR}/nemo/scripts/dataset/data}"
: "${DATADIR:=${ZEPH_DIR}/data/gov_report}"

for f in train-00000-of-00001.parquet validation-00000-of-00001.parquet; do
    [[ -f "${PARQUET_DIR}/${f}" ]] || { echo "ERREUR: ${PARQUET_DIR}/${f} introuvable" >&2; exit 1; }
done
[[ -f "${CONT}" ]] || { echo "ERREUR: conteneur ${CONT} introuvable" >&2; exit 1; }

mkdir -p "${DATADIR}"
tmp_home=$(mktemp -d)
trap 'rm -rf "${tmp_home}"' EXIT

echo "Conteneur : ${CONT}"
echo "Entrée    : ${PARQUET_DIR}"
echo "Sortie    : ${DATADIR}"

apptainer exec --home "${tmp_home}" \
    -B "${ZEPH_DIR}/nemo/scripts:/nv_scripts:ro" \
    -B "${PARQUET_DIR}:/parquet:ro" \
    -B "${DATADIR}:/out" \
    "${CONT}" python3 - <<'EOF'
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/nv_scripts")
from download_dataset import DatasetProcessor  # code NVIDIA, inchangé

transform_row = DatasetProcessor.transform_row
for split in ("train", "validation"):
    df = pd.read_parquet(f"/parquet/{split}-00000-of-00001.parquet")
    data = df.apply(lambda row: transform_row(None, row), axis=1).tolist()
    np.save(f"/out/{split}.npy", data)
    print(f"{split}: {len(data)} séquences -> /out/{split}.npy", flush=True)
EOF

echo
echo "Tailles obtenues (octets) vs attendues (nemo/expected-mounts.csv) :"
while IFS=, read -r key _type rel _n bytes _sha; do
    [[ "${key}" == "DATADIR" ]] || continue
    got=$(stat -c %s "${DATADIR}/${rel}" 2>/dev/null || echo "absent")
    [[ "${got}" == "${bytes}" ]] && status="OK" || status="DIFFÉRENT"
    printf "  %-16s obtenu=%-12s attendu=%-12s %s\n" "${rel}" "${got}" "${bytes}" "${status}"
done < "${ZEPH_DIR}/nemo/expected-mounts.csv"
