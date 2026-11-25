#!/bin/bash
# Chemins propres au nœud courant, sourcé par apptainer_launch.sh et stage_to_tmpdir.sh.
# $TMPDIR est local à chaque nœud : ces valeurs doivent être calculées sur le nœud qui exécute la tâche.
NODE_TMP="${TMPDIR:-/tmp}/zeph_${SLURM_JOB_ID:-manual}"
STAGE_DIR="${NODE_TMP}/stage"
if [[ "${STAGE_TO_TMPDIR:-0}" == 1 ]]; then
    DATA_SRC="${STAGE_DIR}/gov_report"
    CKPT_SRC="${STAGE_DIR}/model"
else
    DATA_SRC="${DATADIR:?DATADIR non défini}"
    CKPT_SRC="${MODEL:?MODEL non défini}"
fi
