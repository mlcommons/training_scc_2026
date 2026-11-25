#!/bin/bash
# Vérifie, depuis le frontal et sans GPU, que tout est prêt pour lancer les runs :
#   1. conteneur NeMo, données .npy et checkpoint présents, tailles conformes à nemo/expected-mounts.csv
#   2. empreintes MLPerf (sparse sha256) des données et du checkpoint (mlperf_common.mountcheck)
#   3. contenu du conteneur (Megatron-Bridge, TE, logging MLPerf, slurm2pytorch, numactl)
#   4. validation à sec des 10 configs (tools/dry_run_config.py)
#   5. avec --slurm : demande Slurm de chaque run validée par `sbatch --test-only` (rien n'est soumis)
# Usage : tools/check_prereqs.sh [--slurm]
set -uo pipefail

HELMA_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
# Préfixe des configs à vérifier (config_<préfixe>_common.sh, config_<préfixe>_Rxx_*.sh) : HelmaH100 sur Helma,
# Cluster pour les modèles du guide (guide_mlperf_llama2_70b_lora/templates/).
CONFIG_PREFIX="${CONFIG_PREFIX:-HelmaH100}"
source "${HELMA_DIR}/configs/config_${CONFIG_PREFIX}_common.sh"

n_ok=0; n_warn=0; n_fail=0
ok()   { echo "  [OK]    $*"; n_ok=$((n_ok + 1)); }
warn() { echo "  [WARN]  $*"; n_warn=$((n_warn + 1)); }
fail() { echo "  [ÉCHEC] $*"; n_fail=$((n_fail + 1)); }

tmp_home=$(mktemp -d)
trap 'rm -rf "${tmp_home}"' EXIT
in_cont() { apptainer exec --home "${tmp_home}" "$@" 2>&1 | grep -v 'ws_find: command not found'; }

echo "== 1. Fichiers"
if [[ -f "${CONT}" ]]; then
    ver=$(apptainer inspect "${CONT}" 2>/dev/null | awk -F': ' '/com.nvidia.pytorch.version/{print $2}')
    ok "conteneur ${CONT} ($(du -h "${CONT}" | cut -f1), PyTorch ${ver:-?})"
    size_gb=$(( $(stat -c %s "${CONT}") / 1000000000 ))
    [[ ${size_gb} -gt 40 ]] && warn "conteneur de ${size_gb} Go : il embarque le checkpoint et un autre .sif (voir docs/01)"
else
    fail "conteneur ${CONT} introuvable"
fi
while IFS=, read -r key _type rel _n bytes _sha; do
    case "${key}" in
        DATADIR) base="${DATADIR}" ;;
        CHECKPOINT) base="${MODEL}" ;;
        *) continue ;;
    esac
    got=$(stat -c %s "${base}/${rel}" 2>/dev/null || echo absent)
    if [[ "${got}" == "${bytes}" ]]; then ok "${key}/${rel} (${bytes} octets)"; else fail "${key}/${rel} : ${got} octets, attendu ${bytes}"; fi
done < "${NEMO_DIR}/expected-mounts.csv"
[[ -f "${MODEL}/iter_0000000/.metadata" ]] && ok "CHECKPOINT/iter_0000000/.metadata (index torch.distributed.checkpoint)" \
    || fail "CHECKPOINT/iter_0000000/.metadata manquant"

echo "== 2. Empreintes MLPerf (mountcheck)"
out=$(apptainer exec --home "${tmp_home}" -B "${DATADIR}:/data:ro" -B "${MODEL}:/ckpt:ro" \
      -B "${NEMO_DIR}/expected-mounts.csv:/tmp/em.csv:ro" "${CONT}" python3 -m mlperf_common.mountcheck \
      --expected_mounts_csv /tmp/em.csv --mounts_to_verify DATADIR:/data CHECKPOINT:/ckpt --verbosity 2 2>&1)
rc=$?
n_match=$(grep -c "mountcheck OK" <<< "${out}")
if [[ ${rc} -eq 0 && ${n_match} -eq 15 ]]; then ok "15/15 fichiers identiques à la référence NVIDIA (sparse sha256)"
else fail "mountcheck : rc=${rc}, ${n_match}/15 OK"; grep -v "mountcheck OK" <<< "${out}" | sed 's/^/          /'; fi

echo "== 3. Contenu du conteneur"
out=$(in_cont "${CONT}" python3 -c "
import megatron.bridge, megatron.core, transformer_engine, mlperf_logging, mlperf_common, hydra, torch
print('torch', torch.__version__, '| TE', transformer_engine.__version__, '| mcore', megatron.core.__version__)")
if grep -q "^torch" <<< "${out}"; then ok "imports Python : $(grep '^torch' <<< "${out}")"; else fail "imports Python : ${out}"; fi
# nvidia-smi et libcuda ne sont pas dans l'image : `apptainer exec --nv` les injecte depuis le pilote de l'hôte.
for tool in slurm2pytorch numactl nsys; do
    if in_cont "${CONT}" bash -c "command -v ${tool}" > /dev/null; then ok "outil ${tool}"; else fail "outil ${tool} absent"; fi
done

echo "== 4. Validation à sec des configs"
for cfg in "${HELMA_DIR}"/configs/config_"${CONFIG_PREFIX}"_R*.sh; do
    res=$( (source "${cfg}"; in_cont -B "${NEMO_DIR}:/workspace/ft-llm:ro" -B "${HELMA_DIR}/tools:/helma_tools:ro" \
            "${CONT}" python3 /helma_tools/dry_run_config.py) )
    rc=$?
    line=$(grep -E "^(mémoire|ERREUR)" <<< "${res}" | head -2 | tr -s ' ' | paste -sd' ')
    name=$(basename "${cfg}" .sh)
    if [[ ${rc} -ne 0 ]] || grep -q "^ERREUR" <<< "${res}"; then fail "${name} : ${line}"
    elif grep -q "OOM probable" <<< "${res}"; then fail "${name} : ${line}"
    elif grep -qE "RISQUE|^ATTENTION" <<< "${res}"; then warn "${name} : ${line}"
    else ok "${name} : ${line}"; fi
done

if [[ "${1:-}" == "--slurm" ]]; then
    echo "== 5. Demandes Slurm (sbatch --test-only, rien n'est soumis)"
    for run in $(ls "${HELMA_DIR}/configs" | grep -E "^config_${CONFIG_PREFIX}_R" | grep -oE '_R[0-9]{2}_' | tr -d _); do
        res=$("${HELMA_DIR}/submit.sh" "${run}" --test-only 2>&1 | grep -E "to start at|error|Error" | head -1)
        if grep -q "to start at" <<< "${res}"; then ok "${run} : ${res}"; else fail "${run} : ${res}"; fi
    done
fi

echo "== 6. Sécurité"
if [[ -n "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN est exporté dans votre shell (~/.bashrc) : run_helma.sub le retire avant les tâches, mais pensez à le révoquer et à utiliser ~/.cache/huggingface/token"
else
    ok "aucun HF_TOKEN dans l'environnement"
fi

echo
echo "Bilan : ${n_ok} OK, ${n_warn} avertissement(s), ${n_fail} échec(s)"
[[ ${n_fail} -eq 0 ]]
