#!/bin/bash
# Vérification complète d'un paquet SCC26, sur le frontal, sans GPU. Reprend les contrôles des slides du comité
# (Benchmarking-Overview-SCC26.pdf, p. 24), lancés directement sur le paquet, comme le fera la notation automatique :
#   1. vérificateurs officiels mlperf-logging (venv .venv-mlperf-logging : tag 6.0.0-rc6, règles training 6.0.0) :
#      rcp_checker, package_checker --werror (conformité de chaque log, graines, description système ; écrit
#      results/<système>/llama2_70b_lora/scaling.json), puis verify_for_v6.0_training.sh : result_summarizer et
#      repo_checker ;
#   2. tools/scc_check.py : arborescence SCC exacte, systems/<système>.json, logs, graines, moyenne olympique du TTT
#      et score normalisé par le facteur de scaling.json.
#
# Usage : tools/scc_verify.sh ngpu8|ngpu16|<chemin de <team_name>/> [--partial]
# Journaux : <système>/runs/checks/<date-heure>/ (chemin direct : <paquet>/../checks/<date-heure>/)
# Code de retour : 0 si tout est valide (vérificateurs officiels, RCP, structure SCC), 1 sinon.
set -uo pipefail

SCC_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "${SCC_DIR}/scc.env"
target="${1:?usage: tools/scc_verify.sh ngpu8|ngpu16|<team_dir> [--partial]}"
shift
if [[ -f "${SCC_DIR}/${target}/config_scc_${target}.sh" ]]; then
    TEAM_DIR="${SCC_DIR}/${target}/${TEAM_NAME}"
    OUT="${SCC_DIR}/${target}/runs/checks/$(date +%Y%m%d-%H%M%S)"
else
    TEAM_DIR=$(readlink -f "${target}")
    OUT="$(dirname "${TEAM_DIR}")/checks/$(date +%Y%m%d-%H%M%S)"
fi
PY="${SCC_DIR}/.venv-mlperf-logging/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "mlperf-logging absent. Installation (frontal, accès réseau) :"
    echo "  python3 -m venv ${SCC_DIR}/.venv-mlperf-logging"
    echo "  ${SCC_DIR}/.venv-mlperf-logging/bin/pip install git+https://github.com/mlcommons/logging.git@6.0.0-rc6"
    exit 1
fi
mkdir -p "${OUT}"
ok=1
# Les vérificateurs sont lancés depuis le dossier parent, avec le nom d'équipe en chemin relatif, comme sur la slide 24.
cd "$(dirname "${TEAM_DIR}")" || exit 1
TEAM=$(basename "${TEAM_DIR}")
systems=()
for desc in "${TEAM}"/systems/*.json; do
    [[ -f "${desc}" ]] && systems+=("$(basename "${desc}" .json)")
done

echo "Paquet   : ${TEAM_DIR}"
echo "Journaux : ${OUT}"
echo
echo "================ 1. Vérificateurs officiels mlperf-logging $("${PY}" -m pip show mlperf-logging 2>/dev/null \
    | awk '/^Version/{print $2}') (tag 6.0.0-rc6), règles training ${MLPERF_RULESET}"

echo "--- 01 rcp_checker (convergence comparée aux points de référence GBS 8/16/32/128)"
for s in "${systems[@]}"; do
    "${PY}" -m mlperf_logging.rcp_checker "${TEAM}/results/${s}/${SCC_BENCHMARK}" --rcp_usage training \
        --rcp_version "${MLPERF_RULESET}" --verbose --log_output "${OUT}/rcp_checker_${s}.log" \
        > "${OUT}/rcp_checker_${s}.stdout" 2>&1
    grep -hE "RCP test|Submission mean|faster than|slower than|ERROR" "${OUT}/rcp_checker_${s}.log" \
        2>/dev/null | sort -u | head -8
    grep -qi "RCP test passed" "${OUT}/rcp_checker_${s}.log" 2>/dev/null || { echo "ÉCHEC RCP : ${s}"; ok=0; }
done

echo "--- 02 package_checker --werror (conformité de chaque log, graines, description système)"
"${PY}" -m mlperf_logging.package_checker "${TEAM}" training "${MLPERF_RULESET}" --werror \
    --log_output "${OUT}/package_checker.log" > "${OUT}/package_checker.stdout" 2>&1
grep -E "ERROR|RCP Test|Seed checker|Results scaling|PACKAGE CHECKER FOUND" "${OUT}/package_checker.log" | uniq
grep -q "PACKAGE CHECKER FOUND NO ERRORS" "${OUT}/package_checker.log" || ok=0
# Un dossier de benchmark au nom inconnu est ignoré en silence, puis « NO ERRORS » est affiché sans qu'aucun log
# n'ait été contrôlé : on exige que chaque système ait réellement été vérifié.
if grep -q "Skipping benchmark" "${OUT}/package_checker.log" \
   || [[ $(grep -c "Running compliance checks in dir" "${OUT}/package_checker.log") -lt ${#systems[@]} ]]; then
    echo "ERREUR : package_checker a ignoré un benchmark (aucun log contrôlé) : voir ${OUT}/package_checker.log"
    ok=0
fi
for s in "${systems[@]}"; do
    f="${TEAM}/results/${s}/${SCC_BENCHMARK}/scaling.json"
    [[ -f "${f}" ]] && echo "${f} (facteur de normalisation RCP, laissé dans le paquet) : $(cat "${f}")"
done

echo "--- 03 verify_for_v6.0_training.sh : result_summarizer et repo_checker"
"${PY}" -m mlperf_logging.result_summarizer "${TEAM}" training "${MLPERF_RULESET}" --csv "${OUT}/results.csv" \
    > "${OUT}/result_summarizer.log" 2>&1 || { echo "ÉCHEC result_summarizer : voir ${OUT}/result_summarizer.log"; ok=0; }
# Cette version écrit le facteur RCP dans results.csv ; le temps est dans results_detailed.csv (<benchmark>:time_to_train).
python3 - "${OUT}/results_detailed.csv" "${MLPERF_BENCHMARK}" << 'EOF'
import csv, sys
try:
    with open(sys.argv[1]) as f:
        rows = list(csv.DictReader(f))
except OSError:
    sys.exit("results_detailed.csv absent (voir result_summarizer.log)")
b = sys.argv[2]
for r in rows:
    ttt = r.get(f"{b}:time_to_train")
    print(f"{r.get('system')} ({r.get('accelerators_count')} GPU) : "
          + (f"{float(ttt):.2f} min ; samples moyens {r.get(f'{b}:samples_to_converge')} ; RCP {r.get(f'{b}:RCP')} ; "
             f"facteur {float(r.get(f'{b}:rcp_scaling_factor') or 1):.4f}" if ttt else "pas de score (trop peu d'essais)"))
EOF
"${PY}" -m mlperf_logging.repo_checker "${TEAM}" training "${MLPERF_RULESET}" > "${OUT}/repo_checker.log" 2>&1
tail -1 "${OUT}/repo_checker.log"
grep -q "ALL CHECKS PASSED" "${OUT}/repo_checker.log" || ok=0

echo
echo "================ 2. Structure SCC (slides p. 23, benchmarks.md § 7) et score"
python3 "${SCC_DIR}/tools/scc_check.py" "${TEAM_DIR}" "$@" | tee "${OUT}/scc_check.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || ok=0

echo
if [[ ${ok} == 1 ]]; then
    echo "PAQUET VALIDE (vérificateurs officiels + RCP + structure SCC)"
    exit 0
fi
echo "PAQUET NON VALIDE : voir ${OUT}/"
exit 1
