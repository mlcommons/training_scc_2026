#!/bin/bash
# Soumission SCC26 MLPerf Llama2-70B LoRA : ngpu8 (2 nœuds x 4 H100) et/ou ngpu16 (4 nœuds x 4 H100), ou tout autre
# système <dir>/config_scc_<dir>.sh (ex. segfault_ngpu16, cluster de compétition). Mode d'emploi : README.md
# Réutilise sans modification le lanceur validé : ${ZEPH_DIR}/helma_runs/run_helma.sub (Apptainer) et ses tools/.
# Sur un autre cluster, SBATCH_EXTRA (dans la config) remplace les #SBATCH propres à Helma (partition, GPU par nœud) :
# les options de la ligne de commande sbatch priment sur celles du script.
set -euo pipefail

SCC_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "${SCC_DIR}/scc.env"
HELMA_DIR="${ZEPH_DIR}/helma_runs"
ALL_SYSTEMS=(ngpu8 ngpu16)

usage() {
    cat << 'EOF'
Usage : ./submit.sh <systèmes> [options] [VAR=valeur ...]

  <systèmes>      ngpu8 (8 GPU), ngpu16 (16 GPU), "all" (les deux), ou un autre dossier <dir> qui contient
                  config_scc_<dir>.sh (ex. segfault_ngpu16)
  --dry-run       n'envoie rien : affiche la commande sbatch et valide la config dans le conteneur (sans GPU)
  --test-only     sbatch --test-only : Slurm valide la demande et estime le démarrage, sans soumettre
  --after JOBID   ne démarre qu'après la fin du job JOBID, réussi ou non (dépendance afterany)
  VAR=valeur      surcharge une variable de la config (ex. NEXP=3 pour compléter un lot interrompu)

Exemples :
  ./submit.sh all --dry-run
  ./submit.sh all                    # 1 job de 10 essais par système
  ./submit.sh ngpu8 NEXP=3           # complément après un job interrompu au 7e essai
EOF
}

systems=(); overrides=(); mode=submit; after=""
while (($#)); do
    case "$1" in
        all) systems+=("${ALL_SYSTEMS[@]}") ;;
        --dry-run) mode=dry ;;
        --test-only) mode=test ;;
        --after) after="${2:?--after attend un JOBID}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *=*) overrides+=("$1") ;;
        *)
            if [[ "$1" =~ ^[A-Za-z0-9_]+$ && -f "${SCC_DIR}/$1/config_scc_$1.sh" ]]; then
                systems+=("$1")
            else
                echo "Argument inconnu : $1 (aucun ${SCC_DIR}/$1/config_scc_$1.sh)" >&2; usage; exit 1
            fi ;;
    esac
    shift
done
[[ ${#systems[@]} -gt 0 ]] || { usage; exit 1; }

# Le job ne garde qu'un instantané de nemo/ et des configs : les empreintes du lanceur et de scc.env sont relevées à la
# soumission, et tools/build_scc_package.py vérifie qu'ils n'ont pas changé depuis.
launcher_files=("${HELMA_DIR}/run_helma.sub" "${HELMA_DIR}"/tools/*.sh "${HELMA_DIR}"/tools/*.py
                "${HELMA_DIR}/configs/config_HelmaH100_common.sh" "${ZEPH_DIR}/nemo/config_common.sh" "${SCC_DIR}/scc.env")

for sys in "${systems[@]}"; do
    cfg="${SCC_DIR}/${sys}/config_scc_${sys}.sh"
    runs_dir="${SCC_DIR}/${sys}/runs"
    [[ -f "${cfg}" ]] || { echo "Config introuvable : ${cfg}" >&2; exit 1; }
    mkdir -p "${runs_dir}/slurm"

    # Nœuds, durée et placement tels que les verra le job (surcharges appliquées avant et après la config).
    # SBATCH_EXTRA (options sbatch sans espace interne) vient en dernier : read lui donne le reste de la ligne.
    read -r nnodes walltime run_name nexp switches exclude team extra < <(
        for kv in "${overrides[@]}"; do export "${kv}"; done
        source "${cfg}"
        for kv in "${overrides[@]}"; do export "${kv}"; done
        echo "${DGXNNODES} ${WALLTIME} ${RUN_NAME} ${NEXP} ${SBATCH_SWITCHES:-none} ${SBATCH_EXCLUDE:-none} ${MLPERF_SUBMISSION_ORG}" \
             "${SBATCH_EXTRA:-}"
    )

    args=(-J "scc_${sys}" -N "${nnodes}" -t "${walltime}" -o "${runs_dir}/slurm/%x-%j.out")
    [[ "${nnodes}" -gt 1 && "${switches}" != none ]] && args+=(--switches="${switches}")
    [[ "${exclude}" != none ]] && args+=(--exclude="${exclude}")
    # shellcheck disable=SC2206
    [[ -n "${extra}" ]] && args+=(${extra})
    [[ -n "${after}" ]] && args+=(--dependency="afterany:${after}")
    cmd=(sbatch "${args[@]}" "${HELMA_DIR}/run_helma.sub" "${cfg}" "${overrides[@]}")

    echo "=== ${sys} (${run_name}) : ${nnodes} nœud(s), ${nexp} essai(s), ${walltime} min, équipe ${team}"
    case "${mode}" in
        dry)
            echo "${cmd[*]}"
            (
                for kv in "${overrides[@]}"; do export "${kv}"; done
                source "${cfg}"
                for kv in "${overrides[@]}"; do export "${kv}"; done
                tmp_home=$(mktemp -d)
                apptainer exec --home "${tmp_home}" -B "${NEMO_DIR}:/workspace/ft-llm:ro" \
                    -B "${HELMA_DIR}/tools:/helma_tools:ro" "${CONT}" python3 /helma_tools/dry_run_config.py \
                    2>&1 | grep -v 'ws_find: command not found'
                rm -rf "${tmp_home}"
            ) || echo "(validation à sec en échec pour ${sys})"
            ;;
        test)
            "${cmd[0]}" --test-only "${cmd[@]:1}" 2>&1 || true
            ;;
        submit)
            out=$("${cmd[@]}")
            echo "${out}"
            jobid=$(awk '{print $NF}' <<< "${out}")
            sha256sum "${launcher_files[@]}" "${cfg}" > "${runs_dir}/slurm/launcher_${jobid}.sha256"
            printf '%s\t%s\t%s\t%s\tNEXP=%s\t%s\n' "$(date '+%F %T')" "${sys}" "${jobid}" "${run_name}" "${nexp}" \
                "${cmd[*]}" >> "${runs_dir}/submissions.tsv"
            ;;
    esac
done
