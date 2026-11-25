#!/bin/bash
# Gardien de nœud (SCC26, 16/11), lancé par run_helma.sub : une tâche par nœud, en arrière-plan, pendant tout le job.
#   au début : arrête les services de STOP_SERVICES actifs sur ce nœud (liste séparée par des virgules, ex.
#              "cc-metric-collector,cc-metric-store"), puis, si GPU_RESET=1, réinitialise les GPU (nvidia-persistenced
#              arrêté le temps du reset) ; écrit <dossier>/<nœud>.ready, ou <nœud>.error en cas d'échec ;
#   à la fin : relance ce qu'il a arrêté et retire le verrou de fréquence (nvidia-smi -rgc) si GPU_CLOCK_LOCK est posé.
# La fin arrive par le fichier <dossier>/stop (fin normale du job, déposé par run_helma.sub) ou par SIGTERM (envoyé par
# Slurm à l'annulation, avant SIGKILL, après KillWait secondes) ; le gardien écrit alors <nœud>.done. Un srun lancé depuis un trap après annulation échoue (« job has expired », job 125) : d'où ce gardien.
# Après un SIGKILL ou un plantage du nœud, relancer à la main : sudo systemctl start <services> ; sudo nvidia-smi -rgc
set -u
dir="${1:?usage: node_guard.sh <dossier>}"
h=$(hostname -s)
log="${dir}/${h}.log"
stopped=()
say() { echo "$(date +%T) ${h} : $*" >> "${log}"; }

restore() {
    trap - TERM INT EXIT
    if [[ -n "${GPU_CLOCK_LOCK:-}" ]]; then
        # GPU par GPU : un GPU en panne (« GPU requires reset ») fait échouer -rgc global (job 137, tritium GPU 5)
        local ok=0 ko=""
        for i in $(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null); do
            if sudo -n nvidia-smi -i "${i}" -rgc > /dev/null 2>&1; then ok=$((ok + 1)); else ko="${ko} ${i}"; fi
        done
        say "verrou SM retiré sur ${ok} GPU${ko:+ ; ÉCHEC sur les GPU${ko}}"
    fi
    for s in "${stopped[@]}"; do
        sudo -n systemctl start "${s}" && say "${s} relancé" || say "ÉCHEC de la relance de ${s}"
    done
    touch "${dir}/${h}.done"
    exit 0
}
trap restore TERM INT EXIT

IFS=, read -ra services <<< "${STOP_SERVICES:-}"
for s in "${services[@]}"; do
    [[ -n "${s}" ]] || continue
    if systemctl is-active --quiet "${s}"; then
        if sudo -n systemctl stop "${s}"; then
            stopped+=("${s}"); say "${s} arrêté"
        else
            say "ERREUR : impossible d'arrêter ${s} (sudo systemctl refusé ?)"; touch "${dir}/${h}.error"; exit 1
        fi
    else
        say "${s} absent ou inactif sur ce nœud"
    fi
done

if [[ "${GPU_RESET:-0}" == 1 ]]; then
    persist=0
    if systemctl is-active --quiet nvidia-persistenced && sudo -n systemctl stop nvidia-persistenced; then persist=1; fi
    if sudo -n nvidia-smi -r >> "${log}" 2>&1; then say "GPU réinitialisés (nvidia-smi -r)"; else say "AVERTISSEMENT : reset des GPU refusé"; fi
    [[ ${persist} == 1 ]] && sudo -n systemctl start nvidia-persistenced
    sudo -n nvidia-smi -pm 1 > /dev/null 2>&1
    say "récupération des GPU : $(nvidia-smi -q | grep 'Recovery Action' | awk '{print $NF}' | sort | uniq -c | xargs)"
fi

say "prêt (services arrêtés : ${stopped[*]:-aucun} ; reset GPU : ${GPU_RESET:-0} ; verrou SM à retirer en fin : ${GPU_CLOCK_LOCK:-non})"
touch "${dir}/${h}.ready"
# Fin normale : run_helma.sub dépose <dossier>/stop (un SIGTERM relayé par srun devient SIGKILL : job 134).
while [[ ! -f "${dir}/stop" ]]; do sleep 2 & wait $!; done
restore
