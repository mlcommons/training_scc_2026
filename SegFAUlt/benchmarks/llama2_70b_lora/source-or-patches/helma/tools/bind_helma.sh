#!/bin/bash
# Lie chaque rang aux cœurs CPU et à la mémoire du nœud NUMA de SON GPU, puis exécute la commande.
# Exécuté DANS le conteneur (BINDCMD de nemo/run_and_time.sh, et devant tools/nccl_check.py).
#
# Pourquoi pas `bindpcie --cpu=exclusive` (fourni par NVIDIA) : sa formule suppose GPU i <-> NUMA i.
# Sur Helma (NPS4, 8 domaines NUMA de 16 cœurs) la topologie réelle est GPU0<->NUMA3 (48-63),
# GPU1<->NUMA2 (32-47), GPU2<->NUMA6 (96-111), GPU3<->NUMA4 (64-79) : bindpcie mettrait les rangs
# du mauvais côté. On lit donc la topologie dans sysfs, sans rien supposer.
# Mémoire : --preferred (et non --membind) pour ne jamais provoquer d'OOM pendant le chargement du
# checkpoint si le domaine NUMA (~96 Go) est plein.
set -euo pipefail

if [[ "${HELMA_BIND:-1}" != 1 ]]; then
    exec "$@"
fi

local_rank="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"
IFS=, read -ra visible <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu="${visible[${local_rank}]}"

bus=$(nvidia-smi -i "${gpu}" --query-gpu=pci.bus_id --format=csv,noheader | tr '[:upper:]' '[:lower:]')
sysfs="/sys/bus/pci/devices/${bus:4}"   # nvidia-smi : 00000000:06:00.0 -> sysfs : 0000:06:00.0
cpus=$(<"${sysfs}/local_cpulist")
numa=$(<"${sysfs}/numa_node")

args=(--physcpubind="${cpus}")
[[ "${numa}" -ge 0 ]] && args+=(--preferred="${numa}")

echo "[bind_helma] $(hostname -s) local_rank=${local_rank} gpu=${gpu} pci=${bus:4} numa=${numa} cpus=${cpus}" >&2
exec numactl "${args[@]}" "$@"
