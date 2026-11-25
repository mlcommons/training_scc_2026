#!/bin/bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${TRAIN_VENV:-$HERE/../venv-train}/bin/python
if [ -n "${DG_JIT_CACHE_DIR:-}" ]; then
    export DG_JIT_CACHE_DIR="$DG_JIT_CACHE_DIR/$(hostname)"
fi
if [ -z "${LOCAL_RANK:-}" ]; then
    echo "rank_exec_cp: LOCAL_RANK unset; cannot resolve the rank's GPU" >&2
    exit 1
fi
_gpu=$LOCAL_RANK
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _gpu=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((_gpu + 1)))
fi
_bdf=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$_gpu" \
    | tr 'A-F' 'a-f' | sed 's/^00000000/0000/')
_node=$(cat /sys/bus/pci/devices/$_bdf/numa_node 2>/dev/null)
case $_node in
    '')
        echo "rank_exec_cp: cannot resolve GPU $_gpu (${_bdf:-no BDF}) to a PCI device" >&2
        exit 1 ;;
    -1)
        echo "rank_exec_cp: rank $LOCAL_RANK on GPU $_gpu has no NUMA domain; running unpinned" >&2
        NA=() ;;
    *[!0-9]*)
        echo "rank_exec_cp: cannot resolve GPU $_gpu (${_bdf:-no BDF}) to a NUMA domain (numa_node='$_node')" >&2
        exit 1 ;;
    *)
        command -v numactl >/dev/null || { echo "rank_exec_cp: numactl not found" >&2; exit 1; }
        echo "rank_exec_cp: rank $LOCAL_RANK on GPU $_gpu -> NUMA node $_node (cpu+mem bound)" >&2
        NA=(numactl --cpunodebind=$_node --membind=$_node) ;;
esac
if [ "${NSYS:-0}" = "1" ] && [ "${GROUP_RANK:-0}" = "0" ] \
   && [[ ",${NSYS_RANKS:-0}," == *",${LOCAL_RANK:-x},"* ]]; then
    NSYS_BIN=$(command -v nsys || echo /usr/local/cuda/bin/nsys)
    _out=${NSYS_OUT:-$PWD/nsys-rank0}
    _out=${_out/rank0/rank${LOCAL_RANK}}
    _gm=()
    if [ "${NSYS_GPU_METRICS:-0}" = "1" ]; then
        _gm=(--gpu-metrics-devices="$_gpu" --gpu-metrics-frequency=20000)
    fi
    exec "${NA[@]}" "$NSYS_BIN" profile --trace=cuda,nvtx --cuda-graph-trace=node "${_gm[@]}" \
      --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=true --output="$_out" "$PY" -u "$@"
fi
if [ -n "${ERRDIR:-}" ]; then
    mkdir -p "$ERRDIR"
    exec "${NA[@]}" $PY -u "$@" 2> "$ERRDIR/node${GROUP_RANK:-x}-rank${LOCAL_RANK}.err"
fi
exec "${NA[@]}" $PY -u "$@"
