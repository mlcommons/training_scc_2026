#!/bin/bash
set -e
H=$(cd "$(dirname "$0")/.." && pwd)
V=${TRAIN_VENV:-$H/venv-train}
NV=$V/lib/python3.12/site-packages/nvidia

if [ -z "${CUDA_HOME:-}" ]; then
    for c in /home/apps/cuda/13.4.2 /usr/local/cuda; do
        [ -x "$c/bin/nvcc" ] && export CUDA_HOME=$c && break
    done
fi
[ -n "${CUDA_HOME:-}" ] || { echo "CUDA_HOME not found"; exit 1; }
export PATH=$V/bin:$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

_rt=$(env -u LD_LIBRARY_PATH "$V/bin/python" -c '
import ctypes
ctypes.CDLL("libstdc++.so.6")
for line in open("/proc/self/maps"):
    if "libstdc++.so.6" in line:
        print(line.split()[-1]); break
' 2>/dev/null)
_rtmax=""
[ -f "$_rt" ] && _rtmax=$(objdump -T "$_rt" 2>/dev/null \
    | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V | tail -1)
TC=""
if [ -n "$_rtmax" ]; then
    for _g in $(ls -d /home/apps/gcc/*/bin/gcc 2>/dev/null | sort -V); do
        [ -x "$_g" ] || continue
        _lib="$(dirname "$(dirname "$_g")")/lib64/libstdc++.so.6"
        _max=$(objdump -T "$_lib" 2>/dev/null | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V | tail -1)
        [ -n "$_max" ] || continue
        [ "$(printf '%s\n%s\n' "$_max" "$_rtmax" | sort -V | tail -1)" = "$_rtmax" ] && TC="$(dirname "$_g")"
    done
fi
if [ -n "$TC" ]; then
    export PATH="$TC:$PATH"
    export CC="$TC/gcc" CXX="$TC/g++" CUDAHOSTCXX="$TC/g++"
else
    export CC=gcc CXX=g++ CUDAHOSTCXX=g++
fi
echo "toolchain: CC=$CC CXX=$CXX (venv libstdc++ max GLIBCXX: ${_rtmax:-unknown})"

export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90a
export NVTE_WITH_USERBUFFERS=1
export NVTE_UB_WITH_MPI=0
export MAX_JOBS=${MAX_JOBS:-140}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-140}

NP=$H/nccl-prefix-${TRAIN_VENV_SUFFIX:-train}
mkdir -p $NP/lib && ln -sfn $NV/nccl/include $NP/include && \
  ln -sf $NV/nccl/lib/libnccl.so.2 $NP/lib/libnccl.so.2 && ln -sf libnccl.so.2 $NP/lib/libnccl.so
export NCCL_HOME=$NP
export CMAKE_PREFIX_PATH=$NP${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
export LIBRARY_PATH=$NP/lib${LIBRARY_PATH:+:$LIBRARY_PATH}

export CPATH=$NV/nccl/include:$NV/cudnn/include:$CUDA_HOME/include:${CPATH:+:$CPATH}
[ -d "$CUDA_HOME/include/cccl" ] && export CPATH=$CUDA_HOME/include/cccl:$CPATH
for E in "$HOME/software/elfutils-dev/usr/include" /usr/include; do
    [ -f "$E/elfutils/libdwfl.h" ] && export CPATH=$E:$CPATH && break
done
export LIBRARY_PATH=$NV/nccl/lib:$NV/cudnn/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export CUDNN_PATH=$NV/cudnn

export NVTE_UB_DEEPGEMM_ROOT=${TE_DEEPGEMM_ROOT:-$H/deepgemm}

export TMPDIR=$H/tmp; mkdir -p $TMPDIR
export NVTE_CMAKE_BUILD_DIR=${TE_CMAKE_BUILD_DIR:-$H/transformerengine/build/cmake-persistent}
export UV_CACHE_DIR=$H/cache/uv

TE_SRC=${TE_SRC:-$H/transformerengine}
cd "$TE_SRC"

if [ "${TE_BUILD_MODE:-editable}" = wheel-only ]; then
    : "${TE_WHEEL_OUT:?TE_WHEEL_OUT not set}" "${TE_BUILD_BASE:?TE_BUILD_BASE not set}" "${TE_CMAKE_BUILD_DIR:?TE_CMAKE_BUILD_DIR not set}"
    case "$TE_BUILD_BASE$TE_WHEEL_OUT$TE_CMAKE_BUILD_DIR" in "$TE_SRC"*) echo "wheel-only outputs must be outside $TE_SRC"; exit 1 ;; esac
    mkdir -p "$TE_WHEEL_OUT" "$TE_BUILD_BASE/egg"
    printf '[build]\nbuild_base = %s\n[egg_info]\negg_base = %s\n' "$TE_BUILD_BASE" "$TE_BUILD_BASE/egg" > "$TE_BUILD_BASE/dist-extra.cfg"
    export DIST_EXTRA_CONFIG=$TE_BUILD_BASE/dist-extra.cfg
    set +e
    uv build --wheel --no-build-isolation --python "$V/bin/python" --out-dir "$TE_WHEEL_OUT" .
    _rc=$?
    set -e
    echo "TE BUILD EXIT: $_rc (wheel-only, no swap)"
    ls -l "$TE_WHEEL_OUT"/*.whl 2>/dev/null || { echo "no wheel in $TE_WHEEL_OUT"; exit 1; }
    exit $_rc
fi

if [ "${TE_BUILD_MODE:-editable}" = wheel ]; then
    _core=libtransformer_engine.so
    _torch=$(basename "$(ls transformer_engine_torch.cpython-*.so 2>/dev/null | head -1)")
    [ -n "$_torch" ] || _torch=transformer_engine_torch.cpython-312-x86_64-linux-gnu.so
    _tag=$(date +%Y%m%d-%H%M%S)${PBS_JOBID:+-${PBS_JOBID%%.*}}
    _tags=$H/transformerengine/build/so-tags
    mkdir -p "$_tags"
    if [ ! -L so-live ]; then
        [ -f "$_core" ] && [ -f "$_torch" ] || { echo "no live .so pair to migrate"; exit 1; }
        _m=$_tags/migrated-$_tag; mkdir -p "$_m"
        cp -p "$_core" "$_torch" "$_m/"
        ln -sfn "build/so-tags/$(basename "$_m")" so-live.new && mv -T so-live.new so-live
        for _f in "$_core" "$_torch"; do ln -sfn "so-live/$_f" "$_f.new" && mv -T "$_f.new" "$_f"; done
        echo "migrated the live pair to so-live -> build/so-tags/$(basename "$_m")"
    fi
    _out=$H/transformerengine/build/wheel-$_tag
    rm -rf "$_out"; mkdir -p "$_out"
    set +e
    uv build --wheel --no-build-isolation --python "$V/bin/python" --out-dir "$_out" .
    _rc=$?
    set -e
    echo "TE BUILD EXIT: $_rc"
    _whl=$(ls "$_out"/*.whl 2>/dev/null | head -1)
    if [ $_rc -ne 0 ] || [ -z "$_whl" ]; then
        rm -rf "$_out"
        echo "BUILD FAILED -- live pair untouched (so-live -> $(readlink so-live))"
        exit ${_rc/#0/1}
    fi
    _new=$_tags/$_tag; mkdir -p "$_new"
    unzip -q -j -o "$_whl" "*$_core" "*$_torch" -d "$_new" \
        || { rm -rf "$_new" "$_out"; echo "wheel lacks the pair -- live pair untouched"; exit 1; }
    [ -s "$_new/$_core" ] && [ -s "$_new/$_torch" ] || { rm -rf "$_new" "$_out"; echo "pair incomplete in the wheel -- live pair untouched"; exit 1; }
    chmod 755 "$_new"/*.so
    rm -rf "$_out"
    _old=$(readlink so-live)
    ln -sfn "build/so-tags/$_tag" so-live.new && mv -T so-live.new so-live
    echo "build OK -- so-live: $_old -> build/so-tags/$_tag (one rename); the previous pair stays for running jobs"
    ls -1dt "$_tags"/* | tail -n +4 | while read -r _d; do
        [ "$_d" = "$_new" ] || rm -rf "$_d"
    done
    echo "so-tags kept: $(ls -1 "$_tags" | tr '\n' ' ')"
    exit 0
fi

_ts=$(date +%Y%m%d-%H%M%S)
_baks=()
for _f in libtransformer_engine.so transformer_engine_torch.cpython-*.so; do
    [ -f "$_f" ] || continue
    cp -p "$_f" "$_f.bak-$_ts"
    _baks+=("$_f" "$_f.bak-$_ts")
done
if [ ${#_baks[@]} -ge 4 ]; then
    echo "entry snapshot: ${_baks[0]}, ${_baks[2]} -> .bak-$_ts"
    _skew=$(( $(stat -c %Y "${_baks[0]}") - $(stat -c %Y "${_baks[2]}") ))
    [ ${_skew#-} -gt 300 ] && echo \
      "WARNING: .so pair mtimes differ by ${_skew#-}s at entry — an interrupted build may have left a mixed pair"
fi

_build_cleanup() {
    _rc=$?
    if [ "${_TE_BUILD_OK:-0}" != "1" ] && [ ${#_baks[@]} -gt 0 ]; then
        _i=0
        while [ $_i -lt ${#_baks[@]} ]; do
            cp -p "${_baks[$_i+1]}" "${_baks[$_i]}" && rm -f "${_baks[$_i+1]}"
            _i=$((_i+2))
        done
        echo "EXIT TRAP (rc=$_rc): .so pair restored to the pre-build state"
    fi
}
trap 'exit $?' TERM INT PIPE
trap _build_cleanup EXIT

set +e
uv pip install --python $V/bin/python \
  --override $H/h200-tuning/setup/$(basename $V)-overrides.txt \
  --no-build-isolation --editable .
_rc=$?
set -e
echo "TE BUILD EXIT: $_rc"
if [ $_rc -ne 0 ]; then
    _i=0
    while [ $_i -lt ${#_baks[@]} ]; do
        cp -p "${_baks[$_i+1]}" "${_baks[$_i]}" && rm -f "${_baks[$_i+1]}"
        _i=$((_i+2))
    done
    _TE_BUILD_OK=1  # the explicit restore above already ran; keep the EXIT trap a no-op
    echo "BUILD FAILED — .so pair restored to the pre-build state; tree left as it was"
    exit $_rc
fi
_i=0
while [ $_i -lt ${#_baks[@]} ]; do
    rm -f "${_baks[$_i+1]}"
    _i=$((_i+2))
done
_TE_BUILD_OK=1
echo "build OK — new .so pair in place; entry snapshots removed"
