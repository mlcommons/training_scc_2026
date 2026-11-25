#!/bin/bash
# setclk.sh <max_MHz|unlock>  -- all 8 GPUs on the LOCAL node, via /mnt/opt/systools/set-gpu-freq.
# A lock of 345..max caps the SM clock at max; the power cap still pulls it lower when needed.
set -u
v=${1:?usage: setclk.sh <max_MHz|unlock>}
if [ "$v" = unlock ]; then ans='a\ns\nu\nyes\n'; want="SM clock  -> unlocked"
else [[ "$v" =~ ^[0-9]+$ ]] || { echo "bad value $v"; exit 2; }; ans="a\ns\nl\n345\n$v\nyes\n"; want="SM clock  -> locked 345-$v MHz"; fi
out=$(printf "$ans" | /mnt/opt/systools/set-gpu-freq 2>&1) || { echo "$out"; exit 1; }
echo "$out" | grep -qF "$want" || { echo "$out" | tail -8; exit 1; }
echo "$(hostname) SM clock: ${v}"
