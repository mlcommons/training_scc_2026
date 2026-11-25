#!/bin/bash
# setcap.sh <gpu_index> <watts> -- non-interactive driver for /mnt/opt/systools/set-gpu-freq
# on the LOCAL node. Answers its prompts: GPU choice (index+1), cap, SM clock "s"(skip), "yes".
set -eu
g=$1; w=$2
[[ "$g" =~ ^[0-7]$ && "$w" =~ ^[0-9]+$ ]] || { echo "usage: setcap.sh <0-7> <200-400>"; exit 2; }
out=$(printf '%s\n%s\ns\nyes\n' "$((g+1))" "$w" | /mnt/opt/systools/set-gpu-freq 2>&1) || { echo "$out"; exit 1; }
echo "$out" | grep -q "power cap -> ${w} W" || { echo "$out"; exit 1; }
echo "$(hostname) GPU$g -> ${w} W"
