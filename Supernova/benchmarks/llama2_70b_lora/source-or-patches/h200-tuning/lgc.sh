#!/bin/bash
set -u
WANT=${1:-}
[[ "$WANT" =~ ^[0-9]+$ ]] || { echo "lgc.sh: usage: lgc.sh <MHz>" >&2; exit 2; }
HOST=$(hostname -s)
if [ "$WANT" = "0" ]; then
    sudo -n /usr/local/sbin/gpu-lgc 0,0 >/dev/null 2>&1 \
        || { echo "lgc.sh: clearing the pin failed on $HOST" >&2; exit 1; }
    echo "lgc: $HOST pin cleared"
    exit 0
fi
sudo -n /usr/local/sbin/gpu-lgc "$WANT,$WANT" >/dev/null 2>&1 \
    || { echo "lgc.sh: gpu-lgc $WANT failed on $HOST" >&2; exit 1; }
sleep 2
GOT=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | sort -u | paste -sd,)
ok=1
for c in ${GOT//,/ }; do
    d=$(( c - WANT )); [ $d -lt 0 ] && d=$(( -d ))
    [ $d -le 60 ] || ok=0
done
[ "$ok" = 1 ] || { echo "lgc.sh: SM clock readback $GOT != pin $WANT on $HOST" >&2; exit 1; }
echo "lgc: $HOST pin $WANT MHz, clocks $GOT"
