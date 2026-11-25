#!/bin/bash
# applycaps.sh w0 w1 ... w7 -- set all 8 caps on the LOCAL node; refuses unless sum == BUDGET (default 2400).
# Lowers first, then raises, so the node total never exceeds the budget mid-update.
set -u
[ "$#" -eq 8 ] || { echo "need 8 caps"; exit 2; }
s=0; for w in "$@"; do s=$((s+w)); done
BUDGET=${BUDGET:-2400}
[ "$s" -eq "$BUDGET" ] || { echo "sum $s != $BUDGET, refusing"; exit 2; }
D=$(dirname "$0")
mapfile -t cur < <(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%d\n",$1+0.5}')
new=("$@"); rc=0
for phase in lower raise; do
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    if { [ $phase = lower ] && [ "${new[$g]}" -lt "${cur[$g]}" ]; } || { [ $phase = raise ] && [ "${new[$g]}" -gt "${cur[$g]}" ]; }; then
      "$D/setcap.sh" "$g" "${new[$g]}" >/dev/null & pids+=($!)
    fi
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
done
got=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%d ",$1+0.5}')
echo "$(hostname) caps: $got"; exit $rc
