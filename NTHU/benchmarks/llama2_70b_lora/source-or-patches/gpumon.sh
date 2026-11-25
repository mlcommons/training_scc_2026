#!/bin/bash
# Log this node's GPU power, module (Grace+Hopper+memory) power, clocks and
# throttle counters every INTERVAL seconds to OUTDIR/gpumon-<node>.csv.
#
#   gpumon.sh OUTDIR [INTERVAL=5]
#
# Started by launch.sh on every node of a job when GPUMON=1, as a second
# (--overlap) step of the same allocation. Field order matches
# probe/throttle_report.py, plus module_power_w at the end; values carry no
# units (nounits), and the event-reason bitmask is hex.
set -u
out=$1; iv=${2:-5}
f=$out/gpumon-$(hostname -s).csv
Q=timestamp,temperature.gpu.tlimit,power.draw,power.draw.instant,enforced.power.limit,temperature.gpu,temperature.memory,utilization.gpu,clocks.current.graphics,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.hw_power_brake_slowdown
echo "$Q,module_power_w" > "$f"
while true; do
  g=$(nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits 2>/dev/null) || break
  m=$(nvidia-smi -q -d POWER 2>/dev/null | awk '/Module Power Readings/{m=1} m&&/Instantaneous Power Draw/{print $5; exit}')
  echo "$g, ${m:-nan}" >> "$f"
  sleep "$iv"
done
