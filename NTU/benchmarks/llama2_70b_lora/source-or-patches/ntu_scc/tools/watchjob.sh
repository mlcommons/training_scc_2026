#!/bin/bash
# watchjob.sh <jobid> : follow one run; scancel on a stalled log; print the parsed result at the end.
JID=${1:?jobid}; D=/work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b; F=$D/slurm-$JID.out; t_start=""; hung=0
echo "WATCH job $JID $(date +%T)"
while squeue -h -j $JID 2>/dev/null | grep -q .; do
  if [ -f $F ]; then
    [ -z "$t_start" ] && t_start=$(stat -c %Y $F) && echo "LOG appeared $(date +%T)"
    age=$(( $(date +%s) - $(stat -c %Y $F) ))
    if grep -q '"key": "run_start"' $F; then
      if [ $age -gt 360 ]; then echo "HUNG job $JID: no log for ${age}s -> scancel $(date +%T)"; scancel $JID; hung=1; sleep 30; break; fi
    elif [ $(( $(date +%s) - t_start )) -gt 900 ] && [ $age -gt 300 ]; then
      echo "STARTUP-HUNG job $JID: no run_start $(( $(date +%s) - t_start )) s after start, log idle ${age}s -> scancel $(date +%T)"; scancel $JID; hung=1; sleep 30; break
    fi
    if grep -qiE 'CUDA error|NCCL error|Traceback|Xid|Segmentation fault|uncorrectable' $F; then echo "ERROR seen in log $(date +%T)"; grep -iE 'CUDA error|NCCL error|Traceback|Xid|Segmentation fault|uncorrectable' $F | head -3; fi
  fi
  sleep 30
done
sleep 20
python3 - "$F" <<'PY'
import re,sys,json,statistics
F=sys.argv[1]; st=[]; t={}; ev=[]; status="nostart"
for line in open(F, errors="ignore"):
    m=re.search(r':::MLLOG (\{.*\})', line)
    if not m: continue
    try: d=json.loads(m.group(1))
    except Exception: continue
    k=d.get("key")
    if k=="run_start": status="running"
    if k=="tracked_stats" and isinstance(d.get("value"),dict) and "train_step_time" in d["value"]: st.append(d["value"]["train_step_time"])
    if k in ("run_start","run_stop"): t[k]=d["time_ms"]
    if k=="run_stop": status=d.get("metadata",{}).get("status","?")
    if k=="eval_accuracy": ev.append((d["metadata"]["samples_count"], round(d["value"],4)))
ttt=f"{(t['run_stop']-t['run_start'])/60000:.2f}" if len(t)==2 else "nan"
step=f"{statistics.mean(st[10:]):.4f}" if len(st)>10 else "nan"
conv=str(ev[-1][0]) if ev and ev[-1][1]<=0.925 else "none"
print(f"RESULT status={status} TTT={ttt} min converged@{conv} step={step} evals={json.dumps(ev).replace(' ','')}")
PY
echo "WATCH done hung=$hung $(date +%T)"
