#!/usr/bin/env bash
# Shows the Slurm queue and the results of all jobs, or the results of one job.
#   scripts/results.sh                    queue and the results of all jobs
#   scripts/results.sh results/<folder>   the results of one job
[ -n "$1" ] && set -- "$(readlink -f "$1")"
cd "$(dirname "$(readlink -f "$0")")/.."

report() {
python3 - "$@" <<'PY'
import glob, json, os, sys
ROW = "  %-9s %6s  %9s  %9s  %9s  %7s  %9s  %9s  %s"
m = lambda x: "-" if x is None else "%.1f min" % x
for folder in [f for f in sys.argv[1:] if os.path.isdir(f)]:
    times = {}   # times.txt: "result_<i> <launch ms> <exit ms>", then "job <start ms> <end ms>"
    if os.path.exists(folder + "/times.txt"):
        for line in open(folder + "/times.txt"):
            k, a, b = line.split()
            times[k] = (int(a), int(b))
    print(os.path.basename(folder.rstrip("/")))
    print((ROW % ("run", "seed", "total", "to start", "trained", "samples", "samples/s", "eval loss", "")).rstrip())
    n = len(glob.glob(folder + "/result_*.txt"))
    trained = []
    for i in range(n):
        name = "result_%d" % i
        e = {}   # last MLLOG event of each key
        for line in open("%s/%s.txt" % (folder, name), errors="ignore"):
            if ":::MLLOG" in line:
                try:
                    d = json.loads(line.split(":::MLLOG", 1)[1])
                    e[d["key"]] = d
                except ValueError:
                    pass
        t = lambda k: e[k]["time_ms"] if k in e else None
        launch, end = times.get(name, (None, None))
        start, stop = t("run_start"), t("run_stop")
        loss = e["eval_accuracy"]["value"] if "eval_accuracy" in e else None
        minutes = (stop - start) / 60000 if start and stop else None
        ok = minutes is not None and loss is not None and loss <= 0.925
        if ok:
            trained.append(minutes)
        speed = e.get("tracked_stats", {}).get("value", {}).get("throughput")
        print((ROW % (name, e.get("seed", {}).get("value", "-"),
                      m((end - launch) / 60000 if launch else None),
                      m((start - launch) / 60000 if launch and start else None), m(minutes),
                      e["eval_accuracy"]["metadata"]["samples_count"] if "eval_accuracy" in e else "-",
                      "%.2f" % speed if speed else "-", "%.4f" % loss if loss is not None else "-",
                      "converged" if ok else "NOT CONVERGED" if minutes else "did not finish")).rstrip())
    if "job" in times:
        print("  job: %.1f min from start to end" % ((times["job"][1] - times["job"][0]) / 60000))
    if os.path.exists(folder + "/summary.json"):   # the competition's checks, run by scripts/mlperf.sbatch after 10 runs
        r = {k.split(":")[1]: v for k, v in json.load(open(folder + "/summary.json"))[0].items() if k.startswith("llama2_70b_lora:")}
        log = open(folder + "/package_checker.log").read() if os.path.exists(folder + "/package_checker.log") else ""
        print("  package checker: %s" % ("passed" if "FOUND NO ERRORS" in log else "FAILED, see package_checker.log"))
        print("  result summarizer: time to train %s, RCP %s, scaling factor %s" % (
            "%.2f min" % r["time_to_train"] if r["time_to_train"] else "-", r["RCP"] or "-", r["rcp_scaling_factor"]))
        continue
    s = sorted(trained)
    if n == 10 and len(s) >= 9:   # the rules: drop the fastest and the slowest run, one failed run counts as the slowest
        print("  score: %.1f min, mean trained time of the middle 8 runs" % (sum(s[1:9]) / 8))
    elif s:
        print("  %d of %d runs converged, mean trained time %.1f min" % (len(s), n, sum(s) / len(s)))
PY
}

if [ -n "$1" ]; then
  report "$1"
  exit
fi
echo "queue:"
squeue -h -o "  %i  %T  %M  %j" | grep . || echo "  none"
echo
echo "results:"
report results/*/
