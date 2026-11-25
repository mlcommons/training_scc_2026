#!/usr/bin/env python3
"""Résumé des runs Helma à partir des logs MLPerf, de la télémétrie GPU et des pré-tests.

Usage :
  summarize_results.py results/R02_baseline_8gpu/123456   # un job   -> tableau + détails
  summarize_results.py results/                           # tous les jobs -> tableau comparatif
  summarize_results.py results/ --json                    # mêmes données en JSON
Python 3.9, bibliothèque standard uniquement (exécutable sur le frontal ou un nœud).

Définitions :
  TTT        run_start -> run_stop (règle MLPerf ; l'init et le chargement du checkpoint sont exclus)
  init       init_start -> run_start (construction du modèle, checkpoint, warmup, capture des CUDA graphs)
  step       médiane des temps de step instantanés (un sur 10) après les 2 premiers relevés
  TFLOP/s    2598 TFLOP par séquence de 8192 tokens (MODEL_TFLOP_PER_SAMPLE de NVIDIA) x débit / nb GPU
  MFU        TFLOP/s par GPU / 989.4 (pic dense BF16 d'une H100 SXM ; le pic FP8 est le double)
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

TFLOP_PER_SAMPLE = 2598.0
H100_BF16_DENSE_TFLOPS = 989.4
SEQ_LEN = 8192
GPUS_PER_NODE = 4


def mllog_events(path):
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            i = line.find(":::MLLOG")
            if i < 0:
                continue
            try:
                events.append(json.loads(line[i + len(":::MLLOG"):]))
            except ValueError:
                continue
    return events


def first(events, key):
    for e in events:
        if e.get("key") == key:
            return e
    return None


def read_env(run_dir, idx):
    env = {}
    path = os.path.join(run_dir, "logs", f"env_run{idx}.txt")
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    env[k] = v
    return env


def analyze_experiment(log_path, run_dir):
    idx = re.search(r"mllog_run(\d+)\.log$", log_path).group(1)
    ev = mllog_events(log_path)
    env = read_env(run_dir, idx)
    r = {"run_dir": run_dir, "experiment": int(idx), "log": log_path,
         "run_id": env.get("RUN_ID", os.path.basename(os.path.dirname(run_dir)).split("_")[0]),
         "job_id": os.path.basename(run_dir)}

    platform = first(ev, "submission_platform")
    nodes = int(platform["value"].split("x")[0]) if platform else int(env.get("DGXNNODES", "0") or 0)
    r["gpus"] = nodes * GPUS_PER_NODE
    for key, name in (("global_batch_size", "gbs"), ("opt_base_learning_rate", "lr"), ("seed", "seed"),
                      ("opt_learning_rate_training_steps", "max_steps")):
        e = first(ev, key)
        r[name] = e["value"] if e else None
    for k in ("TP", "CP", "FP8", "MCORE_CUDA_GRAPH", "STAGE_TO_TMPDIR", "NCCL_IB_HCA", "RECOMPUTE_GRANULARITY"):
        r[k.lower()] = env.get(k)

    t_init, t_start, stop = first(ev, "init_start"), first(ev, "run_start"), first(ev, "run_stop")
    r["init_min"] = (t_start["time_ms"] - t_init["time_ms"]) / 60000 if t_init and t_start else None
    if stop and t_start:
        r["status"] = stop.get("metadata", {}).get("status", "?")
        r["ttt_min"] = (stop["time_ms"] - t_start["time_ms"]) / 60000
        r["samples"] = stop.get("metadata", {}).get("samples_count")
    else:
        r["status"] = "incomplet" if t_start else "pas démarré"
        r["ttt_min"] = r["samples"] = None

    r["evals"] = [(e.get("metadata", {}).get("samples_count"), round(e["value"], 4))
                  for e in ev if e.get("key") == "eval_accuracy"]
    stats = [e for e in ev if e.get("key") == "tracked_stats" and isinstance(e.get("value"), dict)]
    inst = [e["value"]["train_step_time"] for e in stats if "reduced_train_loss" in e["value"]]
    losses = [e["value"]["reduced_train_loss"] for e in stats if "reduced_train_loss" in e["value"]]
    vals = [e["value"]["validation_time"] for e in stats if "validation_time" in e["value"]]
    steady = inst[2:] if len(inst) > 4 else inst
    r["step_s"] = statistics.median(steady) if steady else None
    # La perte loggée par NVIDIA est divisée sur place par le nombre de micro-batches (schedules.py :
    # `output_tensor /= num_microbatches` sur le tenseur aussi mis dans le dict de log) ; les gradients,
    # eux, sont corrects. On la remultiplie pour la rendre comparable d'une config à l'autre.
    micro = max(1, int(env.get("MINIBS", "1") or 1) // max(1, int(env.get("MBS", "1") or 1)))
    r["micro_batches"] = micro
    r["last_train_loss"] = round(losses[-1] * micro, 4) if losses else None
    r["eval_time_s"] = statistics.mean(vals) if vals else None
    if r["step_s"] and r["gbs"] and r["gpus"]:
        sps = r["gbs"] / r["step_s"]
        r["samples_per_s"] = sps
        r["tokens_per_s_gpu"] = sps * SEQ_LEN / r["gpus"]
        r["tflops_gpu"] = TFLOP_PER_SAMPLE * sps / r["gpus"]
        r["mfu"] = r["tflops_gpu"] / H100_BF16_DENSE_TFLOPS
    else:
        r["samples_per_s"] = r["tokens_per_s_gpu"] = r["tflops_gpu"] = r["mfu"] = None

    comp = os.path.join(run_dir, "logs", f"compliance_run{idx}.out")
    r["compliance"] = None
    if os.path.exists(comp):
        text = open(comp, errors="replace").read()
        r["compliance"] = "FAILED" if re.search(r"FAIL", text) else ("SUCCESS" if "SUCCESS" in text else "?")
    return r


def telemetry(run_dir):
    mem, power, busy_power = [], [], []
    for path in glob.glob(os.path.join(run_dir, "telemetry", "gpu_*.csv")):
        with open(path, errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 8 or row[0].startswith("timestamp"):
                    continue
                try:
                    util, used, pdraw = float(row[2]), float(row[4]), float(row[6])
                except ValueError:
                    continue
                mem.append(used)
                power.append(pdraw)
                if util > 50:
                    busy_power.append(pdraw)
    return {"mem_max_gib": max(mem) / 1024 if mem else None,
            "power_max_w": max(power) if power else None,
            "power_busy_mean_w": statistics.mean(busy_power) if busy_power else None}


def preflight(run_dir):
    out = {"nccl": {}, "affinity": [], "staging": [], "nccl_check_ok": None, "net": {}}
    path = os.path.join(run_dir, "logs", "nccl_check.log")
    if os.path.exists(path):
        text = open(path, errors="replace").read()
        out["nccl_check_ok"] = "NCCL_CHECK_OK" in text
        for m in re.finditer(r"NCCL_RESULT op=(\S+) group=(\S+) ranks=(\d+) size_MB=(\d+) \S+ \S+ busbw_GBps=([\d.]+)", text):
            out["nccl"][f"{m.group(1)}/{m.group(2)}/{m.group(4)}MB"] = float(m.group(5))
        out["affinity"] = re.findall(r"AFFINITY (rank=\S+ host=\S+ local_rank=\S+ pci=\S+ ncpu=\S+ cpus=\S+)", text)
    dbg = glob.glob(os.path.join(run_dir, "logs", "nccl_debug", "*"))
    if dbg:
        joined = ""
        for p in dbg:
            joined += open(p, errors="replace").read()
        m = re.search(r"NET/IB : Using (.*?);", joined)
        devs = re.findall(r"\[(\d+)\](\S+?):\d+/(RoCE|IB)", m.group(1)) if m else []
        roce = {i for i, _name, kind in devs if kind == "RoCE"}
        via = re.findall(r"via NET/IB/(\d+)", joined)
        out["net"] = {"ib_devices": m.group(1) if m else None,
                      "net_channels": len(via),
                      "roce_channels": sum(1 for i in via if i in roce),
                      "gdrdma_channels": joined.count("GDRDMA")}
    path = os.path.join(run_dir, "logs", "staging.log")
    if os.path.exists(path):
        out["staging"] = re.findall(r"\[(\S+)\] STAGE_DONE seconds=(\d+) bytes=(\d+) GBps=([\d.]+)", open(path).read())
    return out


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def row(r):
    prec = "?" if r["fp8"] is None else ("FP8" if r["fp8"] == "True" else "BF16")
    graphs = "?" if r["mcore_cuda_graph"] is None else ("oui" if r["mcore_cuda_graph"] == "1" else "non")
    return (f"| {r['run_id']} | {r['job_id']} | {r['experiment']} | {r['gpus']} | TP{r['tp'] or '?'}/CP{r['cp'] or '?'} | "
            f"{r['gbs']} | {prec} | {graphs} | {r['seed']} | "
            f"{r['status']} | {fmt(r['ttt_min'], '.2f')} | {r['samples'] or '-'} | "
            f"{r['evals'][-1][1] if r['evals'] else '-'} | {fmt(r['init_min'], '.1f')} | {fmt(r['step_s'], '.3f')} | "
            f"{fmt(r['samples_per_s'], '.3f')} | {fmt(r['tokens_per_s_gpu'], '.0f')} | {fmt(r['tflops_gpu'], '.0f')} | "
            f"{fmt(r['mfu'] * 100 if r['mfu'] else None, '.1f')} | {fmt(r['mem_max_gib'], '.1f')} | "
            f"{fmt(r['power_busy_mean_w'], '.0f')} | {r['compliance'] or '-'} |")


HEADER = ("| run | job | essai | GPU | parallélisme | GBS | précision | CUDA graphs | graine | statut | TTT (min) | "
          "samples | éval finale | init (min) | step (s) | samples/s | tokens/s/GPU | TFLOP/s/GPU | MFU BF16 (%) | "
          "mém. max (Gio) | puissance (W) | compliance |\n|" + "---|" * 22)


def find_run_dirs(root):
    if glob.glob(os.path.join(root, "mllog_run*.log")):
        return [root]
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*", "mllog_run1.log")))


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    root, as_json = argv[0].rstrip("/"), "--json" in argv
    results, details = [], {}
    for run_dir in find_run_dirs(root):
        tel, pre = telemetry(run_dir), preflight(run_dir)
        details[run_dir] = pre
        for log in sorted(glob.glob(os.path.join(run_dir, "mllog_run*.log"))):
            r = analyze_experiment(log, run_dir)
            r.update(tel)
            results.append(r)
    if as_json:
        print(json.dumps({"experiments": results, "preflight": details}, indent=2, ensure_ascii=False))
        return 0
    if not results:
        print(f"Aucun log MLPerf (mllog_run*.log) trouvé sous {root}")
        return 1

    print(f"# Résumé des runs — {root}\n")
    print(HEADER)
    for r in results:
        print(row(r))
    for run_dir, pre in details.items():
        print(f"\n## {os.path.relpath(run_dir, root) if run_dir != root else os.path.basename(run_dir)}\n")
        for r in (x for x in results if x["run_dir"] == run_dir):
            curve = ", ".join(f"{s}→{v}" for s, v in r["evals"]) or "aucune"
            print(f"- essai {r['experiment']} : évaluations (samples→loss) : {curve} ; dernière perte "
                  f"d'entraînement {r['last_train_loss']} (valeur loggée × {r['micro_batches']} micro-batches)")
        if pre["nccl_check_ok"] is not None:
            print(f"- pré-test NCCL : {'OK' if pre['nccl_check_ok'] else 'ÉCHEC'} ; busbw (Go/s) : "
                  + ", ".join(f"{k}={v}" for k, v in pre["nccl"].items()))
        if pre["affinity"]:
            print(f"- affinité : {len(pre['affinity'])} rangs ; ex. {pre['affinity'][0]}")
        if pre["net"]:
            print(f"- NCCL réseau : {pre['net']}")
        for host, secs, nbytes, gbps in pre["staging"]:
            print(f"- staging {host} : {int(nbytes) / 1e9:.1f} Go en {secs} s ({gbps} Go/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
