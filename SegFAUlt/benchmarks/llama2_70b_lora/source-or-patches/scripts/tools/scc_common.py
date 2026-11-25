"""Fonctions communes aux outils SCC26 (Python 3.9, bibliothèque standard) : lecture des logs MLPerf, TTT, score.

Règles appliquées (benchmarks.md § 6 et 7, slides du comité Benchmarking-Overview-SCC26.pdf p. 17 et 23) :
  * TTT (min) = (run_stop.time_ms - run_start.time_ms) / 60000 ;
  * cible : perte de validation (eval_accuracy) <= 0.925, évaluée tous les 384 samples ;
  * 10 essais ; moyenne olympique : on retire le plus rapide et le plus lent, on moyenne les 8 autres ;
  * au plus 1 essai en échec sur 10 (il compte alors comme le plus lent, donc retiré) ;
  * convergence plus rapide que la moyenne RCP mais crédible : score multiplié par le facteur de scaling.json.
"""
import json
import math
import os
import re
import subprocess

SCC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RUNS = 10
EVAL_QUANTUM = 384
TARGET = 0.925
MAX_FILE_BYTES = 50 * 2**20       # mlperf_logging.repo_checker refuse les fichiers > 50 Mo
RESULT_RE = re.compile(r"^result_(\d+)\.txt$")    # slides p. 23 : result_0.txt .. result_9.txt
SCALING_FILE = "scaling.json"                     # écrit dans results/<système>/<benchmark>/ par package_checker
MLLOG_RE = re.compile(r":::MLLOG (\{.*\})\s*$")
# systems/<student_system_name>.json : description MLPerf complète (slides p. 23, validée par package_checker), plus
# les champs du modèle SCC de benchmarks.md § 7 qui ne s'y trouvent pas. Champs SCC et types, dans cet ordre.
SCC_SYSTEM_FIELDS = [("system_name", str), ("host_processor", str), ("host_processor_count", int),
                     ("accelerator_model", str), ("accelerator_count", int), ("framework", str),
                     ("operating_system", str), ("interconnect", str), ("interconnect_topology", str),
                     ("cuda_version", str), ("nccl_version", str)]
MLPERF_SYSTEM_FIELDS = ["submitter", "division", "status", "number_of_nodes", "accelerators_per_node",
                        "accelerator_model_name", "framework", "other_software_stack"]


def scc_env():
    """Variables de scc.env (TEAM_NAME, ZEPH_DIR, SCC_BENCHMARK, ...), surcharges de l'environnement comprises."""
    names = ["TEAM_NAME", "ZEPH_DIR", "SCC_BENCHMARK", "MLPERF_BENCHMARK", "MLPERF_RULESET"]
    script = f'source "{os.path.join(SCC_DIR, "scc.env")}"; ' + "; ".join(f'echo "${{{n}}}"' for n in names)
    out = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    return dict(zip(names, out.stdout.splitlines()))


def mllog_events(path):
    """Événements :::MLLOG d'un log (le préfixe de rang « 0: » ajouté par srun -l est ignoré)."""
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            m = MLLOG_RE.search(line)
            if m:
                try:
                    events.append(json.loads(m.group(1)))
                except ValueError:
                    pass
    return events


def first(events, key):
    return next((e for e in events if e.get("key") == key), None)


def last(events, key):
    return next((e for e in reversed(events) if e.get("key") == key), None)


def analyze_log(path):
    """Résumé d'un essai : graine, GBS, organisation, évaluations, statut, TTT selon la formule SCC."""
    ev = mllog_events(path)
    start, stop = first(ev, "run_start"), last(ev, "run_stop")
    value = lambda key: (first(ev, key) or {}).get("value")  # noqa: E731
    evals = [(e.get("metadata", {}).get("samples_count"), e.get("value")) for e in ev if e.get("key") == "eval_accuracy"]
    status = (stop or {}).get("metadata", {}).get("status")
    ttt = (stop["time_ms"] - start["time_ms"]) / 60000 if start and stop else None
    target = value("target_accuracy") or TARGET
    converged = status == "success" and bool(evals) and evals[-1][1] is not None and evals[-1][1] <= target
    return {"log": path, "seed": value("seed"), "gbs": value("global_batch_size"), "org": value("submission_org"),
            "platform": value("submission_platform"), "benchmark": value("submission_benchmark"),
            "lr": value("opt_base_learning_rate"), "target": target, "evals": evals,
            "started": start is not None, "status": status or ("interrompu" if start else "pas démarré"),
            "converged": converged, "ttt_min": ttt, "samples": evals[-1][0] if evals else None,
            "final_loss": evals[-1][1] if evals else None,
            "run_start_ms": start["time_ms"] if start else None}


def olympic_mean(trials):
    """(score, nombre d'échecs) : moyenne sans le plus rapide ni le plus lent ; un échec compte comme le plus lent.
    score = None si moins de 3 essais ou si au moins 2 échecs (règle SCC : 1 échec toléré)."""
    scores = sorted(t["ttt_min"] if t["converged"] and t["ttt_min"] is not None else math.inf for t in trials)
    failed = sum(math.isinf(s) for s in scores)
    middle = scores[1:-1]
    if len(scores) < 3 or any(math.isinf(s) for s in middle):
        return None, failed
    return sum(middle) / len(middle), failed


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def trials_table(trials):
    """Tableau Markdown des essais (une ligne par result_<i>.txt)."""
    rows = ["| # | graine | statut | samples | perte finale | TTT (min) |", "|---|---|---|---|---|---|"]
    for i, t in enumerate(trials):
        rows.append(f"| {i} | {t['seed']} | {'convergé' if t['converged'] else t['status']} | {t['samples'] or '-'} | "
                    f"{fmt(t['final_loss'], '.4f')} | {fmt(t['ttt_min'], '.2f')} |")
    return "\n".join(rows)
