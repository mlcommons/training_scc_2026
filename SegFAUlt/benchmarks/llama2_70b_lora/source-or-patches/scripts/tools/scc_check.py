#!/usr/bin/env python3
"""Vérifie un paquet SCC26 <team_name>/ selon les slides du comité (Benchmarking-Overview-SCC26.pdf, p. 17 et 23) et
benchmarks.md (§ 6 et 7), sans GPU ni dépendance.

Usage : tools/scc_check.py <chemin de <team_name>/> [--partial]

Contrôles :
  * arborescence exacte : systems/, benchmarks/llama2_70b_lora/ (README.md, run_and_time.sh, config.sh,
    source-or-patches/), results/<système>/llama2_70b_lora/ ;
  * systems/<système>.json : champs principaux de la description MLPerf (validée en entier par package_checker) et
    les 11 champs du modèle SCC, avec leurs types, system_name = nom du fichier ;
  * results : exactement result_0.txt .. result_9.txt, plus scaling.json (--partial : moins de 10 toléré) ;
  * chaque log : run_start, run_stop, graine (distinctes), GBS identique, évaluations tous les 384 samples ;
  * au plus 1 essai en échec ; TTT = (run_stop - run_start) / 60000 ; moyenne olympique des 8 essais du milieu ;
  * aucun fichier de plus de 50 Mo, aucun lien cassé.
Code de retour : 0 si valide, 1 sinon.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scc_common as sc  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team_dir")
    ap.add_argument("--benchmark", default="llama2_70b_lora")
    ap.add_argument("--partial", action="store_true", help="moins de 10 essais toléré (soumission partielle)")
    args = ap.parse_args(argv)
    team_dir, bench = os.path.abspath(args.team_dir), args.benchmark
    errors, warnings, out = [], [], []

    top = sorted(os.listdir(team_dir)) if os.path.isdir(team_dir) else []
    for d in ("systems", "benchmarks", "results"):
        if d not in top:
            errors.append(f"{d}/ absent de {team_dir}")
    for extra in set(top) - {"systems", "benchmarks", "results"}:
        warnings.append(f"élément inattendu à la racine du paquet : {extra}")

    for root, dirs, files in os.walk(team_dir):
        for name in files:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                errors.append(f"lien cassé : {os.path.relpath(path, team_dir)}")
            elif os.path.getsize(path) > sc.MAX_FILE_BYTES:
                errors.append(f"fichier de plus de 50 Mo : {os.path.relpath(path, team_dir)}")

    bench_dir = os.path.join(team_dir, "benchmarks", bench)
    if not os.path.isdir(bench_dir) or not os.listdir(bench_dir):
        errors.append(f"benchmarks/{bench}/ absent ou vide (code et scripts d'exécution)")
    else:
        for name in ("README.md", "run_and_time.sh", "config.sh", "source-or-patches"):
            if not os.path.exists(os.path.join(bench_dir, name)):
                errors.append(f"benchmarks/{bench}/{name} absent")
    other = set(os.listdir(os.path.join(team_dir, "benchmarks"))) - {bench} if "benchmarks" in top else set()
    if other:
        warnings.append(f"benchmarks/ contient aussi : {sorted(other)}")

    systems = {}
    sys_dir = os.path.join(team_dir, "systems")
    for name in sorted(os.listdir(sys_dir)) if os.path.isdir(sys_dir) else []:
        if not name.endswith(".json"):
            warnings.append(f"systems/{name} : pas un .json")
            continue
        try:
            with open(os.path.join(sys_dir, name)) as f:
                desc = json.load(f)
        except ValueError as e:
            errors.append(f"systems/{name} : JSON invalide ({e})")
            continue
        for key, typ in sc.SCC_SYSTEM_FIELDS:
            if key not in desc:
                errors.append(f"systems/{name} : champ {key} absent")
            elif not isinstance(desc[key], typ) or (typ is str and not desc[key].strip()):
                errors.append(f"systems/{name} : {key} doit être un {typ.__name__} non vide")
        missing = [k for k in sc.MLPERF_SYSTEM_FIELDS if k not in desc]
        if missing:
            errors.append(f"systems/{name} : champs MLPerf absents {missing} (package_checker refusera le paquet)")
        if desc.get("system_name") != name[:-5]:
            errors.append(f"systems/{name} : system_name {desc.get('system_name')!r} différent du nom de fichier")
        systems[name[:-5]] = desc
    if not systems:
        errors.append("systems/ ne contient aucune description de système")

    res_root = os.path.join(team_dir, "results")
    res_systems = sorted(os.listdir(res_root)) if os.path.isdir(res_root) else []
    for s in set(res_systems) - set(systems):
        errors.append(f"results/{s}/ sans systems/{s}.json")
    for s in sorted(systems):
        res_dir = os.path.join(res_root, s, bench)
        if not os.path.isdir(res_dir):
            errors.append(f"results/{s}/{bench}/ absent")
            continue
        names = sorted(os.listdir(res_dir))
        idx = sorted(int(sc.RESULT_RE.match(n).group(1)) for n in names if sc.RESULT_RE.match(n))
        for n in names:
            if not sc.RESULT_RE.match(n) and n != sc.SCALING_FILE:
                errors.append(f"results/{s}/{bench}/{n} : nom hors convention result_<i>.txt")
        if idx != list(range(len(idx))):
            errors.append(f"results/{s}/{bench} : numérotation non continue depuis 0 : {idx}")
        if len(idx) != sc.N_RUNS:
            (warnings if args.partial and idx else errors).append(
                f"results/{s}/{bench} : {len(idx)} logs au lieu de {sc.N_RUNS}")

        trials = [sc.analyze_log(os.path.join(res_dir, f"result_{i}.txt")) for i in idx]
        for i, t in zip(idx, trials):
            if not t["started"]:
                errors.append(f"result_{i}.txt : run_start absent")
            elif t["ttt_min"] is None:
                errors.append(f"result_{i}.txt : run_stop absent")
            if t["benchmark"] not in (None, "llama2_70b_lora"):
                errors.append(f"result_{i}.txt : submission_benchmark {t['benchmark']!r}")
            off = [n for n, _ in t["evals"] if n is not None and n % sc.EVAL_QUANTUM]
            if off:
                warnings.append(f"result_{i}.txt : évaluations hors multiples de {sc.EVAL_QUANTUM} : {off}")
        seeds = [t["seed"] for t in trials]
        if None in seeds or len(set(seeds)) != len(seeds):
            errors.append(f"{s} : graines absentes ou en double : {seeds}")
        gbs = {t["gbs"] for t in trials}
        if len(gbs) > 1:
            errors.append(f"{s} : GBS différents selon les essais : {sorted(gbs, key=str)}")
        orgs = {t["org"] for t in trials}
        if len(orgs) > 1:
            errors.append(f"{s} : submission_org différents : {sorted(map(str, orgs))}")
        score, failed = sc.olympic_mean(trials)
        if failed > 1:
            errors.append(f"{s} : {failed} essais en échec (1 seul toléré)")
        elif failed == 1:
            warnings.append(f"{s} : 1 essai en échec, toléré (compte comme le plus lent)")

        d = systems[s]
        out.append(f"## {s} — {d.get('accelerator_count')} x {d.get('accelerator_model')}, "
                   f"GBS {sorted(gbs, key=str)}, équipe {sorted(map(str, orgs))}\n")
        out.append(sc.trials_table(trials) + "\n")
        out.append(f"Moyenne olympique (TTT, 8 essais du milieu) : **{sc.fmt(score, '.2f')} min**"
                   + ("" if len(trials) == sc.N_RUNS else f" (provisoire, {len(trials)} essais)") + "\n")
        scaling = os.path.join(res_dir, sc.SCALING_FILE)
        if os.path.exists(scaling) and score is not None:
            with open(scaling) as f:
                factor = json.load(f).get("scaling_factor", 1.0)
            out.append(f"Score après normalisation RCP (× {factor:.4f}, {sc.SCALING_FILE}) : "
                       f"**{score * factor:.2f} min**\n")

    print(f"# Vérification SCC26 : {team_dir}\n")
    print("\n".join(out))
    for w in warnings:
        print(f"AVERTISSEMENT : {w}")
    for e in errors:
        print(f"ERREUR : {e}")
    print("\nSTRUCTURE SCC VALIDE" if not errors else "\nSTRUCTURE SCC NON VALIDE")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
