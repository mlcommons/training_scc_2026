#!/usr/bin/env python3
"""Construit le paquet SCC26 d'un système à partir de ses jobs : <système>/<TEAM_NAME>/.

Usage (frontal, Python 3.9, bibliothèque standard ; appelle `apptainer exec` pour lire les versions du conteneur) :
  tools/build_scc_package.py ngpu8              # ngpu8/runs/ -> ngpu8/<TEAM_NAME>/ + ngpu8/package_report.md
  tools/build_scc_package.py ngpu16
  tools/build_scc_package.py ngpu8 --runs <autre dossier de jobs> --run-id S08 --org NHR@FAU \\
      --out /chemin/test --skip-launcher-check  # paquet de test à partir d'anciens jobs

Arborescence écrite (slides du comité Benchmarking-Overview-SCC26.pdf, p. 23) :
  <TEAM_NAME>/systems/<système>.json                               description MLPerf + champs du modèle SCC
  <TEAM_NAME>/benchmarks/llama2_70b_lora/                          README.md, run_and_time.sh, config.sh,
                                                                   source-or-patches/ (code et scripts exécutés)
  <TEAM_NAME>/results/<système>/llama2_70b_lora/result_0..9.txt    copies exactes des logs des essais
  (results/.../scaling.json est écrit ensuite par package_checker, lancé par tools/scc_verify.sh)

Sélection : les essais <runs>/<RUN_ID>_*/<jobid>/mllog_run<k>.log, dans l'ordre (job, puis essai). Un essai qui n'a
jamais atteint run_start (panne avant l'entraînement) est écarté ; parmi les autres, les 10 PREMIERS sont gardés,
convergés ou non : aucune sélection des meilleurs.
Vérifications : graines distinctes, au plus 1 essai non convergé ou non conforme, submission_org = équipe, GBS,
configs et code identiques d'un job à l'autre, lanceur inchangé depuis la soumission, aucun secret, fichiers < 50 Mo.
Code de retour : 0 si le paquet est complet (10 essais) et sans erreur, 1 sinon, 2 si un secret est détecté.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scc_common as sc  # noqa: E402

# Propres au lot : peuvent changer d'un job à l'autre sans changer ni le système ni la recette.
BATCH_VARS = {"NEXP", "SEED_BASE", "WALLTIME", "TRIAL_BUDGET", "SBATCH_EXCLUDE"}
# Outils d'analyse : une modification après la soumission ne change pas l'entraînement (simple avertissement).
ANALYSIS_ONLY = {"summarize_results.py", "dry_run_config.py", "check_prereqs.sh", "prepare_dataset.sh"}
SKIP_REL = {"scripts/models", "scripts/dataset"}   # checkpoint (129 Go) et données brutes : jamais dans le paquet
SECRET_RES = [re.compile(r"hf_[A-Za-z0-9]{30,}"),
              re.compile(r"\b[A-Z0-9_]*(TOKEN|SECRET|PASSW|API_KEY|CREDENTIAL)[A-Z0-9_]*=\S+")]
# Relevés des jobs d'octobre (logs/system_info.txt), utilisés si un job n'a pas laissé ce fichier.
FALLBACK_HOST = {"os": ["AlmaLinux 9.8 (Olive Jaguar)"], "kernel": ["5.14.0-687.52.1.el9_8.x86_64"],
                 "driver": ["595.91.07"], "mem": ["95830"], "plimit": ["500"], "maxclk": ["1980"]}

README_TMPL = """# MLPerf Training Llama 2 70B LoRA — {system} ({team})

Student Cluster Competition 2026 (SC26) submission package, MLPerf Training v6.0 rules (`mlperf-logging`, ruleset 6.0.0).

## System

{system_text} Metadata: `systems/{system}.json` (full MLPerf v6.0 system description, plus the fields of the SCC
template).
{layout_text}
## Recipe (allowed knobs only)

| Knob | Value |
| --- | --- |
| Global batch size | {gbs} ({minibs} micro-batches of 1 per data-parallel rank, DP={dp}) |
| Learning rate | cosine schedule, peak {lr}, at most {max_steps} steps |
| Parallelism | tensor parallel {tp} + sequence parallel inside each 4-GPU NVLink island, data parallel {dp} across islands |
| Precision | FP8 (Transformer Engine, delayed scaling), FP8 attention; CUDA graphs |
| Fixed | Llama 2 70B frozen base weights, LoRA r=16 alpha=32, dropout, weight decay, loss, Scrolls GovReport data |

Implementation: NVIDIA NeMo / Megatron-Bridge from NVIDIA's MLPerf Training v6.0 submission
(`source-or-patches/nemo/`, unchanged), run with Apptainer instead of Pyxis/Enroot
(`source-or-patches/helma/run_helma.sub`). Target: validation loss <= 0.925, evaluated every {eval_every} samples
(ceil(384 / GBS) = {eval_steps} steps); the first {skip_evals} evaluations are skipped (first check at {first_eval} samples).
{rcp_text}
## GPU settings per job

Set by the launcher at the start of each job on every GPU of its nodes and logged in the job's `logs/power_limit.txt`
(`GPU_POWER_LIMIT`, `GPU_CLOCK_LOCK`); `GPU_RESET=1` resets the GPUs first (`tools/node_guard.sh`).

{settings_table}

## Files

| Path | Content |
| --- | --- |
| `run_and_time.sh` | per-trial script run in the container (NVIDIA's, unchanged; copy of `source-or-patches/nemo/run_and_time.sh`) |
| `config.sh` | `{config}`: current configuration of this system (loads `config_HelmaH100_common.sh`, then NVIDIA `config_common.sh`) |
| `source-or-patches/nemo/` | code executed in the container (snapshot taken by the job), with `llama-zeph.def` (Apptainer recipe) |
| `source-or-patches/helma/` | Slurm/Apptainer launcher and its tools (rank placement, CPU/NUMA binding, page-cache eviction, NCCL pre-test, node guard) |
| `source-or-patches/configs/` | system configuration, launcher defaults, NVIDIA `config_common.sh`, `scc.env` |
| `source-or-patches/configs/job_<id>/` | exact configuration files and command-line overrides of each Slurm job used |
| `source-or-patches/scripts/` | `submit.sh`, packaging and verification tools |

## Steps to launch training

```
apptainer build nvidia-zeph.sif source-or-patches/nemo/llama-zeph.def   # NGC PyTorch 26.04 + NeMo, clean build context
# (runs used an image built earlier with the same recipe, its build-context data stripped without rebuilding:
#  source-or-patches/scripts/tools/make_slim_container.sh; same binaries, /workspace/ft-llm = source-or-patches/nemo)
# dataset and checkpoint: source-or-patches/nemo/README.md and nemo/scripts/ (train.npy, validation.npy, checkpoint)
# train.npy / validation.npy re-created inside the container: source-or-patches/helma/tools/prepare_dataset.sh
./submit.sh {sysdir} --dry-run                            # resolved config, memory estimate, no GPU
./submit.sh {sysdir}                                      # one Slurm job, 10 consecutive trials, random seeds
./submit.sh {sysdir} NEXP=<n>                             # additional trials (overrides: see configs/job_<id>/)
tools/build_scc_package.py {sysdir} --jobs <job dirs> && tools/scc_verify.sh <team_dir>
```

## Results

{results}
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root, skip_rel=()):
    """Fichiers de root (chemins relatifs triés), sans outputs/, __pycache__, éléments cachés, .sif, .pyc ni skip_rel."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        dirs[:] = sorted(d for d in dirs if d not in ("outputs", "__pycache__") and not d.startswith(".")
                         and os.path.normpath(os.path.join(rel_dir, d)) not in skip_rel)
        out += [os.path.normpath(os.path.join(rel_dir, f)) for f in sorted(files)
                if not f.startswith(".") and not f.endswith((".sif", ".pyc"))]
    return out


def tree_digest(root):
    h = hashlib.sha256()
    for rel in walk_files(root, SKIP_REL):
        path = os.path.join(root, rel)
        h.update(rel.encode())
        h.update(sha256_file(path).encode() if os.path.exists(path) else b"broken-link")
    return h.hexdigest()


def copy_files(src_root, dst_root, rels):
    """Copie rels de src_root vers dst_root ; renvoie les fichiers écartés (absent, > 50 Mo)."""
    skipped = []
    for rel in rels:
        src, dst = os.path.join(src_root, rel), os.path.join(dst_root, rel)
        if not os.path.exists(src):
            skipped.append(f"{rel} (absent)")
        elif os.path.getsize(src) > sc.MAX_FILE_BYTES:
            skipped.append(f"{rel} (> 50 Mo)")
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
    return skipped


def config_vars(cfg, names):
    """Valeurs de variables après chargement de la config d'un système."""
    script = f'source "{cfg}" > /dev/null 2>&1; ' + "; ".join(f'echo "${{{n}}}"' for n in names)
    out = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, universal_newlines=True).stdout
    return dict(zip(names, out.splitlines()))


def find_jobs(runs_dir, run_id):
    jobs = [d for d in glob.glob(os.path.join(runs_dir, f"{run_id}_*", "*"))
            if os.path.isdir(d) and os.path.basename(d).isdigit()]
    return sorted(jobs, key=lambda d: int(os.path.basename(d)))


def trial_logs(job_dir):
    logs = glob.glob(os.path.join(job_dir, "mllog_run*.log"))
    return sorted(logs, key=lambda p: int(re.search(r"mllog_run(\d+)\.log$", p).group(1)))


def compliance(job_dir, idx):
    """SUCCESS / FAILED / None, d'après logs/compliance_run<k>.stdout écrit par run_helma.sub."""
    path = os.path.join(job_dir, "logs", f"compliance_run{idx}.stdout")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        text = f.read()
    return "SUCCESS" if re.search(r"\bSUCCESS\b", text) else "FAILED"


def config_digest(job_dir):
    """Empreinte des configs copiées par le job (config/*.sh) et des surcharges hors variables de lot."""
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(job_dir, "config", "*.sh"))):
        h.update(os.path.basename(path).encode())
        h.update(sha256_file(path).encode())
    kept = []
    ov = os.path.join(job_dir, "config", "overrides.txt")
    if os.path.exists(ov):
        with open(ov) as f:
            kept = sorted(line.strip() for line in f if line.strip() and line.split("=", 1)[0] not in BATCH_VARS)
    h.update("\n".join(kept).encode())
    return h.hexdigest()


def launcher_check(runs_dir, job_dir):
    """Compare le lanceur, scc.env et la config actuels aux empreintes relevées par submit.sh à la soumission."""
    jobid = os.path.basename(job_dir)
    path = os.path.join(runs_dir, "slurm", f"launcher_{jobid}.sha256")
    if not os.path.exists(path):
        return [f"job {jobid} : empreintes du lanceur absentes ({path})"], []
    errors, warnings = [], []
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, fpath = parts[0], parts[1].strip()
            if not os.path.exists(fpath) or sha256_file(fpath) != digest:
                msg = f"job {jobid} : {fpath} a changé depuis la soumission"
                (warnings if os.path.basename(fpath) in ANALYSIS_ONLY else errors).append(msg)
    return errors, warnings


def scan_secrets(paths):
    hits = []
    for path in paths:
        with open(path, errors="replace") as f:
            for n, line in enumerate(f, 1):
                for rx in SECRET_RES:
                    m = rx.search(line)
                    if m:
                        hits.append(f"{path}:{n}: {m.group(0)[:10]}…")
    return hits


def host_facts(job_dir):
    """OS, noyau, pilote et GPU relevés par run_helma.sub dans logs/system_info.txt (tous les nœuds du job)."""
    facts = {k: set() for k in FALLBACK_HOST}
    path = os.path.join(job_dir or "", "logs", "system_info.txt")
    if job_dir and os.path.exists(path):
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"=== \S+ \| (.+?) \| noyau (\S+)", line)
                if m:
                    facts["os"].add(m.group(1))
                    facts["kernel"].add(m.group(2))
                m = re.match(r"\d+, NVIDIA[^,]*, [0-9A-Fa-f:.]+, (\d+) MiB, ([\d.]+) W, (\d+) MHz, (\S+)", line)
                if m:
                    facts["mem"].add(m.group(1))
                    facts["plimit"].add(str(int(float(m.group(2)))))
                    facts["maxclk"].add(m.group(3))
                    facts["driver"].add(m.group(4))
    return {k: sorted(v) if v else FALLBACK_HOST[k] for k, v in facts.items()}


def container_env(cont):
    """Variables d'environnement du conteneur (versions NGC) + OS_PRETTY_NAME, et version d'Apptainer."""
    tmp = tempfile.mkdtemp()
    try:
        out = subprocess.run(["apptainer", "exec", "--home", tmp, cont, "bash", "-c",
                              'env; echo "OS_PRETTY_NAME=$(. /etc/os-release; echo "${PRETTY_NAME}")"'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                             timeout=600).stdout
        version = subprocess.run(["apptainer", "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f"AVERTISSEMENT : versions du conteneur illisibles ({e})")
        return {}, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line), version.strip()


def scc_system_json(system, nodes, gpn, facts, env):
    """Les 11 champs du modèle SCC de benchmarks.md, mêmes types.
    host_processor_count et accelerator_count sont des totaux pour le système (tous les nœuds)."""
    ngc = env.get("NVIDIA_PYTORCH_VERSION", "26.04")
    return {
        "system_name": system,
        "host_processor": "AMD EPYC 9554",
        "host_processor_count": 2 * nodes,
        "accelerator_model": "NVIDIA H100 SXM5 94GB",
        "accelerator_count": nodes * gpn,
        "framework": f"PyTorch {env.get('PYTORCH_VERSION', '2.12.0a0+0291f96')} "
                     f"(NGC PyTorch {ngc}, NVIDIA NeMo / Megatron-Bridge)",
        "operating_system": f"{', '.join(facts['os'])} (host), {env.get('OS_PRETTY_NAME', 'Ubuntu 24.04')} (container)",
        "interconnect": "NVIDIA InfiniBand NDR200 200Gb/s (4 HCAs per node, one per GPU); NVLink 4 inside a node",
        "interconnect_topology": "Fat-tree, all nodes of the job on one leaf switch (Slurm --switches=1); "
                                 "rail-optimised GPU-local HCAs; 4-GPU NVLink full mesh (NV6) per node, no NVSwitch",
        "cuda_version": ".".join(env.get("CUDA_VERSION", "13.2").split(".")[:2]),
        "nccl_version": re.sub(r"^v|-\d+$", "", env.get("NCCL_VERSION", "2.30.4")),
    }


def mlperf_system_json(team, system, nodes, gpn, facts, env, apptainer_version):
    """Description complète MLPerf v6.0 (champs de mlperf_logging.system_desc_checker), pour les vérificateurs officiels.
    system_name (texte libre pour MLPerf) prend le nom du fichier, comme le demande le modèle SCC."""
    ngc = env.get("NVIDIA_PYTORCH_VERSION", "26.04")
    stack = {k: env.get(v, "") for k, v in [
        ("cuda_version", "CUDA_VERSION"), ("cuda_driver_version", "CUDA_DRIVER_VERSION"),
        ("nccl_version", "NCCL_VERSION"), ("cublas_version", "CUBLAS_VERSION"), ("cudnn_version", "CUDNN_VERSION"),
        ("trt_version", "TRT_VERSION"), ("dali_version", "DALI_VERSION"), ("mofed_version", "MOFED_VERSION"),
        ("openmpi_version", "OPENMPI_VERSION"), ("pytorch_version", "PYTORCH_VERSION"),
        ("transformer_engine_version", "TRANSFORMER_ENGINE_VERSION")]}
    stack.update({"container_runtime": apptainer_version,
                  "kernel_version": ", ".join(f"Linux {k}" for k in facts["kernel"]),
                  "nvidia_kernel_driver": ", ".join(facts["driver"])})
    return {
        "submitter": team, "division": "closed", "status": "Available on-premise",
        "system_name": system, "number_of_nodes": str(nodes),
        "host_processors_per_node": "2", "host_processor_model_name": "AMD EPYC 9554",
        "host_processor_core_count": "64", "host_processor_vcpu_count": "64",
        "host_processor_frequency": "3.1 GHz (boost 3.75 GHz)", "host_processor_caches": "L3 256 MB per socket",
        "host_processor_interconnect": "AMD Infinity Fabric", "host_memory_capacity": "768 GB",
        "host_storage_type": "NVMe SSD", "host_storage_capacity": "15 TB node-local",
        "host_networking": "4x InfiniBand NDR200 (one HCA per GPU)",
        "host_networking_topology": "Rail-optimised: each GPU's HCA sits on the GPU's PCIe switch",
        "host_memory_configuration": "DDR5-4800, 8 channels per socket",
        "accelerators_per_node": str(gpn), "accelerator_model_name": "NVIDIA H100 SXM5 94GB",
        "accelerator_host_interconnect": "PCIe 5.0 x16",
        "accelerator_frequency": f"{'/'.join(facts['maxclk'])} MHz (max SM clock)",
        "accelerator_on-chip_memories": "", "accelerator_memory_configuration": "HBM2e",
        "accelerator_memory_capacity": f"{'/'.join(facts['mem'])} MiB",
        "accelerator_interconnect": "NVLink 4th gen, 6 links per GPU pair (NV6), no NVSwitch",
        "accelerator_interconnect_topology": "4-GPU full mesh", "cooling": "",
        "hw_notes": f"GPU power limit {'/'.join(facts['plimit'])} W",
        "framework": f"NVIDIA NeMo / Megatron-Bridge {ngc} (NGC PyTorch {ngc})", "framework_name": f"ngc{ngc}_nemo",
        "other_software_stack": stack,
        "operating_system": f"{', '.join(facts['os'])} (host), {env.get('OS_PRETTY_NAME', 'Ubuntu 24.04')} (container)",
        "sw_notes": "SCC26 submission. Apptainer instead of Pyxis/Enroot; page-cache eviction with posix_fadvise "
                    "(no sudo); NVIDIA NeMo code unchanged, Helma launcher in "
                    "benchmarks/llama2_70b_lora/source-or-patches/helma.",
    }


# Réglages effectifs d'un job (logs/env_run1.txt, environnement du 1er essai) : ceux qui doivent être identiques entre
# les jobs d'une soumission (erreur s'ils diffèrent) et ceux qui décrivent le matériel (avertissement, documentés).
RECIPE_KEYS = ["DGXNNODES", "DGXNGPU", "TP", "PP", "CP", "SP", "MINIBS", "MBS", "LR", "MAX_STEPS", "FP8", "RANK_LAYOUT",
               "MLPERF_SYSTEM", "MLPERF_SUBMISSION_ORG"]
HW_KEYS = ["GPU_POWER_LIMIT", "GPU_CLOCK_LOCK", "GPU_RESET"]


def job_settings(job_dir):
    vals = {}
    path = os.path.join(job_dir, "logs", "env_run1.txt")
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            for line in f:
                k, sep, val = line.rstrip("\n").partition("=")
                if sep and k in RECIPE_KEYS + HW_KEYS:
                    vals[k] = val
    ov = os.path.join(job_dir, "config", "overrides.txt")
    if os.path.exists(ov):
        with open(ov) as f:
            for line in f:
                k, sep, val = line.strip().partition("=")
                if sep and k in HW_KEYS and k not in vals:
                    vals[k] = val
    return vals


def gpu_settings_text(st):
    txt = f"{st['GPU_POWER_LIMIT']} W" if st.get("GPU_POWER_LIMIT") else "default power limit"
    txt += f", SM clock locked at {st['GPU_CLOCK_LOCK']} MHz" if st.get("GPU_CLOCK_LOCK") else ", SM clock free"
    return txt


def trials_table_en(trials, settings):
    rows = ["| # | Slurm job | seed | status | samples to converge | final eval loss | time-to-train (min) | GPU settings |",
            "|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(trials):
        st = "converged" if t["converged"] else str(t["status"])
        loss = f"{t['final_loss']:.4f}" if t["final_loss"] is not None else "-"
        ttt = f"{t['ttt_min']:.2f}" if t["ttt_min"] is not None else "-"
        rows.append(f"| {i} | {t['job_id']} | {t['seed']} | {st} | {t['samples'] or '-'} | {loss} | {ttt} | "
                    f"{gpu_settings_text(settings.get(t['job'], {}))} |")
    return "\n".join(rows)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_dir", help="ngpu8, ngpu16, ou chemin du dossier du système")
    ap.add_argument("--runs", help="dossier des jobs (défaut : <système>/runs)")
    ap.add_argument("--run-id", help="préfixe des dossiers de jobs (défaut : RUN_ID de la config)")
    ap.add_argument("--out", help="dossier où écrire <TEAM_NAME>/ (défaut : dossier du système)")
    ap.add_argument("--team", help="nom d'équipe (défaut : TEAM_NAME de scc.env)")
    ap.add_argument("--org", help="submission_org attendu dans les logs (défaut : nom d'équipe)")
    ap.add_argument("--skip-launcher-check", action="store_true", help="ne compare pas les empreintes du lanceur")
    ap.add_argument("--jobs", nargs="+", help="dossiers de jobs à empaqueter, dans cet ordre (remplace --runs/--run-id)")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="soumission partielle : nombre d'essais visé (défaut 10, la règle) ; README et bilan l'indiquent")
    ap.add_argument("--finished-only", action="store_true",
                    help="ignore les essais encore en cours (run_start sans run_stop) au lieu de les compter en échec")
    args = ap.parse_args(argv)

    env_scc = sc.scc_env()
    sys_dir = os.path.abspath(args.system_dir if os.path.isdir(args.system_dir)
                              else os.path.join(sc.SCC_DIR, args.system_dir))
    sys_name = os.path.basename(sys_dir)
    cfg = os.path.join(sys_dir, f"config_scc_{sys_name}.sh")
    v = config_vars(cfg, ["RUN_ID", "MLPERF_SYSTEM", "DGXNNODES", "DGXNGPU", "MINIBS", "CONT", "SCC_SYSTEM_PROFILE",
                          "TP", "PP", "CP", "LR", "MAX_STEPS", "RANK_LAYOUT"])
    # Profil matériel d'un autre cluster que Helma (JSON, chemin relatif au dossier du système) : il remplace les
    # descriptions écrites en dur ci-dessous. Clés : "mlperf" et "scc" (champs remplacés), "readme_system" (texte),
    # "a_verifier" (points à contrôler sur place, recopiés en avertissements dans package_report.md).
    profile = {}
    if v["SCC_SYSTEM_PROFILE"]:
        profile_path = os.path.join(sys_dir, v["SCC_SYSTEM_PROFILE"])
        with open(profile_path) as f:
            profile = json.load(f)
    team = args.team or env_scc["TEAM_NAME"]
    org = args.org or team
    runs_dir = os.path.abspath(args.runs or os.path.join(sys_dir, "runs"))
    run_id = args.run_id or v["RUN_ID"]
    system = v["MLPERF_SYSTEM"]
    nodes, gpn = int(v["DGXNNODES"]), int(v["DGXNGPU"])
    scc_bench = env_scc["SCC_BENCHMARK"]
    team_dir = os.path.join(os.path.abspath(args.out or sys_dir), team)

    # ---------------------------------------------------------------- sélection des essais
    errors, warnings = [], [f"profil système, à vérifier : {note}" for note in profile.get("a_verifier", [])]
    jobs = [os.path.abspath(j) for j in args.jobs] if args.jobs else find_jobs(runs_dir, run_id)
    trials, excluded = [], []
    for job in jobs:
        logs = trial_logs(job)
        if not logs:
            excluded.append(f"job {os.path.basename(job)} : aucun essai lancé (voir slurm.out)")
        for log in logs:
            idx = int(re.search(r"mllog_run(\d+)\.log$", log).group(1))
            t = sc.analyze_log(log)
            t.update(job=job, job_id=os.path.basename(job), trial=idx, compliance=compliance(job, idx))
            if not t["started"]:
                excluded.append(f"job {t['job_id']} essai {idx} : run_start jamais atteint")
                continue
            if args.finished_only and t["ttt_min"] is None:
                excluded.append(f"job {t['job_id']} essai {idx} : encore en cours, non retenu (--finished-only)")
                continue
            trials.append(t)
    n_target = args.max_trials or sc.N_RUNS
    selected, extras = trials[:n_target], trials[n_target:]
    used_jobs = [j for j in jobs if any(t["job"] == j for t in selected)]
    if not selected:
        print(f"Aucun essai démarré dans {runs_dir}/{run_id}_*/ : rien à empaqueter.")
        return 1

    # ---------------------------------------------------------------- vérifications
    seeds = [t["seed"] for t in selected]
    dups = sorted({s for s in seeds if seeds.count(s) > 1})
    if dups:
        errors.append(f"graines en double {dups}")
    bad = [f"job {t['job_id']} essai {t['trial']} ({t['status']}, conformité {t['compliance'] or 'absente'})"
           for t in selected if not t["converged"] or t["compliance"] != "SUCCESS"]
    if len(bad) == 1:
        warnings.append(f"1 essai non convergé ou non conforme, toléré (compte comme le plus lent) : {bad[0]}")
    elif bad:
        errors.append(f"{len(bad)} essais non convergés ou non conformes (1 seul toléré) : {', '.join(bad)}")
    orgs = sorted({str(t["org"]) for t in selected})
    if orgs != [org]:
        errors.append(f"submission_org des logs {orgs} différent de {org!r}")
    gbs = sorted({t["gbs"] for t in selected}, key=str)
    if len(gbs) != 1:
        errors.append(f"GBS différents selon les essais : {gbs}")
    settings = {job: job_settings(job) for job in used_jobs}
    for key in RECIPE_KEYS:
        values = {settings[j].get(key, "") for j in used_jobs}
        if len(values) > 1:
            errors.append(f"réglage {key} différent selon les jobs : "
                          + ", ".join(f"{os.path.basename(j)}={settings[j].get(key, '')!r}" for j in used_jobs))
    for key in HW_KEYS:
        values = {settings[j].get(key, "") for j in used_jobs}
        if len(values) > 1:
            warnings.append(f"réglage matériel {key} différent selon les jobs (documenté dans le README) : "
                            + ", ".join(f"{os.path.basename(j)}={settings[j].get(key, '') or '-'}" for j in used_jobs))
    digests = {}
    for job in used_jobs:
        digests.setdefault(config_digest(job), []).append(os.path.basename(job))
    if len(digests) > 1:
        warnings.append(f"fichiers de config différents selon les jobs (copiés dans configs/job_<id>/) : {list(digests.values())}")
    codes = {}
    for job in used_jobs:
        codes.setdefault(tree_digest(os.path.join(job, "code")), []).append(os.path.basename(job))
    if len(codes) > 1:
        errors.append(f"code nemo/ différent selon les jobs : {list(codes.values())}")
    if not args.skip_launcher_check:
        for job in used_jobs:
            e, w = launcher_check(runs_dir, job)
            errors += e
            warnings += w
    big = [t["log"] for t in selected if os.path.getsize(t["log"]) > sc.MAX_FILE_BYTES]
    if big:
        errors.append(f"logs de plus de 50 Mo : {big}")
    hits = scan_secrets([t["log"] for t in selected]
                        + [f for job in used_jobs for f in glob.glob(os.path.join(job, "config", "*"))])
    if hits:
        print("ARRÊT : secret probable dans les fichiers à publier, rien n'a été écrit :\n  " + "\n  ".join(hits))
        return 2

    # ---------------------------------------------------------------- écriture du paquet
    env, apptainer_version = container_env(v["CONT"])
    facts = host_facts(used_jobs[0])
    # Ancienne arborescence (benchmarks.md : Llama2_70B_LoRA/, result_log_<i>.txt) : package_checker ignorerait ce
    # dossier, on le retire pour qu'il ne reste qu'un seul benchmark.
    for legacy in (os.path.join(team_dir, "benchmarks"), os.path.join(team_dir, "results", system)):
        for name in os.listdir(legacy) if os.path.isdir(legacy) else []:
            if name != scc_bench and name.lower() == scc_bench.lower():
                shutil.rmtree(os.path.join(legacy, name))
    res_dir = os.path.join(team_dir, "results", system, scc_bench)
    os.makedirs(res_dir, exist_ok=True)
    for old in os.listdir(res_dir):
        if sc.RESULT_RE.match(old) or old == sc.SCALING_FILE:
            os.remove(os.path.join(res_dir, old))
    for i, t in enumerate(selected):
        shutil.copyfile(t["log"], os.path.join(res_dir, f"result_{i}.txt"))

    os.makedirs(os.path.join(team_dir, "systems"), exist_ok=True)
    desc = mlperf_system_json(team, system, nodes, gpn, facts, env, apptainer_version)
    desc.update(profile.get("mlperf", {}))
    scc_fields = scc_system_json(system, nodes, gpn, facts, env)
    scc_fields.update(profile.get("scc", {}))
    desc.update({k: v for k, v in scc_fields.items() if k not in desc})
    locks = {settings[j].get("GPU_CLOCK_LOCK", "") for j in used_jobs}
    if len(locks) > 1:
        locked = [f"job {os.path.basename(j)}" for j in used_jobs if settings[j].get("GPU_CLOCK_LOCK")]
        desc["hw_notes"] = (desc.get("hw_notes", "") + "; SM clock locked at "
                            + "/".join(sorted(l for l in locks if l)) + f" MHz for {', '.join(locked)} only").lstrip("; ")
    with open(os.path.join(team_dir, "systems", f"{system}.json"), "w") as f:
        json.dump(desc, f, indent=4)
        f.write("\n")

    bench = os.path.join(team_dir, "benchmarks", scc_bench)
    src = os.path.join(bench, "source-or-patches")
    shutil.rmtree(bench, ignore_errors=True)
    code_src = os.path.join(used_jobs[0], "code")
    warnings += [f"code : {s} non copié" for s in copy_files(code_src, os.path.join(src, "nemo"),
                                                              walk_files(code_src, SKIP_REL))]
    helma_dir = os.path.join(env_scc["ZEPH_DIR"], "helma_runs")
    copy_files(helma_dir, os.path.join(src, "helma"),
               ["run_helma.sub"] + [os.path.relpath(p, helma_dir) for p in sorted(
                   glob.glob(os.path.join(helma_dir, "tools", "*.sh")) + glob.glob(os.path.join(helma_dir, "tools", "*.py")))])
    for job in used_jobs:
        jc = os.path.join(job, "config")
        copy_files(jc, os.path.join(src, "configs", f"job_{os.path.basename(job)}"),
                   sorted(os.listdir(jc)) if os.path.isdir(jc) else [])
    job_cfg_dir = sys_dir
    copy_files(sys_dir, os.path.join(src, "configs"), [os.path.basename(cfg)])
    copy_files(os.path.join(helma_dir, "configs"), os.path.join(src, "configs"),
               [os.path.basename(p) for p in glob.glob(os.path.join(helma_dir, "configs", "config_*_common.sh"))])
    copy_files(os.path.join(env_scc["ZEPH_DIR"], "nemo"), os.path.join(src, "configs"), ["config_common.sh"])
    copy_files(sc.SCC_DIR, os.path.join(src, "configs"), ["scc.env"])
    copy_files(sc.SCC_DIR, os.path.join(src, "scripts"),
               ["submit.sh"] + [os.path.relpath(p, sc.SCC_DIR) for p in sorted(
                   glob.glob(os.path.join(sc.SCC_DIR, "tools", "*.py")) + glob.glob(os.path.join(sc.SCC_DIR, "tools", "*.sh")))])
    # Fichiers attendus à la racine du benchmark (slides p. 23) : ceux que le job a réellement exécutés.
    for name, src_dir, src_name in [("run_and_time.sh", code_src, "run_and_time.sh"),
                                    ("config.sh", job_cfg_dir, os.path.basename(cfg))]:
        if os.path.exists(os.path.join(src_dir, src_name)):
            shutil.copyfile(os.path.join(src_dir, src_name), os.path.join(bench, name))
            shutil.copymode(os.path.join(src_dir, src_name), os.path.join(bench, name))
        else:
            errors.append(f"benchmarks/{scc_bench}/{name} : {src_name} absent de {src_dir}")

    score, failed = sc.olympic_mean(selected)
    results = trials_table_en(selected, settings)
    if len(selected) == sc.N_RUNS and score is not None:
        results += f"\n\nOlympic mean time-to-train (8 middle runs of 10): **{score:.2f} min**."
    elif n_target < sc.N_RUNS and len(selected) == n_target and score is not None:
        results += (f"\n\n**Partial submission: {n_target} trials** (MLPerf asks for {sc.N_RUNS}; the SCC rules accept "
                    f"partial submissions). Olympic mean time-to-train of these {n_target} trials (fastest and slowest "
                    f"removed, mean of the {n_target - 2} middle runs): **{score:.2f} min**.")
    elif score is not None:
        results += (f"\n\nProvisional: {len(selected)} of {n_target} trials so far; olympic mean of these "
                    f"(fastest and slowest removed): {score:.2f} min.")
    else:
        results += f"\n\nProvisional: {len(selected)} of {n_target} trials so far."
    if any("encore en cours" in e for e in excluded):
        results += " Trials still running when this package was built are not listed."
    gbs_val = int(gbs[0]) if len(gbs) == 1 and str(gbs[0]).isdigit() else int(v["MINIBS"]) * nodes * gpn // 4
    tp, pp, cpp = int(v["TP"] or 1), int(v["PP"] or 1), int(v["CP"] or 1)
    dp = nodes * gpn // (tp * pp * cpp)
    eval_steps = -(-384 // gbs_val)
    skip = int(0.125 * gbs_val + 2)
    rcp_bs = [8, 16, 32, 128]
    if gbs_val in rcp_bs:
        rcp_text = ""
    else:
        lo = max([b for b in rcp_bs if b < gbs_val], default=None)
        hi = min([b for b in rcp_bs if b > gbs_val], default=None)
        rcp_text = (f"\nGBS {gbs_val} has no reference convergence point of its own: `rcp_checker` interpolates the "
                    f"reference between GBS {lo} and GBS {hi} (as for the GBS 5, 9 and 10 submissions of MLPerf "
                    "Training v6.0).\n")
    layout_text = ""
    if v["RANK_LAYOUT"]:
        layout_text = (f"\nRank placement: `{v['RANK_LAYOUT']}` (host:ranks, `RANK_LAYOUT` in `run_helma.sub`, "
                       "`srun --distribution=arbitrary`); on a node running fewer ranks than it has GPUs, only its first "
                       "GPUs are visible to CUDA and NCCL.\n")
    settings_table = "\n".join(["| Slurm job | trials in this package | GPU settings | GPU reset at start |",
                                "|---|---|---|---|"] + [
        f"| {os.path.basename(j)} | {sum(1 for t in selected if t['job'] == j)} | {gpu_settings_text(settings[j])} | "
        f"{'yes' if settings[j].get('GPU_RESET') == '1' else 'no'} |" for j in used_jobs])
    with open(os.path.join(bench, "README.md"), "w") as f:
        system_text = profile.get("readme_system") or (
            f"{nodes} nodes x 4 NVIDIA H100 SXM5 94GB = **{nodes * gpn} GPUs** on the NHR@FAU Helma cluster "
            "(partition `h100`):\n2x AMD EPYC 9554 per node, 768 GB DDR5, 4-GPU NVLink full mesh per node (no NVSwitch), "
            "one InfiniBand NDR200 HCA per GPU,\nall nodes of the job on a single leaf switch.")
        f.write(README_TMPL.format(system=system, team=team, sysdir=sys_name, system_text=system_text,
                                   layout_text=layout_text, gbs=gbs_val, minibs=v["MINIBS"], dp=dp, tp=tp,
                                   lr=v["LR"], max_steps=v["MAX_STEPS"], eval_every=eval_steps * gbs_val,
                                   eval_steps=eval_steps, skip_evals=skip,
                                   first_eval=(skip + 1) * eval_steps * gbs_val, rcp_text=rcp_text,
                                   settings_table=settings_table, config=os.path.basename(cfg), results=results))

    # ---------------------------------------------------------------- bilan
    lines = [f"# Paquet SCC26 {system} — équipe {team} ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Paquet : `{team_dir}`  ", f"Jobs : {', '.join(os.path.basename(j) for j in used_jobs)} "
             f"({len(selected)}/{n_target} essais visés)", "", sc.trials_table(selected), ""]
    if score is not None:
        label = "définitive" if len(selected) == n_target else f"provisoire, {len(selected)} essais"
        lines.append(f"Moyenne olympique {label} (sans le plus rapide ni le plus lent) : **{score:.2f} min**"
                     f" ; essais en échec : {failed}.")
    elif len(selected) >= 3:
        lines.append(f"Moyenne olympique impossible : {failed} essais en échec (1 seul toléré).")
    lines.append("")
    lines += [f"- écarté : {e}" for e in excluded]
    lines += [f"- non utilisé (au-delà des {n_target} premiers) : job {t['job_id']} essai {t['trial']}" for t in extras]
    lines += [f"- AVERTISSEMENT : {w}" for w in warnings]
    lines += [f"- ERREUR : {e}" for e in errors]
    complete = len(selected) == n_target
    status = "complet et sans erreur" if complete and not errors else ("à corriger" if errors else "incomplet")
    lines += ["", f"Paquet {status}. Vérification : `tools/scc_verify.sh {sys_name}`"]
    report = "\n".join(lines) + "\n"
    print(report)
    if not args.out:
        with open(os.path.join(sys_dir, "package_report.md"), "w") as f:
            f.write(report)
    return 0 if complete and not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
