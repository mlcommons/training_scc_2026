#!/usr/bin/env python3
"""Re-organize ten llama2_70b_lora run logs into the MLPerf Training submission tree.

usage: package_submission.py [--out DIR] [--team Supernova] [--nodes h0,h1,..]
                             [--no-checks] RUNS...

RUNS is either one directory or a list of paths. Each entry is a run dir holding
node0.log, or a log file (result_N.txt, *.log, *.out). A single directory is
searched for run dirs (*/node0.log) first, then for log files. Entries are taken in
sorted order and become result_0..result_9, copied verbatim (the checkers read the
:::MLLOG lines and ignore the rest).

Output (default <repo>/submission/<team>, gitignored; replaced on every call):
  systems/student_system.json            hardware and software detected on the nodes
  benchmarks/llama2_70b_lora/{README.md,run_and_time.sh,config.sh,source-or-patches/}
  results/student_system/llama2_70b_lora/result_N.txt
With --cert N it also writes the SCC certification files of the set to <out>/cert/ (outside the team tree and
the zip): cert-N-mlperf.input (config.sh and the environment of every run), cert-N-mlperf.rslts (per-run summary,
checker output, every result log) and cert-N-mlperf.run (the --run-script files and the launch line of every run).
There is no tstamps file for MLPerf: the runs are timed from the MLLOG run_start / run_stop events of the logs.
scaling.json is written by the MLPerf package checker. After packaging the script runs
rcp_checker, package_checker and verify_for_v6.0_training.sh on the tree (--no-checks
skips them); their output goes to <out>/checks/. A check fails when its log shows errors,
not only on a non-zero exit code. When all pass, <out>/<team>.zip is written.
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNE = os.path.join(REPO, "h200-tuning")
TOKEN = ":::MLLOG "
BENCH = "llama2_70b_lora"
SUBMITTER = "Nanyang Technological University"
ORG_PLACEHOLDER = "SUBMISSION_ORG_PLACEHOLDER"
# GBS -> (warmup steps, decay/max steps) of the recipe, for the packaged README
RECIPES = {8: (0, 800), 12: (40, 533)}

# (source relative to repo, destination relative to source-or-patches)
SOURCES = [
    ("patches", "patches"),
    ("h200-tuning/run_rcp_ref8.sh", "h200-tuning/run_rcp_ref8.sh"),
    ("h200-tuning/run_rcp_het.sh", "h200-tuning/run_rcp_het.sh"),
    ("h200-tuning/lgc.sh", "h200-tuning/lgc.sh"),
    ("h200-tuning/rank_exec_cp.sh", "h200-tuning/rank_exec_cp.sh"),
    ("h200-tuning/config_H200NVL4.sh", "h200-tuning/config_H200NVL4.sh"),
    ("h200-tuning/submission/config.sh", "h200-tuning/submission/config.sh"),
    ("h200-tuning/package_submission.py", "h200-tuning/package_submission.py"),
    ("h200-tuning/setup/setup_sc26.sh", "h200-tuning/setup/setup_sc26.sh"),
    ("h200-tuning/setup/setup_venv_train.sh", "h200-tuning/setup/setup_venv_train.sh"),
    ("h200-tuning/setup/venv_toolchain.sh", "h200-tuning/setup/venv_toolchain.sh"),
    ("h200-tuning/setup/venv-train-overrides.txt", "h200-tuning/setup/venv-train-overrides.txt"),
    ("h200-tuning/setup/stage_checkpoint_ram.py", "h200-tuning/setup/stage_checkpoint_ram.py"),
    ("h200-tuning/optimization.md", "h200-tuning/optimization.md"),
    ("transformerengine/build_te.sh", "transformerengine/build_te.sh"),
    (f"{BENCH}/scripts", "reference/scripts"),
    (f"{BENCH}/configs", "reference/configs"),
    (f"{BENCH}/requirements.txt", "reference/requirements.txt"),
    (f"{BENCH}/README.md", "reference/README.md"),
]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "build")

# Run on every node (locally, or over ssh): one key=value per line.
NODE_PROBE = r"""
lscpu_f() { lscpu | awk -F: -v k="$1" '$1==k {sub(/^[ \t]+/, "", $2); print $2; exit}'; }
echo "host=$(hostname -s)"
echo "cpu_model=$(lscpu_f 'Model name' | sed 's/ [0-9]*-Core Processor//; s/ Processor$//')"
echo "cpu_sockets=$(lscpu_f 'Socket(s)')"
echo "cpu_cores_per_socket=$(lscpu_f 'Core(s) per socket')"
echo "cpu_vcpus=$(nproc --all)"
echo "cpu_max_mhz=$(lscpu_f 'CPU max MHz')"
echo "cpu_l1d=$(lscpu_f 'L1d cache')"
echo "cpu_l2=$(lscpu_f 'L2 cache')"
echo "cpu_l3=$(lscpu_f 'L3 cache')"
echo "mem_kib=$(awk '/^MemTotal/ {print $2}' /proc/meminfo)"
dm=$(sudo -n dmidecode -t memory 2>/dev/null | awk '/^\tSize: [0-9]/ {n++; s=$2" "$3} /^\tSpeed: [0-9]/ && !sp {sp=$2" "$3} /^\tType: DDR/ {t=$2} END {if (n) print n"x "s" "t"-"sp}')
echo "mem_config=$dm"
lsblk -dn -b -o NAME,SIZE,TRAN 2>/dev/null | awk '$1 ~ /^nvme/ {print "nvme_bytes=" $2}'
lsblk -dn -b -o NAME,SIZE,ROTA,TRAN 2>/dev/null | awk '$1 !~ /^(nvme|loop|ram|zram)/ && $2 > 2e11 {print "disk_bytes=" $2 "," $3}'
for d in /sys/class/infiniband/*; do
  [ -d "$d" ] || continue
  st=$(cut -c1 $d/ports/1/state 2>/dev/null); ll=$(cat $d/ports/1/link_layer 2>/dev/null)
  echo "ib=$(basename $d),$(cat $d/hca_type 2>/dev/null),$ll,$st,$(awk '{print $1}' $d/ports/1/rate 2>/dev/null)"
done
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,driver_version,power.limit --format=csv,noheader,nounits 2>/dev/null | sed 's/^/gpu=/'
nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk '$1 ~ /^GPU[0-9]+$/ {print "topo=" $0}'
for d in /sys/bus/pci/devices/*; do
  v=$(cat $d/vendor 2>/dev/null); [ -n "$v" ] || continue
  sp=$(cat $d/current_link_speed 2>/dev/null); wd=$(cat $d/current_link_width 2>/dev/null)
  case $v in
    0x10de) echo "pcie=gpu,$sp,$wd" ;;
    0x15b3) echo "pcie=nic,$sp,$wd" ;;
  esac
done
echo "os=$(. /etc/os-release && echo $PRETTY_NAME)"
echo "kernel=$(uname -sr)"
echo "kmod=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)"
echo "gcc=$(${CC:-gcc} --version 2>/dev/null | head -1)"
echo "nvcc=$(${CUDA_HOME:+$CUDA_HOME/bin/}nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\), V\(.*\)/\2/p')"
"""

HCA_NAMES = {"MT4129": "ConnectX-7", "MT4123": "ConnectX-6", "MT4125": "ConnectX-6 Dx",
             "MT4119": "ConnectX-5", "MT4127": "ConnectX-6 Lx", "MT41692": "BlueField-3"}
IB_NAMES = {100: "EDR", 200: "HDR", 400: "NDR", 800: "XDR"}
HBM = {"H100": "HBM3", "H200": "HBM3e", "B200": "HBM3e", "B300": "HBM3e", "GB200": "HBM3e",
       "A100": "HBM2e", "H800": "HBM3"}


def sh(cmd, **kw):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, **kw).stdout.strip()
    except Exception:
        return ""


def collect_logs(paths):
    if len(paths) == 1 and os.path.isdir(paths[0]):
        d = paths[0]
        entries = sorted(glob.glob(os.path.join(d, "*", "node0.log")))
        if not entries:
            entries = sorted(p for pat in ("result_*.txt", "*.log", "*.out")
                             for p in glob.glob(os.path.join(d, pat)))
        if not entries:
            sys.exit(f"no run logs found under {d}")
        return entries
    out = []
    for p in paths:
        if os.path.isdir(p):
            p = os.path.join(p, "node0.log")
        if not os.path.isfile(p):
            sys.exit(f"not a run log: {p}")
        out.append(p)
    return sorted(out)


def mllog_lines(path):
    lines = []
    with open(path, errors="replace") as f:
        for line in f:
            i = line.find(TOKEN)
            if i >= 0:
                lines.append(line[i:].rstrip("\n"))
    return lines


def run_status(lines):
    for line in reversed(lines):
        if '"key": "run_stop"' in line:
            return json.loads(line[len(TOKEN):]).get("metadata", {}).get("status")
    return None


def log_org(lines):
    for line in lines:
        if '"key": "submission_org"' in line:
            return json.loads(line[len(TOKEN):]).get("value")
    return None


def log_value(lines, key):
    """Scalar value of an MLLOG key, or None when the run never logged it."""
    for line in lines:
        if f'"key": "{key}"' in line:
            return json.loads(line[len(TOKEN):]).get("value")
    return None


def batch_size(parsed, env):
    """Global batch size of the runs.

    The MLLOG record decides, because that is what the checkers read. env.txt is
    only a fallback: a run tree that did not come from run_rcp.sh has no env.txt
    at all, and the old default then silently packaged a GBS 12 set as GBS 8 --
    wrong warmup, wrong decay steps and a config.sh that contradicts the logs.
    """
    vals = {log_value(lines, "global_batch_size") for lines in parsed}
    vals.discard(None)
    if len(vals) > 1:
        sys.exit(f"runs disagree on global_batch_size: {sorted(vals)}")
    if vals:
        gbs = int(vals.pop())
        env_gbs = env.get("VAL_GBS")
        if env_gbs and int(env_gbs) != gbs:
            print(f"warning: global_batch_size is {gbs} in the logs but VAL_GBS={env_gbs} in env.txt",
                  file=sys.stderr)
        return gbs
    return int(env.get("VAL_GBS", 8))


def run_env(logpath):
    """env.txt and nodes.txt written by run_rcp.sh next to node0.log."""
    d = os.path.dirname(logpath)
    env = {}
    if os.path.isfile(os.path.join(d, "env.txt")):
        for line in open(os.path.join(d, "env.txt")):
            k, _, v = line.strip().partition("=")
            env[k] = v
    nodes = None
    if os.path.isfile(os.path.join(d, "nodes.txt")):
        nodes = open(os.path.join(d, "nodes.txt")).read().split()
    return env, nodes


def run_node_count(logpath, env, run_nodes):
    """How many nodes the runs actually used.

    env.txt is absent for a tree that came from another cluster's scheduler, so
    DGXNNODES falls back to the count on job.log's "nodes:" header. Without this
    the count came from whatever host ran the packaging, and a 3-node GBS 12 set
    was described as a single node.
    """
    if env.get("DGXNNODES"):
        return int(env["DGXNNODES"])
    job = os.path.join(os.path.dirname(logpath), "job.log")
    if os.path.isfile(job):
        for line in open(job, errors="replace"):
            if line.startswith("nodes:"):
                return len(line.split(":", 1)[1].split())
    return len(run_nodes) if run_nodes else None


def probe(host):
    local = host.split(".")[0] == socket.gethostname().split(".")[0]
    cmd = ["bash", "-s"] if local else ["ssh", "-o", "BatchMode=yes", host, "bash -s"]
    try:
        r = subprocess.run(cmd, input=NODE_PROBE, capture_output=True, text=True, timeout=120)
    except Exception as e:
        sys.exit(f"cannot probe {host}: {e}")
    if r.returncode != 0 and not r.stdout:
        sys.exit(f"cannot probe {host}: {r.stderr.strip()}")
    info = collections.defaultdict(list)
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        info[k].append(v)
    return info


def merged(values):
    """Distinct values across nodes in first-seen order (one value when homogeneous)."""
    seen = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
    return " / ".join(seen)


def tb(n):
    return f"{n / 1e12:.2f}TB".replace(".00TB", "TB")


def storage(info):
    nv = collections.Counter(int(b) for b in info["nvme_bytes"])
    if nv:
        kind = "NVMe SSD"
        parts = [f"{c}x {tb(b)}" for b, c in sorted(nv.items(), reverse=True)]
    else:
        disks = [(int(b.split(",")[0]), b.split(",")[1]) for b in info["disk_bytes"]]
        kind = "HDD" if disks and all(r == "1" for _, r in disks) else "SSD"
        cnt = collections.Counter(b for b, _ in disks)
        parts = [f"{c}x {tb(b)}" for b, c in sorted(cnt.items(), reverse=True)]
    return kind, ", ".join(parts)


def networking(info):
    ports = collections.Counter()
    for rec in info["ib"]:
        name, hca, ll, st, rate = (rec.split(",") + [""] * 5)[:5]
        if ll != "InfiniBand" or st != "4" or not rate.replace(".", "").isdigit():
            continue
        rate = int(float(rate))
        ports[(HCA_NAMES.get(hca, hca or "HCA"), IB_NAMES.get(rate, ""), rate)] += 1
    return ", ".join(f"{c}x {m} IB {n + ' ' if n else ''}{r} Gb/s" for (m, n, r), c in ports.items())


def nvlink_summary(info):
    """(links-per-peer string, island description) from the nvidia-smi topo matrix rows."""
    rows = [r.split("\t") for r in info["topo"]]
    # nvidia-smi topo -m emits a header row ("<tab>GPU0<tab>GPU1...") whose first
    # field awk prints as "GPU0"; counting it adds a phantom node to the island
    # graph and reports 2x 4-GPU islands as five bogus ones.
    rows = [r for r in rows if len(r) > 1 and
            re.fullmatch(r"(X|NV\d+|SYS|PIX|PXB|PHB|NODE|SOC)", r[1].strip())]
    n = len(rows)
    if not n:
        return "", ""
    adj = {i: set() for i in range(n)}
    links = set()
    for i, r in enumerate(rows):
        for j in range(n):
            cell = r[1 + j].strip() if 1 + j < len(r) else ""
            m = re.fullmatch(r"NV(\d+)", cell)
            if m:
                adj[i].add(j)
                links.add(int(m.group(1)))
    comp, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        stack, c = [i], set()
        while stack:
            x = stack.pop()
            if x in c:
                continue
            c.add(x)
            stack.extend(adj[x])
        seen |= c
        comp.append(len(c))
    if not links:
        return "", "no NVLink"
    sizes = collections.Counter(comp)
    desc = ", ".join(f"{c}x {s}-GPU NVLink island" + ("s" if c > 1 else "") for s, c in sorted(sizes.items(), reverse=True))
    if len(comp) > 1:
        desc += "; islands connected over PCIe/SYS"
    return f"NVLink, {'/'.join(str(l) for l in sorted(links))} links per GPU pair", desc


def processor_interconnect(infos):
    """Socket count plus the socket-to-socket fabric, named only when the vendor is known."""
    socks = merged(i["cpu_sockets"][0] for i in infos if i.get("cpu_sockets"))
    model = " ".join(i["cpu_model"][0] for i in infos if i.get("cpu_model"))
    fabric = ""
    if "AMD" in model or "EPYC" in model:
        fabric = ", Infinity Fabric"
    elif "Intel" in model or "Xeon" in model:
        fabric = ", UPI"
    return f"{socks} sockets{fabric}" if socks else ""


def pcie_summary(info, kind):
    """PCIe link of every GPU (kind="gpu") or NIC ("nic"), from sysfs.

    current_link_speed is the negotiated rate ("32.0 GT/s"), not the part's
    maximum, which is what a submission should report.
    """
    seen = collections.Counter()
    for rec in info["pcie"]:
        k, sp, wd = (rec.split(",") + ["", ""])[:3]
        sp = " ".join(sp.split()[:2])  # sysfs appends the "PCIe" bus name
        if k != kind or not sp:
            continue
        gen = {"2.5 GT/s": "PCIe 1", "5.0 GT/s": "PCIe 2", "8.0 GT/s": "PCIe 3",
               "16.0 GT/s": "PCIe 4", "32.0 GT/s": "PCIe 5", "64.0 GT/s": "PCIe 6"}.get(sp, sp)
        seen[(gen, wd)] += 1
    if not seen:
        return ""
    return ", ".join(f"{c}x {g} x{w}" for (g, w), c in sorted(seen.items(), reverse=True))


def dist_version(py, *names):
    code = ("import importlib.metadata as m,sys\n"
            "for n in sys.argv[1:]:\n"
            "    try: print(m.version(n)); break\n"
            "    except Exception: pass\n")
    return sh(f"{py} -I -c '{code}' {' '.join(names)}")


def software(py):
    sw = {}
    if not os.path.exists(py):
        print(f"warning: {py} does not exist; the software versions are left empty "
              f"(pass --venv to read them)", file=sys.stderr)
        return sw
    sw["torch"] = dist_version(py, "torch")
    sw["cuda"] = sh(f"{py} -I -c 'import torch; print(torch.version.cuda or \"\")'")
    sw["python"] = sh(f"{py} -I -c 'import platform; print(platform.python_version())'")
    for key, names in [("nccl", ("nvidia-nccl-cu13", "nvidia-nccl-cu12", "nvidia-nccl-cu11")),
                       ("cudnn", ("nvidia-cudnn-cu13", "nvidia-cudnn-cu12")),
                       ("cublas", ("nvidia-cublas", "nvidia-cublas-cu12")),
                       ("nvshmem", ("nvidia-nvshmem-cu13", "nvidia-nvshmem-cu12")),
                       ("transformer_engine", ("transformer_engine",)),
                       ("flash_attention", ("flash_attn_3", "flash-attn-3")),
                       ("deepgemm", ("deep_gemm", "deep-gemm")),
                       ("megatron_core", ("megatron-core",)),
                       ("megatron_bridge", ("megatron-bridge",)),
                       ("mlperf_logging", ("mlperf_logging", "mlperf-logging"))]:
        sw[key] = dist_version(py, *names)
    return sw


def system_json(team, nodes, gpus_per_node, infos, sw, py):
    cat = lambda k: [v for i in infos for v in i[k]]
    gpus = [g.split(", ") for g in cat("gpu")]
    gname = merged(g[0] for g in gpus)
    first_name = gpus[0][0] if gpus else ""
    mem_gib = merged(f"{round(int(m) / 2**20)} GiB" for m in cat("mem_kib"))
    skind, scap = storage(infos[0])
    link, islands = nvlink_summary(infos[0])
    mhz = cat("cpu_max_mhz")
    gcc = merged(g.replace("gcc (", "GCC (") if g.startswith("gcc") else g for g in cat("gcc"))
    nvcc = merged(cat("nvcc"))
    hbm = next((v for k, v in HBM.items() if k in first_name), "")
    limits = merged(f"{float(g[4]):.0f} W" for g in gpus if len(g) > 4)
    other = {
        "cuda_version": sw.get("cuda", ""),
        "cuda_driver_version": merged(g[3] for g in gpus if len(g) > 3),
        "nccl_version": sw.get("nccl", ""),
        "cublas_version": sw.get("cublas", ""),
        "cudnn_version": sw.get("cudnn", ""),
        "nvshmem_version": sw.get("nvshmem", ""),
        "transformer_engine_version": sw.get("transformer_engine", ""),
        "flash_attention_version": sw.get("flash_attention", ""),
        "deepgemm_version": sw.get("deepgemm", ""),
        "megatron_core_version": sw.get("megatron_core", ""),
        "python_version": sw.get("python", ""),
        "compiler": gcc,
        "nvcc_version": nvcc,
        "kernel_version": merged(cat("kernel")),
        "nvidia_kernel_driver": merged(cat("kmod")),
    }
    other = {k: v for k, v in other.items() if v}
    total = nodes * gpus_per_node
    return {
        "submitter": SUBMITTER,
        "division": "closed",
        "status": "Available on-premise",
        "system_name": f"{team} ({nodes}x node, {total}x {gname})",
        "number_of_nodes": str(nodes),
        "host_processors_per_node": merged(cat("cpu_sockets")),
        "host_processor_model_name": merged(cat("cpu_model")),
        "host_processor_core_count": merged(cat("cpu_cores_per_socket")),
        "host_processor_vcpu_count": merged(cat("cpu_vcpus")),
        "host_processor_frequency": merged(f"{float(m) / 1000:.1f} GHz (max)" for m in mhz if m),
        "host_processor_caches": merged(f"L1d {a}, L2 {b}, L3 {c}" for a, b, c in
                                        zip(cat("cpu_l1d"), cat("cpu_l2"), cat("cpu_l3"))),
        "host_processor_interconnect": processor_interconnect(infos),
        "host_memory_capacity": mem_gib,
        "host_storage_type": skind,
        "host_storage_capacity": scap,
        "host_networking": networking(infos[0]),
        "host_networking_topology": pcie_summary(infos[0], "nic"),
        "host_memory_configuration": merged(cat("mem_config")),
        "accelerators_per_node": str(gpus_per_node),
        "accelerator_model_name": gname,
        "accelerator_host_interconnect": pcie_summary(infos[0], "gpu"),
        "accelerator_frequency": merged(f"{g[2]} MHz" for g in gpus if len(g) > 2),
        "accelerator_on-chip_memories": "",
        "accelerator_memory_configuration": hbm,
        "accelerator_memory_capacity": merged(f"{g[1]} MiB" for g in gpus if len(g) > 1),
        "accelerator_interconnect": link,
        "accelerator_interconnect_topology": islands,
        "cooling": "Air-cooled",
        "hw_notes": (f"GPU power limit {limits}. " if limits else "") + f"Nodes: {', '.join(merged(i['host']) for i in infos)}.",
        "framework": f"PyTorch {sw.get('torch', '')} + Megatron-Bridge (Megatron-Core {sw.get('megatron_core', '')})",
        "framework_name": f"torch{sw.get('torch', '').split('+')[0]}_megatron-bridge",
        "other_software_stack": other,
        "operating_system": merged(cat("os")),
        "sw_notes": ("Patched TransformerEngine, FlashAttention-3 and DeepGEMM; FP8 delayed scaling, "
                     "full-iteration CUDA graphs, TP communication overlap. Patch set in "
                     "benchmarks/llama2_70b_lora/source-or-patches/patches."),
    }


def strip_comments(text):
    """Drop the comment lines of a shell script, keeping the shebang.

    The working copies carry the experiment record: job ids, arm names, dates
    and the decisions behind each setting. That record belongs in the journal,
    not in a submission. Comment lines only: no heredoc or multi-line string
    in these scripts holds a leading #, so this cannot change what runs.
    """
    out = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.startswith("#!"):
            out.append(line)
            continue
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    # collapse the runs of blank lines the removed comments leave behind
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.lstrip("\n").rstrip() + "\n"


def copy_sources(dst):
    os.makedirs(dst)
    for src, rel in SOURCES:
        s, d = os.path.join(REPO, src), os.path.join(dst, rel)
        if not os.path.exists(s):
            print(f"warning: missing source {src}", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=SKIP)
        else:
            shutil.copy2(s, d)
    for root, _, files in os.walk(dst):
        for name in files:
            if not name.endswith(".sh"):
                continue
            p = os.path.join(root, name)
            with open(p) as f:
                text = f.read()
            with open(p, "w") as f:
                f.write(strip_comments(text))
    commit = sh(f"git -c safe.directory=* -C {REPO} rev-parse HEAD")
    dirty = sh(f"git -c safe.directory=* -C {REPO} status --porcelain --untracked-files=no")
    with open(os.path.join(dst, "COMMIT.txt"), "w") as f:
        f.write(commit + ("  (working tree had uncommitted changes)\n" if dirty else "\n"))


def flat_config(gbs):
    """config.sh of the package: the selected recipe as one self-contained file."""
    base = open(os.path.join(TUNE, "submission", "config.sh")).read()
    if gbs != 12:
        return strip_comments(base)
    over = [l for l in open(os.path.join(TUNE, "submission", "config_ref12.sh")).read().splitlines()
            if not l.startswith("#!") and not l.startswith("source ") and not l.lstrip().startswith("#")]
    # config_ref12.sh's LAYOUT default must precede config.sh's layout switch.
    # config.sh already sets the same default, so insert only when it does not.
    pre = [l for l in over if l.startswith("LAYOUT=")]
    over = [l for l in over if not l.startswith("LAYOUT=")]
    if pre and not any(l.startswith("LAYOUT=") for l in base.splitlines()):
        head, _, rest = base.partition("\n")
        base = head + "\n" + "\n".join(pre) + "\n" + rest if head.startswith("#!") else "\n".join(pre) + "\n" + base
    return strip_comments(base + "\n" + "\n".join(over) + "\n")


LOGGING_URL = "https://github.com/mlcommons/logging.git"
LOGGING_TAG = "6.0.0-rc6"
CHECK_DIR = os.path.join(REPO, "cache", f"logging-{LOGGING_TAG}")


def ensure_checkers():
    """Pin the MLPerf checkers: mlcommons/logging at LOGGING_TAG pip-installed into its own
    venv (the training venv stays untouched), plus the same tag checked out for scripts/.
    Returns (python of the checker venv, verify_for_v6.0_training.sh)."""
    src, venv = os.path.join(CHECK_DIR, "src"), os.path.join(CHECK_DIR, "venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.isdir(os.path.join(src, ".git")):
        os.makedirs(CHECK_DIR, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", LOGGING_TAG, LOGGING_URL, src], check=True)
    uv = shutil.which("uv") or os.path.join(os.path.expanduser("~" + os.environ.get("SUDO_USER", "")), ".local/bin/uv")
    if not os.path.exists(py):
        if os.path.exists(uv):
            subprocess.run([uv, "venv", "-q", "--python", sys.executable, venv], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    have = sh(f"{py} -I -c 'import mlperf_logging.package_checker' 2>&1 && echo ok")
    if not have.endswith("ok"):
        spec = f"git+{LOGGING_URL}@{LOGGING_TAG}"
        cmd = [uv, "pip", "install", "-q", "--python", py, spec] if os.path.exists(uv) else [py, "-m", "pip", "install", "-q", spec]
        subprocess.run(cmd, check=True)
    return py, os.path.join(src, "scripts", "verify_for_v6.0_training.sh")


def run_checks(root, out, py, script):
    """The three MLPerf checks on the packaged tree (logging pinned to LOGGING_TAG); returns {name: returncode}."""
    chk = os.path.join(out, "checks")
    os.makedirs(chk, exist_ok=True)
    resdir = os.path.join(root, "results", "student_system", BENCH)
    env = dict(os.environ, PATH=os.path.dirname(py) + os.pathsep + os.environ["PATH"])
    steps = [
        ("rcp_checker", [py, "-m", "mlperf_logging.rcp_checker", resdir, "--rcp_usage", "training",
                         "--rcp_version", "6.0.0", "--verbose"]),
        ("package_checker", [py, "-m", "mlperf_logging.package_checker", root, "training", "6.0.0",
                             "--werror", "--log_output", os.path.join(out, "package_student_team.log")]),
    ]
    steps.append(("verify_for_v6.0", ["bash", script, root]))
    rcs = {}
    # the package checker appends to its --log_output file: start it empty so an earlier run's errors are not read
    # back as this run's (a passing set was flagged by a stale 'cache_clear' warning from an older package)
    try:
        os.remove(os.path.join(out, "package_student_team.log"))
    except FileNotFoundError:
        pass
    for name, cmd in steps:
        log = os.path.join(chk, f"{name}.log")
        with open(log, "w") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=out)
        text = open(log, errors="replace").read()
        if name == "package_checker":
            text += open(os.path.join(out, "package_student_team.log"), errors="replace").read()
        problems = verdict(name, r.returncode, text)
        rcs[name] = problems
        print(f"[{name}] rc={r.returncode} {'OK' if not problems else 'FAILED'}  log: {log}")
        for pr in problems:
            print(f"    {pr}")
    return rcs


def verdict(name, rc, text):
    """The checkers exit 0 even when they print errors, so the logs decide."""
    bad = []
    if rc != 0:
        bad.append(f"exit code {rc}")
    markers = ("PACKAGE CHECKER FOUND ERRORS", "CRITICAL ERROR", "Compliance check failed", "RCP test FAILED",
               "Traceback")
    for m in markers:
        if m in text:
            bad.append(f"'{m}' x{text.count(m)}")
    errs = sorted({l.strip() for l in text.splitlines() if l.startswith("ERROR")})
    bad += errs[:5]
    if name == "rcp_checker" and "RCP test PASSED" not in text:
        bad.append("no 'RCP test PASSED'")
    if name == "verify_for_v6.0" and "ALL CHECKS PASSED" not in text:
        bad.append("repo_checker did not print ALL CHECKS PASSED")
    warns = sorted({l.strip() for l in text.splitlines() if l.startswith("WARNING")})
    bad += [w for w in warns if "Required AT_LEAST_ONE" in w]
    return bad


def mllog_event(lines, key):
    """The first MLLOG event (dict) with this key, or None."""
    for line in lines:
        if f'"key": "{key}"' in line:
            return json.loads(line[len(TOKEN):])
    return None


def write_cert(out, n, logs, parsed, run_scripts, bench, checks_dir):
    """The SCC certification files of this result set, in <out>/cert."""
    d = os.path.join(out, "cert")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    pfx = os.path.join(d, f"cert-{n}-mlperf")
    starts = [(mllog_event(l, "run_start") or {}).get("time_ms") for l in parsed]
    stops = [mllog_event(l, "run_stop") for l in parsed]
    with open(pfx + ".input", "w") as f:
        f.write("# config.sh of the submission\n")
        f.write(open(os.path.join(bench, "config.sh")).read())
        for i, p in enumerate(logs):
            env = os.path.join(os.path.dirname(p), "env.txt")
            f.write(f"\n# result_{i}: environment of {os.path.dirname(p)}\n")
            f.write(open(env).read() if os.path.exists(env) else "(no env.txt)\n")
    with open(pfx + ".rslts", "w") as f:
        f.write("result    seed   samples  status   time_to_train_s  log\n")
        for i, (p, lines) in enumerate(zip(logs, parsed)):
            stop = stops[i] or {}
            samples = (stop.get("metadata") or {}).get("samples_count")
            ttt = (stop["time_ms"] - starts[i]) / 1000.0 if stop and starts[i] is not None else float("nan")
            f.write(f"result_{i}  {log_value(lines, 'seed')}  {samples}  {run_status(lines)}  {ttt:.1f}  {p}\n")
        if os.path.isdir(checks_dir):
            for name in sorted(os.listdir(checks_dir)):
                if name.endswith(".log"):
                    f.write(f"\n===== {name} =====\n" + open(os.path.join(checks_dir, name), errors="replace").read())
        for i, p in enumerate(logs):
            f.write(f"\n===== result_{i}.txt ({p}) =====\n" + open(p, errors="replace").read())
    with open(pfx + ".run", "w") as f:
        for r in run_scripts:
            f.write(f"===== {r} =====\n" + open(r).read() + "\n")
        f.write("===== launch line of every run =====\n")
        for i, p in enumerate(logs):
            launch = os.path.dirname(p) + ".launch.log"
            head = [l for l in (open(launch, errors="replace").read().splitlines() if os.path.exists(launch) else [])
                    if l.startswith("=== run_rcp:")]
            f.write(f"result_{i}: {head[-1] if head else '(no launch log)'}\n")
    print(f"cert files: {d}")
    return d


def make_zip(out, team):
    import zipfile
    path = os.path.join(out, f"{team}.zip")
    base = os.path.join(out, team)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(base):
            for fn in sorted(files):
                full = os.path.join(d, fn)
                z.write(full, os.path.relpath(full, out))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(REPO, "submission"))
    ap.add_argument("--team", default="Supernova")
    ap.add_argument("--nodes", help="comma-separated hosts to probe (default: nodes.txt of the runs, else this host)")
    ap.add_argument("--venv", default=os.path.join(REPO, "venv-train"))
    ap.add_argument("--allow-n", type=int, default=10, help="required number of runs (default 10)")
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--cert", type=int, help="also write the SCC cert-N-mlperf.* files of this set to <out>/cert")
    ap.add_argument("--run-script", action="append", default=[], help="cert: script(s) that ran the set (repeatable)")
    a = ap.parse_args()

    logs = collect_logs(a.runs)
    if len(logs) != a.allow_n:
        sys.exit(f"expected {a.allow_n} run logs, found {len(logs)}: {logs}")
    parsed = []
    for p in logs:
        lines = mllog_lines(p)
        if not lines or run_status(lines) is None:
            sys.exit(f"{p}: no MLLOG run_stop (incomplete run)")
        parsed.append(lines)
    for p, lines in zip(logs, parsed):
        if run_status(lines) != "success":
            print(f"warning: {p}: run_stop status={run_status(lines)}", file=sys.stderr)
        org = log_org(lines)
        if org != SUBMITTER:
            print(f"warning: {p}: submission_org is {org!r}, not {SUBMITTER!r}", file=sys.stderr)

    env, run_nodes = run_env(logs[0])
    # The probe hosts and the node count are separate questions: the runs may
    # have come from a cluster this packaging host cannot reach, and then only
    # the count is recoverable.
    hosts = a.nodes.split(",") if a.nodes else (run_nodes or [socket.gethostname()])
    nodes = run_node_count(logs[0], env, run_nodes) or len(hosts)
    gpn = int(env.get("DGXNGPU", 0))
    if run_nodes and not a.nodes and socket.gethostname().split(".")[0] not in \
            {h.split(".")[0] for h in run_nodes}:
        print(f"warning: this host is not one of the run nodes ({', '.join(run_nodes)}); "
              f"the hardware in student_system.json is probed from {hosts[0]}. "
              f"Pass --nodes with a live node of the same system.", file=sys.stderr)
    infos = [probe(h) for h in hosts]
    if not gpn:
        gpn = len(infos[0]["gpu"])
    gbs = batch_size(parsed, env)

    py = os.path.join(a.venv, "bin", "python")
    sw = software(py)

    root = os.path.join(a.out, a.team)
    if os.path.exists(a.out):
        shutil.rmtree(os.path.join(a.out, "checks"), ignore_errors=True)
    if os.path.exists(root):
        shutil.rmtree(root)
    bench = os.path.join(root, "benchmarks", BENCH)
    resdir = os.path.join(root, "results", "student_system", BENCH)
    os.makedirs(os.path.join(root, "systems"))
    os.makedirs(resdir)

    for i, p in enumerate(logs):
        shutil.copyfile(p, os.path.join(resdir, f"result_{i}.txt"))  # verbatim, debug prints included

    with open(os.path.join(root, "systems", "student_system.json"), "w") as f:
        json.dump(system_json(a.team, nodes, gpn, infos, sw, py), f, indent=4)
        f.write("\n")

    copy_sources(os.path.join(bench, "source-or-patches"))
    sub = os.path.join(TUNE, "submission")
    steps = log_value(parsed[0], "opt_learning_rate_training_steps")
    factor = log_value(parsed[0], "opt_learning_rate_warmup_factor")
    if steps is None or factor is None:
        warm, steps = RECIPES.get(gbs, RECIPES[8])
    else:
        steps = int(steps)
        warm = int(round(float(factor) * steps))
    readme = (open(os.path.join(sub, "README.md")).read()
              .replace("@GBS@", str(gbs)).replace("@WARMUP@", str(warm)).replace("@STEPS@", str(steps)))
    open(os.path.join(bench, "README.md"), "w").write(readme)
    shutil.copy2(os.path.join(sub, "run_and_time.sh"), os.path.join(bench, "run_and_time.sh"))
    open(os.path.join(bench, "config.sh"), "w").write(flat_config(gbs))

    print(f"submission tree: {root}  (GBS {gbs}, {nodes} node(s) x {gpn} GPUs)")
    for i, p in enumerate(logs):
        print(f"  result_{i}.txt <- {p}")
    checks_dir = os.path.join(a.out, "checks")
    if a.no_checks:
        if a.cert is not None:
            write_cert(a.out, a.cert, logs, parsed, a.run_script, bench, checks_dir)
        return
    chk_py, script = ensure_checkers()
    rcs = run_checks(root, a.out, chk_py, script)
    if a.cert is not None:
        write_cert(a.out, a.cert, logs, parsed, a.run_script, bench, checks_dir)
    if any(rcs.values()):
        print("checks failed: no zip written")
        sys.exit(1)
    print(f"all checks passed; zip: {make_zip(a.out, a.team)}")


def hand_back(path):
    """Under sudo, give the files created here back to the invoking user."""
    user = os.environ.get("SUDO_USER")
    if os.geteuid() == 0 and user:
        subprocess.run(["chown", "-R", f"{user}:", path, CHECK_DIR], capture_output=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        out = next((sys.argv[i + 1] for i, v in enumerate(sys.argv[:-1]) if v == "--out"), os.path.join(REPO, "submission"))
        if os.path.isdir(out):
            hand_back(out)
