#!/usr/bin/env python3
"""Pré-test lancé exactement comme l'entraînement (srun -> apptainer -> slurm2pytorch -> bind_helma.sh).

Vérifie, avant de consommer une heure de GPU :
  * le rendez-vous torch.distributed + NCCL entre tous les rangs (même chemin que train.py) ;
  * l'affinité CPU de chaque rang (doit être le domaine NUMA de son GPU) ;
  * les débits collectifs : global, intra-nœud (NVLink), inter-nœud par rail (InfiniBand), all-to-all.
Lignes lisibles par tools/summarize_results.py :
  AFFINITY rank=.. host=.. local_rank=.. pci=.. cpus=..
  NCCL_RESULT op=.. group=.. ranks=.. size_MB=.. time_ms=.. algbw_GBps=.. busbw_GBps=..
"""
import datetime
import os
import socket

import torch
import torch.distributed as dist

SIZES_MB = [64, 256, 1024, 2048]
WARMUP, ITERS = 3, 10


def cpu_ranges(cpus):
    cpus = sorted(cpus)
    out, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        if c is not None:
            start = prev = c
    return ",".join(out)


def timed(fn, group):
    for _ in range(WARMUP):
        fn()
    dist.barrier(group=group)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench(op, group, ranks, size_mb, label, rank0):
    n = len(ranks)
    numel = size_mb * 2**20 // 2
    dev = torch.cuda.current_device()
    if op == "all_reduce":
        buf = torch.ones(numel, dtype=torch.bfloat16, device=dev)
        fn, factor = (lambda: dist.all_reduce(buf, group=group)), 2 * (n - 1) / n
    elif op == "all_gather":
        out = torch.empty(numel, dtype=torch.bfloat16, device=dev)
        inp = torch.ones(numel // n, dtype=torch.bfloat16, device=dev)
        fn, factor = (lambda: dist.all_gather_into_tensor(out, inp, group=group)), (n - 1) / n
    elif op == "reduce_scatter":
        inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)
        out = torch.empty(numel // n, dtype=torch.bfloat16, device=dev)
        fn, factor = (lambda: dist.reduce_scatter_tensor(out, inp, group=group)), (n - 1) / n
    elif op == "all_to_all":
        inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(inp)
        fn, factor = (lambda: dist.all_to_all_single(out, inp, group=group)), (n - 1) / n
    else:
        raise ValueError(op)
    ms = timed(fn, group)
    algbw = size_mb * 2**20 / (ms / 1e3) / 1e9
    if rank0:
        print(f"NCCL_RESULT op={op} group={label} ranks={n} size_MB={size_mb} time_ms={ms:.3f} "
              f"algbw_GBps={algbw:.2f} busbw_GBps={algbw * factor:.2f}", flush=True)


def main():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    t0 = datetime.datetime.now()
    timeout_s = int(os.environ.get("NCCL_CHECK_TIMEOUT_S", "120"))
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank),
                            timeout=datetime.timedelta(seconds=timeout_s))
    rank0 = rank == 0
    if rank0:
        print(f"INIT_OK world={world} seconds={(datetime.datetime.now() - t0).total_seconds():.1f}", flush=True)

    props = torch.cuda.get_device_properties(local_rank)
    pci = "{:04x}:{:02x}:{:02x}.0".format(getattr(props, "pci_domain_id", 0), getattr(props, "pci_bus_id", 0),
                                         getattr(props, "pci_device_id", 0))
    info = {"rank": rank, "host": socket.gethostname().split(".")[0], "local_rank": local_rank,
            "pci": pci, "cpus": cpu_ranges(os.sched_getaffinity(0)), "ncpu": len(os.sched_getaffinity(0)),
            "gpu": props.name, "mem_GiB": round(props.total_memory / 2**30, 1)}
    infos = [None] * world
    dist.all_gather_object(infos, info)
    if rank0:
        for i in infos:
            print("AFFINITY rank={rank} host={host} local_rank={local_rank} pci={pci} ncpu={ncpu} "
                  "cpus={cpus} gpu={gpu!r} mem_GiB={mem_GiB}".format(**i), flush=True)
        if os.environ.get("HELMA_BIND", "1") == "1" and any(i["ncpu"] > 32 for i in infos):
            print("WARNING affinité non appliquée (plus de 32 cœurs visibles par rang)", flush=True)

    hosts = sorted({i["host"] for i in infos})
    per_node = [[i["rank"] for i in infos if i["host"] == h] for h in hosts]
    per_rail = [list(r) for r in zip(*per_node)] if len(hosts) > 1 else []
    node_group, _ = dist.new_subgroups_by_enumeration(per_node)
    rail_group = dist.new_subgroups_by_enumeration(per_rail)[0] if per_rail else None

    for size in SIZES_MB:
        bench("all_reduce", None, list(range(world)), size, "global", rank0)
    for op in ("all_gather", "reduce_scatter", "all_to_all"):
        bench(op, None, list(range(world)), 1024, "global", rank0)
    bench("all_reduce", node_group, per_node[0], 1024, "intra_node", rank0)
    if rail_group is not None:
        bench("all_reduce", rail_group, per_rail[0], 1024, "inter_node_rail", rank0)
        bench("all_to_all", rail_group, per_rail[0], 256, "inter_node_rail", rank0)

    dist.barrier()
    if rank0:
        print("NCCL_CHECK_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
