"""Load the base checkpoint into FP8 weights without NeMo's restore path.

Imported by the patched train.py (nvidia/patch_nvidia_impl.py, patch 5).

Why. NeMo's restore goes through mcore's dist_checkpointing, whose
`_unwrap_pyt_sharded_tensor` calls `squeeze(0)` on each target parameter. On a
Float8Tensor that dispatches through `__torch_dispatch__`, which dequantises the
whole weight to bf16 on the GPU, and the results are collected into one dict
before anything is written back. At TP=1 that is 138 GB of bf16 on a 96 GB GPU:
measured in the NGC 25.04 container, OOM at 94.7 GiB inside
`selective_restore`. At TP=4, as in NVIDIA's and Krai's submissions, each GPU
holds a quarter and it fits.

What this does instead -- the approach already proven on this cluster by
scripts/convert_checkpoint.load_weights: rank 0 reads the checkpoint's own
stacked tensors into CPU memory one key at a time, through a gloo process group
containing only itself (so torch.distributed.checkpoint's planning never issues
a collective the other ranks are not in), and copies each layer's slice into the
model with `copy_`, which quantises into the Float8Tensor in place. The other
ranks then receive the quantised weights by broadcast: `_data` and `_scale_inv`
for FP8 tensors, the plain data otherwise. 64.7 GiB crosses the fabric instead
of 138.

Parameter names: after the LoRA transform, adapted modules are wrapped, so
`...linear_qkv.weight` in the checkpoint is `...linear_qkv.to_wrap.weight` in
the model. Adapter parameters are not in the checkpoint and are left alone.
"""
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _inner(module):
    m = module
    while not hasattr(m, "decoder") and hasattr(m, "module"):
        m = m.module
    return m


def _broadcast(params):
    for p in params:
        data = getattr(p, "_data", None)
        if data is not None:
            dist.broadcast(data, src=0)
            dist.broadcast(p._scale_inv, src=0)
        else:
            dist.broadcast(p.data, src=0)
    torch.cuda.synchronize()


def load(pl_module, weights_dir, num_layers):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    model = _inner(pl_module)
    by_ckpt_name = {}
    for name, p in model.named_parameters():
        if ".adapter." in name:
            continue
        by_ckpt_name[name.replace(".to_wrap.", ".")] = p
    params = list(by_ckpt_name.values())

    world = dist.get_world_size()
    rank = dist.get_rank()
    solo = dist.new_group(ranks=[0], backend="gloo")  # collective: every rank calls it
    # The other ranks wait for rank 0's read here. On NCCL that is a 10-minute
    # timeout, which job 168 hit while rank 0 was still reading from NFS.
    wait = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))

    if rank != 0:
        dist.barrier(group=wait)
        _broadcast(params)
        _drop_transposes(params)
        return

    t0 = time.time()
    _warm_page_cache(weights_dir)
    print(f"manual_ckpt_load: page cache warmed in {time.time() - t0:.0f}s", flush=True)
    meta = FileSystemReader(str(weights_dir)).read_metadata().state_dict_metadata
    keys = sorted(k for k in meta if "_extra_state" not in k)
    done = set()
    with torch.no_grad():
        for key in keys:
            shape = tuple(meta[key].size)
            buf = {key: torch.empty(shape, dtype=torch.bfloat16, device="cpu")}
            dcp.load(buf, checkpoint_id=str(weights_dir), process_group=solo)
            tensor = buf[key]
            name = key.removeprefix("module.")
            if name.startswith("decoder.layers."):
                suffix = name.removeprefix("decoder.layers.")
                for i in range(min(shape[0], num_layers)):
                    target = by_ckpt_name[f"decoder.layers.{i}.{suffix}"]
                    target.copy_(tensor[i].to(target.device, non_blocking=False))
                    done.add(f"decoder.layers.{i}.{suffix}")
            else:
                target = by_ckpt_name[name]
                target.copy_(tensor.to(target.device))
                done.add(name)
            del buf, tensor
            print(f"manual_ckpt_load: {key} {shape} {time.time() - t0:.0f}s", flush=True)

    missing = sorted(set(by_ckpt_name) - done)
    if missing:
        raise RuntimeError(f"manual_ckpt_load: {len(missing)} parameters not loaded, e.g. {missing[:3]}")
    _drop_transposes(params)
    read_s = time.time() - t0
    t1 = time.time()
    if world > 1:
        dist.barrier(group=wait)
        _broadcast(params)
    print(f"manual_ckpt_load: {len(params)} parameters, read {read_s:.1f}s, "
          f"broadcast to {world - 1} ranks {time.time() - t1:.1f}s", flush=True)


def _warm_page_cache(weights_dir):
    # DCP's reader pulls uncached data off NFS here at 100-200 MB/s (probe job 169:
    # 323 s, and >600 s under contention in job 168); a sequential 64 MB read
    # gets 5 GB/s, and DCP then reads from the page cache at >3 GB/s.
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    def read(path):
        with open(path, "rb", buffering=0) as f:
            buf = bytearray(64 << 20)
            while f.readinto(buf):
                pass

    files = sorted(Path(weights_dir).glob("*.distcp"))
    with ThreadPoolExecutor(len(files)) as ex:
        list(ex.map(read, files))


def _drop_transposes(params):
    for p in params:
        if getattr(p, "_transpose", None) is not None:
            p.update_usage(columnwise_usage=False)
