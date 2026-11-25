"""Profile one training step with torch.profiler (patch 9 in patch_nvidia_impl.py).

TORCH_PROFILE_STEP=N profiles training batch N (0-based, counted by Lightning's
batch_idx) on rank 0 and writes to /results:
  profile-step<N>.txt   kernel table sorted by CUDA time, plus per-category sums
  profile-step<N>.json  Chrome trace (open in https://ui.perfetto.dev)

Used to find where a 3.1 s step goes beyond the GEMMs: attention, the LoRA
adapters' small kernels, norms/activations, offload copies, and idle time.
"""
import os
import re

import torch

_CATS = [
    ("gemm", re.compile(r"gemm|cutlass|cublas|nvjet|Kernel_", re.I)),
    ("attention", re.compile(r"fmha|flash|attn|cudnn_generated|sdpa", re.I)),
    ("transpose/cast", re.compile(r"transpose|cast|quantize|amax", re.I)),
    ("norm", re.compile(r"norm|rmsnorm", re.I)),
    ("activation/rope", re.compile(r"swiglu|silu|gelu|rope|rotary|dropout", re.I)),
    ("copy", re.compile(r"memcpy|copy_", re.I)),
    ("nccl", re.compile(r"nccl", re.I)),
]


def make_callback(step):
    import lightning.pytorch as pl

    class _Profile(pl.Callback):
        def __init__(self):
            self.prof = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.prof == "nsys" and os.environ.get("PROFILE_PERIOD", "0") == "1":
                # Range from batch `step`'s start to the next batch's start, so
                # that the time between batches (optimizer, logging, data) shows.
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(f"torch_profile: nsys range closed at start of batch {batch_idx}", flush=True)
                self.prof = None
                return
            if batch_idx != step or torch.distributed.get_rank() != 0:
                return
            print(f"torch_profile: profiling batch {batch_idx}", flush=True)
            if os.environ.get("NSYS", "0") == "1":
                # Under `nsys profile -c cudaProfilerApi` (PROFILE_CMD in
                # launch.sh) this delimits the captured range.
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                self.prof = "nsys"
                return
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False)
            self.prof.__enter__()
            self.t0 = torch.cuda.Event(enable_timing=True)
            self.t0.record()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Not gated on batch_idx: in job 248 the start hook fired for
            # batch 6 and no end hook ever matched it.
            if self.prof is None:
                return
            print(f"torch_profile: batch end hook, batch_idx={batch_idx}", flush=True)
            if self.prof == "nsys":
                if os.environ.get("PROFILE_PERIOD", "0") == "1":
                    return
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(f"torch_profile: nsys range closed after batch {batch_idx}", flush=True)
                self.prof = None
                return
            t1 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)
            print("torch_profile: profiler stopped, summarising", flush=True)
            wall_ms = self.t0.elapsed_time(t1)
            ka = self.prof.key_averages()
            sums = {c: 0.0 for c, _ in _CATS}
            sums["other"] = 0.0
            total = 0.0
            for e in ka:
                dev = getattr(e, "device_time_total", None)
                if dev is None:
                    dev = getattr(e, "cuda_time_total", 0.0)
                if dev <= 0 or e.key.startswith(("ProfilerStep", "aten::", "autograd::", "torch")):
                    continue
                total += dev
                for c, rx in _CATS:
                    if rx.search(e.key):
                        sums[c] += dev
                        break
                else:
                    sums["other"] += dev
            out = f"/results/profile-step{step}.txt"
            with open(out, "w") as f:
                f.write(f"step {step}: wall {wall_ms / 1000:.3f} s, GPU kernel time (all streams, may overlap) "
                        f"{total / 1e6:.3f} s\n")
                for c, v in sorted(sums.items(), key=lambda kv: -kv[1]):
                    f.write(f"  {c:16s} {v / 1e6:7.3f} s  {100 * v / max(total, 1):5.1f}%\n")
                f.write("\n")
                f.write(ka.table(sort_by="cuda_time_total", row_limit=60))
            self.prof.export_chrome_trace(f"/results/profile-step{step}.json")
            print(f"torch_profile: wrote {out}", flush=True)
            self.prof = None

    return _Profile()
