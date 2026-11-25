"""Offload activations to Grace memory instead of recomputing them (TP=1 on GH200).

Imported by the patched train.py (nvidia/patch_nvidia_impl.py, patch 8) when
CPU_OFFLOADING=True.

Why. With full activation recompute (ACG=full, the configuration jobs 171/173
converged with) every decoder layer runs its forward twice; the second pass is
about a quarter of the 4.18 s step. Megatron-core can instead keep the saved
activations in host memory (`cpu_offloading`, backed by Transformer Engine's
`AsyncDoubleBufferGroupOffloadHandler`), which on GH200 means 480 GB of LPDDR
behind NVLink-C2C rather than a PCIe bus. Megatron refuses to combine the two,
so ACG=none with CPU_OFFLOADING=True; NVIDIA's config already reads both
CPU_OFFLOADING and CPU_OFFLOADING_NUM_LAYERS from the environment.

What TE's handler gets wrong for this model. The frozen FP8 base weights are
saved for backward like any other tensor, and TE's `tensor_push` records the
weight's `[data, transpose]` pair in the handler state. Both stay referenced
until that layer's backward. `fp8_transpose_patch` frees `_transpose` on the
parameter after each layer's forward, but the handler's copy of the reference
keeps the column-major copy alive: +1 byte per parameter for every layer between
forward and backward, the 64 GiB that does not fit at TP=1. (Under recompute the
saved tensors of the throwaway forward are dropped immediately, which is why
this never showed.)

This handler subclass records `[data, None]` for those weights, drops the
transpose reference at push time, and regenerates the transpose from the
row-major data when the tensor is popped for backward (`update_usage`, one
`fp8_transpose` kernel per weight, the same count as under recompute, where the
recomputed forward created it). Each weight is pushed twice per Linear
(`weightmat` and `weight` are the same object), so regenerated transposes are
shared through a cache keyed on the data tensor and cleared at every layer
commit in backward.

Nothing about weights reaches the CPU: `cpu_offloading_weights` stays False.
Offloaded are the tensors TE and NeMo's LoRA adapter tag with
`activation_offloading`. OFFLOAD_UNTAGGED_MIN_MB=N additionally offloads any
untagged saved tensor of at least N MiB that is not a parameter (dropout masks,
for instance); off by default so the first run measures what the tags cover.

The LoRA adapters are NeMo `ParallelLinearAdapter`s built on Megatron's own
`ColumnParallelLinear`/`RowParallelLinear`, which assert when run inside an
offload context (job 227). The reason is real: saved-tensor pack hooks receive
a detached copy of each saved Parameter ("Torch rips off the Parameter
wrapper"), and Megatron's fused wgrad accumulation needs `weight.main_grad` on
the object it gets back. `install_layer_hooks` therefore records the trainable
parameters by data pointer; `tensor_push` hands such a copy back as the
Parameter itself (never offloaded), and the adapters' linears get a copy of the
config with `_cpu_offloading_context` unset so the assertion does not fire. The
adapter input `x`, which NeMo tags, is still offloaded.

Job 234 (tagged tensors only) still grew by 1.56 GiB per layer: Megatron's own
ops save untagged tensors, the largest being the fused SwiGLU's input (fc1
output, 896 MiB in bf16 / 448 MiB with FP8_ACT). OFFLOAD_UNTAGGED_MIN_MB=8 is
therefore the working setting; nothing saved in a layer is mutated between
forward and backward, so offloading untagged tensors is safe.

Forward-time transposes. TE creates the column-major copy of every FP8 weight
in the forward (`_LayerNormLinear` asks for it unconditionally, `Linear` when
the input requires grad), although the forward GEMM only reads the row-major
data; only the backward's dgrad needs it, and this handler regenerates it
there anyway. With OFFLOAD_SKIP_FWD_TRANSPOSE=1 (default) `update_usage` on a
frozen weight ignores requests for column-wise data, which removes one
`fp8_transpose` per weight per forward: 65 GB read + 65 GB written per
microbatch, about 30 ms of a 3.2 s training step and of each 1.1 s eval
microbatch. Only valid together with this handler (recompute needs the
forward-created transpose).

OFFLOAD_TRACE=N prints allocated/peak memory at three points of the backward
pass for the first N steps.
"""
import copy
import os

import torch

_installed = False
_trainable = {}     # data_ptr -> Parameter, for the LoRA weights
_frozen = set()     # data_ptr of every frozen Float8Tensor parameter's _data


def install():
    """Replace TE's offload handler class. Must run before the model is built."""
    global _installed
    if _installed:
        return
    from transformer_engine.pytorch import cpu_offload as co
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    Base = co.AsyncDoubleBufferGroupOffloadHandler
    min_mb = float(os.environ.get("OFFLOAD_UNTAGGED_MIN_MB", "0"))
    trace_steps = int(os.environ.get("OFFLOAD_TRACE", "0"))

    class Handler(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.regen = set()          # tags whose transpose is regenerated at pop
            self.regen_cache = {}       # id(data) -> transpose, one backward layer at a time
            self.trace_left = trace_steps
            if min_mb > 0:
                tagged = self.tensor_need_offloading_checker
                threshold = int(min_mb * 2**20)

                def checker(t):
                    if tagged(t):
                        return True
                    if not isinstance(t, torch.Tensor) or isinstance(t, torch.nn.Parameter):
                        return False
                    if getattr(t, "_no_offload", False):
                        return False
                    return t.numel() * t.element_size() >= threshold

                self.tensor_need_offloading_checker = checker
            print(f"cpu_offload_patch: handler for {self.num_offload_group} of "
                  f"{self.num_layers} layers, untagged>={min_mb}MiB", flush=True)

        @staticmethod
        def _is_frozen_weight(tensor):
            if not isinstance(tensor, Float8Tensor) or tensor._data is None:
                return False
            if _frozen:
                return tensor._data.data_ptr() in _frozen
            # Before _register: FP8 activations carry activation_offloading on
            # their data tensors (set by TE before save); weights carry nothing.
            for t in (tensor._data, tensor._transpose):
                if t is not None and (hasattr(t, "activation_offloading")
                                      or hasattr(t, "weight_offloading")):
                    return False
            return True

        def tensor_push(self, tensor, **kwargs):
            if isinstance(tensor, torch.Tensor) and not isinstance(tensor, Float8Tensor):
                p = _trainable.get(tensor.data_ptr())
                if p is not None and p.shape == tensor.shape and p.dtype == tensor.dtype:
                    tag = (self.current_group, self.tensor_count_current_group)
                    self.tensor_count_current_group += 1
                    assert tag not in self.tensor_tag_to_state
                    self.tensor_tag_to_state[tag] = p    # the Parameter, with main_grad
                    return tag
            if not self._is_frozen_weight(tensor):
                return super().tensor_push(tensor, **kwargs)
            tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1
            assert tag not in self.tensor_tag_to_state
            tensor._data._no_offload = True
            # Drop this object's reference to the transpose. The parameter's own
            # reference goes when the next layer's forward pre-hook runs
            # (fp8_transpose_patch); the backward regenerates it.
            tensor._transpose = None
            tensor._transpose_invalid = True
            self.tensor_tag_to_state[tag] = [tensor._data, None]
            self.tensor_tag_to_buf[tag] = []
            self.fp8_tensor_object_map[tag] = tensor
            self.float8_transpose_cache_valid[tag] = True
            self.regen.add(tag)
            return tag

        def tensor_pop(self, tensor_tag, **kwargs):
            if tensor_tag not in self.regen:
                return super().tensor_pop(tensor_tag, **kwargs)
            self.regen.discard(tensor_tag)
            self.float8_transpose_cache_valid.pop(tensor_tag, None)
            out = super().tensor_pop(tensor_tag, **kwargs)  # restores [data, None]
            # Regenerate on a fresh object sharing the data, so that whatever
            # the pack hook handed us (the parameter itself, or a detached copy
            # of it) is left without a transpose and nothing accumulates across
            # steps. TE's backward only needs the returned object.
            fresh = out.detach()
            key = id(fresh._data)
            cached = self.regen_cache.get(key)
            if cached is None:
                fresh._create_transpose()                   # fp8_transpose from data
                self.regen_cache[key] = fresh._transpose
            else:
                fresh._transpose = cached
                fresh._transpose_invalid = False
            return fresh

        def on_group_commit_backward(self):
            super().on_group_commit_backward()
            self.regen_cache.clear()
            g = self.current_group
            if self.trace_left > 0 and g in (self.num_layers - 1, self.num_layers // 2, 0):
                print(f"offload_trace: backward at layer {g} "
                      f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
                if g == 0:
                    self.trace_left -= 1

    co.AsyncDoubleBufferGroupOffloadHandler = Handler
    _patch_fused_attn()
    _patch_megatron_linear()
    if os.environ.get("OFFLOAD_SKIP_FWD_TRANSPOSE", "1") == "1":
        _patch_update_usage(Float8Tensor)
    if os.environ.get("SKIP_GRAD_TRANSPOSE", "1") == "1":
        _patch_grad_transpose()
    _installed = True


def _patch_grad_transpose():
    """TE's Linear/LayerNormLinear backward always casts grad_output to FP8
    with a column-wise copy, which only the wgrad GEMM reads. Every TE linear
    here has a frozen weight (`ctx.requires_wgrad` False), so the fused
    cast+transpose (320 per step, 0.07 s in job 250's profile) becomes a plain
    cast. Independent of offloading, but installed from here for convenience;
    SKIP_GRAD_TRANSPOSE=0 disables."""
    from transformer_engine.pytorch.module import layernorm_linear, linear

    for module, cls in ((linear, linear._Linear), (layernorm_linear, layernorm_linear._LayerNormLinear)):
        _redefine(cls, "backward", [(
            "columnwise_usage = True\n",
            "columnwise_usage = bool(ctx.requires_wgrad)  # cpu_offload_patch\n",
        )], module)
    print("cpu_offload_patch: grad_output cast without transpose for frozen weights", flush=True)


def _patch_update_usage(Float8Tensor):
    """Frozen weights (registered in `_frozen` by `_register`) get no
    column-wise copy from `update_usage`; the backward pop creates it directly
    with `_create_transpose`."""
    orig = Float8Tensor.update_usage

    def update_usage(self, rowwise_usage=None, columnwise_usage=None):
        if columnwise_usage and _frozen and self._data is not None \
                and self._data.data_ptr() in _frozen:
            columnwise_usage = None
        return orig(self, rowwise_usage=rowwise_usage, columnwise_usage=columnwise_usage)

    Float8Tensor.update_usage = update_usage
    print("cpu_offload_patch: frozen weights skip the forward-time transpose", flush=True)


def _redefine(cls, name, replacements, module):
    """Re-define a (decorated) static method of `cls` from its source with
    text replacements applied, in `module`'s namespace."""
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new in replacements:
        if src.count(old) != 1:
            raise RuntimeError(f"{cls.__name__}.{name}: expected exactly one {old!r}")
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, module.__file__, "exec"), module.__dict__, ns)
    setattr(cls, name, ns[name])


def _patch_megatron_linear():
    """Megatron's LinearWithGradAccumulationAndAsyncCommunication (the LoRA
    adapters' linear_in/linear_out) accumulates the weight gradient into
    `weight.main_grad` on the tensor it gets back from `ctx.saved_tensors`.
    Under any saved-tensor hooks autograd hands the backward a fresh plain
    Tensor for a saved Parameter (measured: the pack hook sees the Parameter,
    the backward sees a Tensor without its attributes), which is what
    Megatron's "CPU Offloading cannot be enabled while using non-TE modules"
    assertion guards against and what job 235 died of. TE keeps the Parameter
    on the ctx for this reason (`ctx.weight_object`); do the same here."""
    from megatron.core.tensor_parallel import layers

    cls = layers.LinearWithGradAccumulationAndAsyncCommunication
    # Indentation is that of the dedented method source: the body sits at 4.
    _redefine(cls, "forward", [(
        "ctx.save_for_backward(input, weight)\n",
        "ctx.save_for_backward(input, weight)\n"
        "    ctx.weight_object = weight  # cpu_offload_patch\n",
    )], layers)
    _redefine(cls, "backward", [(
        "input, weight = ctx.saved_tensors\n",
        "input, weight = ctx.saved_tensors\n"
        "    if getattr(ctx, \"weight_object\", None) is not None:  # cpu_offload_patch\n"
        "        weight = ctx.weight_object\n",
    )], layers)
    print("cpu_offload_patch: Megatron linear keeps the weight Parameter for backward", flush=True)


def _patch_fused_attn():
    """TE 2.2's FusedAttnFunc.forward has never run with offloading and FP8
    attention together (job 232): with `ctx.fp8` it does
    `tensor_list = fp8_tensors` (a tuple) and then `.extend(...)`, and it tags
    the Float8Tensor objects rather than their data tensors, which is what the
    handler's checker looks at. Both lines are rewritten in the function's
    source and the method is re-defined in the module's namespace."""
    from transformer_engine.pytorch import attention as attn
    from transformer_engine.pytorch.cpu_offload import set_offloading_param

    attn.__dict__["_cpu_offload_patch_sop"] = set_offloading_param
    _redefine(attn.FusedAttnFunc, "forward", [
        ("tensor_list = fp8_tensors\n", "tensor_list = list(fp8_tensors)\n"),
        ("tensor.activation_offloading = True\n",
         "_cpu_offload_patch_sop(tensor, \"activation_offloading\", True)\n"),
    ], attn)
    print("cpu_offload_patch: FusedAttnFunc.forward patched for FP8 offloading", flush=True)


def install_layer_hooks(model):
    """Arrange for `_register` to run on the first decoder forward.

    NeMo applies the PEFT transform in `on_train_epoch_start`, after
    `on_fit_start`/`on_train_start` (job 230: only the output layer's linear and
    no adapter existed at fit start), so the adapters can only be found once
    the model actually runs.
    """
    m = model
    while not hasattr(m, "decoder") and hasattr(m, "module"):
        m = m.module
    decoder = m.decoder
    handle = {}

    def once(mod, args, kwargs):
        handle["h"].remove()
        _register(model)

    handle["h"] = decoder.register_forward_pre_hook(once, with_kwargs=True)


def _register(model):
    """After the LoRA transform: record trainable parameters and let Megatron's
    non-TE linears run inside the offload context."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    _trainable.clear()
    _frozen.clear()
    for p in model.parameters():
        if p.requires_grad:
            _trainable[p.data_ptr()] = p
        elif isinstance(p, Float8Tensor) and p._data is not None:
            _frozen.add(p._data.data_ptr())
    n = 0
    for mod in model.modules():
        if isinstance(mod, (ColumnParallelLinear, RowParallelLinear)) and \
                getattr(mod.config, "_cpu_offloading_context", None) is not None:
            mod.config = copy.copy(mod.config)
            mod.config._cpu_offloading_context = None
            n += 1
    print(f"cpu_offload_patch: {len(_trainable)} trainable parameters kept on GPU, "
          f"{len(_frozen)} frozen FP8 weights, "
          f"{n} Megatron linears allowed inside the offload context", flush=True)
