"""NTU add-ons, activated WITHOUT modifying the benchmark source.

Loaded at interpreter start by `zz_dg_dgrad.pth` in site-packages (Python's site module executes `import` lines of .pth
files). Each add-on is inert unless its env var is set; when it is, a meta-path hook waits for the target module to finish
importing, then applies the add-on (which re-binds functions/methods in that module):

  DG_DGRAD=1        on transformer_engine.pytorch        -> dg_dgrad.apply()        (DeepGEMM dgrad + FP8W_T_ONDEMAND)
  NTU_FUSED_LORA=1  on megatron.bridge.peft.lora_layers  -> ntu_lora_fused.apply()  (fused LoRA output path, no a2a)
  NTU_FP8_DY_AG=1   on transformer_engine.pytorch        -> ntu_te_comm.apply()     (MLP fc2 dy all-gather in FP8)
  NTU_DSWIGLU_DG=1  on transformer_engine.pytorch        -> ntu_dg_quant.apply_dswiglu() (dSwiGLU -> DeepGEMM operand)
  NTU_ROPE=1        on transformer_engine.pytorch        -> ntu_rope.apply()        (Triton QKV RoPE, bit-identical)
  NTU_DG_GRADQ=1    on transformer_engine.pytorch        -> ntu_dg_quant.apply_gradq() (qkv/proj/fc2 dy -> DeepGEMM operand)
  NTU_ATTN_BWD=1    on transformer_engine.pytorch        -> ntu_attn_bwd.apply()    (hand-written FP8 attention backward)

Env: DG_DGRAD_DIR = directory holding the add-on modules (default /work/benchmark/MLPerf/patches)."""
import os
import sys


def _addon_path():
    d = os.environ.get("DG_DGRAD_DIR", "/work/benchmark/MLPerf/patches")
    if d not in sys.path:
        sys.path.insert(0, d)


def _activate_dg():
    _addon_path()
    import dg_dgrad
    dg_dgrad.apply()
    import atexit
    atexit.register(lambda: print(f"[DG_DGRAD] calls {dg_dgrad.stats()}", flush=True))


def _activate_lora():
    _addon_path()
    import ntu_lora_fused
    ntu_lora_fused.apply()
    import atexit
    atexit.register(lambda: print(f"[NTU_FUSED_LORA] calls {ntu_lora_fused.stats()}", flush=True))


def _activate_dy_ag():
    _addon_path()
    import ntu_te_comm
    ntu_te_comm.apply()
    import atexit
    atexit.register(lambda: print(f"[NTU_FP8_DY_AG] calls {ntu_te_comm.stats()}", flush=True))


def _activate_dswiglu():
    _addon_path()
    import ntu_dg_quant
    ntu_dg_quant.apply_dswiglu()
    import atexit
    atexit.register(lambda: print(f"[NTU_DSWIGLU_DG] calls {ntu_dg_quant.stats()}", flush=True))


def _activate_rope():
    _addon_path()
    import ntu_rope
    ntu_rope.apply()
    import atexit
    atexit.register(lambda: print(f"[NTU_ROPE] calls {ntu_rope.stats()}", flush=True))


def _activate_gradq():
    _addon_path()
    import ntu_dg_quant
    ntu_dg_quant.apply_gradq()
    import atexit
    atexit.register(lambda: print(f"[NTU_DG_GRADQ] calls {ntu_dg_quant.stats()}", flush=True))


def _activate_attn_bwd():
    _addon_path()
    import ntu_attn_bwd
    ntu_attn_bwd.apply()
    import atexit
    atexit.register(lambda: print(f"[NTU_ATTN_BWD] calls {ntu_attn_bwd.stats()}", flush=True))


_TARGETS = {}      # module name -> add-on activators, run in order once the module has executed
for _flag, _mod, _fn in (("DG_DGRAD", "transformer_engine.pytorch", _activate_dg),
                         ("NTU_FP8_DY_AG", "transformer_engine.pytorch", _activate_dy_ag),
                         ("NTU_DSWIGLU_DG", "transformer_engine.pytorch", _activate_dswiglu),
                         ("NTU_ROPE", "transformer_engine.pytorch", _activate_rope),
                         ("NTU_DG_GRADQ", "transformer_engine.pytorch", _activate_gradq),
                         ("NTU_ATTN_BWD", "transformer_engine.pytorch", _activate_attn_bwd),
                         ("NTU_FUSED_LORA", "megatron.bridge.peft.lora_layers", _activate_lora)):
    if os.environ.get(_flag, "0") == "1":
        _TARGETS.setdefault(_mod, []).append(_fn)

if _TARGETS:
    import importlib.abc
    import importlib.machinery

    class _PostImportLoader(importlib.abc.Loader):
        """Wraps the real loader of a target module; runs its add-on once the module has executed."""
        def __init__(self, real, fns):
            self._real, self._fns = real, fns

        def create_module(self, spec):
            return self._real.create_module(spec)

        def exec_module(self, module):
            self._real.exec_module(module)
            for fn in self._fns:
                fn()

        def __getattr__(self, name):          # resource readers etc. go to the real loader
            return getattr(self._real, name)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            fns = _TARGETS.pop(name, None)
            if fns is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader, fns)
            return spec

    sys.meta_path.insert(0, _Finder())
