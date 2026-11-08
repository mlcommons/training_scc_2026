#!/bin/bash
# Reproduces the Transformer Engine bulk wgrad bug with frozen weights (README, Change 2, item 1) using TE's own
# tests/pytorch/distributed/run_layer_with_overlap.py plus a --freeze option (all weights requires_grad=False,
# like the LoRA base weights). Writes run_layer_frozen.py into the current directory (run it from a scratch
# directory) and runs three cases.
# usage: te-bulk-wgrad-repro.sh <TransformerEngine checkout at the installed TE commit> [GPUs, default 4]
# Expected: LayerNormLinear trainable passes, LayerNormLinear frozen fails (input gradient not reduced),
# LayerNormMLP frozen passes (it has the check). Set UB_SKIPMC=1 on GPUs without CUDA multicast.
# Observed with TE 2.14 (f031cf87) on 4x A100: LayerNormLinear frozen input.grad 7.03125 vs 28.125 in the
# reference (1/4, i.e. 1/TP), exit 1; the other two cases pass.
TE=${1:?usage: $0 <TransformerEngine checkout> [GPUs]}; N=${2:-4}
python3 - "$TE/tests/pytorch/distributed/run_layer_with_overlap.py" <<'PY' || exit 1
import sys
s = open(sys.argv[1]).read()
a = '    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")\n'
b = '    dist_print("Copied parameters from test model to reference model...", debug=True)\n'
assert a in s and b in s, "run_layer_with_overlap.py has changed, add --freeze by hand"
s = s.replace(a, a + '    parser.add_argument("--freeze", action="store_true", help="Freeze all weights.")\n', 1)
s = s.replace(b, b + "    if opts.freeze:\n        for p in list(test_model.parameters()) + list(ref_model.parameters()):\n"
                     "            p.requires_grad_(False)\n", 1)
open("run_layer_frozen.py", "w").write(s)
PY
DIMS="-n 64 -d 128 -s 4096 -b 1 --use-bf16-params --seed 1234"
run() { echo "=== $1"; shift; torchrun --standalone --nproc_per_node=$N run_layer_frozen.py $DIMS "$@"; echo "=== exit code $?"; }
run "LayerNormLinear, bulk overlap, trainable" -l layernormlinear --linear-parallel-mode column
run "LayerNormLinear, bulk overlap, frozen" -l layernormlinear --linear-parallel-mode column --freeze
run "LayerNormMLP, bulk overlap, frozen" -l layernormmlp --freeze
