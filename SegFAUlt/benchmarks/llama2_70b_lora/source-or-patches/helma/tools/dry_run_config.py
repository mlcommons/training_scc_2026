#!/usr/bin/env python3
"""Validation à sec d'une config de run, sans GPU (frontal) : lancé dans le conteneur par submit.sh --dry-run
et tools/check_prereqs.sh, avec l'environnement déjà chargé par la config.

Compose la config Hydra exactement comme nemo/train.py (mêmes resolvers que utils.resolve), puis affiche
le découpage (TP/CP/DP, GBS, accumulation), le calendrier d'évaluation MLPerf, des contrôles de cohérence
et une estimation grossière de la mémoire par GPU (ordre de grandeur, pas une garantie).
Code de retour 1 si une erreur bloquante est détectée.
"""
import math
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = os.environ.get("NEMO_CONF_DIR", "/workspace/ft-llm/conf")
GIB = 2**30
# Mémoire utilisable par GPU : H100 94 Go ≈ 93 Gio (défaut) ; H100/A100 80 Go ≈ 79 ; H200 141 Go ≈ 139.
H100_USABLE_GIB = float(os.environ.get("GPU_MEM_GIB", "93"))
LINEAR_PARAMS = 68.45e9         # 80 couches x (QKV + proj + FC1 + FC2) de Llama2-70B
EMB_OUT_PARAMS = 0.524e9        # embeddings + couche de sortie (vocab 32000 x 8192 x 2), en BF16
HIDDEN, LAYERS, SEQ = 8192, 80, 8192
# Calibré sur R01 (job 910321, TP4/SP/FP8, FP8_ACT=1) : 34,7 Gio/GPU après chargement du checkpoint
# (≈ 2 octets/poids malgré FP8=True), pic de 79,9 Gio en entraînement. D'où ~35 octets par token et par
# couche d'activations (FP8_ACT), ~44 en BF16, et ~3 Gio fixes. fp8_param=1 (1 octet/poids) : non mesuré (R07).
ACT_BYTES_FP8_ACT, ACT_BYTES_BF16 = 35, 44
OVERHEAD_GIB = 3.0              # contexte CUDA, tampons NCCL, workspaces


def register_resolvers():
    for name, fn in {
        "add": lambda x, y: x + y,
        "multiply": lambda x, y: x * y,
        "floor_div": lambda x, y: x // y,
        "ceil_div": lambda x, y: math.ceil(x / y),
        "floor": lambda x: math.floor(x),
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def memory_estimate(m, tp, cp):
    fp8 = bool(m.fp8)
    weights = (LINEAR_PARAMS * (1 if (fp8 and m.fp8_param_gather) else 2) + EMB_OUT_PARAMS * 2) / tp
    bytes_per_token_layer = ACT_BYTES_FP8_ACT if (fp8 and m.activation_func_fp8_input_store) else ACT_BYTES_BF16
    shard = tp if (tp > 1 and m.sequence_parallel) else 1
    s_local = SEQ // cp
    act_layer = s_local * HIDDEN * bytes_per_token_layer / shard
    n_rec = 0
    if m.recompute_granularity == "full":
        n_rec = LAYERS if m.recompute_method == "uniform" else int(m.recompute_num_layers or 0)
    acts = (LAYERS - n_rec) * act_layer + n_rec * s_local * HIDDEN * 2 / shard
    total = weights / GIB + acts / GIB + OVERHEAD_GIB
    return weights / GIB, acts / GIB, total


def main():
    register_resolvers()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="megatron_gpt_peft_tuning_config")
    OmegaConf.resolve(cfg)

    t, m = cfg.trainer, cfg.model
    tp, pp, cp = m.tensor_model_parallel_size, m.pipeline_model_parallel_size, m.context_parallel_size
    world = t.devices * t.num_nodes
    dp = world // (tp * pp * cp)
    gbs, mbs = t.global_batch_size, t.micro_batch_size
    minibs = int(os.environ.get("MINIBS", "1"))
    start_eval = (cfg.skip_evals + 1) * t.val_check_interval
    eval_iters = math.ceil(173 / t.val_global_batch_size)
    errors, warnings = [], []

    if world % (tp * pp * cp):
        errors.append(f"{world} GPU non divisible par TP*PP*CP={tp * pp * cp}")
    if 8 % tp or (8 // tp) % cp or (64 // tp) % cp:
        errors.append(f"têtes d'attention (64 Q / 8 KV) non réparties par TP={tp} x CP={cp}")
    if m.sequence_parallel and tp == 1:
        warnings.append("SP=1 avec TP=1 : ignoré par Megatron")
    if tp > 1 and not m.sequence_parallel:
        warnings.append("TP>1 sans SP : activations non partagées entre rangs TP")
    if tp > t.devices:
        warnings.append(f"TP={tp} > {t.devices} GPU/nœud : le TP traverse l'InfiniBand")
    if m.fp4:
        errors.append("FP4 (NVFP4) demandé : impossible sur H100 (Hopper n'a pas d'unités FP4)")
    if minibs % mbs:
        errors.append(f"MINIBS={minibs} non multiple de MBS={mbs}")
    if gbs != minibs * dp:
        warnings.append(f"GBS={gbs} différent de MINIBS x DP={minibs * dp}")
    if t.max_steps < start_eval:
        warnings.append(f"MAX_STEPS={t.max_steps} < 1re éval (step {start_eval}) : aucune évaluation")

    w, a, total = memory_estimate(m, tp, cp)
    verdict = "OK" if total < H100_USABLE_GIB - 8 else ("RISQUE" if total < H100_USABLE_GIB - 2 else "OOM probable")
    seed = os.environ.get("SEED") or f"{os.environ.get('SEED_BASE', m.seed)} (essai 1)"
    recompute = (f"{m.recompute_granularity} {m.recompute_method or ''} {m.recompute_num_layers or ''}".strip()
                 if m.recompute_granularity else "non")
    recipe = f" (recette {m.fp8_recipe})" if m.fp8 else ""

    print(f"run            : {os.environ.get('RUN_ID', '?')} {os.environ.get('RUN_NAME', '')} ({os.environ.get('DGXSYSTEM', '')})")
    print(f"GPU            : {t.num_nodes} nœud(s) x {t.devices} = {world}")
    print(f"parallélisme   : TP={tp} PP={pp} CP={cp} SP={bool(m.sequence_parallel)} -> DP={dp}")
    print(f"batch          : GBS={gbs} (MINIBS={minibs} x DP={dp}), MBS={mbs}, micro-batches/step={minibs // mbs}")
    print(f"optimisation   : LR={cfg.optim.lr} MAX_STEPS={t.max_steps} graine={seed} séquence={m.encoder_seq_length}")
    print(f"précision      : FP8={bool(m.fp8)}{recipe}, FP8 attention={bool(m.fp8_dot_product_attention)}, "
          f"FP8 activations={bool(m.activation_func_fp8_input_store)}, FP4={bool(m.fp4)}")
    print(f"CUDA graphs    : {os.environ.get('MCORE_CUDA_GRAPH', '0') == '1'} (portée {m.cuda_graph_scope}), "
          f"recompute={recompute}")
    print(f"évaluation     : tous les {t.val_check_interval} steps, {cfg.skip_evals} sautées -> 1re au step {start_eval} "
          f"({start_eval * gbs} samples), {eval_iters} itérations/éval, cible {cfg.custom.target_log_ppl}")
    print(f"mémoire/GPU    : poids {w:.1f} Gio + activations {a:.1f} Gio + ~{OVERHEAD_GIB:.0f} Gio "
          f"= {total:.1f} / {H100_USABLE_GIB:.0f} Gio -> {verdict}")
    for msg in warnings:
        print(f"ATTENTION      : {msg}")
    for msg in errors:
        print(f"ERREUR         : {msg}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
