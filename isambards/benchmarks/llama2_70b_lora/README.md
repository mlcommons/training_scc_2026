# isambards: Llama 2 70B LoRA patches

These patches fix the FP8 activation-storage configuration key and support multiple micro-batches during synthetic warm-up in [AMD's v5.1 MI300X implementation](https://github.com/mlcommons/training_results_v5.1/tree/main/AMD/benchmarks/llama2_70b_lora/implementations/MI300X_EPYC_9575F_pytorch_llama2_70b).

## 1. Fix FP8 activation storage key

AMD's config sets `FP8_ACTIVATION=True`. YAML maps this to `fp8_activation_input_store` which is not a field recognised by Megatron. The field Megatron actually reads is [`activation_func_fp8_input_store`](https://github.com/ROCm/Megatron-LM/blob/bc0be960c2f1fb2c4a57ecd7dec6235c79f85412/megatron/core/transformer/transformer_config.py#L96). This patch matches NVIDIA's [H200 config](https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/h200_ngc24.09_nemo/conf/megatron_gpt_peft_tuning_config.yaml#L105) and their [B200 config](https://github.com/mlcommons/training_results_v5.0/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo/conf/megatron_gpt_peft_tuning_config.yaml#L46).

Correcting this reduced peak memory from approximately 175 GiB to 150 GiB on our system.

## 2. Support multiple micro-batches during warm-up

The synthetic warm-up batch uses a single `token_count` entry. As a result warm-up fails when the batch is split into multiple micro-batches, e.g. with `CP=2` at global batch size 8.


The patch sets one token count per sample by using NVIDIA's [H200 implementation as a reference](https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/h200_ngc24.09_nemo/custom_callbacks.py#L426) (whose result ran with [`CP=2`](https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/h200_ngc24.09_nemo/config_DGXH200_1x8x2xtp1pp1cp2.sh)).

This change only affects untimed warm-up. In our tests without the patch, splitting into two micro-batches raised `ZeroDivisionError` whereas with the patch it succeeded. We checked it on CPU against NeMo's [`get_iterator_k_split`](https://github.com/NVIDIA/NeMo/blob/v2.3.0/nemo/collections/nlp/modules/common/megatron/utils.py#L424).

## Applying the patches

Both patches target `AMD/benchmarks/llama2_70b_lora/implementations/MI300X_EPYC_9575F_pytorch_llama2_70b` in [training_results_v5.1 at commit `441d65f`](https://github.com/mlcommons/training_results_v5.1/tree/441d65f4445cd59687466533fd9f55399136e7a4), where `PATCHES` is set to `isambards/benchmarks/llama2_70b_lora/source-or-patches`.  

```bash                                                        
patch -p1 < $PATCHES/0001-fp8-activation-store-config-key.patch
patch -p1 < $PATCHES/0002-warmup-token-count-per-sample.patch
```              
