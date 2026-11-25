"""Evaluate each validation sample only up to its last loss token (patch 10).

Every validation sample in NVIDIA's validation.npy is 8192 real tokens (a
GovReport document truncated to the window), but the loss mask covers only the
summary: 1070 tokens on average, and the last masked position averages 5737
(min 1212, max 8192). With causal attention the logits at position i depend
only on positions <= i, so the tokens after the last loss token cannot change
the eval loss. The reference and NVIDIA nevertheless run all 8192 positions.

With EVAL_TRUNCATE=1, `GPTSFTPackedDataset.__getitem__` cuts a validation
sample after the token that is the label of its last loss position. The collate
already pads to the longest item in the (VAL_MBS=1) batch rounded up to
`pad_seq_length_to_mult`, which is raised to EVAL_TRUNCATE_MULT (default 256)
for the validation set to keep the number of distinct shapes (cuDNN attention
plans, cuBLAS heuristics) small. Positions are unchanged, so RoPE is unchanged;
padded positions carry loss_mask 0. Training data is untouched (3901 samples,
all 8192 tokens, all with loss).

Expected: ~30% fewer tokens per eval microbatch on average, more in attention
(quadratic), for an identical loss up to FP8 quantization noise.
"""
import os

_installed = False


def install():
    global _installed
    if _installed:
        return
    # NeMo 2.x's FineTuningDataModule builds nemo.collections.llm.gpt.data.core's
    # own copy of GPTSFTPackedDataset, not the one under collections.nlp
    # (which is what a first version of this patched, to no effect: job 247).
    # Patch both.
    import nemo.collections.llm.gpt.data.core as core
    from nemo.collections.nlp.data.language_modeling.megatron import gpt_sft_dataset as nlp_ds

    mult = int(os.environ.get("EVAL_TRUNCATE_MULT", "256"))
    for cls in {core.GPTSFTPackedDataset, nlp_ds.GPTSFTPackedDataset}:
        _wrap(cls, mult)
    _installed = True
    print(f"eval_truncate: validation samples cut after their last loss token, "
          f"padded to a multiple of {mult}", flush=True)


def _wrap(GPTSFTPackedDataset, mult):
    orig = GPTSFTPackedDataset.__getitem__

    def is_validation(ds):
        return "validation" in os.path.basename(str(getattr(ds, "file_path", "")))

    def __getitem__(self, idx):
        item = orig(self, idx)
        if not is_validation(self):
            return item
        if getattr(self, "pad_seq_length_to_mult", 0) < mult:
            self.pad_seq_length_to_mult = mult
        mask = item["loss_mask"]
        last = -1
        for i in range(len(mask) - 1, -1, -1):
            if mask[i]:
                last = i
                break
        if last < 0:
            return item
        keep = last + 2                       # the label of position `last` is token last+1
        if keep >= len(item["input_ids"]):
            return item
        return {
            "input_ids": item["input_ids"][:keep],
            "loss_mask": mask[:keep],
            "seq_boundaries": [0, keep],
        }

    GPTSFTPackedDataset.__getitem__ = __getitem__
