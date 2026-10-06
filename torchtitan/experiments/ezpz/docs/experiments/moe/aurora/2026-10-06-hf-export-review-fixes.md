# MoE HF export review fixes and pipeline validation

Purpose: validate the corrected shared-expert layout, exact FP32 routing bias,
tokenizer checks, and the existing conversion/lm-eval wrapper on Aurora.
This follows the [HF export guide](../../../guides/agpt-moe-hf-eval.md).

## Changes

- Split and reconstruct native interleaved shared gate/up rows through the
  model adapter, including the converter's checkpoint-schema selection path.
- Preserve persistent expert bias in FP32 during lower-precision export.
- Require the training tokenizer explicitly for MoE conversion; reject
  out-of-range vocabularies and incompatible Llama special tokens before load.
- Anchor converter imports to the checkout and allow separate conversion and
  lm-eval interpreters; support device selection and bounded task runs.
- Add real native and legacy small-DCP conversions, native round-trip and
  shared-forward checks, full registered-model shape checks, tokenizer failure
  cases, BF16 remote loading with exact bias, and native/HF forward comparison.
- Check shared-expert DTensor restoration with two CPU ranks, including shards
  that split gate/up pairs; these distributed tests run inside the allocation.

## Validation plan

CPU tests use the existing Torch 2.13 XPU venv on the login node, with no XPU
execution. Transformers 4.57.6, huggingface-hub and tokenizers are exposed from
the frameworks installation through an ignored test overlay; torch is not
installed or replaced.

The compute job uses `validate_agpt_moe_hf_1n.pbs`, which invokes the actual
`convert_and_eval.sh` pipeline. It uses one debug node, a pinned clean source
commit, the Torch 2.13 converter and frameworks 2025.3.1 for inference.
The submitted environment records these inputs:

- DCP: `torchtitan/outputs/agpt_dense_moe_256n_50k/CODEX_AGPT_MOE256_SAFE_20260810T164655Z_7dee5484/moe/train/checkpoint/step-27000`
- Tokenizer: `torchtitan/outputs/agpt_2b_50k_native_ddp_compile_wandb_ab/8703043/source/assets/hf/llama-2-32k-sp`
- Native logits: `torchtitan-moe-eval-current/outputs/evals/moe-12b2a-step27000/native-smoke/8886033/logits.pt`
- Frozen Sonic logits: `torchtitan-moe-eval/outputs/evals/moe-12b2a-step27000/full-sonic-oracle/8882709/logits.pt`

All input paths are relative to `/lus/flare/projects/AuroraGPT/sww/new_tt_aurora`.
Outputs are under this checkout's
`outputs/evals/moe-12b2a-step27000/hf-export-runs/<PBS job ID>/`.

The task smoke runs eight documents each for HellaSwag, ARC-Easy,
ARC-Challenge, Winogrande, PIQA, OpenBookQA and BoolQ, batch size one,
zero shots. These are integration results, not complete benchmark metrics.
Numerical gates: relative RMS <= 2%, cosine >= 0.999, top-1 agreement >= 95%
on the saved 64-token reference. Every loaded routing-bias buffer must equal
the corresponding source DCP FP32 tensor exactly.

## Results

The login-node CPU suite passed 17 tests with Torch
`2.13.0.dev20260430+xpu` and Transformers `4.57.6`. Both small-DCP layouts
exported successfully; the saved weights and FP32 biases matched expectations
exactly. HF BF16 loading kept the bias in FP32 without changing its values.
The full registered model passed a meta-device round trip with all keys and
shapes restored. The real tokenizer and source DCP metadata also passed
preflight checks (336 layer tensors; FP32 source routing bias).

Compute submission is pending the pinned-source preflight.
