# MoE HF export review fixes and pipeline validation

Purpose: validate the corrected export through the [existing HF pipeline](../../../guides/agpt-moe-hf-eval.md) on Aurora.

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

The compute job invokes `convert_and_eval.sh` from
`validate_agpt_moe_hf_1n.pbs`, using one debug node, pinned clean source,
the Torch 2.13 converter and frameworks 2025.3.1 for inference. Inputs:

- DCP: `torchtitan/outputs/agpt_dense_moe_256n_50k/CODEX_AGPT_MOE256_SAFE_20260810T164655Z_7dee5484/moe/train/checkpoint/step-27000`
- Tokenizer: `torchtitan/outputs/agpt_2b_50k_native_ddp_compile_wandb_ab/8703043/source/assets/hf/llama-2-32k-sp`
- Native logits: `torchtitan-moe-eval-current/outputs/evals/moe-12b2a-step27000/native-smoke/8886033/logits.pt`
- Frozen Sonic logits: `torchtitan-moe-eval/outputs/evals/moe-12b2a-step27000/full-sonic-oracle/8882709/logits.pt`

All input paths are relative to `/lus/flare/projects/AuroraGPT/sww/new_tt_aurora`.
Outputs are under this checkout's
`outputs/evals/moe-12b2a-step27000/hf-export-runs/<PBS job ID>/`.

The smoke covers eight documents each for HellaSwag, ARC-Easy, ARC-Challenge,
Winogrande, PIQA, OpenBookQA and BoolQ (batch one, zero shots). Numerical
gates on the saved 64-token reference: relative RMS <= 2%, cosine >= 0.999,
top-1 agreement >= 95%, and exact source FP32 routing biases.

## Results

The final login-node CPU suite passed 20 tests, including the three existing
dense converter schema tests, with Torch `2.13.0.dev20260430+xpu` and
Transformers `4.57.6`. Both small-DCP layouts
exported successfully; the saved weights and FP32 biases matched expectations
exactly. HF BF16 loading kept the bias in FP32 without changing its values.
The full registered model passed a meta-device round trip with all keys and
shapes restored. A combined run exposed uninitialized toy expert weights;
explicit model initialization fixed the fixtures. Source metadata and tokenizer
checks also passed (336 layer tensors; FP32 source routing bias).

Job `8907007` ran on `x4504c0s6b0n0` at source `b024aa791`, then exited 1
after 1m44s. The two spawned CPU test processes failed to import the fixture
through the unqualified `conftest` module name. The pipeline correctly stopped
before conversion. The test now imports a plain configuration factory through
the full repository package path; the failed logs are retained under
`outputs/evals/moe-12b2a-step27000/hf-export-runs/8907007/validation/`.

Replacement job `8907019` ran on `x4112c2s6b0n0` at source
`3df50f0f7c615ddc8691089d9201a93cdee8ed08` and finished with `job_state=F`,
`Exit_status=0`, `resources_used.walltime=00:16:34`. Both two-rank DTensor
cases passed. The wrapper exported the real checkpoint and completed all seven
tasks on `xpu:0` using Torch `2.10.0a0+git449b176`, Transformers `4.57.6`,
and lm-eval `0.4.10`: eight effective documents per task, 176 likelihood requests.
These are integration results, not complete benchmark metrics.

| Reference | Relative RMS error | Cosine similarity | Top-1 agreement |
|---|---:|---:|---:|
| Current native TorchTitan | 0.4401% | 0.9997711 | 96.875% |
| Frozen Sonic | 0.5275% | 0.9997683 | 98.4375% |

Both final-token top-10 lists match exactly. All 24 loaded routing biases
remained FP32 and equaled the original DCP tensors bit for bit. An independent
CPU read of safetensors also verified exact source biases. The header contains
2,835 BF16 tensors and 24 FP32 tensors.

Artifacts relative to this checkout:

- `outputs/evals/moe-12b2a-step27000/hf-export-runs/8907019/hf/`: complete HF export.
- `outputs/evals/moe-12b2a-step27000/hf-export-runs/8907019/results/`: lm-eval
  results JSON and per-task samples.
- `outputs/evals/moe-12b2a-step27000/hf-export-runs/8907019/validation/`:
  `comparisons.json`, `routing-bias.json`, `serialized-routing-bias.json`,
  `logits.pt`, and `dtensor-tests.log`.
- `outputs/evals/moe-12b2a-step27000/review-fixes/`: failed `pbs.log`, successful
  `pbs-retry-1.log`, `submission.json`, failed `cpu-tests.log`, and passing
  `cpu-tests-final.log`.

## Remaining gate

The upstream CPU workflow excludes experiment paths. A CPU workflow is prepared
under ignored outputs; placing it in `.github/workflows/` awaits the requested
exception to the ezpz AGENTS.md restriction on edits outside `experiments/ezpz/`.
Full benchmark metrics remain separate from the bounded integration smoke.
