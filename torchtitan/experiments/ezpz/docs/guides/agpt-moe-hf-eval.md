# AGPT MoE Hugging Face export

The `AGPT_2B_50K_MOE_sdpa_aurora_full_sonic` flavor can be exported with the
same DCP-to-Hugging-Face converter used by dense AGPT models. The artifact is a
standard `AutoModelForCausalLM` remote-code checkpoint and runs through
lm-eval's existing `hf` backend; there is no separate lm-eval adapter.

The HF model retains the training architecture:

- 24 Llama-style GQA layers, width 2048, 16 query heads and 4 KV heads;
- ComplexRoPE with head width 128 and theta 50000;
- 36 routed SwiGLU experts per layer, top-3 routing and width 2112;
- original softmax probabilities without selected-score renormalization;
- the FP32 expert bias affects selection only;
- one always-on shared SwiGLU branch of width 4224.

The exporter understands both current fused TorchTitan state dictionaries and
the historical August 2026 Sonic DCP layout. Historical Q/K projections are
permuted to Hugging Face's RoPE convention, and Sonic expert matrices are
transposed into per-expert HF linear weights. Source and destination key sets
are validated exactly.

## Convert and evaluate

Run this inside the established Aurora lm-eval environment. The tokenizer
directory must contain the `tokenizer.model` used for training.
MoE conversion requires an explicit `--tokenizer-dir`; it has no tokenizer
default. The exporter checks that the SentencePiece vocabulary fits the model
and that UNK/BOS/EOS are the Llama IDs 0/1/2. These checks cannot establish
whether a different tokenizer with the same vocabulary size matches training.

```bash
bash torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh \
  --model 12b2a \
  --step 27000 \
  --dcp-dir /path/to/checkpoint/step-27000 \
  --tokenizer-dir /path/to/llama-2-32k-sp \
  --output-root /path/to/eval-output \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq
```

Conversion writes safetensors, `config.json`, tokenizer metadata, the
SentencePiece model, and the two Python files referenced by `auto_map`.
lm-eval loads the result with `trust_remote_code=True`. The implementation has
been host-tested with Transformers 4.57.6, the version in Aurora frameworks
2025.3.1; the evaluation environment uses lm-eval-harness 0.4.10.

For conversion alone, add `--convert-only`. To evaluate an existing export,
add `--eval-only` and use the same `--output-root`.

The converter needs the current TorchTitan runtime (Torch 2.13 on Aurora).
Evaluation can use a separate Python environment with Transformers 4.57.6 and
lm-eval 0.4.10. Set `CONVERT_PYTHON` and `LM_EVAL_PYTHON` to their interpreter
paths, and `CONVERT_PYTHONPATH` to any converter dependency overlay. The wrapper
adds its own checkout to the converter import path. Evaluation retains the
calling environment's `PYTHONPATH`. `--device` selects the HF device (default
`xpu:0`); `--limit` bounds documents per task for integration checks.

BF16 exports keep the persistent routing bias in FP32. Current shared experts
are unpacked from interleaved `[2F, D]` gate/up rows, and HF import restores
that physical layout.

The matched dense baseline uses `--model 2b_50k`, the same tokenizer directory,
and a separate output root. This selects the 24-layer, vocabulary-50304 dense
model and writes a standard Llama HF configuration from the model registry.
It also uses the HF evaluation backend. Plain `--model 2b` selects the separate
12-layer Gemma-tokenizer model.

For paired benchmark runs, use identical `--tasks`, `--batch-size`,
`--num-fewshot`, `--max-length 2048`, and `--seed 1234,1234,1234,1234` for both
models, and omit `--limit`. Per-example logs are retained by the wrapper.
The [full paired HF report](../experiments/moe/aurora/2026-10-06-hf-dense-moe-paired.md)
records the checkpoint identities and comparison settings.

Run the focused regression suite explicitly; the upstream CPU workflow
excludes experiment paths. On Aurora, run the two-rank CPU cases inside a PBS
allocation:

```bash
python -m pytest torchtitan/experiments/ezpz/tests/moe/test_agpt_moe_hf_*.py \
  torchtitan/experiments/ezpz/tests/moe/test_convert_to_hf_output.py \
  torchtitan/experiments/ezpz/tests/moe/test_dense_hf_conversion.py
```

## Validation

Job `8907019` exported the historical step-27000 checkpoint in BF16 with exact
FP32 routing biases, then evaluated it through the wrapper on XPU. On the same
64-token input, relative RMS error was 0.440% against current native TorchTitan
and 0.528% against the frozen Sonic training implementation. Cosine similarity
was at least 0.999768 and top-1 agreement was 96.875% / 98.438%, respectively.
All paths produced the same final-token top-10. The seven-task commonsense
smoke completed 176 likelihood requests over eight documents per task.
See the [review-fix validation report](../experiments/moe/aurora/2026-10-06-hf-export-review-fixes.md)
for source SHA, runtime versions, artifacts, and the initial test-fixture failure.

Job `8907189` completed all seven full evaluation splits for the matched dense
and MoE step-27000 checkpoints, with 20,465 documents per model and identical
HF settings and tokenizer assets. MoE's equal-task mean was 54.41% versus
51.68% for dense; its document-weighted gain was +4.23 percentage points.
See the [full paired HF report](../experiments/moe/aurora/2026-10-06-hf-dense-moe-paired.md)
for per-task scores, paired statistics and dense conversion validation.
