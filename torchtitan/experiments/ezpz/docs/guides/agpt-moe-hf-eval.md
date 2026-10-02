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
