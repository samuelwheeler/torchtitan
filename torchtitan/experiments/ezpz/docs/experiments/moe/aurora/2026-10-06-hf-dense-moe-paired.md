# Matched dense 2B and MoE 12B2A HF evaluation

Purpose: compare the data-matched step-27000 checkpoints through the same
HF/lm-eval workflow, extending the earlier MoE integration smoke to complete
benchmarks with a dense baseline.

Both checkpoints consumed 169,869,312,000 training tokens and use the frozen
32K SentencePiece tokenizer. Their common backbone has 24 layers, width 2048,
16 query heads, four KV heads, vocabulary 50304 and ComplexRoPE theta 50000.
The dense FFN width is 10496. The MoE has 36 routed experts of width 2112,
top-3 selection and shared width 4224.

## Implementation and checks

`convert_and_eval.sh --model 2b_50k` selects the dense baseline, derives its
Llama configuration from the registered model, and uses the same tokenizer
asset writer and HF backend as the MoE. `--max-length` and `--seed` make the
comparison settings explicit. New CPU tests verify full registered-model HF
shapes, tokenizer/BOS behavior, actual native and historical DCP conversions,
exact FP32 exported weights, and native/HF logits. Nine focused tests passed.

## Run settings

- Zero-shot HellaSwag, ARC-Easy, ARC-Challenge, Winogrande, PIQA, OpenBookQA,
  and BoolQ; complete evaluation splits with no document limit.
- HF backend, BF16 weights, batch size 8, context length 2048, all four seeds
  1234; per-example logs for identity checks and paired statistics.
- Converter: existing Torch 2.13 venv; inference: frameworks 2025.3.1 with
  Transformers 4.57.6 and lm-eval 0.4.10.
- One debug node; six tiles per model. Each tile handles one task, except
  OpenBookQA and BoolQ share a tile sequentially. PBS launch artifacts and
  validation helpers stay under ignored outputs.

Inputs, relative to `/lus/flare/projects/AuroraGPT/sww/new_tt_aurora`:

- Dense: `torchtitan/outputs/agpt_dense_moe_256n_50k/CODEX_AGPT_DM256_20260802T194045Z_0b96f486/dense/train/checkpoint/step-27000`.
- MoE: validated HF export at `torchtitan-moe-hf-export/outputs/evals/moe-12b2a-step27000/hf-export-runs/8907019/hf`.
- Tokenizer: `torchtitan/outputs/agpt_2b_50k_native_ddp_compile_wandb_ab/8703043/source/assets/hf/llama-2-32k-sp`.

Outputs: `outputs/evals/pair-step27000/hf-full/<PBS job ID>/` in this checkout.
Submission records, helpers and CPU logs:
`outputs/evals/pair-step27000/hf-full-validation/`.

## Results

Compute evaluation pending. The previous native-model results are a reference;
the eight-document MoE HF smoke is insufficient for a model-quality comparison.

Initial submission `8907124` stayed queued because the user's debug-scaling
running-job limit was reached. It was cancelled while queued and replaced by
the one-node debug layout, preserving all evaluation settings.
