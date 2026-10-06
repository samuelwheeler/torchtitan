# Matched dense 2B and MoE 12B2A HF evaluation

Purpose: compare the data-matched step-27000 checkpoints through the same
HF/lm-eval workflow, extending the earlier MoE integration smoke to complete
benchmarks with a dense baseline.

Both checkpoints consumed 169,869,312,000 training tokens and use the frozen
32K SentencePiece tokenizer. Their common backbone has 24 layers, width 2048,
16 query heads, four KV heads, vocabulary 50304 and ComplexRoPE theta 50000.
The dense FFN width is 10496. The MoE has 36 routed experts of width 2112,
top-3 selection and shared width 4224.

## PR validation note

The HF conversion and existing lm-eval workflow were validated on Aurora with
matched dense 2B and MoE 12B2A step-27000 checkpoints, each trained on 169.9B
tokens. Both completed all seven zero-shot benchmarks (20,465 documents per
model) using the same tokenizer, HF backend, BF16 weights, batch size 8,
context length 2048 and seeds. Paired document/prompt/target hashes and task
settings matched; job `8907189` exited successfully.

MoE scored higher on six of seven tasks. The equal-task mean was **54.41%
versus 51.68%** for dense (**+2.73 percentage points**); the document-weighted
mean was **60.20% versus 55.97%** (**+4.23 points**, paired 95% CI
**+3.61 to +4.84**). HellaSwag showed the largest gain (+7.02 points), while
dense led BoolQ by 0.89 points. These means use `acc_norm` where available and
`acc` otherwise. The dense export also passed exact checks of all 219 tensors
after BF16 casting and checkpoint mapping, native/HF logit checks and focused
regression tests. Per-task scores, source SHA and artifacts are recorded below.

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

Job `8907189` completed on `x4112c4s4b0n0` with `job_state=F`,
`Exit_status=0`, and PBS walltime **49m38s**. The checkout was clean and pinned
to `4e6f3c99dd8665ad178c81d178b1cda9efae59d9`. All 14 model/task evaluations
finished: **20,465 documents per model**, or 40,930 model/document records.

Scores are percentages; deltas are MoE minus dense in percentage points.
`acc_norm` is length-normalized multiple-choice accuracy; BoolQ and Winogrande
use ordinary `acc`.

| Task | Documents | Metric | Dense 2B | MoE 12B2A | Delta (pp) | Exact McNemar p |
|---|---:|---|---:|---:|---:|---:|
| HellaSwag | 10,042 | acc_norm | 56.33 | 63.35 | +7.02 | 2.67e-98 |
| ARC-Easy | 2,376 | acc_norm | 54.59 | 56.73 | +2.15 | 0.0299 |
| ARC-Challenge | 1,172 | acc_norm | 28.92 | 31.66 | +2.73 | 0.0416 |
| Winogrande | 1,267 | acc | 56.04 | 59.83 | +3.79 | 0.0199 |
| PIQA | 1,838 | acc_norm | 73.12 | 75.84 | +2.72 | 0.000606 |
| OpenBookQA | 500 | acc_norm | 33.40 | 35.00 | +1.60 | 0.332 |
| BoolQ | 3,270 | acc | 59.33 | 58.44 | -0.89 | 0.476 |
| **Equal-task mean** | — | — | **51.68** | **54.41** | **+2.73** | — |
| **Document-weighted mean** | **20,465** | — | **55.97** | **60.20** | **+4.23** | **4.05e-41** |

MoE has higher point accuracy on six of seven tasks. The document-weighted
gain has a paired normal-approximation 95% confidence interval of
**+3.61 to +4.84 percentage points**. There are 2,517 examples where only MoE
is correct and 1,652 where only dense is correct. HellaSwag supplies 10,042
of 20,465 examples; the equal-task mean gives each task equal weight.
Per-task p-values are two-sided and unadjusted for multiple comparisons.

## Conversion and comparison validation

The dense HF export contains **219 tensors**, all exactly equal to the DCP
after the requested BF16 cast and layout/RoPE transformations. On a fixed
64-token CPU input, native versus stock HF Llama logits have relative RMS
error **1.1917%**, cosine **0.9997004**, and top-1 agreement **61/64 (95.3125%)**.
These pass the preselected 2% RMS, 0.999 cosine and 95% top-1 gates. MoE uses
the real-checkpoint export validated in
[job 8907019](2026-10-06-hf-export-review-fixes.md).

All paired document IDs, document/prompt/target hashes and filters match.
Task configurations, versions/hashes, counts, special-token IDs, runtime
versions and seeds match. Checkpoint paths are excluded from task metadata
comparison, and serialized function memory addresses are normalized.
The SentencePiece model, tokenizer configuration and special-token map are
byte-identical. SentencePiece SHA-256:
`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`.

Converter/reference Torch: `2.13.0.dev20260430+xpu`; HF inference Torch:
`2.10.0a0+git449b176`, Transformers `4.57.6`, lm-eval `0.4.10`.
After the pinned run, `0eeb83681` moved the dense export model to module scope
for stable configuration class identity. Three dense regression cases passed
again, covering class serialization, full shapes, both DCP layouts, exact
FP32 weights and native/HF logits. Full-config pickle serialization also
encounters preexisting initializer closures; only class serialization is tested.

## Artifacts

Under `outputs/evals/pair-step27000/hf-full/8907189/`:

- `pair-summary.json` / `.md`: full-precision metrics and paired statistics,
  using the existing summarizer from the earlier evaluation checkout.
- `validation/pair-identity.json`: comparison identities/settings, tokenizer
  hashes, counts and summarizer source hash.
- `validation/dense-validation.json`, `.log` and `dense-logits.pt`: numerical
  and checkpoint checks; `validation/pbs-terminal.json`: scheduler evidence.
- `{dense,moe}/<task>/results/`: original harness results and per-example logs;
  `<task>/results.json`: normalized copy for paired statistics.
- `dense/hf`: link to the complete dense export from job `8907155`;
  `moe/hf`: link to the validated export from job `8907019`.

Submission records, PBS script, launch log (`pbs-reference-fixed.log`),
post-processing helper and CPU logs are under
`outputs/evals/pair-step27000/hf-full-validation/`, outside tracked PR source.
The requested seven-task evaluation gate is complete.

## PR preparation

Sam Foreman's `ezpz` base `d4449ba0ec254449b874ebc1b114fa84f66200f2` was
integrated in `c1c2847cc`. The only conflict was the journal; both entries
were preserved. The upstream changes concern RL transport, so the evaluated
HF implementation is unaffected. All **23 focused CPU tests passed** after
integration, including the existing dense converter schema cases; shell syntax
checks also passed. The two distributed DTensor cases passed in job `8907019`.
Commit `da71a884f` adds standard license headers; Python syntax-tree comparison
confirmed that these additions preserve executable code.

CPU evidence and the prepared PR description are under
`outputs/evals/pair-step27000/pr-readiness/`. The target is
`saforem2/torchtitan:ezpz`, with head
`samuelwheeler:feature/aurora-moe-hf-export`. The existing GitHub CPU workflow
excludes experiment paths and uses private PyTorch runners, so these tests
remain manually validated. Local pre-commit verification is pending:
[ezpz policy](../../../../AGENTS.md) assigns that check to the user.

## Earlier attempts

Initial submission `8907124` stayed queued because the user's debug-scaling
running-job limit was reached. It was cancelled while queued and replaced by
the one-node debug layout, preserving all evaluation settings.

Job `8907144` stopped during module initialization because `set -u` exposed
an optional module-system variable. Moving nounset after module loading fixed
the launcher. Job `8907155` exported all 219 dense HF tensors exactly after
BF16 conversion, then stopped at the numerical check before any benchmark
scoring. The validation
helper had loaded historical logical projections without restoring their
packed native counterparts; GQA splitting can produce copies. The reference
loader now explicitly repacks and strictly loads them. Dense CPU tests also
cover multiple KV groups (three checks passed). The complete dense HF export
is reused by the replacement job.
