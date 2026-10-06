# HF export PR feedback fixes

Purpose: address correctness, publication, reproducibility and test issues
from [PR #63](https://github.com/saforem2/torchtitan/pull/63).

## Changes

- Legacy Sonic metadata now excludes only the known optimizer, scheduler,
  training-state and dataloader components. Unknown top-level model tensors
  are rejected rather than silently omitted. The real checkpoint contains
  5,073 metadata entries, including 339 model tensors.
- The converter holds a destination lock, writes weights and assets into a
  sibling staging directory, records a completion manifest, and publishes by
  atomic rename. Nonempty exports are never overwritten. Failed staging
  directories remain available for inspection. The wrapper validates file
  names, sizes, assets and weight shards before evaluation or eval-only reuse.
  Older dense static config/tokenizer assets are copied inside staging too.
- Both preflight and the validation PBS launcher call the same source guard,
  rejecting an incorrect SHA, tracked edits and untracked files. Ignored
  output/cache directories remain allowed.
- Small native and legacy dense/MoE exports are compared against an independent
  reference that decodes physical QKV, gate/up and expert storage directly.
  Expected values and legacy checkpoint fixtures no longer call the adapter.

The user deferred the expert-dispatch performance suggestion. The HF forward
pass remains unchanged: each routed expert processes only its selected tokens;
the Python loop still scans the expert list.

## CPU validation

All 180 CPU cases passed: 176 MoE/converter/schema/publication/source-guard
cases and four retained JSON configuration factories on the meta device.
The static-asset fixture initially lacked the fast tokenizer JSON required by
the existing dense workflow; saving its HF tokenizer supplied that file, and
both native/legacy static-asset cases then passed. The failed run is retained.
The protected Torch installation remains `2.13.0.dev20260430+xpu`.

Logs and compute submission helpers are ignored artifacts under
`outputs/evals/moe-12b2a-step27000/pr-review-fixes/`. The helpers are excluded
from the PR. The inherited repository-wide lint limitations are recorded in
the [paired evaluation report](2026-10-06-hf-dense-moe-paired.md#pr-preparation).

## Initial compute run

Aurora job `8907545`, pinned to
`872cb25c07416b39e8d6bf484bcbf424fa58e8c0`, passed both two-rank cases and
published the complete real MoE export, then exited 1 after 7m01s. The new
wrapper validation imported the experiment package, whose initializer requires
`ezpz`; that dependency is absent from the separate inference environment.
The check now loads the standard-library validation file directly with `runpy`.
A regression runs the wrapper validation with site packages and dependency
paths disabled. The failed job, complete export and terminal evidence are
retained in the ignored artifacts.

## Compute validation

Aurora debug job `8907576` ran on `x4013c7s4b0n0`, pinned to
`a800018fdee8d03b759ac43487cd1e711ed9f04f`. Terminal evidence:
`job_state=F`, `Exit_status=0`, walltime 25m35s. Both two-rank DTensor cases
passed again. The existing wrapper freshly converted both real step-27000
checkpoints and evaluated eight documents from each of the seven tasks for
each model (56 documents and 176 likelihood requests per model). Settings:
BF16, batch one, context 2048, zero shots, seeds `1234,1234,1234,1234`.
Document/prompt/target hashes and task settings matched between models.

The MoE native/Sonic comparisons passed on the same 64-token input:

| Reference | Relative RMS | Cosine | Top-1 agreement |
|---|---:|---:|---:|
| Current native | 0.449% | 0.9997712 | 96.875% |
| Frozen Sonic | 0.536% | 0.9997684 | 98.438% |

Both references produced the same final-token top-10. All 24 loaded routing
biases were exact FP32 matches to the source checkpoint.

Every exported tensor matched the previously validated exports exactly in
value, shape and dtype: 2,859 MoE tensors from job `8907019` and 219 dense
tensors from the export used by job `8907189` (created in `8907155`). Model and
tokenizer JSON assets matched, SentencePiece bytes matched, and the MoE model
and configuration Python files had identical syntax trees. The inference
implementation is unchanged, so these export/publication fixes preserve the
artifacts used for the [full paired benchmark](2026-10-06-hf-dense-moe-paired.md).
The bounded evaluations are regression checks, not replacement benchmark scores.

Run artifacts are under
`outputs/evals/moe-12b2a-step27000/pr-feedback-validation/hf-export-runs/8907576/`:
`validation/{comparisons,routing-bias,export-equality,smoke-identity}.json`,
`validation/dtensor-tests.log`, `hf/`, `dense/hf/`, and both `results/` trees.
Submission inputs, script/helper SHA256 digests, terminal scheduler evidence,
CPU and pre-commit logs remain under `pr-review-fixes/`. The scripts and large
artifacts remain ignored and are excluded from the PR.

Validation is complete. Expert-dispatch optimization remains deferred.
