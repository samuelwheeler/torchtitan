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

All 179 CPU cases passed: 175 MoE/converter/schema/publication/source-guard
cases and four retained JSON configuration factories on the meta device.
The static-asset fixture initially lacked the fast tokenizer JSON required by
the existing dense workflow; saving its HF tokenizer supplied that file, and
both native/legacy static-asset cases then passed. The failed run is retained.
The protected Torch installation remains `2.13.0.dev20260430+xpu`.

Logs and compute submission helpers are ignored artifacts under
`outputs/evals/moe-12b2a-step27000/pr-review-fixes/`. The helpers are excluded
from the PR. The inherited repository-wide lint limitations are recorded in
the [paired evaluation report](2026-10-06-hf-dense-moe-paired.md#pr-preparation).

## Compute validation plan

Use one Aurora debug node and a clean pinned checkout. Re-run both two-rank
DTensor cases, export the same real dense and Sonic step-27000 checkpoints,
and run eight documents from each of the seven tasks for each model through
the existing wrapper. Settings: BF16, batch one, context 2048, zero shots,
seeds `1234,1234,1234,1234`.

Compare every exported tensor and HF asset against the previously validated
exports from jobs `8907019` (MoE) and `8907155` (dense). Recheck the MoE native
and Sonic logit references and all 24 exact FP32 routing biases. The full
20,465-document paired benchmark remains the
[existing result](2026-10-06-hf-dense-moe-paired.md); these are regression smokes.

Compute validation is pending.
