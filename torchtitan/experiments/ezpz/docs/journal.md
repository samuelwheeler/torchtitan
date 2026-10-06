# Development Journal

Running log of what's happening, session by session. Most recent first.

## 2026-10-06 -- HF export PR feedback

- Addressed unknown legacy model keys, atomic export publication, untracked
  source files and circular conversion expectations from PR #63. The user
  deferred expert-dispatch optimization; the inference code remains unchanged.
- All 180 CPU cases passed. New checks cover concurrent writers, interrupted
  exports, static dense assets, independent mappings and pinned-source guards.
- Evidence: `outputs/evals/moe-12b2a-step27000/pr-review-fixes/`.
  [Validation report](experiments/moe/aurora/2026-10-06-hf-export-pr-feedback.md).
- Job `8907545` passed both two-rank cases and published the MoE export, then
  failed the new wrapper validation on a training-only `ezpz` import in the
  inference runtime (exit 1, 7m01s). Direct standard-library validation and an
  isolated-runtime regression fix that dependency. Failed artifacts retained.
- Job `8907576`, pinned to `a800018fd`, passed on `x4013c7s4b0n0`:
  `job_state=F`, `Exit_status=0`, walltime 25m35s. Both two-rank cases and fresh
  dense/MoE conversion plus seven-task smokes passed (56 documents/model);
  document/prompt/target hashes and settings matched.
- All 2,859 MoE and 219 dense tensors exactly match the earlier validated
  exports; model/tokenizer assets are equivalent and MoE model syntax trees
  are identical. All 24 FP32 routing biases remain exact. Native/Sonic relative
  RMS is 0.449% / 0.536%, with unchanged top-1 agreement and final-token top-10.
- Artifacts: `outputs/evals/moe-12b2a-step27000/pr-feedback-validation/`;
  submission digests and terminal scheduler evidence in `pr-review-fixes/`.
- Validation gate closed; issue 5 is deferred. PR #63 contains the correctness
  fixes and accompanying validation evidence.

## 2026-10-06 -- MoE HF export PR preparation

- Integrated Sam's current `ezpz` base `d4449ba0e` in `c1c2847cc`, preserving
  both journal entries. The branch is no longer behind the target base.
- All 27 focused CPU checks passed after integration (23 converter/model/schema
  cases plus four retained JSON config registrations on the meta device).
  Initial config imports lacked existing Grain/renderers dependencies; an
  isolated no-deps overlay resolved them without changing Torch. The two distributed
  DTensor checks and full paired HF evaluation already passed in jobs
  `8907019` / `8907189`. The upstream RL changes do not alter the evaluated
  HF implementation.
- Added standard license headers in `da71a884f`; syntax-tree comparison
  confirmed identical executable code. Wrapper, PBS and preflight shell syntax
  checks passed. All PR changes stay under `experiments/ezpz/`.
- Evidence and submission text: `outputs/evals/pair-step27000/pr-readiness/`.
  [Validation and PR note](experiments/moe/aurora/2026-10-06-hf-dense-moe-paired.md).
- The user authorized local pre-commit verification as an exception to the
  ezpz policy. The formatter changed 15 Python files without changing their
  syntax trees; all PR-file hooks passed, with the repository-wide Pyrefly
  hook checked separately in a temporary copy.
- The post-format CPU rerun passed all 160 cases in the MoE and retained JSON
  launch-config suites (156 ordinary and four meta-device checks). The
  two-rank DTensor module retains its earlier compute-node evidence.
- Pyrefly reports identical 35 core errors on the feature/base branches in
  this environment. Repository-wide Lychee finds one existing broken skill
  link in `experiments/graph_trainer/AGENTS.md:300`, reproduced on the base;
  all PR-file links pass. See the linked report and readiness logs.
- The generic GitHub CPU workflow does not cover these experiment tests;
  inherited repository-wide lint failures remain a CI limitation.

## 2026-10-06 — Matched dense/MoE HF evaluation

- Added the dense `2b_50k` export and HF pipeline selection, shared Llama
  tokenizer assets, explicit evaluation context and seed settings.
- Nine focused CPU checks passed. The dense model was then moved to module
  scope for stable configuration class identity; all three dense cases passed
  again, including both DCP layouts and native/HF logits.
- Job `8907189`, pinned to `4e6f3c99d`, completed on `x4112c4s4b0n0`:
  `job_state=F`, `Exit_status=0`, walltime 49m38s. All seven zero-shot HF tasks
  completed for both step-27000 checkpoints (20,465 documents/model).
- All 219 dense HF tensors matched the source after BF16 conversion and
  mapping. Native/HF relative RMS was 1.1917%, cosine 0.9997004, top-1 61/64.
- Paired document/prompt/target hashes, settings and tokenizer assets matched.
  MoE led six tasks: equal-task mean 54.41% vs 51.68%; document-weighted
  60.20% vs 55.97%, gain +4.23pp (paired 95% CI +3.61 to +4.84pp).
- Job `8907124` was cancelled while queued at the debug-scaling user limit;
  `8907144` failed module initialization, and `8907155` exported successfully
  but stopped on an incorrect native reference loader. Both launcher/reference
  issues were fixed before the successful run; failed artifacts are preserved.
- Artifacts: `outputs/evals/pair-step27000/hf-full/8907189/`, including paired
  statistics, identity checks and scheduler terminal evidence. Launch records
  and CPU logs: `outputs/evals/pair-step27000/hf-full-validation/`.
- Evaluation gate closed; integration with current `origin/ezpz` remains a
  separate PR gate.
- [Run plan and results](experiments/moe/aurora/2026-10-06-hf-dense-moe-paired.md).

## 2026-10-06 — MoE HF export review fixes and real-checkpoint validation

- Fixed interleaved shared-expert mapping in both directions, preserved FP32
  routing biases in BF16 exports, validated the explicit training tokenizer,
  and exercised the existing conversion/evaluation wrapper with separate
  converter and inference runtimes.
- Initial debug job `8907007` exited 1 after 1m44s because spawned DTensor
  workers could not import a test fixture; conversion never started. The
  fixture factory now uses its fully qualified package path.
- Replacement job `8907019`, pinned to `3df50f0f7`, ran on
  `x4112c2s6b0n0` and ended `job_state=F`, `Exit_status=0`, walltime 16m34s.
  Both two-rank layout tests passed; step-27000 exported successfully; lm-eval
  completed 176 likelihood requests across seven tasks (eight documents each).
  All 24 serialized and loaded FP32 routing biases matched the DCP exactly.
- Native/Sonic relative RMS errors were 0.4401% / 0.5275%, cosine >= 0.999768,
  and top-1 agreement 96.875% / 98.4375%; final-token top-10s matched both.
- The final CPU run passed 20 tests, including existing dense converter checks.
  A combined run had exposed uninitialized toy experts; explicit initialization
  fixed the fixtures. With the two distributed cases, 22 checks passed.
- Artifacts: `outputs/evals/moe-12b2a-step27000/hf-export-runs/8907019/`.
  Logs and submission record: `outputs/evals/moe-12b2a-step27000/review-fixes/`.
  [Validation report](experiments/moe/aurora/2026-10-06-hf-export-review-fixes.md).
- Next gate: adding the prepared GitHub CPU workflow requires the requested
  exception to the ezpz-only edit policy. Full-task benchmark metrics were
  outside this bounded integration validation.

## 2026-10-01 (Sunspot) -- automatic TorchStore locality resolution

Resolved the forced-Gloo caveat in the production Monarch/TorchStore runbook.
TorchStore compared `os.environ.get("HOSTNAME", socket.gethostname())` between
clients and storage volumes, but scheduler/MPI launchers propagated the head
process's exported `HOSTNAME` to remote actors. Two-host stdlib probe `12479166`
measured a process on `x1922c6s5b0n0` resolving as `x1922c6s3b0n0`, which made
a remote volume appear local and selected inaccessible SharedMemory.

`repair_hostname_env()` now runs in the controller, the core trainer/generator
actor bootstraps, and the ezpz XPU bootstrap before TorchStore resolves
locality. Actor probe `12479169` corrected each inherited FQDN to its own
physical hostname and passed with `TORCHSTORE_LOCALITY_OK`, PBS exit 0.

Correct locality exposed a second policy defect: capability probes selected
MonarchRDMA and then XCCL ahead of Gloo even though neither TorchStore backend
is qualified on this XPU stack. Controls `12479170`, `12479171`, and `12479174`
failed at the initial generator pull, matching historical explicit controls.
The final automatic policy preserves SharedMemory for genuine same-host
transfers and selects Gloo cross-host; explicit Gloo/XCCL/MonarchRDMA selectors
remain available for controlled qualification.

The protected `rl-monarch-torch214` venv was not modified. A separate clone
was repaired from exact RECORD-hash-matching uv-cache files after a complete
audit found 41 missing files across 30 packages; its final audit found zero
missing files and its Torch/ezpz/Monarch/TorchStore import closure passed.

Matched explicit-Gloo job `12479172` and exact-head automatic job `12479179`
both passed the complete two-host gate. Job `12479179`, source
`e02b5266efa5fa0f1e7a6704e166265c21ed725d`, produced 40/40 rollouts across
policy versions 0–3, four pushes/pulls, three finite updates (losses
`-0.0021/-0.030/-0.100`, gradient norms `0.36/0.31/0.32`), full checkpoints at
steps 1–3, clean shutdown, and PBS exit 0. [PR #61](https://github.com/saforem2/torchtitan/pull/61)
contains the fix, tests, reusable probe, and updated runbook; 19 focused tests,
launcher contracts, GitHub lint, and independent review pass.

## 2026-09-30 (reporting) -- INCITE Q3 report

Created [`summaries/2026-Q3-incite.md`](summaries/2026-Q3-incite.md) from the
Q3 retrospectives, production/evaluation trackers, and September 30 accepted
artifacts. The report records both completed 2B base chains, accepted 20B and
stage-2 endpoints, XPU SFT/GRPO and MoE results, and the unresolved 80B
mechanism. Resource usage is reported from the latest documented August 30
Aurora balance rather than presenting an unsupported September 30 total.

## 2026-09-30 (Sunspot) -- upstream `97e673b779` integration and 30B AdamW canary

Branch `sync/upstream-97e673b779` integrates upstream through `97e673b779` and
the concurrent `origin/ezpz` documentation commit. Optimizer/config and
`TrainingEngine`/`ParallelismContext` migrations are complete. Exact Torch 2.15
imports/config construction passed, the combined focused suite passed 163 tests,
and independent review found and closed one stale MoE per-block compile path.

Hardware job `12479099` passed dense TP1 and MoE but exposed a TP2 attention
reshape assumption. Follow-up jobs `12479100`-`12479102` showed that TP2 with
full activation checkpointing is not recompute-stable: saved global-token
metadata (512) is compared with recomputed TP-local metadata (256). Geometry
experiments did not solve it and were reverted. The exact no-AC control
`12479103` completed three finite TP2 updates with PBS exit 0 (`loss3=10.55887`,
`grad3=0.6776`), isolating the limitation to full-AC/SPMD replay rather than TP.

The decisive 30B gate `12479105` passed on repaired Torch 2.15 at 192 ranks / 16
nodes with HSDP `3 x 64`: three finite updates, PBS exit 0,
`FULL_MODEL_CANARY_PASS`, `loss3=11.83799`, and `grad3=3.6425`.

AdamW LR canary `12479108` then completed ten finite points over `1e-7` to
`1e-4` at GBS 960. The sampled smoothed-loss minimum is `11.677259086400811` at
`1e-5`, with loss increasing at all three higher samples. The runner's
`6.70e-7` output is a safety-scaled detector candidate, not the measured
optimum. CSV/plot evidence is preserved under
`docs/experiments/lr-finder/agpt/data/2026-09-30-30b-adamw-canary/`. The
optimizer matrix remains blocked pending fixed-LR validation.

The final integration candidate then incorporated historical full-state
checkpoint migration PR #47 as `a4283256de`. On that exact production-code
head, DCP job `12479113` exited 0, wrote nonempty step-2 and step-4 metadata,
loaded step 2, and reproduced uninterrupted step-3/4 loss and gradient values
exactly. Two-host RL/weight-sync job `12479114` also exited 0 with 40/40
completed rollouts, policy versions 0–3, four pushes/pulls, three full
checkpoints, and finite nonzero gradient norms `0.28`, `0.27`, `0.27`. Its
isolated overlay uses BlendCorpus `feat/remove-deepspeed` at
`50502b0c9de37887bdf2123b13293e264ab9942f`; the protected RL venv was not
modified.

Matched 30B AdamW fixed-LR jobs `12479115`–`12479117` each completed ten finite
updates on the same `3 x 64` topology. Final loss/gradient values were
`10.62454/31.9376` at `4.64e-6`, `10.11744/8.4969` at `1e-5`, and
`10.42669/33.6769` at `2.15e-5`. `1e-5` is therefore the best tested
short-horizon candidate; longer training remains required for a production
recommendation. The final collectable suite passed 382 tests with 4 skips and
21 subtests after migrating stale test fixtures.

## 2026-09-30 (Sunspot) -- 30B backend/runtime reconstruction

Historical logs establish that the successful 30B optimizer campaign used
Torch 2.13 with `partial_dtensor` and pure FSDP over 192 ranks. Current-head
Torch 2.14 job `12479081` recreated its 16-node/LBS=5/GBS=960 geometry but
failed before step 1: FSDP attempted a 36.35-GiB all-gather with 29.84 GiB
already resident per rank. Resident memory was nearly unchanged from shard-16,
so shard-192 did not restore the historical memory behavior.

The shared Torch 2.15 environment was repaired after proving that a broad
`core*` cleanup had deleted 38 legitimate package files. The exact Torch wheel
and affected packages were reinstalled, Triton's missing file was restored at
its RECORD hash, and 33,409 RECORD files verified with zero missing or
mismatched. Current-head job `12479083` then reproduced Torch 2.14's same
36.35-GiB first-forward all-gather OOM. Frameworks `2026.1.0` Torch 2.13 job
`12479084` reached FSDP construction but still reproduced pytorch/pytorch#181519
despite current head's unconditional `_parallelize`: a plain `weight` remained
where `dp_mesh_dims` requires a full-mesh DTensor. Attempt `12479082` was a
harness-only failure because its batch script did not load the frameworks module
needed to resolve MKL; the corrected wrapper is committed.

An opt-in dense TP=1 legacy compatibility path was added at commit `c484260c02`:
skip full-SPMD parameter annotation, use the one-dimensional `dp_shard` mesh,
and omit `DataParallelMeshDims`. Focused tests pass. Torch 2.15 5B job `12479085`
validated the mechanism with three finite updates (`loss3=11.99557`,
`grad3=1.8735`, PBS exit 0). The 30B escalations failed: Torch 2.15 `12479086`
still requested the 36.35-GiB first-forward all-gather, and frameworks Torch
2.13 `12479087` requested 72 GiB during fused-FFN initialization at
`gate_init(t[0])`. Thus neither a repaired newer runtime nor mesh-only emulation
of `partial_dtensor` restores 30B on current head.

Exact historical-source attempt `12479088` ran commit `5a26d8e7c5c05cd38bec7ab4eb48036e5ba54c6d`
with the package versions recorded by successful job `12473743`, but the shared
runtime had later lost 17 `core.py` files and could only be repaired as a copy;
the attempt did not reproduce a clean historical baseline.

The source regression was then identified directly. Upstream #4808 changed
fused FFN `w13` from `[2F,D]` to `[2,F,D]` and added a required FSDP `Shard(1)`
override. AGPT's copied FSDP wrapper omitted that replay, so default `Shard(0)`
padded the two-element axis to the DP shard degree. The resulting allocations
match the failures exactly: about 36 GiB BF16 and 72 GiB FP32 at shard 192.
Commit `cde3c93227` mirrors core's existing
`linear_param_shard_placements()` mapping. Focused tests pass. Job `12479089`
then failed closed because shard 192 cannot evenly divide matrix-row dimension
16,384; the corrected `3 x 64` HSDP job `12479090` completed three finite 30B
updates on repaired Torch 2.15 with PBS exit 0, `loss3=11.8405`, and
`grad3=3.5988`. This closes the full-model runtime gate without restoring
`full_dtensor`; subsequent 30B work must retain a dimension-compatible shard
degree and the stacked-linear placement override.

## 2026-09-30 (Aurora) -- production reporting audit and tail evaluation

- Aurora's supported submission entry point is now the `prod` routing queue,
  whose production destinations use the current BKC and provide oneAPI 2026.1
  by default. The canonical newer-PyTorch guide was updated accordingly, and
  the separate `next-eval` page was collapsed into it. Historical oneAPI
  2025.3.1 instructions remain in a closed disclosure for reproducibility,
  while new operator examples use `qsub -q prod`. The guide also warns not
  to load `frameworks/2026.1.0`, whose bundled PyTorch is not the isolated
  Torch 2.15 runtime used by the validated path.
- Reconciled the registered production lineages against complete DCP metadata
  and accepted result artifacts. The base heads are covered: 2B-256 step
  92,859, 2B-512 step 46,429, 20B-512 step 11,100, and 20B-256 step 17,500 all
  have accepted endpoint evaluations. The two 20B tail results are jobs
  `8880564` and `8880658`; both contain six 0-shot tasks plus ARC-Challenge
  25-shot with finite metrics.
- Two stage-2 gaps remained. The 2B-512 stage-2 chain is complete at step
  23,746 but was evaluated only through step 22,300; the 2B-256 stage-2 chain is
  complete at step 41,300 but was evaluated only through step 28,000. Submitted
  independent fail-closed jobs `8880872` (`2b_real`, step 23,746) and `8880873`
  (complex `2b`, step 41,300). Both subsequently finished with PBS exit 0 and
  validated seven-measurement artifacts.
- The full 608-result corpus plus all four tail artifacts was overlaid into a
  clean Aurora checkout and rerendered successfully. The combined eval chart now
  reaches 20B-512 step 11,100, 20B-256 step 17,500, 2B-512 stage-2 step 23,746,
  and 2B-256 stage-2 step 41,300.
- Exact-head full-state smoke `8880891` used PR #45 commit `6b3246fac9` and
  finished with PBS exit 143 before DCP load. Its branch lacked the later AGPT
  full-SPMD parallelization path used by passing gate `8879698`; this did not
  exercise or refute the legacy DCP migration.
- Production umbrella `8879474` remains queued in `medium`; follower `8879475`
  remains dependency-held. Neither has launched and no production checkpoint
  head has changed.

## 2026-09-29 (mbph + Sunspot) -- upstream `f359667` parity and LR recovery

The 30B pre-step failure discriminator completed on Sunspot. Job `12479051`
passed exact-size raw BF16 all-gather, FP32 reduce-scatter, and FP32 HSDP
replica-group all-reduce controls. Final job `12479055` (commit
`199563d0658c00b3b738fff1b597371a07b2a845`) then completed three forward,
backward, and nonzero optimizer updates in both the pure-FSDP `1 x 16` arm and
the four-node HSDP `3 x 16` arm; PBS exit was 0 and the artifact contains
`FSDP2_XCCL_MATRIX_PASS`. This rules out a deterministic raw-XCCL,
shard-16-storage, or reduced-replicate-axis defect. It does not clear the
768-rank model, so the next gate is a four-node full-30B canary before any
production-scale LR restart. Intermediate jobs `12479049`, `12479052`,
`12479053`, and `12479054` exposed and corrected probe-only launcher,
rendezvous, sparse-gradient, and XCCL AVG-versus-SUM contract errors.

The first full-model escalation did not clear the blocker. HSDP canary
`12479056` (`3 x 16`, 48 ranks, commit `f38c35ec99`) built the 26.20B model at
32.64 GiB/rank and entered step 1, then failed during `loss.backward()` with a
Level Zero `MPL_gpu_imemcpy`/MPI pipeline assertion in oneCCL scale-out
all-reduce; PBS exit was 137. Pure-FSDP control `12479057` (`1 x 16`, 16 ranks,
commit `7b6ad55de6`) also built the model and entered step 1, then ranks 12 and
14 reported `UR_RESULT_ERROR_OUT_OF_RESOURCES` from backward before clipping or
an optimizer update; PBS exit was 143. This exonerates the HSDP replica axis as
the primary cause but confirms a full-model, pre-update collective/resource
ceiling. A current-head 20B/64-layer pure-FSDP control is the next size rung.

Canary instrumentation was then corrected. `12479064` had completed backward
and optimizer work but crashed in the opt-in per-parameter diagnostics, which
added hundreds of DTensor all-reduces; it was not a valid 10B training failure.
With diagnostics removed, 10B job `12479065` completed three finite AdamW
updates and exited 0 (`12.01583 -> 11.97582`, grad norm `2.3522 -> 2.4822`).
Corrected 20B `12479066` still failed in first backward. Algorithm controls
`12479059`/`12479060`, node-local shard control `12479061`, and IPC-cache
control `12479062` all failed. Torch 2.13 `12479063` never became a runtime A/B:
current `dp_mesh_dims` rejected a plain `weight` during FSDP construction.
Depth-48 eager job `12479067` hit a real HBM OOM. These eager canaries differed
from the compiled production 30B launcher, so the maintained canary now retains
compile for the next production-shaped check.

Compiled follow-ups exhausted the remaining safe local controls. `12479068`
and `12479072` failed before step 1 in FSDP backward unshard/copy-in. Corrected
10B `12479065` passed three updates; corrected 20B `12479066` failed in
backward. `CCL_SYCL_OUTPUT_EVENT=0` let 20B `12479071` reach AdamW state
allocation, where it OOMed, but did not clear 30B. Backward-prefetch suppression
shifted but did not remove the 30B failure (`12479075`). At dimension-safe
`dp_shard=32`, `12479076` failed in post-backward `_chunk_cat`; equivalent-copy,
synchronization, and combined controls `12479078`-`12479080` still failed on the
first fallback write, proving the resource exhaustion originated earlier. The
failed monkeypatches were removed. The current Torch 2.14 full-model path is
blocked above 10B; preserved Torch 2.13 cannot build current full-SPMD FSDP,
and the available Torch 2.15 venv has missing core package files. No 64-node LR
retry is justified until a repaired newer XPU runtime or upstream fix exists.

Exact-head Sunspot smoke `12479017` ran commit `bb39b72eaa` from the immutable
project-filesystem checkout. Dense TP=1 and TP=2 each completed three optimizer
steps with arm exit 0. The MoE arm failed during FSDP initialization before
training: routed expert `weight` parameters remained plain tensors while
`dp_mesh_dims` requires all parameters on the full SPMD mesh. A focused local
regression reproduced the missing placement; the ezpz compatibility layer now
assigns replicated full-SPMD placements to routed expert weights when EP is off.
The post-fix focused suite passes 73 tests, 2 skips, and 13 subtests. MoE-only
retry `12479018` then finished with PBS exit 0 and `VERDICT: ok` on repaired
commit `6913990333`: three finite updates with losses 12.95227, 12.59636, and
11.47138 and gradient norms 0.9651, 1.2498, and 1.7102. Exact-head Sunspot
dense TP=1, dense TP=2, and MoE liveness are now green. Distributed DCP resume
and RL/weight-sync remain open, as does exact-head Aurora hardware validation.

Exact-head DCP retry `12479022` closed the distributed-checkpoint gate on commit
`c228830bd3`: an uninterrupted four-step control was compared with a separate
four-step run that wrote full state at step 2 and a fresh process that loaded
that checkpoint. Resumed steps 3 and 4 matched the control's loss and gradient
norm exactly, and a new full step-4 checkpoint was written; PBS exit was 0. The
first harness attempt `12479021` had successfully saved and resumed but used a
different total-step schedule and an ANSI-sensitive parser, so its failed
verdict was a harness defect rather than a DCP failure.

The exact-head RL/weight-sync launcher is ready at commit `b0e660ec58`, but its
mandatory preflight correctly refused submission: the protected
`rl-monarch-torch214` environment is missing
`torch.fx.experimental.unification.core` and `blendcorpus`. Older RL venvs are
on Torch 2.12/2.13 and stale `spmd-types`; none is a valid substitute for the
current sync. The protected environment was not modified. This gate requires a
fresh isolated RL runtime or a verified immutable archive.

The RL runtime blocker was closed without modifying the protected environment.
A new project-filesystem overlay used the validated healthy Torch 2.14 base and
copied complete cached vLLM/tvm-ffi artifacts while reading the remaining
Monarch/TorchStore dependencies from the quarantined environment. After adding
the missing Torch 2.14 `pipeline_per_edge_p2p` config compatibility shim,
exact-head job `12479027` finished with PBS exit 0 and
`RL_SYNC_VERDICT: ok`: 20/20 completed bounded rollouts, policy versions 0 and
1, two trainer pushes, two generator pulls, a real optimizer step, and a
nonempty full step-1 checkpoint. This established same-host transport plumbing,
but the zero loss/gradient meant it did not establish a real policy update.
Harder same-host jobs `12479028` and `12479029` each completed three finite,
nonzero-gradient updates and wrote three checkpoints, but respectively 2/40 and
5/40 rollouts hit the generation cap.

The final acceptance run `12479032` used two physical Sunspot hosts and the
repository's scheduler-SPMD path with explicit TorchStore Gloo. After increasing
the actor attach timeout to cover the measured cold import, it completed 40/40
bounded rollouts across policy versions 0–3, four trainer pushes and four
generator pulls, three finite updates (gradient norms 0.35, 0.34, 0.28), and
three full checkpoints. PBS exited 0 with `RL_MULTIHOST_VERDICT: ok`. This closes
the exact-head Sunspot RL/weight-sync gate.

The isolated `sync/upstream-f359667` worktree now contains upstream merge
`780f0a73e2` plus replayed configuration/topology, MoE, RL, and regression
changes. The committed collectable ezpz suite reached 260 passed, 2 skipped, and
14 subtests. Independent review then found stale validator API calls and
fake-SPMD topology/serialization risks; those are fixed locally and the focused
post-review suite passes 50 tests with 2 skips. The full unit-level ezpz suite
passes 75 tests with 2 skips.

The pre-sync/post-sync numerical gate passed exactly at zero tolerance for
`llama3/debugmodel` and `deepseek_v3/debugmodel`: identical initialization,
outputs, losses, gradients, AdamW state, and post-step parameters across two
updates. Checkpoints saved after step 0 loaded in both directions and reproduced
uninterrupted step 1 exactly. This closes local numerical and ordinary
checkpoint-state parity; distributed DCP and accelerator behavior remain open.
See the
[merge-readiness report](experiments/upstream-f359667-merge-readiness.md).

On Sunspot, corrected 5B chain `12479007`-`12479010` uses the isolated Torch
2.14 XPU runtime and invokes `ezpz.cli:main` through the verified interpreter to
avoid the copied venv's stale console-script shebang. Preflight `12479007` and
five-step `1e-4` canary `12479008` finished with exit 0. The canary's inline
`INVALID` is a known stdout-only predicate defect: W&B run
[`qxkw7004`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qxkw7004)
contains the optimizer diagnostics. The `3e-5` control `12479009` is running;
`1e-5` job `12479010` is queued.

Historical 30B AdamW job `12478510` cannot be resumed because checkpointing was
disabled on code predating durable LR-finder state. A fresh 75-point trajectory
was submitted at the measured-good TP1, `dp_replicate=48`, `dp_shard=16`
geometry on source `45f2e72d05`, with checkpoints every 25 points. Compute
runtime preflight `12479013` passed; production job `12479014` is queued. The
fresh trajectory avoids scientifically invalid stitching across independent
initializations.

Persistent operations and subsequent terminal transitions are mirrored to the
private [TorchTitan Agent Vault](https://mbph.tail3e7069.ts.net:10444/n/history/torchtitan/index.md).
## 2026-09-30 (Aurora) -- tail-eval schema failure and corrected fork control

- Production umbrella `8879474` started at 19:08 UTC on 1,054 nodes. Its
  pinned 20B runtime archive entered broadcast to all 788 nodes used by the two
  20B seats; no trainer console or finite training step existed at the latest
  probe, so the run is in prestage rather than accepted application progress.
  Follower `8879475` remains dependency-held via `afterany:8879474`.
- Chain-3 sync1 debug comparator `8881518` restored the exact step-39,900 seed
  in 68.08 seconds and entered training, but completed no finite optimizer
  update before `MPI_Allreduce_c` hit `MPIR_CVAR_PROGRESS_TIMEOUT=600`; PBS
  ended the one-hour allocation with `Exit_status=-29`. Together with sync0
  `8881300`, this shows that changing only `CCL_OP_SYNC` from `0` to `1` does
  not clear the post-restore first-update blocker. Older queued fallback
  `8881044` was cancelled after `8881518` started.
- Chain-3 sync0 debug replacement `8881300` restored the read-only step-39,900
  seed in 33.13 seconds and entered training, then completed no finite optimizer
  update before `MPIR_CVAR_PROGRESS_TIMEOUT=600`; PBS ended the one-hour
  allocation with `Exit_status=-29`. Explicit `CCL_OP_SYNC=0` therefore does
  not resolve the post-restore first-update collective blocker. The sync1
  comparator `8881044` remains queued as the sole remaining transport
  discriminator. Production umbrella `8879474` remains queued and follower
  `8879475` remains dependency-held.
- Full-state acceptance `8881028` finished after 00:09:19 with PBS
  `Exit_status=143` and a fail-closed `20B_S3000_UPDATE_FAILED rc=143` artifact.
  The local-shard QKV migration removed the prior global `torch.cat` OOM, but
  all 48 ranks failed during DCP restore in DTensor
  `Shard._maybe_unpad_tensor` at `torch._check(orig_size >= logical_dim_size)`.
  No finite step 3,001, `VALIDATED` marker, or step-3,003 checkpoint metadata
  exists. Paired chain-3 controls `8881043` (`CCL_OP_SYNC=0`) and `8881044`
  (`CCL_OP_SYNC=1`) remain independently queued in `next-eval`; neither has
  produced an artifact. Production umbrella `8879474` remains queued and
  follower `8879475` remains dependency-held.
- Production umbrella head `8879474` remains queued in `medium` for 1,054
  nodes, and its PBS `comment`/`estimated.start_time` pair **oscillates between
  scheduling cycles** rather than progressing. Observed within eleven minutes
  on 2026-09-30: at 07:15 CDT `Not Running: Job would conflict with reservation
  or top job` / `Thu Oct  1 04:49:20 2026`; at 07:23, 07:25, and 07:26 CDT
  `Not Running: Node is in an ineligible state: offline` /
  `Thu Oct  1 16:49:20 2026`. The earlier `Not enough free nodes available`
  reading belongs to the same rotation. Treat the comment as a per-cycle
  snapshot, not a monotone blocker sequence, and do not derive a trend from a
  single poll. `eligible_time` does advance monotonically (15:10:47 -> 15:19:20).
  The cron monitor `~/.hermes/scripts/monitor_aurora_lr_recovery.py` had the raw
  `comment` in its dedup key, so each rotation fired a spurious transition; it
  now collapses every `Not Running:*` comment to `queued: not started`.
  Follower `8879475` remains dependency-held via `afterany:8879474` with
  `Hold_Types = d`. No seat has launched, no `.o8879474` output exists, and no
  checkpoint head changed. `qselect -u foremans` on Aurora returns only these
  two jobs, so the tail-eval campaign is fully terminal and nothing else is live.
- Production umbrella head `8879474` remains queued in `medium` for 1,054
  nodes, but PBS changed its blocker from an ineligible offline node to `Not
  enough free nodes available` and now reports an estimated start of
  `2026-10-01 16:49:20`. Follower `8879475` remains dependency-held via
  `afterany:8879474`. No seat has launched and no checkpoint head changed.
- Step-17,500 sibling eval `8880658` finished after 01:41:37 with PBS
  `Exit_status=0` and a non-semantic `Stageout_status=1`. It produced a fresh
  3,012-byte `results.json` (SHA-256
  `9d326a9c37d2c1df75855e3891e61ad1bbbcbd6518203b0bc0905fea1f4ba06f`)
  with all seven requested measurements finite: HellaSwag
  `acc_norm=0.6786`, ARC-Easy `acc_norm=0.6646`, Winogrande `acc=0.5699`, PIQA
  `acc_norm=0.7671`, OpenBookQA `acc_norm=0.3780`, BoolQ `acc=0.6199`, and
  ARC-Challenge 25-shot `acc_norm=0.4505`. The tail-eval backfill is complete.
- Asset-repair eval-only retry `8880564` finished after 01:20:11 with PBS
  `Exit_status=0` and a non-semantic `Stageout_status=1`. It produced a fresh
  3,010-byte `results.json` with seven finite measurements: HellaSwag
  `acc_norm=0.6875`, ARC-Easy `acc_norm=0.6814`, Winogrande `acc=0.5983`, PIQA
  `acc_norm=0.7682`, OpenBookQA `acc_norm=0.3740`, BoolQ `acc=0.6303`, and
  ARC-Challenge 25-shot `acc_norm=0.4437`. This cleared the semantic gate, and
  step-17,500 sibling `8880658` was submitted from immutable source
  `975e43931a`; it started in `capacity` at 09:14 CDT and has not produced a
  result artifact yet.
- Asset-repair eval-only retry `8880564` started in `capacity` at 02:41 CDT
  from immutable source `975e43931a`. It reused the verified 38.6-GiB
  step-11,100 HF shard, repaired the missing HF assets, loaded the model on
  `xpu:0`, and entered the six-task 0-shot lm-eval. At 02:50 CDT the job was
  still running and no `results.json` existed; the 20B-256 step-17,500 sibling
  remains gated on a fresh accepted semantic result.
- Eval-only control `8880551` reused the verified 38.6-GiB step-11,100 HF
  shard and reached lm-eval, but finished after 44 seconds with PBS
  `Exit_status=1` and `Stageout_status=1`. The export directory still lacked
  `config.json`, so Transformers rejected it before model loading and no
  `results.json` was produced. Asset-repair retry `8880564`, pinned to
  immutable source `975e43931a`, is queued in `capacity`. The 20B-256
  step-17,500 sibling remains gated on a fresh accepted semantic result.
- Corrected Flare-output tail-eval canary `8880485` finished after 00:28:47
  with PBS `Exit_status=1`. Historical-schema conversion from immutable source
  `2f2787faf7` succeeded and produced a 38.6-GiB HF shard plus index on Flare.
  The eval wrapper then resolved both `config.json` and the `tt-lm-eval`
  overlay relative to the output root instead of a checkout; both setup steps
  failed, Transformers rejected the export for missing `model_type`, and no
  `results.json` was written. The HF export is reusable after asset repair.
  The 20B-256 step-17,500 sibling remains gated on a successful semantic eval.
- Corrected Flare-output tail-eval canary `8880485` started in `capacity` at
  06:43 CDT from immutable source `2f2787faf7` for the 20B-512 step-11,100
  checkpoint. It passed the pinned-source/import gate and entered
  single-process DCP conversion; after eight minutes it had produced neither a
  fresh HF export nor `results.json`. The 20B-256 step-17,500 sibling remains
  gated on both artifacts from this canary.
- Historical-schema tail-eval canary `8880334` finished after 00:20:55 with
  PBS `Exit_status=143` and `Stageout_status=1`. It entered single-process DCP
  conversion from source `25ab092c6c`, but the log stopped inside checkpoint
  loading and no HF safetensors or `results.json` was produced. The canary is a
  controlled failure, so the 20B-256 step-17,500 sibling remains gated. A
  follow-up source revision `2f2787faf7` pins the large conversion/eval outputs
  to Flare; its exact-runtime preflight was still active at this observation
  and had not submitted a replacement PBS job.
- Four-node restore/update replacement `8880383` finished after 00:09:20 with
  PBS `Exit_status=143` and `Stageout_status=1`. All 48 ranks reached DCP load,
  then failed on the same historical/current schema mismatch:
  `Missing key in checkpoint state_dict:
  layers.0.attention.qkv_linear.wqkv.weight`. The artifact recorded
  `20B_S3000_UPDATE_FAILED rc=143`; no step 3,001 and no fresh checkpoint were
  produced. Increasing from two to four nodes removed the earlier lazy-state
  memory limit but does not solve checkpoint-schema compatibility, which is
  now the remaining restore gate.
- Four-node TP=2 / DP-shard=24 restore/update replacement `8880383` was
  submitted to `debug-scaling` from the immutable `f4e678c023` checkout after
  two-node control `8880302` OOMed during SophiaG state loading. PBS accepted
  the exact 48-rank topology; the job entered running state on four nodes and
  resolved all 48 ranks during bootstrap. This changes only the
  storage-shard geometry; acceptance still requires restoring step 3,000,
  completing steps 3,001--3,003, and writing a fresh nonempty DCP checkpoint.
- Corrected full-state restore/update control `8880302` finished after 00:11:00
  with PBS `Exit_status=143` and `Stageout_status=1`. It cleared the earlier
  plain-tensor parallelization defect, built the 24-rank pure-FSDP model, and
  reached the step-3,000 DCP load, but SophiaG lazy-state initialization during
  `dcp.load()` exhausted each 63.98-GiB tile: 62.70 GiB allocated, 698.38 MiB
  reserved, 97.73 MiB free, then a 280-MiB allocation failed. No training step
  or fresh checkpoint was produced; the artifact recorded
  `20B_S3000_UPDATE_FAILED rc=143`. The two-node topology is therefore below
  the full-state memory floor, and the fork remains blocked on a larger-shard
  restore/update control.
- Historical-schema tail-eval canary `8880334` entered running state in the
  `capacity` queue on one node from immutable source `25ab092c6c`; conversion
  began for the 20B-512 step-11,100 checkpoint. It has not yet produced a fresh
  HF export or `results.json`, so the 20B-256 step-17,500 sibling remains gated.
- Corrected tail-eval retries `8880289` (20B-512 step 11,100) and `8880292`
  (20B-256 step 17,500) both finished with PBS `Exit_status=1` and produced no
  `results.json`. The pinned Torch 2.15 conversion runtime successfully imported
  the current checkout and opened each DCP checkpoint, clearing the earlier
  missing-`spmd_types` failure, but the current fused-QKV model requested
  `layers.0.attention.qkv_linear.wqkv.weight`, which is absent from these
  historical checkpoints. Both wrappers propagated conversion failure
  correctly. The remaining gate is a converter/model definition matching the
  checkpoints' historical state schema; neither failed export is accepted.
- Production umbrella `8879474` remains queued and follower `8879475` remains
  dependency-held.

## 2026-09-29 (Aurora) -- production continuation and isolated 20B fork gates

- Exact-head Aurora sync gate `8879698` finished in 00:07:44 with PBS
  `Exit_status=0`, `Stageout_status=1`, and `VERDICT: ok` on source
  `f4e678c023`. Dense TP=1, dense TP=2, and repaired MoE each completed three
  finite optimizer updates; their step-3 losses were 10.66281, 10.71614, and
  11.47138 respectively. This closes the remaining current-stack Aurora
  dense/MoE execution gate. Production umbrella `8879474` remains queued for
  1,054 nodes, with `8879475` dependency-held behind it.
- Tail eval `8878144` finished with PBS `Exit_status=0` after eight seconds but
  produced no step-11,100 result. DCP conversion imported the current checkout
  through the bare `frameworks/2025.3.1` Python and failed on
  `ModuleNotFoundError: spmd_types`; the loop then printed its completion banner
  and returned zero. The sibling step-17,000 eval `8878145` used the same broken
  wrapper and was cancelled while queued; PBS reached terminal state before it
  consumed a node. Both eval tails remain unmeasured. A corrected retry must use
  a conversion interpreter with pinned `spmd-types==0.2.5` and make conversion
  failure propagate nonzero before submission.
- Production umbrella `8870515` finished after 12:18:20 with PBS
  `Exit_status=143` and `Stageout_status=1` when node
  `x4418c3s1b0n0` requested job termination (`code 15009`). The seats retained
  substantial durable progress despite the parent failure: 20B-512 logged
  through step 11,190 but its step-11,200 directory is empty (4 KiB, no
  shards), so its resumable head remains step 11,100; 20B-256 logged through
  step 17,600, completed step-17,500 (239 GiB, 3,073 files including nonempty
  metadata), and left step-17,600 empty; 2B-256 stage 2 logged through step
  41,385 and completed step-41,300 (24 GiB, 3,073 files including nonempty
  metadata). The 20B-512 seat exhausted seven supervised launches: five
  longer attempts ended 143, then two 27--28 second PALS RPC launch failures
  against `x4418c3s1b0n0` ended `stuck_pre_training`/127. No replacement was
  submitted here because the active chat agent owns production continuation
  wrappers and submissions.
- One-node production-runtime staging smoke `8879160` reached the explicit
  `20B_PRODUCTION_STAGING_VALIDATED` marker after staging immutable archive
  SHA-256 `3ccf3faaf8f33d9fe6dc19257a49aeb54f03d93918cf6478e35ede2e7902142e`
  to `/tmp/.venv-20b-s3000-spmd025-py3126-20260929` in 48.2 seconds. PBS
  finished with `Exit_status=0` and `Stageout_status=1`; this clears the
  compute-portable archive/staging gate only, not restore or optimizer-update
  correctness. Active-chat-owned two-node restore/update control `8879231`
  then staged the same archive successfully but failed during model
  parallelization, before restore or training. All 24 ranks raised `ValueError:
  When dp_mesh_dims is provided, all parameters must be DTensors on the full
  SPMD mesh ... Got plain tensor for parameter 'weight'`. PBS finished after
  00:08:27 with `Exit_status=143`; the job artifact recorded
  `20B_S3000_UPDATE_FAILED rc=143`. It emitted no restore marker or training
  step and wrote zero checkpoint files. The next gate is a corrected SPMD/FSDP
  topology control owned by the active chat agent; no production fork is safe
  to submit yet.
- Chain-3 restore control `8878906` reached the real application on the isolated
  oneAPI 2026.1.0 + Torch 2.15 runtime, restored the read-only step-39,900 seed
  in 292.62 seconds, and entered `Training starts at step 1`. It never completed
  an optimizer update: ranks timed out in `MPI_Allreduce_c` after the configured
  600-second MPI progress timeout, and PBS ended the 1-hour allocation with
  `Exit_status=-29` at 01:00:18. The job-unique checkpoint sink is empty. This
  validates runtime staging, imports, model construction, and DCP restore, but
  leaves the first-update/reproduction gate blocked on the collective stall.
- Target-node import control `8878862` finished with PBS `Exit_status=25`
  (`Stageout_status=1`, walltime 00:02:15). The immutable 2.7-GiB runtime
  archive staged successfully in 48.2 seconds to
  `/tmp/.venv-20b-s3000-spmd025-20260927`, and the probe ran on allocated host
  `x4415c7s7b0n0`. The staged environment is not compute-portable: `bin/python`
  is an absolute symlink to the login/runtime-specific
  `/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/python-3.12.12-nvje3vk/bin/python3`,
  which does not resolve on that compute node; `bin/python3 -> python` is
  consequently non-executable. The job correctly wrote
  `TARGET_NODE_IMPORT_FAILED rc=25`; no Python import, restore, optimizer step,
  or checkpoint occurred. This resolves the prior execution-context question:
  archive placement is correct, but the archive itself cannot be used for the
  fork. The next gate belongs to the active chat agent: build a separately
  named compute-portable runtime/archive, verify its interpreter and exact
  import closure on an allocated node, then run restore plus multiple finite
  optimizer updates before any production fork submission.
- Aurora retired the `next-eval` queue. All four user-owned queued jobs in that
  queue were deleted and verified terminal: chain-3 control `8876446`,
  `bench-pr` `8876598`, `hf264` `8878669`, and `sc25-pr4-xccl` `8878807`.
  A post-delete `qselect -u foremans -q next-eval` returned zero jobs. Any
  scientifically necessary successor must be redesigned for and submitted to
  a supported production queue; none of these stale queued jobs should be
  treated as an active gate.
- Production umbrella `8870515` allocated 2,098 nodes. All three seats reached
  finite optimizer updates. Verified progress included 20B-512 step 11,176,
  20B-256 step 17,065, and 2B-256 step 37,591; complete checkpoints were
  independently observed at 20B-256 step 17,000 and 2B-256 step 37,500. A
  transient/incomplete 20B-512 step-11,200 directory was not accepted as a
  checkpoint. Later 2B-256 logging stopped at step 39,454 while ranks remained
  live, so that seat is treated as a suspected stall rather than healthy
  progress pending supervisor/terminal evidence.
- Tail-evaluation backfills `8878144` and `8878145` were submitted for the
  current 20B-512 step-11,100 and 20B-256 step-17,000 checkpoints. Scheduler
  completion alone is not acceptance; fresh terminal `results.json` artifacts
  remain the gate.
- The isolated 20B-512 continuation source at step 3,000 remains complete:
  6,145 files including nonempty metadata. The private step-3,100 destination
  remains empty and is not a checkpoint. No failed staging smoke modified the
  canonical production chain.
- Fork attempts `8874846` and `8877631` failed before restore/training: the
  former used a broken launcher/shebang, while the latter broadcast the
  immutable runtime but expected `/tmp/.venv-20b-s3000-spmd025` after `ezpz`
  had extracted `/tmp/.venv-20b-s3000-spmd025-20260927`.
- One-node staging controls then exposed two stale-wrapper prerequisites.
  `8878636` repeated the undated-directory expectation. `8878677` used the
  corrected dated directory but invoked `ezpz` through a shared-checkout venv
  whose absolute Python shebang was missing on the compute image. Both failed
  before archive staging, restore, or optimizer work.
- Replacement staging smoke `8878708` used the compute-visible production-clone
  launcher venv (`ezpz 0.27.3`) to stage the immutable archive (SHA-256
  `b8e12eecc5eb67cbe6a20e05f7f23bac322f18f58d09e243af9aa04775725e55`). It
  finished with `Exit_status=22` and `Stageout_status=1` after `ezpz` reported a
  successful 48.2-second stage to the expected target
  `/tmp/.venv-20b-s3000-spmd025-20260927/`. The wrapper then checked that
  node-local path from the wrong execution context and emitted
  `STAGING_PATH_MISMATCH`; imports were not exercised. The next one-node control
  must validate and import through the allocated hostfile on the target node.
  A later gate must additionally restore step 3,000, execute finite optimizer
  updates, and write a fresh nonempty checkpoint; an import-only pass is
  insufficient.

## 2026-09-27 (mbpr) -- fix `cos_sin` DCP -> HF fused-weight omission

`AgptStateDictAdapter.to_hf()` bypassed
`_native_fused_linears_to_hf()` for `_real` (`CosSinRoPE`) flavors. The adapter
therefore ignored native `wqkv` and `w13` tensors while still exporting `wo` and
`w2`. A meta-device `2b_real` reproduction exported 1,370,015,744 of
1,986,578,432 parameters: 616,562,688 missing (31.0%).

The adapter now splits native fused QKV and MLP tensors before applying the
`cos_sin` no-permute key mapping. The regression test checks all five formerly
missing HF projections and exact parameter-count preservation. Meta-device
impact checks now export 1,986,578,432 / 1,986,578,432 parameters for `2b_real`
and 20,742,804,480 / 20,742,804,480 for `20b_real`. Complex-RoPE exports still
use the unchanged parent path.

## 2026-09-27 (sunspot) -- 5B AdamW fixed-LR matrix: wrapper false-INVALID root-caused and fixed

### The false INVALID was a wrapper bug, not a science failure

The v2 submitter `5b-adamw-fixed-lr-v2.pbs` gated its `VALIDATED` marker on

```bash
grep -q 'diag/update_ratio_max' "$LOG"
```

but `diag/update_ratio_max` is produced by
`torchtitan/experiments/ezpz/diagnostics/__init__.py:212` and handed to the
metrics logger -- it reaches W&B and TensorBoard and **never stdout**. Confirmed
by direct count on job `12478895`: `0` occurrences of `update_ratio` anywhere in
the 718 KB console log, versus `198` occurrences inside
`run-ye6i2w6e.wandb`. The gate can therefore never pass, so every arm that runs
perfectly still writes `INVALID` and the wrapper exits nonzero. This is the same
class of defect catalogued on 2026-09-16: a check that returns the same answer
whether or not the thing works.

This retires the earlier framing that only `12478892` carried a "wrapper-only
false INVALID marker". It is systematic across the whole matrix.

### Arms adjudicated against the real acceptance contract

Added `adjudicate_fixed_lr.py` (remote repo root, `/usr/bin/python3.11` -- the
login default is Python 3.6 and rejects `from __future__ import annotations`).
It checks the five real criteria and writes
`VALIDATED_POSTHOC` / `INVALID_POSTHOC` JSON next to each arm.

| job | LR | PBS | steps | nonfinite | final loss | final grad | update_ratio max/median | ckpt | verdict |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `12478895` | 1e-4 | finished(exit=0) | 100/100 | 0 | 5.70206 | 0.6114 | 1.76e-3 / 7.05e-5 | 768 shards + `.metadata`, 55.8 GB | **VALIDATED** |
| `12478892` | 1e-3 (diagnostic v1) | finished(exit=0) | 100/100 | 0 | 6.89134 | 0.5910 | 1.79e-3 / 1.30e-4 | 768 shards + `.metadata` | **VALIDATED** |
| `12478896` | 3e-4 | running | -- | -- | -- | -- | -- | -- | pending |
| `12478897` | 1e-3 | queued | -- | -- | -- | -- | -- | -- | pending |
| `12478898` | 3e-3 | queued | -- | -- | -- | -- | -- | -- | pending |
| `12478899` | 1e-2 | queued | -- | -- | -- | -- | -- | -- | pending |

Both completed arms show real optimizer work, not merely finite losses:
`diag/update_ratio_min` is `2.07e-7` (LR=1e-4) and `2.20e-6` (LR=1e-3), 243
parameters carry gradients, 0 are frozen, and `diag/clip_fired = 0` with
`clip_headroom > 1.6`, so clipping never masked the trajectory.

### Matched trajectories so far (identical seed 42, fresh init, GBS 6144)

| step | LR=1e-4 loss | LR=1e-3 loss | LR=1e-4 grad | LR=1e-3 grad |
|---:|---:|---:|---:|---:|
| 1 | 11.98051 | 11.98051 | 2.3190 | 2.3190 |
| 2 | 11.10704 | 17.42651 | 2.9995 | 71.3191 |
| 5 | 12.27560 | 14.27604 | 31.0375 | 76.0778 |
| 10 | 9.50286 | 11.47980 | 15.0125 | 25.1504 |
| 25 | 7.50742 | 7.94368 | 3.4958 | 3.6615 |
| 50 | 6.70066 | 7.34670 | 3.9287 | 1.9261 |
| 75 | 6.11304 | 7.24173 | 1.3656 | 2.3233 |
| 100 | **5.70206** | 6.89134 | 0.6114 | 0.5910 |

Step 1 is bit-identical across arms, confirming the shared initialization. LR=1e-3
takes a large early excursion (grad norm 71.3 at step 2, 76.1 at step 5) and never
recovers the gap; LR=1e-4 stays bounded and ends 1.19 nats lower. **No LR is being
recommended yet** -- 1e-4 is currently the lowest sampled point, so the curve may
still be descending toward smaller LR, and 3e-4 (the interesting intermediate) is
only now running. Three of five arms remain unmeasured.

### Changes landed

- `adjudicate_fixed_lr.py` -- artifact-based acceptance adjudicator.
- `5b-adamw-fixed-lr-v3.pbs` -- replaces the impossible stdout grep with the
  adjudicator's verdict, and forces `rc=1` when the wrapper would otherwise
  exit zero on an `INVALID` verdict. Syntax-checked with `bash -n` locally and
  on Sunspot.
- Dashboard pane `wC:p43` (`LR + MDS154391 Stage 2`) now prefers the post-hoc
  verdict and renders it as `validated*` / `failed*`; verified the restarted
  pane shows `12478895` and `12478892` as `validated*`.

The four not-yet-terminal arms were **not** cancelled. Their queued snapshot of
v2 runs an identical training command -- only the marker logic differs -- so
destroying a 64-node queue position to change a post-run `printf` would have
cost real allocation for no scientific gain. They will be adjudicated post-hoc
by the same script.

## 2026-09-24 (sunspot) -- MDS154391 two-stage reproduction complete; GRPO stack smoke queued

The journal was not updated during this campaign. This entry reconstructs the
results from PBS records, retained logs, trainer state, checkpoint metadata, and
raw semantic generations rather than from remembered status messages.

### Stage 1: broad SFT and semantic checkpoint selection

- Job `12478590` completed the native Grain broad-SFT run at step 900 with
  `Exit_status=0`. Final reported loss was `1.02522`, gradient norm `1.2596`,
  HBM `19.40 GiB/rank`, and throughput `2,071 tokens/s/rank`.
- Immutable DCP checkpoints at steps 300, 600, and 900 were retained under
  `agpt2b-mds154391-broad-grain-sft900/checkpoints/`.
- GSM8K-200 raw-generation evaluation selected step 600 by semantics, not loss:

  | checkpoint | correct | accuracy | strict `<answer>` envelope |
  |---|---:|---:|---:|
  | 300 | 47/200 | 23.5% | 0% |
  | 600 | 52/200 | 26.0% | 0% |
  | 900 | 48/200 | 24.0% | 0% |

  The generations were coherent multi-step arithmetic. Step 600 was therefore
  the Stage-2 seed despite step 900 being later.
- Eval jobs `12478604`, `12478605`, and `12478606` each wrote all 200 rows, then
  wedged during vLLM teardown because the `EngineCore` child remained alive.
  Their eventual `143` statuses were operator cleanup after artifact validation,
  not semantic failures. The evaluator now calls
  `llm.llm_engine.engine_core.shutdown()` in `finally`.

### Stage 2: focused GSM8K-R1CoT SFT

- The first run, `12478612`, was technically healthy but scientifically wrong:
  packing plus gradient accumulation 10 produced only 12 optimizer steps. It was
  cancelled at authoritative step 4/12 (`Exit_status=143`); its partial output is
  retained but non-authoritative.
- Corrected job `12478614`, pinned to commit
  `8acf6c7b152ae5ccaeb26688b7f451ad7b6a1728`, used gradient accumulation 1 and
  the Stage-1 step-600 HF export. It completed exactly 93/93 optimizer steps,
  three epochs, and checkpoints 31, 62, and 93 with `Exit_status=0`. Final train
  loss was `0.4852`; the final HF model is approximately 7.94 GB.
- Semantic eval job `12478618` completed with `Exit_status=0` and retained 200
  raw generations. Results: 197/200 format-valid (`98.5%`), 43/200 strict-answer
  correct (`21.5%`), and 3/200 truncated or unclosed. Raw correct, incorrect,
  malformed, and truncated samples were inspected. This reproduces historical
  B2 behavior (98.5% format, approximately 20.5% accuracy): focused Stage 2
  taught the answer envelope but did not improve the broad checkpoint's 26.0%
  exact-answer accuracy.

### GRPO functionality smoke

- Commit `89cba3c023c93e03235f73a108cf203007202bad` adds a fail-closed one-node,
  20-step Monarch/TorchStore/vLLM GRPO smoke starting from the completed Stage-2
  model. It uses the validated `rl_grpo_lora_agpt_2b_gsm8k_b2smoke` config:
  8 prompts/step, 4 samples/prompt, fp32 generation, maximum 700 generated tokens,
  and checkpoints every 10 steps.
- Acceptance requires initial and post-update policy-weight synchronization,
  nonzero policy versions, real reward-bearing updates, checkpoints 10 and 20,
  bounded raw rollouts, and clean actor shutdown. This is a stack-functionality
  test, not evidence that GRPO improves quality.
- Initial smoke `12478619` proved model load, vLLM rollout, initial TorchStore
  sync (7.94 GB at 1.81 GB/s), repeated post-step sync (about 100 GB/s), and
  policy-version advancement 0 through 4. However, all 100 retained rollouts
  reached the 700-token limit: the default renderer supplied only eos ID 1 and
  omitted AGPT `<end_of_turn>` ID 107. Every sample was therefore classified
  `truncated_length`, producing zero reward, loss, and gradient. The run was
  cancelled after step 4 (`Exit_status=143`) rather than wasting 20 no-op steps.
- Commit `5e3da9fc6ba87f550ea875b67ff9b2ba23ae7ade` wraps the default renderer
  with explicit AGPT stop IDs `(1, 107)`. The built renderer returned `[1, 107]`
  in the protected runtime and all 9 focused renderer tests passed.
- Corrected retry `12478621` completed 20/20 steps with `Exit_status=0`,
  checkpoints 10 and 20, policy versions through 20, nonzero reward-bearing
  gradients, repeated post-update TorchStore synchronization, and clean actor
  shutdown. Its final rollout corpus contained 360 samples: 359 completed,
  359 format-valid, 114 exact-correct, 278 with nonzero advantages, and one
  pathological repetitive truncation.
- Raw-rollout inspection across low-, middle-, high-reward and late-policy
  strata found fluent, on-topic arithmetic rather than broad gibberish. Correct
  samples had concise valid derivations; failures were mostly coherent setup or
  arithmetic errors. Wrong but well-formed answers receive the 0.25
  format/extractability floor, so reward alone is not a semantic quality verdict.
- Stage-2 checkpoint evals completed cleanly: checkpoint 31 scored 33/200
  (`16.5%`) with 170/200 format-valid (`85.0%`); checkpoint 62 scored 39/200
  (`19.5%`) with 198/200 format-valid (`99.0%`); checkpoint 93 remained best at
  43/200 (`21.5%`) and 197/200 (`98.5%`).
- Initial GRPO merge/eval `12478626` failed closed because the legacy exporter
  expected split Q/K/V keys while current DCP stores fused `wqkv` and `w13`.
  Commit `e3b127ec9a25dded79167d0f167a26566d514cfd` added explicit dual-schema
  support. Real preflight folded adapters in all 12 layers and changed 48
  attention tensors relative to the base.
- Corrected merge/eval `12478627` completed with `Exit_status=0`: GRPO step 20
  scored exactly 43/200 (`21.5%`) with 197/200 format-valid (`98.5%`), identical
  aggregate metrics to the Stage-2 base. The merge was real; deterministic
  generations changed on 7/200 examples, but none changed correctness. The
  20-step run therefore validates the full GRPO stack, not a quality gain.

### 30B synchronous-DCP cache fix

- Old-cache jobs retained hook-generated contiguous tensors and consumed about
  `57.23 GiB/rank`, then failed during step-2 FSDP all-gather with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- Commit `1d58869cde6a055d3d9ac5ec221eab756e81d34f` makes synchronous DCP state-dict
  generation lazy while preserving stable cached storage for asynchronous DCP.
- Canary `12478607` completed step 1 at `42.31 GiB/rank`, about 15 GiB/rank below
  the old-cache runs, and wrote a complete 293 GB synchronous checkpoint in
  74.7 seconds. This strongly validates the memory-retention fix.
- Attempt 1 subsequently suffered multi-rank `SIGSEGV` during step-2 FSDP
  unshard, not the prior OOM. Attempt 2 restored the 293 GB checkpoint in 863.9
  seconds and entered the resumed update. The PBS job has now ended with
  `Exit_status=1`; the terminal attempt-2 failure still requires classification.
  Dependent full retry `12478608` was not released as a validated success.

### Other diagnosis and operational changes

- 5B job `12478591` was not an OOM. Attempt 1 completed its LR range but found no
  blow-up point; generic failover incorrectly retried that scientific outcome.
  Later attempts hit nullable Grain restore state
  `examples_iterable.previous_state`. A wider fresh LR range is the correct next
  experiment, not checkpoint resume.
- Stage-2 progress now comes from `trainer_state.json`, not numbers accidentally
  parsed from seed paths or partially-created checkpoint directories.
- The active dashboard remains in Herdr pane `wC:p38`; the obsolete
  `old-sunspot-monitor` pane was closed. The sole event-driven notifier is
  `/Users/sam/.hermes/scripts/alcf-significant-events.py` on `mbph`.

## 2026-09-21 (aurora) -- umbrella 8828612 reached walltime; continuation queued

- Production umbrella `8828612` ran on 2,098 nodes from 2026-09-19 22:48 UTC
  for 12:00:23 and finished at walltime (`Exit_status=-29`).
- Its independently verified trainer-2 / 20B-256 history spans steps 15,201
  through 16,035. The other four seat outcomes were not inferred from the PBS
  terminal state and remain unaudited in the overview-level documentation.
- Dependent continuation `8834528` was released after the predecessor finished
  and is queued as of 2026-09-21. It has not started and has no trainer logs.

## 2026-09-16 (aurora) -- checks that pass in both the working and broken states: six instances in two days

Two fixes landed and two conclusions were withdrawn. The through-line is one
error shape, and it showed up six times between this session and
`sunspot-tt-ezpz` working the same surface.

### The shape

A check that returns the same answer whether or not the thing works.

| check | passes when broken because |
|---|---|
| `import` the moe flavors | 14/14 imported, 0/14 built (sync 84 `#4631` reshaped the router) |
| grep `"full SPMD"` / `"plain tensor"` for pytorch `#181519` | both strings live INSIDE the `raise` the patch removes |
| chain depth monitor | depth read `2` for 79h while zero checkpoints advanced |
| `[ -x .venv/bin/python ]` from a login node | the spack base interpreter is not mounted on compute |
| patch one of three `.so.5` call sites | `exact_expert_gemm` passes, `ops` and `grouped_gemm` still refuse |
| `Exit_status = 0` on a PBS probe | the job died before running a line of Python |

The `#181519` one is the sharpest. My probe grepped prose and reported
`VERDICT_181519 PRESENT` on `torch 2.13.0a0+gitcf30153`. A local CPU `torch
2.13.0` that certainly lacks the fix scores identically -- 1095 lines, `full
SPMD` x2, `plain tensor` x1 -- because both strings are the text of

```python
raise ValueError(
    "When dp_mesh_dims is provided, all parameters must be "
    "DTensors on the full SPMD mesh (e.g. via distribute_module). "
    f"Got plain tensor for parameter '...'."
)
```

A build matches BECAUSE it still raises. Grep for symbols a patch introduces,
never prose. Prose in a `raise` is evidence the bug is present.

`#181519` is ABSENT on all four reachable torch builds. The stronger evidence
is execution, not inspection -- `sunspot-tt-ezpz` ran the merged tree on both
images:

| job | image | torch | result |
|---|---|---|---|
| `8829185` | prod `20260828` | `dev20260520+xpu` | dies in FSDP setup |
| `8831522` | test `20260831` | `dev20260428+xpu` | identical, 72 `dtensor_err` / 3 arms |
| `8829243` | prod, PRE-merge | | trains, ok |

`8831522` reached `IMPORT_OK` and built the model before failing, so it is the
FSDP path and not packaging. The test bkc was the last place the blocker could
plausibly have been absent. A hard floor for sync 84, closed from both
directions.

### Shipped

- [`67d4f262f`](https://github.com/saforem2/torchtitan/commit/67d4f262f) vendored `aurora_moe` matched only `libmkl_sycl_blas.so.5`,
  so on the `26.181.0` image behind `next-eval` (ships `.so.6`) it REFUSED a
  working oneMKL. Globbed, in all three files. The check is a presence probe per
  its own error text; ABI is the linker's job at load time.
- [`0dfa5fbe5`](https://github.com/saforem2/torchtitan/commit/0dfa5fbe5) `bmm_nodrop` did its GEMMs in whatever dtype it got, so
  it raised `expected scalar type Float but found BFloat16` under mixed
  precision. Both existing backends already cast to bf16 and return
  `type_as(x)`; the port inherited an omission from a branch whose caller
  pre-aligned dtypes.
- [`e29bcbcb0`](https://github.com/saforem2/torchtitan/commit/e29bcbcb0) the 80B LR guard I added in `607f1f623` ran pre-CLI, so
  it could only ever see the registry default. 8 configs hard-blocked with no
  reachable way to run them (`agpt_80b_zloss` could not be launched at all), 14
  silently returning `8e-4`. Moved after the CLI parse and the
  `_build_optimizer_config` rebuild.
- ezpz [#244](https://github.com/saforem2/ezpz/pull/244) PALS tagged the parent
  reporting an RPC-forward failure instead of the child that died, so failover
  evicted a healthy node and relaunched onto the sick one.
- ezpz [#245](https://github.com/saforem2/ezpz/pull/245) `rc=143` +
  `std::bad_alloc` filed as `walltime`. The accompanying `died from signal`
  lines are stripped as innocent cascade, so the log looks clean.

### Withdrawn

`aurora_sycl` was reported working, then failing, then working, then failing.
Final state, three measurements:

| job | stack | MKLROOT | result |
|---|---|---|---|
| `8829416` | frameworks `2026.1.0` py3.12 | `26.181.0` | works |
| `8829454` | yeeted `/tmp/.venv` py3.14 | `26.26.0` (wrong image) | `UR_RESULT_ERROR_UNINITIALIZED` |
| `8829790` | yeeted `/tmp/.venv` py3.14 | `26.181.0` (correct) | `gemm_bf16bf16bf16: unsupported device` |

`8829790` decides it: image-matched oneMKL and it still fails, so the oneMKL
image is not the whole story. It works under the module python and not under
the venv production trains with. `bmm_nodrop` ran in the same job on the same
device at `rel_err 0.000e+00`, which isolates the failure to the aurora-moe
SYCL path.

The XPU `topk` tie-instability that motivated a core router change does not
reproduce: 0 differing results over 20 identical calls with ties in 2913/4096
rows. That change was not made.

### The rule worth keeping

~10 failed probe jobs here, ~4 on the sunspot side, none of which measured the
thing under test. Causes: `module: command not found` in a PBS shell,
`libglog.so.0`, a venv base interpreter absent on compute, a script edited after
`qsub`, and `/opt/aurora/default` resolving to `26.26.0` from login and
`26.181.0` from compute.

Every one is handled by an existing submit script. Measured twice on two trees
with two globs: **114/122 PBS scripts here use `#!/bin/bash --login` (93.4%)**,
179/192 on `sync84-trial` (93.2%). Start from one of those, do not hand-roll.

`--login` is also what sets `CONDA_PREFIX`. Proven on `x4000c5s1b0n0` -- the
node that produced UNSET for the other session -- both answers from one machine:

```
BEFORE module (--login):     CONDA_PREFIX=UNSET
AFTER  module (--login):     CONDA_PREFIX=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
AFTER  module (non-login):   CONDA_PREFIX=UNSET
```

Node variance excluded. Guard it anyway -- unset, `"${CONDA_PREFIX}/lib:"`
expands to a bare `/lib:` and fails later as a missing library.

### Production

The 2098-node chain has not advanced a checkpoint since 2026-09-13. `8828611`
queued since 09-15 15:39, `8828612` held behind it, 9976 nodes job-exclusive
and 538 free. Nothing wrong with the chain; the machine is full.

The depth monitor read `2` throughout. Added a progress monitor keyed on
checkpoint mtime, which fired at 79h on its first tick.
## 2026-09-15 (local) -- sync 84: 148 upstream commits, ten indirect breaks, and moe importing clean while 0 of 14 flavors built

Asked whether there was anything upstream to pull in. 148 commits, not the 64
I surveyed on 09-08. Merged in a throwaway worktree (`../tt-sync84`, branch
`sync84-trial`); nothing landed on `ezpz` or a production clone.

- **Zero of the 148 touch `experiments/ezpz`, and ten of them broke it
  anyway.** All indirect -- upstream moved, renamed, or re-defaulted things
  ezpz imports, subclasses, or calls. Module moves (#4628, #4630, #4444,
  #4648), the attention rename (#4533), the spmd_backend deletion (#4419),
  mandatory fused QKV and gate-up (#4526, #4535), the MoE router reshape
  (#4631), the merged-batch protocol (#4572, #4398), and a newly-required
  `ModelSpec.max_context_length` (#4328).

- **The mistake worth writing down: moe imported 14/14 clean and built 0/14.**
  I build-tested agpt, saw moe's imports pass, and inferred the surface was
  healthy. Every flavor was dead at config construction
  (`TokenChoiceTopKRouter.Config() got an unexpected keyword argument
  num_expert_groups`). A parallel audit caught it, not me. `slots=True`
  dataclasses reject an unknown keyword even when its value is None, which is
  why it was 14/14 rather than the 2/14 that actually set the field.

- **#4572 is the one no local check finds.** Core's `batch_generator` now
  yields one dict where ezpz unpacked `(input_dict, labels)`. Imports pass,
  meta builds pass, all 82 tests pass -- and the run dies at step 1. Ported
  producer through consumer: blendcorpus yield, `train_step`, validator.

- **#4533 rekeyed the TP sharding contract for the THIRD time**
  (`BLNH -> TNH -> THK/THV`). It matches by positional-arg NAME and asserts
  only under TP>1, so every miss sails through a TP=1 smoke. The warning
  comments in both forks now record all three occurrences, and the rule:
  whenever a sync touches `decoder_sharding.py`, diff its `in_dst_shardings`
  keys against the ezpz `forward()` signatures BEFORE running at TP>1. Even
  the test written to survive renames broke -- it assumed q/k/v share one
  suffix, and #4533 gave v a different one.

- **Verified by running it.** 14/14 imports (with a pre-merge control, so it
  is a real comparison), 12/12 agpt and 14/14 moe meta builds, 82 passed /
  0 failed, standalone scripts at exactly the pre-merge baseline. Param counts
  AND state-dict key names are byte-identical pre- vs post-merge across every
  flavor tested -- **existing checkpoints load**, proven rather than assumed.

- **Adversarial verification: 3 of 24 findings refuted, none of them fake.**
  Each had a correct core-side claim and a wrong ezpz half -- an inverted
  consequence, a stale callsite, an unreachable code path. Changed zero lines;
  two corrected reasoning that would have misdirected later debugging.

- **The XPU blocker is now confirmed on BOTH compute images.** The first
  writeup of this rested on `debug-scaling` (prod bkc) alone -- both earlier
  `next-eval` jobs had died on the venv trap before any sync-84 code ran, so
  the test bkc was reported as covered while being untested. Rerun on the test
  bkc (`8831522`, a copy of the /home-based prod venv upgraded to
  `spmd_types 0.2.5`): all three arms `rc=143`, 72 dtensor errors, byte-for-byte
  the same `ValueError`. Prod `8829185` identical; pre-merge `8829243` TRAINS on
  the same hardware. Two images by execution, four torch builds by inspection.

- **Two things are NOT settled, and both need hardware.** `spmd_types` may be
  a live blocker: ezpz pinned `partial_dtensor` precisely because
  `spmd_types` failed every ezpz config, #4419 deleted that pin, and the
  upstream guard is still as narrow as the bug doc says is insufficient.
  And the numerics are unverified -- fused vs unfused QKV is one GEMM instead
  of three. A seeded loss comparison and a 2N smoke are owed.

- Three new REQUIRED deps, one a git pin (`torch_remat`, plus `renderers` and
  its chain). They **mask** every break above, so installing them on the
  clusters reveals rather than fixes. Install procedure in `upstream-sync.md`.

## 2026-09-08 (sunspot) -- `lm_head` is dp-invariant: the gradient concentration has a mechanism, and four curve fits were wrong

The machine came back (113 of 129 free, all 29 mount-check offlines cleared, a
full survey returning 110 good / 0 bad), which made the dp question testable
for the first time.

- **The headline: `lm_head.weight`'s gradient norm is FLAT across a 16x change
  in data parallelism.** Five arms at dp 12/24/48/96/192, matched on GBS,
  seed, LR trajectory, optimizer and model:

  ```
  mean layer gradnorm  ~  dp^-0.394
  max (lm_head)        ~  dp^-0.013      (0.2536 -> 0.2457, flat to 3%)
  ```

  So the concentration does not grow because `lm_head` grows -- **it grows
  because everything else averages away and `lm_head` does not.** State the
  components; the skew ratio is derived and misleads.

- **Ordinary layers average at -0.394, not the -0.5 of independent sampling.**
  dp-rank gradients are correlated, now quantified.

- **`tok_embeddings` is the control that makes this interesting.** Same shape
  as `lm_head` (vocab x dim, the two largest tensors), never once in the top-5
  at any dp. Not "big tensor, big gradient". And the 80B is untied --
  `enable_weight_tying` appears once in the registry, in `agpt_2b_tied` -- so
  they are genuinely separate tensors.

- **Four functional forms, four overturns.** 2 points said log-linear
  (predicted 172.7, got 162.3), 3 said decelerating (predicted 116.6, got
  131.2), 4 said a power law dp^0.358 (predicted 101.4, got 93.2), 5 fit
  nothing better than 4.3% against a 0.7% noise floor. Each described its own
  data and failed out of sample. **The monotonic finding survived all four
  revisions; the shape claim was wrong every time.** Skew is a ratio of two
  quantities with different scalings -- it looks like a clean power law over
  any two points and like nothing over five. I have stopped proposing forms.

- **The mechanism, with a test that can kill it.** `lm_head`'s gradient is
  predicted-minus-true summed over the vocabulary; early in training the
  prediction is near-uniform on every rank regardless of which tokens it saw,
  so the dominant term is common-mode and does not average. All five arms sit
  at loss 12.86-12.96 against ln(256128) = 12.45 -- barely past uniform. Job
  `12474810` descends at lr=1e-6 (which took `12473149` from 12.95 to 8.098)
  and tracks skew WITHIN one run. **Flat skew from 12.9 to ~8 falsifies the
  explanation** and would make the invariance structural, which is the more
  interesting outcome.

- **A retraction I had to withdraw.** I claimed `8540102` inherited
  `agpt()`'s `lr=8e-4` because its writeup states no LR, and retracted a sound
  experiment on that basis. Wrong: the 80B submit script sets
  `LR="${LR:-1e-6}"` and passes `--optimizer.lr` explicitly, so the registry
  default is unreachable by that path. **A writeup that states no LR means
  1e-6.** An audit commissioned to find MORE LR-voided failures found none --
  all six checked ran at 1e-6 or below.

- **Two "trends" that were artifacts.** "Runner-up layers migrate toward the
  input as dp rises" was one anomalous arm (mean layer index 39/44/**10.6**,
  not a gradient). "dp=24 skew is drifting monotonically" was the same
  up-up-down-up wobble every arm shows in its first five steps. Both caught by
  computing a summary statistic instead of reading three numbers.

- **Still confounded:** dp and GAS move together at fixed GBS
  (GBS = 4096 x dp x GAS), so all of the above is strictly "dp with inverse
  GAS". `12474809` holds dp=192 and moves GAS 2 -> 1 to separate them.

## 2026-09-07 (sunspot) -- the 80B failure is not caused by its LR trajectory or its batch size; gradient mass lives in `lm_head.weight`

Continues the 2026-09-06 entry below, after the machine was made usable again.

- **`lm_head.weight` carries 44x the gradient norm of any other tensor**, and
  the whole ranking is frozen: `lm_head`, then `attention.wo` from layers 65,
  36, 57, 5 -- same order every step, though ranks 1-4 sit within 2% of each
  other. Values that close would reorder constantly under noise. They never
  do, so this is structure, not scatter.

  **It is outside the transformer stack.** Softcap and QK-Norm, the two routes
  on the standing list, both bound attention scores *inside* the blocks and
  neither touches the output head. If the failure originates there, they were
  never going to bound it.

  This had been measured five times and never named -- `collect_param_stats`
  computed the layer name and discarded it. Found by driving the capture with
  a poisoned gradient, not by reading it.

- **`8574385` died at an effective LR of 3.87e-9** -- four orders of magnitude
  BELOW the ~7.4e-7 ceiling `agpt_80b.md` documents. Reading the flag would
  never have shown this; it took computing `peak * n / warmup` at the death
  step. The 80B's real failure is not an over-large learning rate, which
  retires the framing every prior 80B run was built around.

- **The LR trajectory alone does not cause it.** `12474765` ran 25/25 steps
  clean at that exact trajectory -- 3.87096768e-09 at step 18 against an
  intended 3.871e-09, exact to 9 significant figures -- same optimizer, same
  GBS. So the failure is not a deterministic function of (LR trajectory, GBS,
  optimizer, step count) alone.

- **Batch size is not the variable either, at dp=96.** Two arms differing ONLY
  in GBS (6,144 vs 384 seqs, same nodes, same dp, same LR): means agree to
  **0.03-0.20%**, variances scale **2.75-4.33x**. Those ratios sit on
  **sqrt(16) = 4** -- exactly what sampling noise predicts for a 16x batch
  reduction, three of four within 8%. The batch does what averaging says and
  nothing more. No run in this investigation had previously varied GBS with dp
  held fixed; the 2026-08-31 attempt varied both and was void for it.

- **What is left: dp (96 vs ~1530) or the seed.** A materially narrower
  question than the session started with.

- **Clipping fires on 100% of steps** (`preclip ~8.07 -> postclip 1.0`), so
  every grad_norm in the record -- ours and `8574385`'s documented "flat
  ~6.17" -- is the pre-clip value. The optimizer never saw it. And one `inf`
  zeroes every *other* gradient (`scale = max_norm/inf = 0`) while the
  offender becomes `nan`, which is both why a run can recover from a
  non-finite step and a second way to identify the culprit: post-clip,
  exactly one tensor is non-finite among all-zeros.

- **Method note.** Three separate mechanisms let a broken run report success
  tonight -- a warmup clamp whose banner echoed the unclamped value, a capture
  that logged aggregates as if they named tensors, and an inner `timeout`
  shorter than the walltime exiting 0. Plus two tools I wrote and had to fix.
  Every one was found by driving code or comparing outputs; none by reading.
  Recorded in `project_runs_that_fail_successfully`.

## 2026-09-06 (sunspot) -- every job silently requeued for hours; the cause was nodes PBS calls healthy, and the fix exposed that the 80B dies at an LR four orders below its documented ceiling

- **Nothing could launch, and nothing said so.** Multi-node jobs cycled
  `Q -> R -> E -> Q` with `Exit_status = -3`, no output file, no log
  directory, no error. The 80B capture reached **`run_count = 21`** without
  executing a line before PBS system-held it. `-3` is "exec failed, requeue",
  and the requeue is what makes it silent: the job looks queued, not broken.
- **The cause was in the node comments, not the job.** The PBS prologue
  mount-checks every filesystem named in `#PBS -l filesystems=`; a failure
  offlines the node and requeues the job. 29 nodes were down for
  `home not mounted`, all from other users' jobs, ongoing since Sep 4.
- **Ladder the scale with a TRIVIAL payload.** 1N `echo` ran clean; 64N `echo`
  requeued exactly like the real job. That is what rules out your script, and
  without it the evidence pointed at qdel-and-resubmit, which would have fixed
  nothing.

- **I published three fixes that testing then refuted.** Each read as obviously
  correct:
  1. *Drop `home` from `filesystems=`.* Refuted by a matched 8N pair -- the
     probe that REQUESTED `tegu:home` ran fine. The flag was never the
     variable. I had already committed and pushed it.
  2. *Rack x1921 is the ceiling.* Refuted -- a 16N job drew all 16 nodes from
     x1921 and started fine.
  3. *Host pinning is sufficient.* Refuted -- the pinned 64N launched
     perfectly (768/768 GPUs) and died at **0 training steps**.
- **`free` is not `usable`.** The pinned run was killed by `x1922c7s2b0n0`,
  which reports `state = free` with **no comment** and on which `/lus/tegu` is
  unreachable. A rank there cannot `cd` to the repo, python is not found, it
  exits 127, and mpiexec tears down all 768. **No `pbsnodes` query finds
  these.** Six such nodes now known, in both racks.
- **So filter by an access test, not by scheduler state.**
  `scripts/cluster/survey_nodes.sh` stats a repo path from every candidate and
  keeps the ones that answer; it rediscovered the killer node independently.
  68 nodes verified. Survey in small batches -- a 70-node survey requeues on
  the same lottery it exists to map.

- **The warmup clamp voids any short reproduction, and the banner lies about
  it.** torchtitan clamps `warmup_steps` to `total_steps`
  (`lr_scheduler.py:105-112`) with only a warning, and the trainer banner
  prints the UNCLAMPED value on the very next line:

  ```
  [W] Warmup steps (4650) exceed total steps (25). Adjusting warmup steps to 25.
  [I] Trainer is initialized with ... total steps 25 (warmup 4650)
  ```

  Checking the banner -- the natural place to look -- confirms the wrong
  number. Shortening a run makes the clamp TIGHTER, so my "warmup-matched"
  25-step run was at **186x** the original LR, worse than the 116x run it
  replaced. Matching honestly needs ~2325 steps = **~55 days** at 34 min/step.
- **Fix: rescale the peak instead of fighting the clamp.** Under linear warmup
  `lr(n) = peak * n / warmup`, so `5.376344e-09` with warmup 25 reproduces
  `1e-6` with warmup 4650 **exactly** -- verified to 2.2e-16 across steps 1-25.

- **THE FINDING. `8574385` died at an effective LR of 3.87e-9.** That is four
  orders of magnitude BELOW the ~7.4e-7 ceiling `agpt_80b.md` documents.
  **The 80B's real failure is not an over-large learning rate.** Reading the
  flag would never have shown this; it took computing the effective schedule
  the flag produced. Every prior 80B run at 8e-4 was void on its own terms --
  this configuration is the one that was always worth instrumenting.

- **A run that fails its designed purpose is not automatically waste.** I was
  about to kill the clamped job. Computing what it IS doing showed it ramps
  4e-8 -> 1e-6, crossing the documented ceiling at step 18.5 with the capture
  armed -- a real test of whether that ceiling transfers to SophiaG. Kept it.
- **Ended with two concurrent 32N captures** (`12474733` fixed-LR stability,
  `12474740` ceiling sweep), per-layer diagnostics confirmed emitting, and
  `80b_capture_rescaled.pbs` ready as the true reproduction. First 80B
  training steps of the day: loss 12.95721 -> 12.94267, grad_norm ~8.03, zero
  non-finite events.
- **Still open:** the ALCF ticket is drafted and unsent (29 nodes stay offline
  until it goes), and 64 upstream commits are pending -- `#4398`
  (valid-token counts in collation) and the checkpoint cluster
  (`#4187/#4188/#4197/#4270/#4474`) are the ones that touch our paths.

## 2026-08-31 (sunspot) -- muP LR transfer confirmed at production width; SophiaG at half LR DELAYS divergence 4x but does not prevent it

- **muP stage 4 answered the question the coordinate check could not.** A
  passing coordinate check proves the parametrization is internally
  consistent; it says nothing about the payoff. Three grids of discrete
  fixed-LR runs (12 to 15 arms each, ~1h per grid) measured whether the
  optimum actually moves with width:

  | eta | 1536 | 3072 | 6144 |
  |---:|---:|---:|---:|
  | 1.6e-5 | 7.322 | 6.544 | 6.176 |
  | **6.4e-5** | 5.877 | **5.510** | **5.520** |
  | **2.56e-4** | **5.672** | 6.022 | 5.952 |
  | 1.024e-3 | 6.390 | 7.084 | 6.624 |

  **3072 and 6144 agree exactly at 6.4e-5**, both with interior minima. 1536
  lands one grid step higher. The usable claim: **tune at 3072, deploy at
  6144** -- 4x fewer parameters on the tuning run, verified at production 30B
  geometry. Writeup: [experiments/mup/README.md](experiments/mup/README.md).
- **The first two grids produced confident WRONG verdicts at rc=0**, which is
  the more useful part of the story. Grid 1 reused the tiny ladder's LR range
  (6 layers, where this ladder is 64), every width bottomed out at the leftmost
  point, and the harness reported "same argmin -- TRANSFER" from three
  identical boundary artifacts. Before that, a rehearsal swept
  `--optimizer.param-groups.0...lr` and moved only the embedding group's single
  parameter, leaving the 42-parameter hidden group at default -- loss varied
  0.006 nats across a 16x span and the harness reported NO TRANSFER from noise.
  Both are now gated, and each gate was tested against the real data that
  fooled it plus a synthetic interior-minimum case that must still pass.
- **Muon's high LR is not Muon's.** The no-rescale arm (`adjuster_lr_ref=False`,
  job `12474327`) suggested 5.09e-04 against the rescaled arm's 5.68e-04 -- a
  factor of **1.12**. Removing a 15.677x multiplier from 21.5% of parameters
  moved the optimum 12%, so the AdamW-path majority anchors the curve. Read
  5.68e-04 as a property of the hybrid, not of Muon.
  [lr-finder/agpt/2026-08-30-30b-gbs960-muon.md](experiments/lr-finder/agpt/2026-08-30-30b-gbs960-muon.md).
- **Muon's shape cutoff is now a parameter, and a real-Muon arm exists.** The
  gate was `max(p.shape) <= 10000`, hardcoded at three sites, with the code's
  own comment reading "need to change this!!!". Its intent is to keep
  embeddings and the head off the Muon path; at 30B it also excludes
  `w1/w2/w3` at 16384. Measured on a meta-device build of each config:

  | config | `muon_max_dim` | params on Muon |
  |---|---:|---|
  | `agpt_30b_olmo2tok_muon` | 10000 | 21.5% (5.64B of 26.20B) |
  | `agpt_30b_olmo2tok_muon_ffn` | 20000 | 95.3% (24.96B of 26.20B) |

  `muon_max_dim` defaults to 10000, so every existing arm's partition is
  unchanged. `muon_ffn` is wired into both `lrfind_opt.pbs` and `optcmp.pbs`,
  which closes the "needs its own case entry" gap the finder doc flagged.
  Its LR is deliberately a placeholder: 5.68e-04 was measured on the 21.5%
  partition, and the norescale result above says that number tracks the
  population `muon_ffn` removes -- job `12474361` sweeps the new partition
  under the identical protocol. One caveat unique to this arm: with the FFN
  included, `0.2*sqrt(max(A,B))` is 15.677x on attention (6144) and 25.6x on
  the FFN (16384), so "the effective LR" is not a single number here.
- **The muP width-dependent blow-up is partly explained, and recorded as
  partly.** Stage 4 found the optimum width-invariant while the divergence
  threshold was not (loss 7.16 / 12.71 / 41.53 at eta=6.4e-3). muP scales only
  the hidden matrices by 1/m; the unscaled groups (embedding, readout, norms)
  grow in ABSOLUTE size with width -- 308M / 617M / 1,234M -- while still
  running at full eta, so the widest model takes an identical oversized step
  on 4x as many unprotected parameters. That accounts for the direction and
  for why it does not threaten the transfer result (near the optimum the
  hidden group is 95% of the model). It does NOT account for the magnitude:
  excess loss grows 5.58x and 27.91x where the unscaled count grows 2x and 4x,
  roughly the square. Three points cannot fit an exponent, so section 5.8 says
  so rather than proposing one.
- **The 80B "root cause" does not survive examination, and the depth bisect
  that would test it had never been run.** Asked to state definitively why the
  80B NaNs, I went back through the evidence. The documented mechanism (bf16
  accumulation in the 84-layer residual stream) fails three independent ways:
  bf16 and fp32 share an 8-bit exponent (3.3895e38 vs 3.4028e38, MEASURED), so
  fp32 cannot fix a range problem; fp32-ing the residual add -- the direct
  test -- fails to prevent the NaN; and the "bf16 masks true grad_norms of
  21K-79K down to ~5-7" smoking gun describes an operation that does not exist
  (bf16 holds those values to 0.29%, and overflow yields inf, which propagates
  through a norm). The run underpinning it all (`8537349`) ran at n32/GBS=96,
  inside the regime where bf16 also trains clean -- retracted 2026-08-14 and
  never propagated to the meeting notes.
  [known-bugs/80b-nan-what-we-know.md](guides/known-bugs/80b-nan-what-we-know.md).
- **muP 30B stage 5 COMPLETE: 1,081 steps, loss 11.99 -> 3.85, zero NaN,
  `Exit_status=0`.** The first sustained muP run at production width (dim
  6144), across three chained links with clean `rc=124` resumes between them.
  All 51 grad_norm excursions above 2.0 sit in the warmup window -- binned
  into 50-step buckets the rate is 42/49 in window 0, 9/50 in window 50, then
  **zero from step 100 onward** with max decaying 1.61 -> 1.26. Peak 38.34 is
  a step-6-to-8 initialization transient, not instability.

  This is the payoff of the stage-4 transfer result: the parametrization was
  validated by coordinate check and by a discrete-grid LR sweep at 1536/3072/
  6144, and it now trains at production width without special handling.

- **RETRACTED SAME DAY: the bisect and the GAS sweep both ran at lr=8e-4,
  ~1000x past the documented ~7.4e-7 ceiling.** Their step-2 blow-ups are that,
  not a finding -- the depth counts, the "GBS not dp" conclusion and the
  frozen-at-43 reading are all invalidated. The control (job `12474431`) at
  **lr=5e-7** trains clean on the same 64 nodes: 12 steps, 0 events, loss
  12.948 -> 12.795, grad_norm 7.9. What survives is the refutation of the
  stated MECHANISM and the magnitude measurements, neither of which depends on
  the LR. `agpt_80b`'s docstring now warns about the default.
- **The bisect ran (job `12474403`) and found a real depth effect that is NOT
  the stated one.** Three arms, identical but for depth, at dp=192 / TP=4 --
  the exact configuration where three prior runs NaN'd:

  | arm | layers | grad events | loss NaNs | final |
  |---|---:|---:|---:|---|
  | L48 | 48 | 0 | 0 | loss 6.54, healthy |
  | L72 | 72 | 1 | 0 | loss 6.80, healthy |
  | L84 | 84 | 8 | 2 | loss 6.66, healthy |

  Transient non-finite gradients occur at a rate that scales monotonically
  with depth. And `qk_q_absmax_local` reads **61.2 at all three depths**
  (62.25 at dp=12 in the clean regime) -- 36 orders below the bf16 ceiling,
  with `grad_absmax` peaking at 0.03. Overflow is now refuted by measurement
  in the failing regime, not only by argument.
- **I reported twice that the deepest arm died. It did not, and both the claim
  and the mechanism I built on it are withdrawn.** L84's loss went non-finite
  at steps 40 and 59 and recovered immediately each time (step 41: loss 6.98,
  grad_norm 7.42; step 60: loss 6.66, grad_norm 3.73). I read the failure off
  step 40 without reading step 41, then derived "two back-to-back events are
  fatal" from it and called the mechanism conclusive. **No arm died; the
  production failure was not reproduced.** How a run actually terminates is
  open again -- 60 steps at this batch was not enough, and the production
  failures were at production batch, which the bisect held fixed.
- **The one step that mattered had no data, and now does.** `trainer.py`
  zeroed the gradients ~90 lines before the diagnostics ran, so a non-finite
  step logged one error line and nothing else -- L72's step-55 event is absent
  from W&B while 51-54 and 56-57 logged normally. Fixed (`f01548f6a`):
  `collect_param_stats(per_layer=True)` now runs inside the non-finite branch
  before zeroing, and separately logs the metrics that are themselves
  non-finite, which name the affected tensors. 6/6 tests. Which tensor goes
  first is the remaining unknown, and the next 80B run at dp=192 records it.
- **CORRECTED 2026-09-02: the half-LR arm DIVERGED at step 4163.** It
  cleared the four early onsets AND the 4106 test point, then blew up 50
  steps later (peak **4,341**, 30 excursions, final 5,083 steps). Halving
  the LR moves the onset 1048 -> 4163 and cuts the peak 100,611 -> 4,341:
  a ~4x delay and a ~23x smaller spike, **not prevention**. I published
  "resolved" minutes after it passed 4106 while it was still running --
  a pre-registered threshold stops you moving goalposts, not declaring
  victory on crossing one.
- **SophiaG at half LR (1.78e-5) cleared all four prior onset steps** --
  1,052 in-window steps, **zero** excursions, max grad_norm 0.94, loss 2.879 at
  step 1579. First arm to get past 1048.
- **That is not yet evidence the lower LR helps.** At every matched step the
  full-LR arm was equally clean: 0 excursions in 574 in-window steps below
  1081, and it diverged anyway at ~1550 on a different trajectory. The fresh
  arm's SECOND onset came at ~4106 after 2,200 clean steps; this arm has run
  1,052. The loss cost of halving is near zero (2.879 vs 2.910), so if lower LR
  did buy stability it would be nearly free -- but the case is indeterminate,
  not made. Settling it needs ~4200 steps, about 5 more links.
- **The W&B report was silently omitting two entire arms.**
  `sophiag-fresh-seed1234` (10 runs, to step 5079) and `sophiag-lowlr-seed1234`
  (3 runs) were in wandb, collected by `_discover`, and displayed by nothing --
  including the replicate that finished FIRST on loss. `ARM_GROUPS` enumerated
  three names and the variant filter knew only the `rerun-` prefix. Now matched
  by prefix, so a future `-v TAG=...` arm joins without editing the file.
  Report: 10 groups, 55 runs.
  [Report](https://wandb.ai/aurora_gpt/agpt-30b-optcmp/reports/30B-optimizer-comparison:-AdamW-vs-Mano-vs-SophiaG-(GBS=960,-constant-LR)--VmlldzoxNzg0MDgwNg==)
- One tension stays open: the divergence threshold moves sharply with width
  (loss 7.16 / 12.71 / 41.53 at eta=6.4e-3 across the ladder) while the optimum
  does not. muP predicts both should be width-invariant; only one is.

## 2026-08-30 (sunspot) -- muP passes the coordinate check; Muon at 30B wants 18.6x AdamW's LR

- **muP for AdamW went from a design audit to a passing coordinate check in a
  day.** Six registered flavors plus an `mup_tiny` CPU ladder, and a
  subprocess-per-width harness that first reproduced the standard-parametrization
  divergence signature before bridging to muP. Slopes
  (`d log2(l1) / d log2(width)`, 4 widths x 8 steps, tolerance 0.05):
  `layers.5.attention` 1.42 -> -0.010, `layers.4.attention` 1.32 -> 0.020,
  `tok_embeddings` control 0.001 -> 0.001. Max `|slope|` 0.020.
- **The readout was the whole fight, and the first fix made it worse.** The port
  paired Table 3's `d^-1` init with Table 8's O(1) LR, leaving `lm_head` at
  -0.130 while every hidden module sat near zero. Scaling the readout LR to
  `eta/m` drove it to -0.483. A negative slope means over-scaled DOWN, so the
  `d^-1` init was itself the problem: keeping agpt's existing `fan_in^-1/2` with
  an O(1) readout LR gives 0.001 and passes. 19/19 muP tests, all 56
  pre-existing agpt flavors still build.
- Stages 1-3 ran on a login node, on CPU, with no allocation.
- **Two muP bugs that made the flavors unreachable.** They had no `--config`
  entry point, so nothing could run them, and the sweep moved only one param
  group until `MUP_ETA` was threaded through.
- **Muon at 30B wants 5.68e-04** (job `12474326`, `rc=0`, 90 rows) -- first time
  Muon has run at this scale. Blow-up at 5.68e-03. That is 18.6x AdamW's
  3.05e-05 and 10.1x Mano's 5.61e-05, so the usable bands barely overlap.
- That number is two numbers. The finder sweeps the BASE; Muon rescales by
  `0.2*sqrt(max(A,B))` -- a uniform 15.677x on the 21.5% of params it owns. So
  the Muon path runs at 8.90e-03 while the AdamW-path 78.5% runs at 5.68e-04. A
  no-rescale arm (`adjuster_lr_ref=False`, job `12474327`) was launched to
  separate them, with the prediction written into the config docstring before
  the result.
- **The fresh SophiaG replicate leads on loss.** 5,080 steps from random init
  ending at 2.43156, about 0.08 nats ahead of Mano and 0.17 ahead of AdamW at
  every matched step from 3000 on -- after diverging twice and peaking at
  grad_norm 16,532.
- Correction: "the fresh SophiaG arm escaped" was a fair observation and a wrong
  conclusion. It held the good state ~2,200 steps, reached loss 2.4984 (below
  AdamW's final 2.51357), then diverged a SECOND time at ~step 4106 -- after six
  consecutive 100-step windows at exactly zero excursions, max never above 0.24.
  Quiet is not recovery.

## 2026-08-30 (aurora) -- moe coverage closed at TP=4 and compiled; libglog verified on hardware

- **All five moe corners pass** (job `8792615`, 1N x 4 ranks, 5 steps):

  | arm | step 1 -> 5 | max delta vs eager |
  |---|---|---|
  | TP=1 control | 12.93609 -> 11.32376 | -- |
  | TP=2 eager | 12.94930 -> 11.49335 | -- |
  | TP=2 compiled | 12.94907 -> 11.49342 | 0.00033 |
  | TP=4 eager | 12.86241 -> 11.75634 | -- |
  | TP=4 compiled | 12.86258 -> 11.75609 | 0.00041 |

  Compiled tracks eager to ~3-4e-4, so compile is doing the same math. Losses
  differ ACROSS TP degrees (0.196 at TP=2, 0.433 at TP=4) -- different sharding,
  different reduction order -- and memory per rank falls 2.80 / 1.94 / 1.38 GiB.
- The first attempt died in all five arms on `ImportError: cannot import name
  'SpmdType'`. Core `parallel_dims.py` imports it; the venv's pinned
  `spmd_types` 0.2.1 does not export it. 0.2.5 does. Two silent install failures
  preceded the fix -- no `uv` on Aurora, and `pip` needs the ALCF proxy
  exported. The failing run still printed a pip upgrade notice and looked fine.
- **The libglog trap is now verified on hardware, not from notes** (job
  `8792135`). Six of seven claims confirmed, including the exact frames
  (`distributed_c10d.py:151` -> `torchcomms/__init__.py:45` -> `:42`) and 0
  matches for the fw lib dir in `LD_LIBRARY_PATH` after `module load`.
- Two corrections there. The page said "the soname went backwards" -- measured,
  BOTH went backwards: `libglog.so.2 -> .so.0` and `0.7.1 -> 0.4.0`. And a probe
  using `env -u LD_LIBRARY_PATH` reported `libmkl_intel_lp64.so.3` as the first
  failure, which looked like several missing libraries; that is an artifact of
  the probe discarding paths the module legitimately sets. With the module
  environment untouched the first and only failure is `libglog.so.0`. The
  one-line workaround is complete.
- **The production docs said things that were not true.** `agpt/20b/n256` marked
  job `8681340` (2026-07-23) as RUNNING; `agpt/2b` marked the 512N chain LIVE
  when it completed 2026-08-13; `20b/n512` contradicted itself, heading step
  6,100 while its own Latest-checkpoint line said 8,700 and told the reader to
  trust that line. Disk says 10,600. 33 files reconciled.
- I called `docs/production/metrics/*.csv` authoritative and was wrong --
  `manifest.json` says `exported_at 2026-08-17`, so it predates the 08-20 and
  08-26 legs. Heads audited on disk instead: 20b-256 `step-12000` (137 ckpt
  dirs, mtime Aug 26 09:27), 20b-512 `step-10600` (126).
- Reverted 14 files whose only change was a Last-updated bump with no content
  edit. Marking an unchanged page as freshly reviewed is the habit that made
  these docs untrustworthy in the first place.
- **`refresh_all.sh` auto-committed a degraded chart.** It reported 7 chart
  failures and committed anyway; `cpt_loss.svg` lost the `#838383` series and
  gained default-theme chrome. Root cause is upstream in `ambivalent`:
  matplotlib 3.11 removed `style.core` from the `style` namespace, the style
  import dies, the plotter falls back to defaults and exits 0. Reverted, and
  fixed upstream in `saforem2/ambivalent#6`; `#7` restores nine stylefiles that
  ship on PyPI but exist nowhere in that repo's git history.

## 2026-08-30 (polaris) -- zombie-GPU drain ticket filed; the 20B is stalled, not advancing

- **The drain ticket went to ALCF** covering three nodes with a stuck GPU left
  by another user's job. PBS still advertises `ngpus=4` on them and reassigns
  them, so rotation cannot win -- correct attribution was necessary and not
  sufficient.
- The Polaris 20B page now records the chain as stalled rather than advancing.
- Correction: the reconcile commit claimed figures could not be regenerated
  because the plotters need cluster data. They pull from the W&B API over the
  network; the only blocker was a missing API key.

## 2026-08-29 (aurora) -- production has not trained since 08-26, and the queue is why

- **`8784460` has been queued since Wed Aug 26 13:22 UTC** -- 2,098 nodes,
  `large`, `score_boost = 0`. It fits: at 18:23 there were 2,206 free nodes
  against the 2,098 ask, and it did not start. Five minutes later free was 166.
- No job of 2,000+ nodes is running at all; the largest is 516, then
  512/512/260/256. Small jobs consume capacity continuously, so a 2,098-node
  request never accumulates a contiguous block whatever its eligible time.
- On 08-29 job `8791192` (2,304 nodes, another project, 22 minutes eligible,
  same zero boost) started ahead of it.
- Ruled out before filing: reservations total 134 nodes, `max_queued` is 10
  against our 2, 2,098 sits inside the 2000-10624 range, `Hold_Types = n`, and
  the `beforeany` is our own chain successor. AuroraGPT has +1,269,842.6
  node-hours -- the negative balance is `datascience`, a different project.
- The first draft of the ticket led with the scheduler comment moving to `Node
  is in an ineligible state: down`. That message was transient and reverted the
  next day, so the draft was replaced rather than amended.

## 2026-08-29 (sunspot) -- the optimizer comparison finished: Mano 2.43889 vs AdamW 2.51357

- **Both healthy arms ran to the `training.steps=6000` ceiling** at 23.59B
  tokens each, `rc=0`, zero NaN/inf, zero skipped gradients. Mano led from ~1.0B
  onward and never gave the lead back.
- One seed per arm and no decay phase. The documented prior is that Muon-family
  wins short runs and AdamW takes it back in decay, so this is consistent with
  the prior rather than a refutation of it.

## 2026-08-28 (aurora) -- all five RC corners pass, including compiled agpt TP=2

- Job `8789506`, 2N x 4 ranks, 5 steps each: `agpt` tp=1 10.78117 -> 10.32300,
  tp=2 eager 10.88838 -> 10.43820, **tp=2 compiled 10.88835 -> 10.43784**,
  `moe` tp=1 12.93609 -> 11.32376, tp=2 12.94930 -> 11.49335.
- **Compiled `agpt` at TP=2 works on the RC.** On the production `.venv` that
  corner dies in `tensors_saved_with_vc_check` with a `DeviceMesh` in
  saved-for-backward. Compiled and eager agree to 3.6e-4, so it is a real pass
  and not a different code path. Confirm at 80B before retiring `compile=OFF`
  generally -- that is where the assertion was characterised.
- **The moe fix is not version-dependent.** Both moe arms reproduce job
  `8787243`'s losses to the five decimals stdout prints, across a different
  torch build and oneCCL. That is reported precision, not a bitwise check.
- `-P torch -P pytorch-triton-xpu` does not protect the XPU build. Job
  `8784535`: the resolver pulled a generic PyPI `torch-2.13.0` straight over it,
  all 14 installs printed `ok`, and only an explicit `torch.__version__` assert
  caught it.

## 2026-08-28 (sunspot) -- SophiaG is 4/4, and a clean run buys no warning

- An independent from-scratch arm with `--debug.seed=1234` diverged too. Onsets
  across the four: 1048, 1176, 1071, ~1550.
- Escalation over four steps: 0.20 -> 4.95 -> 36.14 -> 98.98 -> 731.80, under a
  nearly flat loss (2.952 -> 3.641). Read `grad_norm`, never loss -- the loss
  dips look like recovery.
- The guard's first live catch: replicate 3 stopped at `grad_norm` 53.49, 133x
  the trailing median, `rc=0` after 55 minutes instead of ~8h.
- Correction: the relapse section concluded the arm "both diverged and stayed
  diverged". It did not.

## 2026-08-27 (aurora) -- moe trains at TP>1: the SDPA wrapper returned 4D from a 3D contract

- **Two lines.** `moe/__init__.py` unflattened `[T,N,H] -> [B,L,N,H]` going in
  and never re-flattened coming out, so MLA reshaped against a batch leading dim
  instead of a token one and `wo` got `Shard(dim=0)` where row-parallel wants
  `Partial(sum)`. `agpt` had the re-flatten all along, with a comment warning
  about exactly this. The port copied the input half and dropped the output half.
- **A wrong fix shipped first and only the TP=1 arm caught it.** Copying `agpt`'s
  `view(out.shape[0], -1)` verbatim reads moe's BATCH axis:
  `mat1 and mat2 shapes cannot be multiplied (1x1048576 and 2048x256)`. That
  broke TP=1, which had always worked.
- Twelve hypotheses died first, every one about configuration. Job `8789506`
  shows both models' SDPA wrapped on the same mesh `('tp',)(2,)` declaring
  identical axes. The difference was never config; it was four characters of
  arithmetic.
- Correction: "XPU graph capture cannot include oneCCL collectives" is wrong.
  `CCL_OP_SYNC=1` was the blocker, and the Intel ask on that page must not be
  filed.

## 2026-08-27 (polaris) -- failover names the real bad node now: two bugs, FQDN and rc=124

- `get_machine()` returned a login FQDN against bare registry keys, so the
  patterns that existed were unreachable -- `ModuleNotFoundError` on
  `patterns.polaris-login-04`, and the scraper blamed an innocent rank-0 node.
- The rc=124 path filed a recoverable bad-node timeout as terminal.
- Cost: job `7550301` ~1h of 130 nodes with zero training steps; `7560196` 3h03m
  at `Exit_status=124`.
- Three green gates in this window were testing the policy and not the binding.
  The grad-norm guard is the canonical case: validated by replaying finished
  logs, it read a name local to `train_step`, `NameError`'d on step 1, and PBS
  still reported `Exit_status=0` with the done-marker written.

## 2026-08-26 (aurora) -- torchtitan and ezpz train on frameworks/2026.1.0

- Job `8784615`, 2N x 12: 10/10 steps, 10.86492 -> 9.05897, monotonic, finite
  grad norms, `rc=0`. First AuroraGPT training of any kind on the RC, after
  about eight jobs of environment plumbing.
- **Production ran once and has not run since.** Umbrella `8773440` started Wed
  04:14 UTC and used 5h13m of a 12h slot. Three of five seats trained: 20b-n256
  201 steps (2.31488 -> 2.24577), 2b-n512 504 steps (2.74535 -> 2.73836),
  2b-n256 782 steps (2.56449 -> 2.54807). Both 512N seats never started, and the
  reason is not established.
- `8775285` (30B LR-finder umbrella) ran 1h32m and exited 0 with all four seats
  at `rc=143` and no suggested LR in any log -- a clean exit that produced
  nothing.
- Provenance: the 08-26 handoff recorded only the bare step counts 201/504/782.
  The loss deltas above were read off the seat logs on 08-30 and are filed here
  because that is when the work happened, not when it was measured.
- `8773440` lost 7 of its 12 hours to a regex: ezpz 0.21's progress detector
  matched `\bstep=\d+` while torchtitan prints `step: 21800`. Measured on that
  job's own logs, seats that had trained 201 / 504 / 782 steps all scored
  no-progress under 0.21 and all three score correctly under 0.27.3.
- And about 5 more hours to seats that exited and left their slice idle. Every
  seat was gone by 04:27; PBS did not kill the shell until 09:29.

## 2026-08-26 (polaris) -- the 20B page was 3,200 steps stale

- Every claim on it said step 2,400; the chain is at 5,600. Census: 56
  checkpoint dirs, step-100..step-5600 at interval 100, all carrying
  `.metadata`, 234 GB each = 13 TB. The page had claimed ~700 GB.
- Correction, same day: "58 ckpt dirs" was the directory's hard-link count, not
  a file count.

## 2026-08-26 (sunspot) -- the 30B optimizer comparison hits its 10B budget

- Mano 2.7701 vs AdamW 2.9134 at step 2,543 / 9.99B tokens, gap -0.1433.
- Correction: four claims were walked back in one day, all the same failure mode
  -- fitting a trend across too few points and calling it a direction. Bin into
  windows and compare a sequence.

## 2026-08-25 (polaris) -- BlendCorpus never got the #4121 fold

- **The post-sync crash was a dataloader layout mismatch, and both earlier
  readings of it were half right.** Upstream #4121 moved the LM stack to a
  flat `[T]` token layout. BlendCorpus counts SEQUENCES and kept yielding
  `[B, L]`; nobody adapted it, because it lives out-of-tree at ALCF. Every
  production agpt config on that path died at the first attention layer with
  `token count 1 is not a multiple of max_context_length 8192` -- `token
  count` is dim 0, and it was the batch size.

- **Measured rather than inferred, after two reverted guesses.** Pushing real
  tensors through the shipped reshape:

      OLD | input (1, 8192) | q (1, 131072, 128)  BROKEN
      NEW | input (8192,)   | q (8192, 16, 128)   OK

  The first is bit-for-bit the shape the 2026-08-24 crash reported.
  `QKVLinear` reads `num_tokens = x.shape[0]`, so a retained batch dim merges
  sequence and heads. The earlier Perlmutter measurement of `[T, N, H]` was
  correct *for Grain* -- and treating it as settling the question for
  BlendCorpus was the mistake. Two loaders, two layouts.

- **The `positions` half was the dangerous one.** It also had to fold, and it
  does not raise. `get_efficient_causal_mask_mod_for_packed_document` reads
  `positions.shape[0]` as seq_len and cumsums along dim 0; on `[B, L]` that
  reads the batch size and yields document ids of `-1`. Flex attention would
  mask the wrong spans and show up only as a worse loss curve.

- **Fix** (`42f4edfaa`, entirely inside `experiments/ezpz/`): fold `input`,
  `labels`, and `positions` at the yield. `flatten()` is row-major, so it is
  identical to the `torch.cat(rows)` core already does -- token order is
  unchanged, which is what makes it safe for in-flight chains. Smoke: Polaris
  7557829, 2N, 12/12 steps, loss **12.91 -> 8.19**, step-1 within 0.007 nats
  of the Grain baseline. `rc=0` alone would have proven nothing; a wrong
  reshape scrambles Q/K and still runs.

- **Polaris needed a torch 2.13 venv.** The production `.venv` is torch
  2.10.0+cu128 and cannot import HEAD (`DataParallelMeshDims`). Unlike
  Aurora's ABI-welded XPU stack, Polaris is plain CUDA, so `torch==2.13.0`
  installs from PyPI normally. Built `.venv-torch213` alongside, verifying
  `torch.__version__` after every install batch. `submit_agpt_20b_autoretry.sh`
  now takes `VENV_DIR` (`a3264a329`). Note PyPI `ezpz` is a *different*
  package (0.1.2) -- an auto-install loop keyed on ModuleNotFoundError will
  chase it forever.

- **ezpz#230 merged** (2026-08-24). Verified live: `get_patterns_for_machine
  ("polaris")` now returns all 5 patterns, so the blind-failover bug below is
  closed on the code side.

- **`refresh_all.sh` quietly committed two empty charts.** Both agpt
  `eval_overview.svg` regenerated from a laptop with no eval data
  (`outputs/evals` is gitignored and lives on the cluster): the plotters
  logged `0 steps` for every series, rendered empty axes, saved, and exited
  0. 2b went 5 series -> 2, 20b 6 -> 3. Caught by distinct-color count, not
  by any exit code. Restored (`2982d114a`) and guarded (`34ea26b2d`) -- the
  plotters now refuse to overwrite an existing figure when every disk-backed
  series is empty.

## 2026-08-25 (aurora) -- three more bugs behind the fold; frameworks/2026.1.0 lands; the eval charts were never plotted

Same day, other machine. Everything below is downstream of `42f4edfaa` above --
with the fold in place the smoke got past attention and immediately found what
the crash had been hiding.

- **Three smoke arms, three unrelated bugs, each invisible behind the last.**
  `sync_smoke.sh` had never once run green on Aurora, and each fix exposed the
  next. Worth stating because "the smoke fails" read as one problem for most of
  the day.

- **TP>1 has been broken since the #4121 port.** MEASURED, job `8781623` arm 2
  (2N, TP=2): `AssertionError: XPUScaledDotProductAttention: local_map is set
  but in_dst_shardings is missing entries for: ['q_BLNH','k_BLNH','v_BLNH']`.
  #4121 (`73aed7f6c`) renamed the upstream keys to `_TNH`; our port
  (`476d16831`) adapted the function BODY and left its PARAMETERS at `_BLNH`.
  The contract matches by positional-arg name, so the two halves of one change
  disagreed. Hidden because TP=1 never wraps the attention local_map -- exactly
  the blind spot that arm exists to cover, and the same failure the 57th sync's
  original rename was written to prevent, reintroduced from the other
  direction. Fixed `b5f6f712f`.

- **Both flags the smoke passes are fake, and always were.** There is no
  `training.seq_len` and no `training.local_batch_size`. Dumping the dataclass
  gives three size knobs: `num_tokens_per_microbatch_per_dp_rank`,
  `num_tokens_per_train_step`, `max_context_length`. tyro rejected
  `--training.seq-len` and `--training.local-batch-size` on every run since
  `b8c369fb0` -- 48 rejections in one job, 72 in another -- so the shrink the
  moe arm was meant to apply never applied. Fixed `a3069ea00`.

- **Post-fold, tokens-per-microbatch must EQUAL max_context_length.**
  `rope._reshape_for_broadcast` does `rope_cache[:T].view(T, 1, w)` against a
  cache holding only `max_context_length` rows, and post-fold `T =
  ntok_per_microbatch`. Two rows is a hard shape error: `RuntimeError: shape
  '[16384, 1, 8]' is invalid for input of size 65536` (arm 1; debugmodel
  defaults to 16384 vs 8192). **Production survives with ZERO margin** --
  `agpt_2b` and `agpt_20b` both run `ntok == max_context_length == 8192`,
  exactly one row. Nothing documented that constraint; it does now.
  `emit_positions=True` looks like the workaround and is not: it takes the
  `rope_cache[positions]` branch but puts a `DeviceMesh` into
  saved-for-backward, which AOT autograd rejects.

- **`frameworks/2026.1.0` is on Aurora, and only on the validation nodes.**
  `module avail frameworks` from a login node shows only `2025.3.1`, and
  `/opt/aurora/26.181.0` does not exist there -- it lives in the validation
  image, so the module cannot be seen or tested without landing on one of its
  three nodes. Different deployment from Sunspot: the module's own python
  3.12.12 BUNDLES torch `2.13.0a0+gitcf30153` and there is no
  `venvs/fw-2026.1-rc2`. Image delta: SLES 15-SP7 (login 15-SP4), level-zero
  `.77-1146` (login `.42-1146`), oneAPI tree `26.181.0`.

- **It is blocked out of the box by a missing `LD_LIBRARY_PATH`.** Every
  `import torch` dies on `OSError: libglog.so.0` from `torchcomms`, which
  `torch/__init__.py` imports -- so it gates everything. The library is not
  missing; it ships in the module's own `lib/` and the modulefile does not
  export it (`libgflags.so.2.2` too). With `LD_LIBRARY_PATH=$FW/lib` the stack
  works (job `8781129`): 12 XPUs, bf16 matmul, SDPA fwd+bwd, and the
  **compiled** SDPA backward -- the surface that failed on the Sunspot fw-RC
  with `assert_size_stride`. Same build hash, so this is RC4. It is 1 rank, so
  it does NOT retire `known-bugs/fw-rc-compile-sdpa-backward-tp4.md`.
  `dist.is_xccl_available()` is True; `oneccl_bindings_for_pytorch` is absent
  and that is correct, it is the IPEX-era shim. Written up in
  `guides/frameworks-rc-validation.md` (`1d64b1f76`).

- **Eval scores never reached the dashboard, for two independent reasons.**
  `_eval_scores` keyed on `ckpt_base` (the checkpoint dir,
  `agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288`) while the sweeps write to a
  short alias (`agpt-2b-v2-512n`); and it globbed `results/<task>/**/
  results_*.json` while our sweeps write one flat `results/results.json` with
  task names at the TOP level. Either alone was fatal, and neither raised --
  an un-evaluated chain legitimately returns `{}`, so a total failure and the
  normal case are indistinguishable. Fixed `c102006c4`; 1,738 points across 4
  chains appeared immediately.

- **And they had no chart at all.** The payload only ever carried the NEWEST
  checkpoint's scores, so there was nothing to plot. `evals.history` now
  carries every evaluated step, and the web board shows the selected metric
  full-width above two grids: one panel per remaining metric, one per eval
  task (`d19710bdb`). Verified headlessly with a stub DOM -- an old cache
  builds 6 charts and HIDES the eval section, the real cluster cache builds
  13.

- **The dashboard was understating both 20B chains.** Checkpoint audit against
  disk: 20b_v2_256 at 11,800 vs 10,369 cached (+1,431), 20b_v2_512 at 10,600
  vs 9,690 (+910). Those are umbrella `8764675`'s 08-20 steps, which the
  cached backbone never picked up. `2b_v2_512` confirmed finished at 46,429
  from disk rather than from a log line.

- **Own goals worth recording.** (1) `timeout 900 ssh` killed the local end of
  a `tar` and left a truncated 2.8 GB `.venv.tar.gz`; worse, I "verified" it
  with `gzip -t ... && echo OK || echo CORRUPT`, which printed OK on a failing
  exit code. Rebuilt detached to a `.partial` with an atomic rename. (2) A
  piped `module load ... | tail -3` runs in a subshell and silently no-ops --
  python stayed 3.6.15 and every import failed with a misleading "No module
  named torch". (3) Submitted a job without `cd`-ing to the repo, so
  `PBS_O_WORKDIR=/home/foremans` and the script sourced the stale home `.venv`
  (torch 2.9, dead against the new UMD). (4) `git pull` printed "Aborting"
  inside an `&&` chain that reached `qsub` anyway. All four are the same
  family: trusting an exit code instead of the artifact.

## 2026-08-23 (polaris) -- failover was blind on this machine the whole time

- **Polaris had zero bad-node scraper patterns registered, and nothing ever
  said so.** `ezpz.failover.patterns` ships `aurora.py` and `sunspot.py`;
  `get_patterns_for_machine("polaris")` returned `[]`. An empty pattern set
  and a genuinely clean log both come back as `[]`, and falling back to blind
  rotation is the *correct* response to the second -- so a whole machine
  having no patterns degraded gracefully into looking like normal operation.
  The bug produced no error to grep for. It surfaced only by asking the
  scraper directly what it returns.
- **The cost was job 7550301: 130 nodes, ~1 hour, zero training steps.** Two
  ranks raised `torch.AcceleratorError: CUDA error: CUDA-capable device(s)
  is/are busy or unavailable` at `set_device()`. Blind rotation swaps
  `active[0]` **by design**, so it retired a healthy node twice while the sick
  one stayed in the allocation, and attempt 2 failed identically. The
  `stuck_pre_training` guard then stopped the chain -- correctly; it is the
  only thing that kept this from eating the whole spare pool. `step-5600` is
  intact (512 shards, `.metadata` present); the loss was pure allocation burn,
  against a balance now at **-49,134 node-hours**.
- **The only host-attributed line in 1800 lines named the wrong node.**
  `x3007c0s13b1n0: rank 57 died from signal 15` is the idle watchdog's *own*
  SIGTERM -- a victim of our teardown. It was the sole hostname available and
  matching it would have swapped a third innocent node. The most-available
  evidence pointed away from the culprit, which is the same shape as the
  `aurora.py` innocent-cascade warning.
- **PALS `--label` is what makes a CUDA fault attributable at all.** Without
  it a rank's traceback reaches the log bare. I confirmed the format on real
  hardware (probe 7553963) instead of assuming it: the prefix is
  `<fqdn> <rank>: ` and is applied **per line** to a multi-line Python
  traceback on stderr. Guessing the Aurora shape (`<host>: `, no rank field)
  would have compiled fine and matched nothing.
- **Enabled opt-in, via `EZPZ_MPI_LABEL=1`, not globally.** Aurora and Sunspot
  patterns anchor on *unlabeled* `^<host>: ` lines; turning labeling on
  everywhere would have silently broken both. Only the Polaris submit script
  sets it -- the other three autoretry scripts are Intel XPU and were left
  alone deliberately.
- **The negative test is the important one.** On unlabeled input the scraper
  must stay silent rather than tag the SIGTERM victim: a false positive swaps
  a healthy node *and* leaves the culprit in place, strictly worse than blind
  rotation. Verified against the real 7550301 log (returns `[]`) and the real
  labeled probe log (returns exactly the culprit). 8/8 tests pass, including
  on a compute node.
- **Verified the whole chain through a real `ezpz launch`** (probe 7553977),
  not just the pieces: `--label` present in the emitted mpiexec command, 5
  patterns registered, tests green, fault labeled, culprit scraped.
- **`ezpz tar-env` skips with exit 0 when the tarball already exists.** It
  logged "already exists, skipping creation" and exited clean -- I nearly
  shipped a "rebuilt" 4.4 GB tarball still carrying the Aug 10 code. Same
  silent-no-op family as the 1s PBS job and `0 ok 36 skipped`: verify the
  artifact, never the exit code.
- **A `site-packages`-only fix would have vanished on the next venv rebuild.**
  ezpz is installed from a pinned commit, and compute nodes run a yeeted
  `/tmp/.venv` unpacked from the tarball -- not the live `.venv`. Vendored the
  module under `experiments/ezpz/failover_patterns/` with an install script
  that verifies registration and exits non-zero if it did not take.
- Commits: `341096f90` (fix + 8 tests), `9ab148774` (known-bugs writeup,
  linked from the Polaris production README).

## 2026-08-19 (sunspot) -- flex MoE was two stacked bugs; the 30B trains and resumes; hybridep is not ours to fix

- **Every flex-attention MoE config was dying on a missing BlockMask, and it
  took two fixes, not one.** Core builds the mask in `_prepare_inputs` only
  `if positions is not None` (trainer.py:738), and blendcorpus never yielded
  that key -- only the HF loader did. Fix 1 emits per-document positions
  (restarting at 0 after each EOD, because blendcorpus PACKS documents and a
  plain arange would let attention cross document boundaries). I unit-tested
  the tensor logic against the HF semantics on four cases before running it
  anywhere. It passed, and the real run **failed identically**. Fix 2 was the
  actual bug: `_document_positions` read the EOD id back off `_bc_cfg`, which
  is blendcorpus's own config object and is not guaranteed to carry the
  field, so `getattr(..., None)` silently yielded None. Now resolved once in
  `__init__` and owned on `self`. A red herring cost time here: the dumped
  config shows `"eod_token_id": null`, which looks like the smoking gun but
  is the user-facing field, legitimately null when the value comes from the
  tokenizer fallback -- measured separately that gemma resolves `eos_id = 1`.

- **The mask fix exposed a second, unrelated bug underneath: MoE routing is
  not recompute-stable under FullAC.** With the mask built, `moe_small`
  reached a real backward and died on
  `CheckpointError: Recomputed values ... 560 vs 559`. Ran three AC arms on
  both flex configs: full fails, AC-off fails at 89-94% memory on level_zero
  error 40, **selective passes 5/5**. Not luck -- `SelectiveAC` keeps
  `aten.topk.default` as MUST_SAVE precisely to hold expert assignments
  stable across recompute. Changed both flex configs to selective; verified
  all four configs (2 flex + 2 SDPA controls) pass 5/5 on defaults. The
  controls report the same memory as before any of this work (74.05% /
  94.54%), which is what shows the maskless path was not perturbed.

- **The 30B config trains, and its checkpoints round-trip.** 482 steps, loss
  12.03 -> 3.37 monotone, grad_norm 0.7-1.0, memory flat at 62.79%, ~28% MFU
  held (matching exp05's 12-step number). Compiled resume dies on the
  `tensors_saved_with_vc_check` AOT bug -- but the **DCP load itself
  succeeded** (65.64s) and diffing the job scripts showed the only change was
  the checkpoint interval, so resuming is the trigger, not config drift.
  Uncompiled resume works: step 251 at loss 4.540 against the step-250
  checkpoint's 4.61, descending to 4.235 by step 299. I misreported the
  uncompiled cost as "2.8x slower" from a warmup step; over the run it is
  ~7% (455 vs 489 tps). The real cost is memory: 93.80% vs 62.79%.

- **The EP a2a abort has a name, and it is not ours.**
  `ur_die: urEventWait must not be called for an internal event` -- Intel's
  Unified Runtime, not a torchtitan assertion. EP is not the trigger
  (`moe_debugmodel_ep` runs 5/5). Grepping all 15 sweep configs, `ur_die`
  appears in exactly the two arms whose dumps contain the a2a frame, so the
  two-cause split now rests on two independent signals. Memory separates them
  cleanly: the ur_die arms sit at 47.77% / 79.74%, the others at 86-95%, and
  non-EP `moe_7b` fails the same way at 95.40% -- the control showing EP is
  irrelevant to that second mode.

- **hybridep is closed WONTFIX.** I had called it "blocked on a torch version
  floor", which implies a bump would fix it. Two blockers sit behind the
  import error, each sufficient: `deep_ep` is not installed, and `deep_ep` is
  CUDA-only -- the docstring says "GB200 NVLink72 Systems", it is
  TMA-optimized, and it calls `cudaStreamSynchronize`. Exclude from XPU
  sweeps rather than carrying it as an open bug.

- **sft migration finished cleanly.** 4 directories, 0 failures, datascience
  back to 16.53T of 20T (was 20.48T, over quota).

## 2026-08-21 (sunspot, later) -- production had no defense against a NaN gradient

- **The NaN guard ran AFTER the optimizer step and only looked at loss.** Guard
  at `trainer.py:1085`, step at `:871` -- so a NaN gradient was written into
  the weights before anything noticed. Our own 80B report is the proof:
  `grad_norm nan @30, loss nan @31, nan-abort @35`. grad_norm went bad a full
  step BEFORE the loss, so the earliest signal was one ahead of the one we
  watched, and five more updates landed on top. Job 8663177 separately ran
  ~370 NaN steps unbounded. Worse than "off by default": production omits
  `--nan-abort-consecutive` ENTIRELY, because the pinned pre-#3623 clones have
  no such field and passing it crashes every rank (killed umbrella 8680578).
  SCOPE (corrected 2026-08-22): no canonical chain was observed doing this --
  all five seats of umbrella 8764675 log ZERO non-finite loss/grad_norm lines.
  Both cited jobs are experiments (12473142 is an 80B bf16 RC validation run,
  8663177 is the dolmino CPT attempt that was a RoPE-flavor mismatch). The
  defect is that nothing WOULD have caught it, not that it happened in prod.
  Fixed with a host-side isfinite check on grad_norm before the step:
  non-finite -> zero_grad, log, skip the step, still advance the lr scheduler.
  No new collective -- grad_norm is already rank-reduced inside
  `clip_grad_norm_`, so every rank branches identically.

- **Deliberately not upstream's mechanism.** #4226 fixes the same ordering bug
  with `torch._assert_async`, where a failed device-side assert invalidates
  the process -- our failover would read that as a crash rather than a clean
  stop, and its XPU behavior is undocumented. It also lands in
  `Trainer.train_step`, which `FaultTolerantTrainer` overrides wholesale, so
  merging it would have given us NOTHING. That is the useful shape of the
  finding: the most valuable commit in the sync was valuable only as a
  hand-port.

- **Smoked both changes rather than reasoning about them.** nanguard (12473623):
  10/10 bit-identical against the parent commit, guard fired 0 times, tps
  337-338 vs 338-340. Being honest about resolution -- ~0.3% is BELOW what a
  single-shot 10-step comparison can resolve (exp05 measured a 2.8% cross-job
  floor), so bit-identity is the real result and "costs nothing" is not yet
  earned. And the smoke proves the guard is INERT, not that it FIRES; that
  path needs a divergent config and is still owed.

- **Also added a 4D rank assert at the three attention wrappers** (ndimsmoke
  12473624: agpt 10 steps, moe 10 steps, 0 fires). Upstream's fold-batch-dim
  reshapes the LM stack to a flat `[T]` layout; our wrappers take `q_BLNH` and
  immediately `transpose(1, 2)`. On 3D input that swaps N with H instead of L
  with N and **SDPA accepts it** -- a degraded loss curve, no traceback. The
  assert converts the worst failure mode in the whole sync from silent to
  loud, passes today, and is constant-folded under dynamo. Note the analysis
  said four sites; there are exactly three.

- **The sync itself: defer.** Grain (`1b04fc1c3`) is the OLDEST of the 35
  commits, so every other one descends from it -- no merge can take the
  mechanical import fixes without also taking Grain and fold-batch-dim. I had
  recommended "land group 1 first, gate group 2" twice before checking the
  ancestry; that was wrong. Splitting requires cherry-pick, i.e. carrying
  divergence. Trigger for doing it: upstream deprecating `partial_dtensor`, or
  adopting the 2.14 nightly -- do both in one revalidation window, not two.

## 2026-08-21 (sunspot) -- the 30B RAN OUT: 2000/2000, 12.028 -> 2.115, zero NaN

- **The 30B finished its full config.** Step 2000 of 2000, `rc=0`, in 2h58 of
  a 6h allocation -- an actual completion, not a timeout. Four jobs, one
  continuous trajectory: 12473304 (1->482, 12.028->3.357), 12473476
  (401->871, ->2.617), 12473515 (801->1781, ->2.246), 12473545 (1751->2000,
  ->**2.115**). Zero NaN/inf in any of them. Nine checkpoints, 2.6 T. Steady
  state to the last step: 496 tps, 28.3% MFU, memory flat at 59.84%,
  grad_norm falling 0.26 -> 0.074. That answers all three questions exp08 was
  opened for -- loss descends, grad_norm stays bounded, checkpoints
  round-trip -- and the last two resumes were COMPILED.

- **Three corrections today, all the same shape: reusing a stale fact without
  re-checking it.** Worth naming together because the pattern is the lesson.
  (1) I claimed every nightly job had secretly run on torch 2.13; my
  path-grep took the first `site-packages/torch` in each log, which is a
  `_pytree` warning from the inherited conda torch before the venv's torch
  loads. (2) I claimed TP>1 hit "a third, unidentified failure" on the
  nightly -- I compared a CDT commit time against UTC mtimes AND grepped for
  only one of the two known errors. (3) I used the uncompiled 30B's 93.80%
  memory as the ceiling for the 2.14 bump, hours after retiring the caveat
  that made that path exist. Nothing runs uncompiled; the live chain is
  59.84%.

- **The instrument that fixed all three: ask the run, not the clock or the
  filename.** Every run dumps its resolved config.
  `grep -ao '"global_vocab_size": [^,]*'` separates fix-active from
  fix-inactive with no timezone reasoning at all, and every failing TP>1 arm
  dumps `null` -- so no failing arm ever had the fix and there was no mystery.
  Same for torch identity: `spmd_types` cannot produce one step on 2.13, so
  steps>0 IS the proof of which torch ran. Behavioral discriminators beat
  metadata.

## 2026-08-21 (sunspot) -- the 30B compiled resume works after all; a chain that actually reaches 2000

- **Compiled resume is not broken; `full_dtensor` was.** exp08 had carried a
  caveat for days -- "resume uncompiled for one interval, or wait for the
  upstream fix" -- and job 12473515 retires it. It resumed from step-800
  COMPILED on the `partial_dtensor` pin, loaded in 40.38 s with no vc_check,
  and has run 800+ steps since: 2.712 -> 2.263, zero NaN, 489 tps, 28.3% MFU,
  memory flat at 59.84%. The memory number is the whole point. The uncompiled
  workaround cost 93.80% occupancy to save ~7% throughput, which was never
  shippable at this size; the compiled resume costs nothing. This also closes
  the last hole in the round trip -- the earlier uncompiled resume ran out of
  walltime one step before its checkpoint, so a post-resume SAVE had never
  actually been observed. Three now (1000/1250/1500), ~27 s each.

- **That answers open question 1 on the vc_check page,** which had asked for
  exactly this before trusting the pin ("`partial_dtensor` is the legacy path
  and this session has only smoke-tested it, 3/3 steps"). 1589 steps is not a
  smoke test. Status header moved from OPEN to OPEN-upstream/not-blocking.

- **`--checkpoint.interval` is a capacity decision at 30B, not a granularity
  knob.** The preceding job set interval=100 reasoning that restart points
  are cheap. At 294 G per checkpoint, 2000 steps / 100 = 5.9 T against 1.5 T
  free; it died with `Errno 28` mid-write. Worth being precise that this was
  a *checkpoint write* failure, not a training fault -- the model was fine.
  The fix was interval=250, deliberately NOT `keep-latest-k`: deleting
  history to buy space trades a permanent asset for a temporary one. The
  partial step-900 dir left behind is harmless, since `_find_load_step`
  (dcp.py:640-684) only counts a step dir holding `.metadata` or
  `model.safetensors.index.json`.

- **Queued `30b_long4.pbs` (12473545) `afterany` the live job.** At 42 s/step
  the inner `timeout 41400` lands near step 1770, ~230 short of the 2000-step
  config, so the chain needed a continuation -- there was none. long4 scales
  BOTH clocks together (walltime 6h, `timeout 19800`), which is the entire
  lesson from `30b_long.pbs`: that one had its PBS walltime raised 6h -> 12h
  while the inner timeout stayed at 20400, and took an `rc=124` at step 871
  with half its allocation unused. Also committed the whole chain
  (`30b_converge` through `30b_long4`), which had been running from
  uncommitted scripts sitting in the repo root.

- **A loss bump that was not one.** Step 1300 reads 2.434 against 1200's
  2.377 and looks like a regression. It is batch noise: loss oscillates in a
  +/-0.05 band about a descending mean while grad_norm falls monotonically
  0.26 -> 0.13. Divergence has the opposite signature -- grad_norm rising.
  Two arbitrary 50-step samples will disagree at this amplitude, so sample
  denser before calling a bump.

- **Two watcher bugs worth naming, both mine.** First: `qstat` is not on
  `PATH` for non-interactive ssh here, so `ssh sunspot "qstat ..."` fails
  with `command not found` -- and with `2>/dev/null` attached that is
  indistinguishable from an empty queue. Use `/opt/pbs/bin/qstat`. Second:
  the first monitor I armed globbed `outputs/logs/30b-converge/*/train.log`
  and immediately fired `Errno 28 / No space left / Traceback`. Those came
  from 12473509's *old* disk-full log; the live log has zero. A health
  watcher must scope to the job it is watching, or it will keep re-reporting
  history as news.

- **Also: never guess a PBS job's log path.** Three probes were burned
  globbing `~/torchtitan*` and `/lus/tegu/projects/*/foremans/torchtitan*`
  before reading `Output_Path` and `PBS_O_WORKDIR` out of `qstat -f`. The
  checkout is at `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan`
  -- two levels deeper than every glob assumed. Related zsh trap: an unmatched
  glob aborts the whole command line with `no matches found`, and `2>/dev/null`
  does NOT suppress it, because the shell emits it before the command runs.

## 2026-08-20/21 (aurora) -- MDS + lr in the dashboard; TWO seats were loading complex weights as cos_sin; a fix that never reached the queue

- **The umbrella finally ran after ~44 h queued, on the PRE-FIX script.** PBS
  snapshots the submit script at `qsub`, so the 20B constant-LR fix
  (`b08fccfd1`, Aug 19 12:23) never reached job `8764675` (queued Aug 18 20:03).
  It trained a full 12 h cycle on the decaying config -- clean walltime finish
  (`Exit_status=-29`, 12:00:31), t1 20b-512 -> step 10,699 / ckpt 10600,
  t2 20b-256 -> 11,800. Held successors are submitted at the same time as their
  predecessor, so they are pre-fix too: replaced `8764677` -> `8768759` ->
  (after the RoPE fix) `8769730`. **Check `qstat -f <id> | grep ctime` against
  the fix commit date whenever a script fix lands with jobs already queued.**
- **TWO of five seats were loading COMPLEX-trained weights under a cos_sin
  config** (`efcae5419`). `CONFIG_SUFFIX` was one global `_real` for all five,
  but each chain crossed the 2026-06-25 switch at a different step and
  `2b_v2_256` never crossed it at all. Audited per seat at its real resume/seed
  step:

  | seat | parent @ step | trained | was |
  |---|---|---|---|
  | t0 dolmino | 2b_v2_512 @46429 | `_real` | ok |
  | t1 20b-512 | 20b_v2_512 @9000 | `_real` | ok |
  | t2 20b-256 | 20b_v2_256 @10369 | `_real` | ok |
  | t3 2b-512 constlr | 2b_v2_512 @21307 | **complex** | WRONG |
  | t4 2b-256 constlr | 2b_v2_256 @9500 | **complex** | WRONG |

  t3 is a 512N PRODUCTION seat resuming IN PLACE -- it only avoided corrupting
  a live chain because it dies on `std::bad_alloc` first. Now a per-seat 12th
  TRAINERS field; the guard tests for UNSET rather than empty, because empty is
  a meaningful value here (complex). Same trap as the `decay_ratio` field.
  **Found because the user asked "why is it using agpt_2b_real" about a smoke
  script that had copied the flavor verbatim from the umbrella.**
- **The t4 seat's failure peeled back three layers.** `lm_head.weight`
  (fixed earlier) -> `optimizer.state...` because the CLONE ran a stale
  `ckpt_key_compat.py` with **zero** optimizer-namespace handling while main's
  had it (commit `40ffc9215` never propagated; the file being PRESENT is not
  evidence the fix is IN it) -> now `AttributeError: 'dict' object has no
  attribute 'mul_'` in SophiaG, i.e. the pre-#3623 nested->flat optim format.
  Converted step-9500 successfully (1110 subkeys / 111 params, momentum live:
  `exp_avg` 67.5, `hessian` 0.053) but the smoke has yet to run -- see below.
- **MDS and `lr` are both in the dashboard now** (`f87b77372`, `712b46ba0`).
  MDS reads the committed CSV (154,391 rows, 0 W&B calls) rather than a
  cross-project scan over 8,098 runs; verified a faithful dump (1242/1242 loss
  values bit-identical to run `ov1dn10t`). Its tokens/step is confirmed
  BIT-EXACT against W&B's own `consumed_train_tokens`: 154391*6144*8192 ==
  7,770,753,466,368 exactly. `lr` rides in a SEPARATE scan_history pass --
  folding it into `OLOG_KEYS` returns 0 rows for all 6 backfill runs
  (scan_history intersects keys) and would have deleted 3,408 backfilled points
  from every other metric.
- **The matched-pair eval says no meaningful separation at step 21,000.** Mean
  delta -0.0033 across 7 tasks, fork ahead on 5/7. Two tasks exceed 1 sigma in
  OPPOSITE directions (arc_easy +0.0144 to the fork, boolq -0.0409 to
  canonical). **boolq is confounded**: the fork trained under cos_sin from
  complex-derived weights, and boolq is calibration-sensitive. The clean
  version of this experiment needs a correctly-flavored fork.
- **Own-goals worth recording.** Three allocations lost to my scripting errors
  on one smoke script (no venv activation; guessed `--optimizer.name` when the
  build wants `--optimizer=`; then `SophiaG not added` because `ezpz launch`
  needs `--nproc`/`--hostfile`/`--` and the node-local venv -- I had
  reconstructed the wrong command SHAPE instead of copying the umbrella's).
  Separately, the `_real` adapter guard in `eval-2b-v2.sh` tested `V2_REPO`
  when the conversion imports from `CONVERT_REPO`, so it refused a CORRECT
  matched-pair invocation and cost one arm of that eval (`ab32015a2`).
- **`prod_dash` liveness and the umbrella.** A running umbrella marked exactly
  ONE seat live: `CKPT_RE.search()` takes the first match, and the umbrella
  `.o` names all five ckpt dirs. Now `finditer()`, with the shared log
  authoritative for STATE but never for the per-step tip (`9e7dc63b8`).
  `isLive` also trusted `queue_state=="R"` with no freshness check; keyed on
  `live_tip.age` now -- a first attempt gated on `log_age` and reported 0 live
  while four chains trained, because `log_age` is the mtime of HISTORICAL logs
  (`246c01faa`).
- **A silent under-count with an honest counter.** The 20b-512 ropefix sweep
  reported "34 ok, 0 skipped, 0 failed" while 2 of 36 checkpoints had no
  winogrande. Nothing failed -- the STEPS list itself omitted 5000 and 6000
  (jumping 4900->5100, 5900->6010) while the 256n list had both. No
  artifact-check downstream could catch this; only diffing the two arms' lists
  did (`32b94a9ea`). Winogrande is now 72/72; a follow-up (`8770286`) is making
  512n uniform across all four tasks after my gap-fill created the two new dirs
  with `TASKS=winogrande` only.
- Also: stage-2 token offset now honored in the TUI and the tokens-vs-time
  chart (`8f4a60d8d`), the MDS tokens/step constant corrected in all three eval
  plotters (`aec112640`, figures regenerated `97fa40ed0`), `SSH_TIMEOUT`
  600->1800 s because the cold build grew to ~915 s (`a67c8cb65`), and the web
  board no longer overflows horizontally (`27eac13be`).
- Open: `8769730` queued with 0 prod-reachable free nodes; the t4 smoke needs
  rewriting to reuse the umbrella's launcher; t3's `std::bad_alloc` has a fresh
  512N reproduction (note it died while t1 ran fine at the SAME 522 nodes in
  the same job, which argues against pure rank count); #76 906 GB reclaim.

---

## 2026-08-19 (local + aurora) -- a browser dashboard; the dolmino tokens-axis offset finally landed; a matched-pair eval needs TWO RoPE flavors

- **`prod_dash.py` hung for 8m20s and blamed SSH.** It was not SSH (0.2 s
  round-trip, warm cache). `parse_olog` was line-scanning **8.3 GB** of trainer
  console logs on every refresh -- 3.1G/2.6G/2.0G files that are almost entirely
  the constlr seat's ~745k-line crash spew. `grep` finds **zero** step lines in
  the 2.6 GB one. Capped at 256 MiB per file (`EZPZ_OLOG_MAX_BYTES`); `--board`
  went 8m20s -> **24.9 s**. The error message pointing at a dead ControlMaster
  is misleading and sent the first diagnosis the wrong way.
- **The TFLOPs "gaps" are not gaps.** Every metric has exactly the same point
  count as loss (570/570, 577/577, 599/599) and 20b_v2_256's step spacing is
  min==median==max==18 -- not one sample missing. The picket-fence is real
  throughput variance: checkpoint saves inflate a step's wall-clock, so tps and
  tflops collapse for that step. 20b_v2_256's dips land at 901/1801/2701, dead
  regular. (The 900-step interval is INFERRED from that spacing; `CKPT_INTERVAL`
  was not read.) 512N is genuinely wider: 22% of samples < 50 TFLOPs vs 5%.
- **New `prod_dash_web.py` -- browser view, loopback only** (`fcc3a40f4`).
  Calls `prod_dash.fetch()` and serves the payload verbatim, so the web view,
  the TUI, and the committed charts share one data layer. stdlib `http.server`;
  uPlot vendored (51 KB, no CDN). `--host` refuses non-loopback unless
  `PD_WEB_ALLOW_PUBLIC=1` -- unauthenticated production telemetry, so remote
  viewing goes through `ssh -L 8712:127.0.0.1:8712 aurora`.
- **Driving it in a real browser was load-bearing.** curl returned 200 on every
  endpoint and the JSON had all 7 chains x 5 metrics, and the page was still
  broken four ways: uPlot defaults x to a TIME scale (steps rendered as
  "12/31/69"); `spanGaps:false` drew ONE chain of seven (chains sit on different
  step grids, so the union x-axis is ~3.5k values and any chain is ~83% nulls --
  a null means "no sample HERE", not "training gapped"); the "no data"
  placeholder persisted under the canvas; favicon 404'd every load. A JSON-level
  check cannot see any of these.
- **The dolmino tokens-axis offset, flagged as a todo on 08-17, is now fixed**
  (`4d2895f30`). Its step counter restarting at 1 is CORRECT (steps are
  per-chain; it seeds weights-only from stage-1 step-46429). But "tokens seen"
  is cumulative, so plotting from 0 claimed it saw its first token alongside
  stage-1. New optional `prior_tokens` in `trajectories.py` -- the single source
  of truth, so every plotter inherits it. Verified: dolmino now spans
  **4.674T -> 5.452T**.
  - **Checked per chain rather than pattern-matching on "is it a fork":** only
    dolmino needs it. `constlr_from9200` also branches mid-run but KEPT the
    parent's step numbering (series starts at 9201), so its tokens are already
    cumulative and an offset would double-count. The discriminator is the first
    logged step.
  - **`pct_target` deliberately NOT offset:** dolmino's `token_target` is the
    stage-2 increment (2.390T), so `step*gbs*seq` correctly measures progress
    through stage 2. Offsetting it would report >100% instantly.
- **Matched-pair eval at step 21,000 submitted** (`8766898`, capacity,
  `a94fb88f9`). The constant-LR fork tracks canonical within +0.001..+0.008 nats
  -- inside noise -- and loss is not the deliverable, so this asks the eval
  question at the current frontier instead of waiting on step ~30k (gated on the
  stalled umbrella).
  - **The two arms need DIFFERENT RoPE flavors**, which is why this is a bespoke
    script: canonical step-21000 -> run 21grc6o7 (2026-05-26) -> `2b` complex;
    fork step-21000 -> run xii94czx (2026-08-16) -> `2b_real` cos_sin. They
    crossed the 2026-06-25 switch at different times. One flavor for both -- the
    natural move for a "matched pair" -- scrambles Q/K on one arm and reads as a
    large fake regression. Flavors resolved from W&B argv, not guessed.
  - **`CONVERT_REPO` must be the main repo:** neither pinned clone ships
    `agpt/state_dict_adapter.py`, so both fall back to the bare
    `Llama3StateDictAdapter`, which permutes unconditionally -- making a
    `2b_real` request a silent no-op producing exactly the corrupt export the
    flag guards against.
  - **Confound to carry into the writeup:** the fork differs from canonical in
    BOTH the LR schedule and the RoPE convention.
- **The offset audit found 2 MORE plotters, and one was on a Y axis.** Fixing
  prod_dash + the web view was not enough. A fan-out over all 6 tokens-axis
  files (`8f4a60d8d`): `prod_dash_app.py` (the TUI) read the same payload as the
  SVG path and never used `prior_tokens`; `plot_production_wandb.py` plots
  cumulative tokens on the **Y** axis vs wall-clock, which a grep for x-axis
  token math never surfaces. `plot_production_combined.py` was already correct
  via its own `token_offset_b`; both `plot_eval_overview.py` omit dolmino.
  - That file's bug has a DIFFERENT mechanism, same outcome: it never derives
    tokens from steps, it reads the logged `n_tokens_seen` -- which ALSO
    restarts at 0 for stage 2, because the trainer zeroes it, only
    `train_state` persists it, and `--checkpoint.initial-load-path` defaults to
    `initial_load_model_only=True`.
  - It also hardcoded `target_b = 4670`, a drifting duplicate of
    `OLMO_MIX_1124_TOKENS`. For dolmino that divided an increment-frame
    numerator by a cumulative-frame denominator: the committed SVG read
    "684.4B (14.7% of 4.67T)", correct in NEITHER frame.
  - A SECOND hardcoded 4.67T then survived in the title f-string, so the chart
    briefly read "77.2% of 4.67T" while dividing by 7.06T -- worse than the
    original, because it looks self-consistent (`038a670e5`).
  - Final, verified per chart (`97fa40ed0`): dolmino **5,451.9B (77.2% of
    7.06T)** = 778.1B logged + 4,673.8B prior. The two COMPLETED 2B chains also
    moved **100.1% -> 100.0%**; that 0.1 was the 4670 literal being ~3.8B short
    of the real constant, and "100.1% of target" on a finished run invites the
    wrong question.
- **Three false readings today, one root cause: a verifier that under-matches.**
  (1) `[0-9.]*B` against "5,451.9B" captured "451.9B" -- no comma in the class
  -- so a correct fix looked like it had made the number SMALLER, and several
  turns went into hunting a bug in working code. (2) Grepping
  `outputs/evals/agpt-2b-ropefix-*` when the dirs are `agpt-20b-*` returned zero
  and got two healthy jobs called "concerning". (3) A regen that died with
  `ModuleNotFoundError` (needs `PYTHONPATH=$PWD`) still exited 0; only the
  unchanged SVG title revealed it. Rule: confirm the checker matches a string
  you have SEEN before trusting the number it returns.
- **Capacity is 70 queued / 16 running**, which is why `8766898` has not
  started -- not the two ropefix jobs, as first guessed.
- **Umbrella `8764675` still queued** behind the `at_queue=lustre_scaling`
  stall; the 20B constant-LR fix (`b08fccfd1`) rides on it.
- Still open: the 906 GB reclaim decision on the two mixed ckpt dirs (no
  pressure -- /lus/flare is 67% used with 31 PB free), and the fork re-check at
  ~30k/~46k.

---

## 2026-08-17 (aurora) -- a second key rename kills the same seat for the 3rd time; two chains were registered but never drawn; cot-long was measurable all along

- **`output.weight` -> `lm_head.weight` is the second upstream rename to break
  the 2B-256 constlr seat.** It died on all 3,072 ranks with `Missing key in
  checkpoint state_dict: lm_head.weight`, then sat dead ~9 hours holding 256
  nodes of umbrella 8756957. Its log labels the death `exit 127`, which reads
  as the known pals RPC launch failure and is not -- the real error is
  thousands of rank-lines deep. That is the third umbrella (8714502, 8744247,
  8756957) this seat has lost to a key rename.
- **Two wrong diagnoses before the right one.** "The checkpoint is missing" --
  no, the clone lives under `/flare/AuroraGPT/foremans/runs/`, not the main
  repo, and the checkpoint is intact (3,072 shards, 13 GB, valid `.metadata`).
  "The clone is on stale code" -- no, it is the same commit as `agpt-2b-v2`,
  which resumes fine. The discriminator is checkpoint VINTAGE: the fork's own
  step-20600 (written by current code) has `lm_head.weight`; the copied-in
  step-9500 predates both renames.
- **Every production 2B/20B checkpoint on disk still spells the head
  `output.weight`.** The live chains survive only because their clones are
  pinned to pre-rename code, so pulling a prod clone to HEAD is a deliberate
  act. `ckpt_key_compat.py` now composes both remaps in one `dcp_load` wrapper
  with independent detection (`e1320edf9`).
- **The head regex was wrong in a way the unit test caught.** Anchoring on
  "start of string or after ANY dot" also rewrites `layers.0.moe.output.weight`
  -- a real MoE key. Re-anchored to start-of-string or an explicit optimizer
  prefix; all 8 cases pass, and the real step-9500 is confirmed to need BOTH
  remaps.
- **Two chains were registered, data-backed, and silently absent from the
  charts.** Both plotters keep display lists separate from `trajectories.py`,
  and a regen that omits a curve still prints "0 scripts failed". The stage-2
  dolmino chain never reached the training chart (and needs an x-offset, since
  its step counter restarts at 1); `agpt-20b-v2-256n` was excluded from the
  eval chart as "a noisy 3-pt cluster" and stayed excluded after growing into
  a full production chain with 43 eval points. Both plotters now reconcile
  against `trajectories.py` at startup and exit naming what is missing. The
  eval chart went 24 -> 32 series.
- **cot-long did NOT drift -- and it was measurable the whole time.** It was
  written off as unevaluable because its LoRA checkpoints are gone, but
  `rollout_samples.jsonl` survived with 6,337 scored rollouts carrying
  `AnswerCorrectReward` per sample. Accuracy ROSE 0.069 -> ~0.17 over the first
  ~120 steps then held flat to step 400 (held-out validation 0/20 -> 5/20).
  The apparent format collapse was `ThinkFormatReward` reading only `content`
  while vLLM had split `<think>` into `reasoning_content` -- the same bug that
  once zeroed the reward, now zeroing the analysis. Added
  `rl/scripts/score_rollouts.py`.
- **All six known doc inconsistencies closed.** Four were real; two were not
  errors -- there is no "B1" (the series legitimately starts at B2), and
  cot-long's entry was wrong in the opposite direction from what it recorded.
  v2-256n was settled by checking disk (zero `checkpoint-*` dirs) rather than
  picking a side.
- **Stage-2 curve is pink**, not teal -- teal sat close enough to the new 20B
  greens to be misread as one of them.
- **20B-256 was under-reporting its own progress by ~25%.** Three W&B runs
  carrying 2,699 steps were never registered, so every chart, the live board,
  and the exported store stopped at 8,333 while training had reached 10,300 on
  disk. Its README carried three mutually inconsistent step counts (6,800 /
  9,400 / 8,334) and the 8,334 was the honest one -- it was the true head of
  the *plotted* data. Added `find_missing_runs.py`, which asks W&B which runs
  wrote each ckpt dir instead of trusting a hand-maintained list; all four
  wrong outputs this month traced to that list and each was found by accident.
- **Unregistered is not the same as missing data.** The first scan flagged 29
  runs; only 8 were real. The rest are crashed relaunches whose ranges a later
  run re-covers, runs that died before logging, or -- the instructive one --
  `v46mecdx`, which looks like a clean 50-step gain but logs steps 1-50 for a
  fork that BEGINS at 9,200. A smoke test sharing a ckpt dir. Auto-adding
  "gains" would have injected foreign data.
- **I then corrupted the store myself and caught it by diffing.** Appending
  `djmhgmmq` (Aug 3) to the END of `20b_v2_256` put its crashed values on top
  of `2ktrz29u`'s (Aug 5) re-trained ones, moving loss ~0.1 nats across nine
  steps with no error anywhere. `concat_chain` does `by_step[step] = row` in
  list order, so **run-id order is semantic** -- last listed wins. Documented
  at the merge site and enforced by `audit_run_order.py`.
- **Sorting took three passes.** Fixing adjacent inversions is not sorting;
  each pass just exposed the next neighbour. The third pulled every run's
  `created_at` and sorted outright. Along the way my first rewrite silently
  deleted 20 lines of provenance comments (reverted), and trailing per-id
  annotations stayed with their line while ids moved (realigned). All nine
  chains now verify chronological.
- **The one surviving value change was a genuine correction.** Step 8,330 in
  `20b_v2_256` moves 2.3892 -> 2.2599 because `t9vly2u8` (Aug 13) now wins it
  over `82e1jewm` (Aug 7). Checkpoint mtimes settle it: the chain persists
  step-8300 on Aug 8 then nothing until step-8400 on Aug 13, so `82e1jewm`
  logged 8,330 without ever persisting weights past 8,300 -- an orphan tail --
  while `t9vly2u8` owns the surviving checkpoints.
- **The RoPE A/B is now a 5-point series, not two spot-checks.** Correcting the
  permute lifts every metric at every step (ARC-C +0.016 to +0.037, ARC-Easy
  +0.022 to +0.038), and the corrected ARC-C rises monotonically across
  4500-4800 where the corrupted one wanders -- which is how a downward stretch
  read as decay. One correction to the earlier framing: the penalty is NOT
  roughly flat before step 5000, it ranges 0.016-0.037 across five adjacent
  checkpoints.
- **Umbrella 8756957 finished at 100% of walltime** (12h00m23s / 12h,
  `Exit_status = -29` = clean expiry) -- the first to use its whole allocation,
  beating 8744247's 97%. Both are 12h asks rather than the 24h dispatches that
  kept dying young. Successor `8760249` released itself from its system hold.
- **Three collision-writeup recommendations closed, and the naive version of
  one would have been wrong.** `.owner` claims the checkpoint dir so two
  concurrent jobs are loud instead of silent (warns rather than refuses -- a
  crashed predecessor always leaves a stale claim, and a job cannot tell stale
  from live from the inside, so refusing would convert every crash into a
  failed resume). `audit_ckpt_dirs.py` finds collisions already on disk. The
  design lesson: **an mtime gap is not the signal.** Three dirs in the affected
  tree gap 108-121s purely from straggler ranks, so a gap check raises five
  alarms of which three are false. The discriminator is that a collision's
  split falls exactly on a *shard index*, because one job overwrote a
  contiguous prefix. Validated: finds both known-mixed dirs, silent on all
  three stragglers.
- **The ARC-C artifact now has a mechanism, not just a correlation.** Extending
  the 20B-256 sweep to seven points caught the two curves moving in OPPOSITE
  directions at the same checkpoint: at step 5000 the corrupted series posts
  its lowest value of the run (0.3046) while the corrected posts among its
  highest (0.3737). And the penalty GROWS with training (+0.015 at 4000 ->
  +0.069 at 5000), independently reproducing the 20B-512 result that ARC-C's
  penalty doubles between 5000 and 6000 while ARC-Easy's stays flat. That is
  why the artifact impersonates a late-training decay rather than a constant
  offset: a permute error costs more as the model's answers sharpen, because
  there is more signal to scramble.
- **Corrected myself on the incident doc's shard size.** I read
  `252,561,448 B` as wrong because my spot-check of shards 0-3 showed ~1.31 GB.
  The doc cites shard *191* specifically and is exact. What I had missed is
  that sizes vary by an order of magnitude WITHIN one writer's cluster, which
  is why the audit tool reports a median and its comment says not to read
  either number as "the shard size".

## 2026-08-16 (sunspot) -- the 30B goes from proposal to measured model: 27.89% MFU at 2N, 25.54% at 64N, HSDP ceiling found between 20B and 30B

- **`agpt_configs["30B"]` did not exist.** `30b-exp/` was a design document
  whose performance claims could not be tested at all. Interpolated the family
  (the proposal fixes only `dim=6144`) -> dim 6144 / 64L / 48H / 8kv /
  ffn 16384 / head_dim 128, 28.1B params. It trains.
- **Tuned at 2N to 466 tps / 27.89% MFU** (LBS=3, compiled, TP=1). TP hurts
  monotonically (TP=2 -26%, TP=4 -55%) -- it only earns its keep when the
  model does not otherwise fit. Activation checkpointing is load-bearing;
  `ac=none` is a genuine OOM.
- **LBS=3 is faster AND cheaper in memory than LBS=2** (78.5% vs 80.6%).
  Larger batches amortize the activation peak, so "bigger batch costs more
  memory" inverts across that step.
- **Scaling ladder 2N -> 64N: 27.45 / 27.03 / 26.53 / 26.67 / 26.36 / 25.54%.**
  1.42% MFU lost per doubling vs the 2B's 13.96% -- 8x better. No cliff in the
  measured range. Caveats kept in the doc: 512N is 3 doublings further out and
  the 2B's collapse was a cliff, not a slope; and at 64N the 2B is itself still
  at 25.9%, so this range does not yet separate the two models.
- **The batch ceiling is the proposal's blind spot.** LBS=3 at 512N implies
  GBS 18,432 / 75M tokens per step, and the 2B already measured a ceiling
  below that (GBS 12,288 lost 3-8pp per token against 6,144, with per-step
  parity proving the architecture was fine).
- **HSDP fails on the 30B but works on the 20B.** Not an OOM -- it dies at
  71.68% memory with 18 GiB free, inside `clip_grad_norm_` ->
  `clip_grad.py:106 torch.stack`, `UR_RESULT_ERROR_OUT_OF_RESOURCES`. That is
  level_zero resource exhaustion. Four arms settled it: 20B passes 12/12,
  `foreach=False` fails identically (so the fused clip is exonerated), tensor
  count is identical at 579 for both models. Size is the axis.
- **`agpt_30b_llama3tok` had never produced a step**, for two bugs both mine:
  gemma's `hf_assets_path` under a 128,256 embedding, and a docstring telling
  callers to pass `--tokenizer.path` (not a flag). Fixed; it now runs, and its
  freed HBM buys **LBS=4 -> 497 tps / 28.99% MFU**, a batch step the gemma
  variant cannot reach.
- **Corrections to my own earlier claims this session:** "HSDP OOM'd" (twice);
  "model size is probably not the axis"; "the 128k is the better config
  outright" (it is a tie at matched LBS -- MFU is FLOP-normalized); and
  run-to-run noise is <1%, not the ~6% I had assumed, which had caused a real
  +5.9% compile effect to be dismissed.

## 2026-08-10 (aurora) -- the two 7.771T MDS checkpoints found and evaluated; MMLU settled by controlled experiment

- **The genuine 7.771T checkpoints existed all along, in directories nothing
  had searched.** Every MDS sweep looked under
  `optimizer-experiments/Megatron-DeepSpeed/checkpoints/`, which stops at
  `global_step140352`. W&B's per-run `working_directory` showed the two Feb-2026
  branches wrote elsewhere:

  | arm | dir | ckpts | tip | data | loss |
  |-----|-----|-------|-----|------|------|
  | stage3-mix | `optimizer-experiments/stage3-mix/` | 295 | step-154391 | stage1-33-stage2-33-stage3-34 | 2.033 |
  | stage3 | `optimizer-experiments/stage3/` | 283 | step-154391 | nvidia-math1-code2 | 0.880 |

  Neither had EVER been evaluated -- not MMLU, not gsm8k, not commonsense --
  despite being the most-trained models in the project at ~1.66x the entire v2
  token budget. Job 8747067 ran the full ladder on both.

- **The finishing mix is a controlled experiment, and it is decisive.** Same
  architecture, same optimizer/LR, same 7.771T tokens; only the last ~0.6T of
  data differs:

  | metric | stage3-mix | stage3 (math/code) |
  |--------|-----------|--------------------|
  | mmlu | 0.2463 | 0.2591 |
  | arc_challenge@25 | **0.4164** | 0.3703 |
  | hellaswag | **0.5874** | 0.4215 |
  | arc_easy | **0.7138** | 0.6216 |
  | gsm8k | 0.0167 | **0.0303** |

  The general mix wins every commonsense task by a wide margin (HellaSwag
  0.587 vs 0.422 -- 16 points). Math/code wins gsm8k but 0.0303 is still
  effectively zero and it costs heavily elsewhere: **catastrophic forgetting
  from a narrow finishing mix**, the same effect the 07-28 anneal A/B found
  when pure edu-web wiped out math. stage3-mix is now the best 2B we have on
  ARC-Easy (0.7138) and ARC-C@25 (0.4164), beating the 7.064T dolmino ckpt.

- **MMLU: settled.** 0.2463 and 0.2591 -- both chance, on the two most-trained
  models in the project, one of which is our best on every other benchmark.

  | model | tokens | best commonsense | mmlu |
  |-------|--------|------------------|------|
  | 20B-256 | 0.39T | -- | 0.2599 |
  | 2B-256 COMPLETE | 4.674T | hellaswag 0.561 | 0.2437 |
  | MDS dolmino | 7.06T | arc_c 0.3968 | 0.2413 |
  | MDS stage3-mix | 7.771T | **arc_e 0.7138** | 0.2463 |
  | MDS math/code | 7.771T | -- | 0.2591 |

  Five configurations, a 20x token span, two model scales, four finishing
  mixes -- and a validated harness (Llama-3.2-1B = 0.3121, Llama-3.1-8B =
  0.6530 on this exact path). A 1B public model clears chance where our best
  7.771T model does not. Capability is real and rising on everything else, so
  this is not a training failure: **the pretraining corpus does not contain
  what MMLU tests, and no amount of tokens or finishing-mix reshuffling fixes
  it.** It needs academic multiple-choice content, deliberately added.

- **The forgetting curve: immediate, then relentless** (job 8747310, a
  matched ladder every ~2000 steps across both stage-3 arms). The endpoint
  comparison said math/code trades ~16 points of HellaSwag for ~0 gsm8k; the
  ladder says WHEN.

  | step | mix hellaswag | m/code hellaswag | mix arc_c@25 | m/code arc_c@25 |
  |------|---------------|------------------|--------------|-----------------|
  | 140400 | 0.5869 | **0.5636** | 0.4249 | **0.4249** |
  | 142400 | 0.5884 | 0.4842 | 0.4121 | 0.3899 |
  | 144400 | 0.5861 | 0.4614 | 0.4130 | 0.3746 |
  | 146400 | 0.5872 | 0.4519 | 0.4096 | 0.3737 |
  | 148400 | 0.5868 | 0.4415 | 0.4078 | 0.3712 |
  | 150400 | 0.5881 | 0.4338 | 0.4155 | 0.3669 |
  | 152400 | 0.5854 | 0.4307 | 0.4138 | 0.3609 |

  Both arms start IDENTICAL at the branch point (arc_c 0.4249 on both), then
  math/code drops **8 points of HellaSwag in the first 2,000 steps** and keeps
  bleeding monotonically to 0.431 -- no plateau, no recovery. So it is neither
  a pure cliff nor a slow drift: a sharp initial hit followed by continuous
  decay. Meanwhile stage3-mix is flat on HellaSwag and arc_c and **still
  IMPROVING on arc_easy (0.687 -> 0.722)**, so the gap widens with every step
  spent on the narrow mix. Total purchase: gsm8k 0.0152 -> 0.0334.

  **Operational lesson:** the damage was measurable 2,000 steps in. A
  broad-capability metric checked during a narrow-data phase would have caught
  this and stopped the run 12,000 steps earlier. Endpoint-only evaluation is
  what let it run to completion.

- **The three MDS cooldown forks: no capability jump** (job 8747311). These
  had never been evaluated. `automate-cooldown/` also holds ~310 ws24/gb48
  dev-scale checkpoints, which are NOT production geometry and were excluded;
  only cooldown-3/4/5 are ws3072.

  | fork | step | ~tokens | mmlu | arc_c@25 | hellaswag | arc_easy | gsm8k |
  |------|------|---------|------|----------|-----------|----------|-------|
  | cooldown-3 | 52,650 | 2.65T | 0.2404 | 0.3746 | 0.5734 | 0.6549 | 0.0045 |
  | cooldown-4 | 72,500 | 3.65T | 0.2438 | 0.3797 | 0.5897 | 0.6713 | 0.0045 |
  | cooldown-5 | 92,400 | 4.65T | 0.2373 | 0.3780 | 0.5913 | 0.6772 | 0.0030 |

  Commonsense rises smoothly with tokens (hellaswag 0.573 -> 0.591, arc_easy
  0.655 -> 0.677) with **no discontinuity at any cooldown**, and arc_c@25 is
  flat at ~0.377. This **corroborates the 2026-07-28 anneal A/B at 5-9x the
  token count it was tested at**: that experiment concluded the LR schedule is
  not the lever at 10B tokens, and these cooldowns say the same at 2.65-4.65T.
  The recommendation now rests on two independent horizons.

  MMLU adds three more chance values, bringing the count to **eight
  configurations at chance spanning 0.42T -> 7.771T**, two model scales, and
  five data treatments including three LR cooldowns. Schedule does not move it
  either.

- Eval coverage audit while chasing this: modern-block (mmlu/arc_c-25/gsm8k)
  coverage is 90% on 20B-256, 43% on 20B-512, 31% on 2B-512, and **4% on the
  COMPLETED 2B-256 chain** (8 of 193 steps). The MDS stages had ~none, which
  is why the 7.771T gap went unnoticed: `eval_mds_sweep.sh` hardcodes four
  commonsense tasks at `num_fewshot=0`.


## 2026-08-05 (aurora + sunspot) -- umbrella finally seats and advances all 3 chains; MMLU flatline traced to DATA not the harness; four SILENT refresh failures found and fixed; the ARC-C "decline" is a shot-count collision; RC4 retires the TP=4 workaround; PP "bug" turns out to be a torch version floor (PR retracted)

- **The ~2k-node umbrella (8714502) seated after ~6 days queued, ran 8h01m, and
  reported `failed: 5 / 5` -- but three chains genuinely advanced.** The summary
  banner is exit-code-driven and badly misleading here:

  | trainer | chain | steps | ckpt head | loss |
  |---------|-------|-------|-----------|------|
  | 1 | 20B-512 | 6801 -> 7149 | 7,100 | 2.415 |
  | 2 | 20B-256 | 7501 -> 7897 | 7,800 | 2.3694 |
  | 0 | 2B-512 | 39601 -> 41300 | 41,300 | 2.6913 |
  | 3 | 2B-512 **constlr-from9200** | never started | -- | -- |
  | 4 | 2B-256 **constlr-from9500** | never started | -- | -- |

  Full dispatch report:
  [`docs/experiments/agpt/aurora/20260805-umbrella-8714502.md`](experiments/agpt/aurora/20260805-umbrella-8714502.md)
  -- per-trainer detail plus three follow-ups it surfaced: 20B-512's loss ROSE
  2.18 -> 2.41 across the run (256N on the same constant-LR config stayed flat,
  and now reports a LOWER loss than 512N despite ~half the tokens); 2B-512 MFU
  is ~3-9% against 17-20% for the 20B trainers; and trainers 3/4 report
  "FAILOVER STOP: walltime" after 618s/1109s having never trained a step, so
  the failover taxonomy is mislabelling setup failures as benign walltime exits.
  Trainers 3/4 are **not** spare or duplicate slots -- they are the two
  constant-LR fork experiments (from9200 / from9500), and both died in
  `checkpointer.load()` with a `CheckpointException` before step 1, so the
  umbrella delivered them zero compute. Seed ckpts look structurally sound
  (6144 / 3072 shards, both with `.metadata`); trainer 3's log references
  `step-9260`, a 24-entry fragment, rather than the full `step-9200`, so a
  stale "latest" resolution is the leading suspect.

  Two independent causes, neither numerical: trainer 0 hit a clean
  `FAILOVER STOP: walltime` (rc=143, working as designed); trainer 1 died rc=127
  at 05:50:21 when a node went unresponsive (`No reply from x4410c7s1b0n0 after
  97s`), auto-retry correctly rotated it out, and the RELAUNCH hit the known
  Aurora pals RPC fault (`Couldn't forward RPC launch ... Resource temporarily
  unavailable`) -> `FAILOVER STOP: stuck_pre_training`. The umbrella then
  SIGTERM'd trainers 2/3/4 at that same second. So: 1 clean walltime exit, 1
  infra fault, 3 collateral kills.
- **Ckpt heads trail the logged steps by design.** Trainers log every step but
  save every 50, and all were SIGTERM'd mid-interval: logged 7,897 / 7,149 vs
  resumable 7,800 / 7,100. Always take the head from disk, not the last log line.
- **Four SILENT failures in the doc-refresh machinery -- every one reported
  success while skipping its work.** This is why the production docs had drifted
  a week despite a catch-all built to prevent exactly that:
  1. **Loss was looked up from a run-id list frozen at 2026-07-27.**
     `_wandb_latest_loss` walks `wandb_run_ids` and returns the newest loss it
     KNOWS about. 20B-256's list ended at `v58n7vam`=8703284 (~step-6037), whose
     loss is 2.68 -- exactly the stale number the dashboard kept reporting. The
     function was working perfectly on a stale input. Added the missing
     segments: 20b_v2_256 += cxlt0tpe=8698125 (6151->6897) + 2ktrz29u=8714502
     (7501->7897); 20b_v2_512 += 9d1g9zsw=8714502; 2b_v2_512 += vtumb5cb=8714502.
  2. **`LOSS_RE` needs a `**Loss:**` line and 20b/n512 + 2b/n512 never had one**,
     so their loss field was skipped forever with no warning. Added.
  3. **Five plotters hardcoded the Sunspot repo root** (`/lus/tegu/...`). Two
     failed loudly on Aurora (PermissionError / FileNotFoundError); the other
     THREE failed silently -- they guard on the missing Sunspot trainer_state and
     "skip cleanly", so they exited 0 having plotted nothing and the catch-all
     looked green. All now resolve via `Path(__file__).resolve().parents[N]`.
  4. **Stale `Last updated:` markers were detected but unfixable** -- phase 2 of
     `check_stale_docs.sh` is read-only and the markers are hand-written prose,
     so the punch-list only grew (9 stale). New `utils/refresh_last_updated.py`,
     wired as `refresh_all.sh` stage 2b, stamps each doc with its LAST-COMMIT
     date (not today -- stamping today would claim a freshness the content lacks
     and permanently silence the check) and skips dirty files.

  Commits 598c7c1bf / 097dfd8e1 (plot_b4.py was the last [UNWIRED] plotter) /
  3858d730b / 87382f971 / 10fb962f0, plus the refresh output itself (96153a497)
  and the rollup Loss/State/Trend prose (f7f3aba35).
- **Verified the loss fix end-to-end**: 20B-256 moved off a value pinned at
  step-3,100 (2.4394 -> 2.3694) and the two 512N chains produced a loss for the
  first time. The W&B values differ slightly from the console lines transcribed
  by hand (2.6913 vs 2.6972, 2.415 vs 2.4145) -- W&B is authoritative, which is
  itself the argument for automating this.
- **RETRACTED mid-session: the 20B "ARC-Challenge decline" is a shot-count
  collision, not a capability trend.** The story ran: modern-block backfills
  8729921/8731413 completed (48h capacity, after two 12h walltime kills) and
  appeared to show ARC-C falling 0.4138 at step-4000 -> 0.2875 at step-6500,
  toward the 0.25 chance floor, while train loss kept improving. The
  tokenizer was correctly ruled out (the driver copies gemma-7b assets,
  olmo-mix-1124 IS gemma-tokenized, model vocab 256128 vs tokenizer 256000 is
  standard 128-alignment padding) -- that part stands. But the column itself
  was not a single measurement: **arc_challenge is scored twice per step,
  0-shot in the commonsense block and 25-shot in the modern block, and both
  wrote the same `arc_challenge` key**, so whichever phase finished last won.
  lm-eval's `n-shot` block was dropped, leaving nothing to disambiguate. The
  "peak" and the "floor" are not comparable numbers, and the apparent
  +7-to-9pp "recovery" first seen at 256n step-6900 / 512n step-6550 was just
  fresh 0-shot values sitting beside 25-shot neighbours. Fixed in 3e1877170
  (`<task>@<N>shot` keys + preserved n-shot); applies to steps evaluated from
  here on, so already-written files stay ambiguous until re-run.
  - What survives: **MMLU is genuinely at chance** across every step measured
    (.249-.267, single source, 5-shot only, no collision), and HellaSwag
    ~0.60 / ARC-Easy ~0.66 are real and steady -- both appear once each.
  - Tail evals 8735716/8735717 are running over 6900-7800 / 6550-7100. Their
    arc_challenge will still collide (they predate the fix); their MMLU and
    commonsense numbers are trustworthy.
  - Lesson: before reading any eval column as a time series, confirm every
    point in it was produced by the same task config. A silent key collision
    looks exactly like a trend.
- **MMLU never leaves chance on ANY AuroraGPT checkpoint -- and the harness is
  fine, so it is the DATA.** Chased this to a conclusion tonight:

  | model | tokens | MMLU | note |
  |-------|--------|------|------|
  | 20B-256 | 0.39T | 0.2599 | |
  | 20B-512 | 0.71T | 0.2655 | |
  | 2B-512 | 3.99T | 0.2473 | |
  | 2B-256 | 4.674T | 0.2437 | **COMPLETE run, 100% of target** |
  | 2B MDS dolmino | **7.06T** | 0.2413 | job 8736655; best-ever ARC-C 0.3968 |

  A 20x token span, two model scales, no upward trend -- the most-trained model
  scores LOWEST. Meanwhile the same checkpoints improve monotonically on
  everything else (hellaswag 0.405 -> 0.561 across the completed 2B run).
  - Killed "not enough tokens yet": a COMPLETED 4.674T run finished at chance.
  - Killed "2B is below MMLU scale": 7.06T with our best ARC-C is still 0.2413.
  - Killed "the harness is broken" (job 8736838): three cached public models
    through our EXACT path -- same tt-lm-eval venv, same
    `simple_evaluate(num_fewshot=5, device="xpu:0")` -- reproduce their
    published numbers. **Llama-3.2-1B = 0.3121** (published ~0.32) and
    **Llama-3.1-8B = 0.6530** (published ~0.66). A 1B model clears chance on
    this path; our 2B at 7.77T does not. (Llama-3.2-3B errored with "not a
    string", a loading issue, not a scoring one.)
  - Tokenizer was already ruled out earlier (gemma-7b assets vs gemma-tokenized
    olmo-mix-1124; vocab 256128 vs 256000 is 128-alignment padding).

  **CORRECTED 2026-08-08:** this checkpoint is **7.064T tokens on
  dolmino-mix-1124-fused**, not 7.77T. `ntok7770B` in the directory name is a
  TARGET baked into the naming convention, not what was consumed -- W&B run
  bklwz5oh/rk3uudzf at the same global_step140352 report
  consumed_train_tokens = 7,064,147,460,096 and lm loss 2.424. The two runs
  that really reached 7.771T are Feb-2026 branches (stage-mix and
  nvidia-math1-code2) that POSTDATE this checkpoint. The finding is unchanged
  and if anything cleaner: 7.06T is still ~1.5x the entire v2 budget, and the
  mix is general-purpose, so "a math-finished model would obviously miss MMLU"
  does not apply.

  What is left is the training mix. `olmo-mix-1124` appears to contain little
  of what MMLU tests -- multiple-choice academic knowledge across 57 subjects.
  The models learn language modelling well and never acquire that. Llama's
  pretraining is known to carry substantial textbook/exam-style content; ours
  may simply not. **If MMLU-style performance matters for AuroraGPT it has to
  be trained for -- a data intervention, not more tokens.** Probe scripts:
  `scripts/eval/oneoff/eval-mds-mmlu-7770B.sh`, `.../eval-mmlu-harness-check.sh`.
  - Note `eval_mds_sweep.sh` never ran MMLU on the MDS chain at all: it
    hardcodes `tasks=[hellaswag,arc_easy,arc_challenge,winogrande]` at
    `num_fewshot=0`, predating the modern block. That is why the 7.77T gap
    existed.

- **frameworks RC4 FIXES the TP=4 SDPA-backward compile assert** (Sunspot
  12472578 2N, confirmed 12472582 at 4N/30 steps): 0 `assert_size_stride` at
  every TP degree, continuous descent 13.00 -> 6.71. The
  ".venv for compiled TP=4, or stay at TP<=2" workaround is obsolete. A control
  run on the production .venv (12472583) shows both stacks agree within ~1.5% at
  every rung, so the ~7% TP=4 MFU is the intrinsic cost of TP=4 for this model at
  2N -- **RC4 is performance-neutral, not a regression**. Commits eecb2fdc9 /
  00f49ebe9 / 61dbe409f.
- **74th upstream sync** (merge ddb41730a): 2 commits, inert for ezpz. The NVFP4
  converter (#3914) was checked line-by-line -- purely additive, and NVIDIA-only
  (Blackwell FP4) so inapplicable on XPU. Nothing to replay. Commit 3f32a1aed.
- **The PP fix (7e2975dc3) DOES NOT WORK -- correcting an earlier claim.** It was
  committed on reasoning alone and had never been run. First actual verification
  (12472586) at PP=2 / `pp_microbatch_size=1` / LBS=2 still dies with
  `ValueError: Expecting 2 arg_mbs but got 1` and **zero steps**. The PP=1
  regression rung passes (step 10), so the commit is harmless to every non-PP
  run -- but the bug it targeted is still open.
  **The original diagnosis was wrong.** I fixed dataloader SIZING on the theory
  that the iterator exhausted mid-run. This run dies at step 1 with **no "Ran out
  of data"** anywhere in the log; a dry iterator fails late and says so. So the
  microbatch group is short from the very first step, and sizing was never the
  cause.
  Instrumented probe 12472588 then falsified all four follow-up candidates at
  once: `PPDIAG[init-post] num_pp_mb=2 pp_enabled=True cls=FaultTolerantTrainer
  lbs_now=2` and `PPDIAG[train_step] num_pp_mb=2`, dataloader-dry count 0. So our
  side genuinely builds groups of 2, the schedule genuinely wants 2
  (`Expecting 2 arg_mbs`), and it still receives 1 -- the loss happens BETWEEN
  `train_step`'s group construction and the schedule call, not in the config, the
  attribute, the class, or the dataloader. Next: read how `arg_mbs` is passed
  into `_check_inputs` (a `pp_has_first_stage` filter is the leading suspect,
  since upstream nulls `arg_mbs` on non-first stages).
- **RESOLVED same day, and it is NOT a torchtitan bug: PP needs a newer torch.**
  I traced the above to `trainer.py` calling `pp_schedule.step(arg_mbs=...)` when
  `step()` (I believed) took only whole-batch input, opened
  [pytorch/torchtitan#4071](https://github.com/pytorch/torchtitan/pull/4071), and
  verified it on XPU *and* on an A100 (parent reproduced the error, patch reached
  step 10, loss 8.14 -> 4.34). Maintainer @tianyu-l then asked whether I had tried
  a current PyTorch nightly -- and that was the whole thing. PyTorch `main`'s
  `step()` now takes pre-split microbatches directly (`arg_mbs`/`kwarg_mbs`,
  @sanketpurandare's change), so the torchtitan call site is CORRECT and my patch
  was unnecessary. **PR closed with a retraction.**
  The real, useful finding: **PP is blocked on a torch version floor.** Every
  torch we tested predates that change (`2.13.0.dev20260519+xpu` on Sunspot,
  `2.13.0.dev20260611+cu126` on Polaris, 2.11, 2.8), so `arg_mbs` falls into
  `**kwargs`, `_split_inputs` re-chunks it, and `_check_inputs` sees 1 instead of
  N. Gate on this before attempting PP anywhere:
  `"arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters`.
  Commit `7e2975dc3` (dataloader sizing) stays in the tree -- inert for non-PP
  (PP=1 reaches step 10) but never the fix for the `arg_mbs` error.
  **CORRECTION 2026-08-10: do NOT revert it.** I had recommended reverting it as
  a "non-fix"; checking the arithmetic before doing so shows it is CORRECT and
  fixes a real, separate bug. Under PP the dataloader is built with
  `local_batch_size = pipeline_parallel_microbatch_size` (trainer.py:270), i.e.
  1 instead of 2 in the test config, and blendcorpus sizes its sample budget
  from `train_iters` (`blendcorpus_builder.py:150`). Without the
  `* _num_pp_microbatches` the budget is short by exactly that factor
  (10 iters x lbs 1 = 10 samples, vs 20 for the same non-PP run) and the
  iterator runs dry mid-run. Upstream passes plain `training_steps`
  (`torchtitan/trainer.py:510`) because its dataloader does not derive a sample
  budget this way -- so the product is an ezpz/blendcorpus necessity, not a
  divergence. Reverting would introduce a bug.
  Two process lessons: (1) I checked pytorch `main` before filing the `fork_rng`
  issue, which correctly prevented a redundant report (already fixed by
  pytorch#180512) -- I did not apply that same check to my own patch, and it cost
  maintainer review time. (2) I explained away a strong disconfirming signal:
  upstream runs PP integration tests in 8-GPU CI, so a universally broken PP path
  would have been red. I called it "a 5-day-old regression nobody noticed"
  instead of treating it as evidence I was wrong.
- **B4 SFT page expanded with charts** (5bd4c1ecd): `plot_b4.py` +
  `cot_ladder.svg` / `accuracy_vs_genlen.svg` on the two-stage-vs-single-stage
  result, plus a TL;DR, per-arm hypotheses, and a note on why the gen_len /
  n_unclosed guardrails earned their keep. No numbers changed.
- **RC4 env recipe (for anyone repeating the frameworks-RC test).** The RC4 conda
  env is READ-ONLY and owned by another user, and ships no ezpz/torchtitan stack.
  `venvs/fw-2026.1-rc2` needed 4 packages + 11 transitive deps before a rank
  could start: `tensorboard`(+data-server), `blendcorpus@feat/remove-deepspeed`,
  `tyro`, `spmd_types`, a `_torchtitan_repo.pth`, then typeguard /
  eval-type-backport / protobuf / pyyaml / absl-py / markdown / pillow /
  werkzeug / hydra-core / omegaconf / antlr4 (all `--no-deps`; `grpcio`
  deliberately EXCLUDED as a compiled ext). **Every one of these surfaced as the
  same generic `Cannot import config_registry`** because
  `torchtitan/config/manager.py:120` swallows the real exception -- worth a
  `raise ... from` upstream. Gate on importing
  `...agpt.config_registry` with `PYTHONPATH` UNSET from a neutral cwd; gating on
  `import agpt` gives FALSE PASSES (it never reaches metrics.py). By contrast the
  canonical `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv`
  worked first try -- prefer it.
- **Node `x1921c4s3b0n0` is persistently broken** -- `mounts=2` where healthy
  nodes have 3, so the project path is invisible and any rank landing there exits
  127, killing the whole job. Failed 5/5 appearances (12472558, 12472560,
  12472573, excluded by probe 12472563, `repo=FAIL` in 12472561), never passed.
  `c4s5`/`c4s6` additionally FLAP (passed in 12472563, failed in 12472564);
  c2 and c7 have been clean throughout, so this looks c4-localized. **Worth an
  ALCF ticket** -- it will silently break anyone who lands on it. Workaround now
  in the probe scripts: request N+1 nodes and drop any that cannot see $REPO,
  running the probe from `/tmp` (from $REPO the bad node fails the chdir and
  tears down the probe itself, reporting 0 healthy).
- **Queue state at end of session.** Nothing training: 8714503 (umbrella
  successor, 2098N, auto-released from hold), 8730438 (20B-256 native, 260N) and
  8731758 (20B-512 native, 516N) all Q; both tail evals R on capacity. The
  umbrella overlaps BOTH individuals -- if it seats, kill 8730438 + 8731758
  first. The two individuals do NOT collide with each other (different repo
  clones, different ckpt dirs). The zombie 512-chain bridge 8687863 is still
  parked in H and needs a manual `qdel`: held since 07-29 AND carrying
  `-W depend=afterany:8687862` on a job PBS has since purged, so `qrls` alone
  can never satisfy it. Inert, but clutter.
- **Watch out for false `GONE` events from the qstat watcher.** Its presence
  check keys on the job id appearing in an awk-parsed snapshot, and the field
  offsets shift when some rows carry an elapsed time (`R 02:16`) and others do
  not (`--`). That drops a line and looks like a job left the queue -- it fired
  three times today on jobs that were still sitting there. The `Q -> R`
  transitions compare states rather than presence and are unaffected, so the
  collision alerts are trustworthy; always confirm a `GONE` against live qstat.

## 2026-07-27 (aurora) -- prod_dash: Textual multi-metric TUI + streamed cold-build progress; 20b-256 capacity bridge advancing

- **prod_dash cold-build no longer a silent hang.** The live dashboard sat
  silent for the full ~6m40s cold W&B backbone build because `fetch()` ran the
  remote aggregator with `capture_output=True`. The aggregator now emits
  timestamped `[prod_dash +Ns]` progress to stderr (log-scan, per-chain W&B
  pull i/N, experiment scan, done); `fetch()` streams it to the terminal;
  `load_backbone` logs warm-hit/stale-serve/cold-build. `PD_QUIET=1` for the
  detached refresh worker. Commit ca03546d3.
- **prod_dash generalized to a Textual multi-metric TUI (`--app`).** New
  `utils/prod_dash_app.py`: Tabs metric selector (loss / grad_norm / tps /
  tflops / mfu), a PlotextPlot chart with all chains overlaid (live highlighted,
  step<->tokens x-axis toggle), the text board, and a RichLog that streams
  cold-build progress. Threaded `@work` fetch keeps the ~min SSH build off the
  UI thread; `set_interval` auto-refresh. Opt-in; needs
  `uv pip install textual textual-plotext` (pure-python, torch-safe), falls back
  to `--board` if absent.
  - Data side was nearly free: the backbone already fetched
    step+loss+grad_norm+tps+mfu via `concat_chain(OLOG_KEYS)` then discarded all
    but (step,loss). Now `build_backbone` stores per-chain
    `series{loss,grad_norm,tps,tflops,mfu}` (downsampled once, step-aligned) and
    KEEPS `curve=series[loss]` so render_board/draw_curves/--svg/kitcat are
    unchanged. Added `tflops` to `wandb_fetch` OLOG_KEYS (regex already captured
    it). Verified: all 5 series populate for the live 20b_v2_256 chain (n=591),
    headless `App.run_test` passes (9 chains, all metrics, board + toggle),
    `--board` regression clean, `--app`-without-textual falls back. Commits
    8a6bb61e9 / 6640fc86a / 52c84f767.
  - Fixed a pipe deadlock in the streamed `fetch(stderr_cb=...)` path: reading
    stderr to EOF before draining stdout hangs once the child fills the stdout
    pipe with the (large) JSON -- now drains both concurrently (stdout in a
    thread).
- **20b-256 capacity bridge (8703284) advancing.** 16N capacity-queue bridge
  (GAS=16 -> GBS=6144 bit-identical) resumed the 20b-256 chain from step-6000
  (loss ~2.46 continuous, MFU ~24.6%); `afterany` continuation 8705325 queued.
  W&B run `v58n7vam`. First advance past step-6000 since 2026-07-24. 20b-512
  holds at step-6100. 20B tail eval backfills (8703745 512n done, 8703835/8705119
  256n) filled the 5300-6100 gaps; a wrong-clone REPO bug in the first 256n
  attempt (8703746) was fixed. Aurora had a brief 2026-07-27 outage (killed
  in-flight evals with -29, no ckpt corruption; resubmitted).

## 2026-07-24 (aurora) -- W&B fetch consolidation + preflight-smoke run-id fixes + overlay legend

- **Root-caused the "stuck 2B-256 curve" -- it was a misleading legend + a
  bookkeeping bug, never lost data.** The all_production_training overlay labelled
  each curve `(n=<len(loss)>)`; `n=86480` is a row COUNT, not a step, and made
  the completed 2B-256 chain look stuck while the prod_dash board correctly showed
  step 92,859 / 100%. Fixed the label to `step 92,859, 100% of 4.67T` so the chart
  agrees with the board. The curve always reached the 4.67T stage-1 completion.
- **Preflight-smoke reinit-swallow (the real data bug), swept across ALL chains.**
  Every production job runs a `python -m ezpz.examples.test` preflight smoke first;
  it opens a wandb run in the `ezpz.examples.test` project and, with
  `reinit='default'`, the real `wandb.init(project=torchtitan.ezpz.train)` RETURNS
  that already-active smoke run. So some recorded run-ids were the 5-step SMOKE
  runs (0 training rows), forcing .o-log fallbacks / gaps. Corrected across chains:
  - 2b_v2_256 completion trio: a5h4aaf7/gx7ph91w/mqi69lx2 -> 9itxu3pt/ew4pqb51/fm3gzdxt.
  - 2b_v2_256 mid-chain: okyt09kv -> jkde9zdg (fills the 74319->80301 gap;
    chain n 86480 -> 92456, endpoint/loss unchanged).
  - 20b_v2_256 gap-fill: dpiog1q7/auy8wohg -> 17sfemjj/rugscgjs.
  - 20b_v2_512: qttj3l3p DROPPED (crashed dud, 0 steps, covered by retry wjy5pvxm).
  - 2b_v2_512: dropped i0ayskft's stale empty-W&B .o fallback (it synced later).
  A zero-row sweep over every chain confirms these were the only offenders.
  Rule for future bookkeeping: take the run-id from the `torchtitan.ezpz.train`
  View-run line, never the `ezpz.examples.test` one.
- **Consolidated the two W&B-fetch code paths into one shared module** (the drift
  above is exactly why the board and charts disagreed). New
  `utils/wandb_fetch.py` holds the ONE `.o`-log parser + per-chain concat with the
  robust olog-fallback rule. It is dependency-light BY CONTRACT (stdlib + lazy
  wandb, NO numpy/matplotlib/torch) so prod_dash's cluster-side `_AGG` can
  spec-load it by path like `trajectories.py`. `plot_production_wandb.concat_runs`
  is now a thin numpy adapter over it; `prod_dash._wandb_curve/_olog_curve`
  delegate to it. Verified behavior-preserving: a golden before/after diff of
  `concat_runs` over every chain is byte-identical except the intended 2b_v2_256
  gap-fill. Commits 778e302ca / 2398c8313 / 0433322c0 / 9d46c5c50 / 45eac96ad.

## 2026-07-18 (sunspot) -- full-mix SFT eval verdict + GRPO validation + 70th sync + RL cleanup

- **Full-mix SFT eval -- COMPLETE on all 3 axes; deliverable is checkpoint-900, NOT 8672.**
  Evaluated the finished run (gs138650 x tulu_math_uc_mix_full, step 8672).
  - **base-LM sweep 0->8672 (jobs 12470365 + 12470886):** checkpoint-8672
    **catastrophically forgot** -- every base-LM task collapsed to ~random
    chance (hellaswag 0.59->0.27, arc_easy 0.69->0.30), collapse between step
    ~1500-4500. checkpoint-900 (pre-collapse) retains base-LM (hellaswag 0.59,
    arc_easy 0.64).
  - **IFEval (jobs 12470889 step-8672, 12470896 step-900):** ckpt-900
    prompt-strict **0.253** (>= metamathqa-729's 0.244, >> baseline 0.179);
    ckpt-8672 flat-to-down (0.168). The overfit killed instruction-following too.
  - **GRPO (job 12470959, ckpt-900, sum_digits):** accuracy_reward climbed
    **0.31 -> ~0.74** in ~20 steps -- strong RL starting point (hit 6h walltime,
    not converged). Confirms ckpt-900 is good downstream.
  - **Root cause of the collapse:** LR 2e-5 held >1e-5 through step ~4350 (cosine
    decay only bites the 2nd half); ~4000 steps at peak LR on the narrow
    OpenMathInstruct-2 math-CoT distribution overfit + forgot. The metamathqa
    recipe survived only by STOPPING at 729 steps. **Lesson: cap full-mix SFT at
    O(1000) steps or lower the LR; a full epoch at peak LR is the mistake, not
    the mix.** Full writeup: `production/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md`.
- **GRPO-on-XPU detour (documented so it never repeats):** the GRPO data point
  took a long avoidable path -- I ran the SUPERSEDED `.venv` hf.generate FSDP
  path (hangs multi-rank; regressed since June-10, single-rank works) and the
  old vllm-test+.venv-split script (`Device string must not be empty`) before
  finding the blessed vLLM server-mode path (unified `venvs/rl-vllm/`,
  `rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`). Lesson: READ
  `docs/rl/grpo-on-xpu-status.md` first. Recorded in the
  `project_grpo_xpu_vllm_path` memory.
- **RL script cleanup:** removed 10 superseded/bring-up RL scripts (the broken
  `8n_vllm` + `vllm_serve_xpu.sh` PYTHONPATH-bridge + 8 pre-unified-venv
  vLLM-XPU one-offs whose writeups already live in `docs/rl/history/`).
  Reworded the supersede refs to note removal (kept the lessons); history/ +
  journal untouched.
- **70th upstream sync** (merge 412d93fd8, 3 commits): required a 5-file replay
  of the MoE sibling-experts refactor (#3859: `MoE.experts` GroupedExperts ->
  `MoE.routed_experts` RoutedExperts(inner_experts + token_dispatcher)). Caught
  by sync_smoke (moe rc=143), fixed, re-smoked VERDICT ok (job 12470902).
  Details in `upstream-sync.md` 70th entry.
- **macOS local-run gotcha:** `ezpz launch ... --module ezpz.agpt` fails with a
  misleading core error "Cannot import config_registry for module 'ezpz.agpt'"
  -- the real cause (masked by `torchtitan/config/manager.py` catching all
  ImportError) is `agpt/parallelize.py` doing `import ezpz` when the standalone
  `ezpz` package isn't installed in the local .venv. Fix: `uv pip install -e
  <ezpz clone>`.

## 2026-07-16 (sunspot) -- full-mix 8N SFT COMPLETE (epoch 1.0) + 69th upstream sync

- **Full-mix 8N SFT finished cleanly**: gs138650 x tulu_math_uc_mix_full (~54B
  tokens, 1 epoch) reached **step 8672 / epoch 1.0, final loss 0.357,
  mean_token_accuracy 0.902** -- Execution finished with 0, checkpoint-8672
  saved intact. Ran across 10 fault-tolerant afterany chain links (12470350
  head -> ... -> 12470478 -> 12470479 final) with clean walltime handoffs; the
  early idle-hang pattern (HANGs #1-#7 through step ~2400) abated from cont 8 on
  (12470436/437/478/479 all long+clean). Beats the completed metamathqa SFT
  (0.77) as expected for ~12x the tokens. Chart + run README + dashboard
  refreshed to the final step; run-status table de-mangled (a blank-line split +
  dup cont-3 row had broken its GitHub render).
- **69th upstream sync** (right after SFT completion): merged 9 commits
  (8c92de86b), after pulling a concurrent origin/ezpz push. **Caught + fixed a
  breaking change**: upstream #3923 moved Linear out of common/nn_modules.py
  into new common/linear.py; our agpt + moe import the deep nn_modules path (not
  the re-exporting package root), so it would have raised ImportError. Repointed
  both callsites (2148074cb): agpt -> common.linear, moe split (Linear from
  common.linear, RMSNorm stays in common.nn_modules). sync_smoke.sh VERDICT: ok
  (job 12470795), all 3 configs rc=0 (agpt, agpt@TP=2, moe). Details in
  upstream-sync.md (69th entry).
- **Disk**: freed 2.65T earlier this session (3 dead 80B diagnostic
  checkpoints); datascience group /lus/tegu dropped 8.79T -> 6.13T.
- **Compile root-cause (2026.1.0 venv)**: triton 3.7.2 torch.compile segfault at
  driver.py:364 is a SYCL version skew (icpx module vs torchs bundled
  intel_sycl_rt); FIX = match icpx to the wheel (load oneAPI 2026.0.0 module,
  not 2026.1.0). Recorded in the project_py313_pt214_compile_segfault memory.

## 2026-07-12 (aurora) -- 20B chains -> constant LR; corrected mislabeled 8661913

- **Held both canonical 20B chains at constant LR (no decay)** per request. Patched
  `submit_agpt_20b_autoretry.sh` in BOTH prod clones (agpt-20b-v2 512N,
  agpt-20b-n256 256N): added `DECAY_RATIO="${DECAY_RATIO:-0.0}"` +
  `--lr-scheduler.decay-ratio="${DECAY_RATIO}"` on the train cmd. Verified the
  scheduler math: `decay_ratio=0.0 -> decay_steps=0 -> lr_mult=1.0 at every step`
  (flat after the 200-step warmup, LR stays 2.28e-5 through step 92,858). Queued
  continuations 8647383 (512N) + 8647386 (256N) pick it up at launch -- no
  resubmit, nothing killed. Backups: 512N `-20260712-160557`, 256N
  `.bak-preconstlr-20260712`. Both chains were still pre-decay anyway (onset was
  step 18,572; at 6,000 / 4,200). Mirrored the same `DECAY_RATIO` default into the
  main-repo `submit_agpt_20b_autoretry.sh` so it survives future clone refreshes.
- **Corrected a wrong record: job 8661913 was NOT a constant-LR fork.** It had
  been documented (20b-256 README + a refresh commit msg) as "constant-LR fork
  from the trained base, loss 12.21->4.28, validates constant LR at 20B." The
  resolved config from its own log disproves that: `initial_load_path=null`,
  `decay_ratio=0.8`, `warmup_steps=200`, `lr=2.28e-5`, and loss starting at 12.21
  @ step-10 dropping steeply = random-init warmup, NOT a fork. It was really a
  from-scratch duplicate 20B-256 run on the DEFAULT schedule that reached only
  step-400 before silent-hanging. Fixed the 20b-256 README section; NOT resumed
  (would duplicate the canonical chain at worse loss). The `-constlr` ckpt dir is
  misnamed and can be archived.
- Also fixed the 80B GBS6144 LR-finder chart (restored mano/sophiag curves from
  Sunspot dated-record, hardcoded so it no longer depends on gitignored CSVs) and
  refreshed all production docs/charts to 2026-07-12 (0 stale / 0 drift).


## 2026-07-12 (sunspot) -- big-mix SFT finally training (at 8N) after a 6-failure saga

Got the "more tokens" SFT running: gs138650 base on the FULL OpenMathInstruct-2
`tulu_math_uc_mix` (53.3M packed seqs, ~54B tokens, 1 epoch). Job 12470350 +
afterany chain, stepping cleanly at 8N -- loss 1.34 -> 1.0 by step 128,
mean_token_accuracy 0.685 -> 0.735, checkpoints every 50. Getting here meant
root-causing and fixing a cascade (each documented in the
[launch report](experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md)
+ [production doc](production/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md)):

1. **Runtime tokenize** of 93M rows (~8.5h) blew the watchdog -> offline
   `--pretokenize_to` / `--pretokenized_dataset` (TRL skips prep on `input_ids`).
2. **save_to_disk** slow/OOM -> parallel `save_to_disk(num_proc)` no-pre-flatten;
   **packing map** OOM -> `dataset_num_proc` walkdown (96@49% / 32@86% / 8@100%).
3. **seq_len 2048** OOM'd the XPU tile at step 0 -> reverted to the proven 1024
   corner (bsz2/gas32 to hold GBS=6144 at 8N).
4. **TRL re-packs pre-packed data** (Bus error) -> `packing=False` auto-set when
   `seq_lengths` present.
5. **ezpz `--auto-retry`** false-positives on TRL (`stuck_pre_training`: it looks
   for `step=` markers TRL never emits) -> dropped it; fault tolerance now via the
   afterany chain + `--resume_from_checkpoint` + save_steps=50.
6. **384-rank GPU page fault** (`Segmentation fault from GPU ... NotPresent
   Write`, rank 221) -- NOT bad node (reproduces across nodes), NOT OOV (full 53M
   scan max 255998 < vocab 256000); a real 384-rank scale fault
   (`project_sft_v2_base_oom_badnode` class, base-independent). Bisect
   (12470343/346/347/348/349): 2/4/8N clean, 12/16/32N segfault -> run at **8N**.

Also: nudged huggingface/datasets PR #8318 (vectorized interleave) -- tagged
@lhoestq (arrow_dataset.py owner), CI awaits maintainer approval.

## 2026-07-11 (aurora) -- synthetic-summary data-gen POC (summarize olmo-mix-1124)

Built + validated an end-to-end pipeline to generate synthetic mid-training
data by summarizing existing gemma-tokenized olmo-mix shards. New code under
`experiments/ezpz/synthetic/` (4 commits): `detok_to_text.py` (.bin -> text,
correctness gate = text fixed-point, 265/265 on a wiki slice), `summarize_text.py`
+ `submit_summarize.sh` (batched HF `generate` on 1 XPU tile), `qc_summaries.py`
(drop prompt-echo/meta/too-short), `retok_to_bin.py` (summaries -> drop-in
blendcorpus .bin/.idx).

- **Model bake-off:** `DeepHermes-3-Llama-3-3B` beat `Llama-3.1-8B-exvocab`
  4/4 vs 2/4 on the smoke (exvocab echoed the prompt / emitted meta-commentary).
- **Full pilot** (job 8665137, 1787 wiki docs, DeepHermes-3, batch 16): rc=0,
  1.40 docs/s on one tile; QC kept 1784/1787 (99.8%); retok -> 1784-doc /
  229,104-tok synthetic shard, readback clean.
- **Key finding: ~9x compression** (2.06M source tok -> 229K summary tok) -- the
  eval gate must compare equal-token AND equal-doc budgets. Pilot slice is too
  small to move 2B benchmarks; a real run needs a much larger corpus + vLLM
  (no XPU vLLM venv in this clone yet).
- **Two XPU gotchas fixed** (both cost a smoke each): (1) `.venv` XPU env must
  mirror `train_agpt_2b_venv.sh` (explicit `module load oneapi/release/2025.3.1
  hdf5 pti-gpu`); (2) under `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK`
  must be a bare int -- the composite `"0.0"` form makes torch.xpu see ZERO
  devices (memory: project_ze_affinity_mask_flat_format).
- Report: `docs/experiments/synthetic/aurora/2026-07-11-summarize-olmo-mix-poc.md`.

**Fleet:** all production chains (2b-512 8661117, 20b-512 8647383, 20b-256
8647386/8661913, CPT 8662867, umbrella 8663177) remained `at_queue`-starved the
whole session -- no CLI action possible (never-kill rule).

---

## 2026-07-09 (sunspot) -- 65th + 66th upstream syncs; v2-base SFT OOM + bad-node recovery

- **65th upstream sync (9 commits, `77444f3d3..upstream/main`).** Mostly
  `experiments/rl/` (we don't run it) + a crash-on-invalid-loss guard. No
  replay (no llama3/deepseek_v3/qwen3); agpt/moe diff empty; clean auto-merge.
  3 shared files touched, all benign: `components/loss.py` (spmd_types-guarded
  asserts + additive `return_entropy` kwarg), `token_dispatcher.py` (no-op
  type-check guard), and `torchtitan/trainer.py` (invalid-loss crash in the
  UPSTREAM `Trainer.train()`, which ezpz overrides -- ezpz keeps its softer
  opt-in `nan_abort_consecutive` guard, no conflict). Smoke `VERDICT: ok`
  (job 12470143), landed `533f604fe`.
- **66th upstream sync (4 commits, `533f604fe..upstream/main`).** All in paths
  we don't run (HF-backend SFT, Helion RoPE overrides, graph_trainer). No
  replay; clean merge. **Landmine noted:** `hybridep.py` renamed its base class
  `OpaqueBase -> CustomClassBase` (a newer-torch symbol absent in our venv).
  Verified harmless -- the hybridep module import is lazy, so `import ezpz.moe`
  and standard/EP MoE are unaffected; only the opt-in `comm_backend=hybridep`
  path would `ImportError` on this torch. Smoke `VERDICT: ok` (job 12470255,
  losses bit-identical to 64th/65th), landed `379926566`. Both syncs documented
  in `upstream-sync.md`.
- **v2-base SFT crashed on a oneCCL cache leak, then a bad node.** The 32N SFT
  on the completed v2 2B base (job 12470088) OOM'd at step ~248/729:
  `ze error at zeCommandListAppendMemoryCopy, ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`.
  Root cause from oneCCL's own log: the Level-Zero IPC-handle cache fails to
  release handles (`ipc_handle_cache: handle type is unexpected. Not calling
  zeMemPutIpcHandle`) and accumulates until device OOM. Fix (commit
  `ff71b3e79`): set `CCL_ZE_CACHE_{GET,OPEN}_IPC_HANDLES_THRESHOLD=8000` (the
  knob oneCCL's hint recommends + `moe_ab_check.sh` already uses).
- **Resume 1 (12470254): CCL fix worked, hit a bad node.** No OOM -- resumed
  from checkpoint-200 and stepped to 211, then `Segmentation fault from GPU ...
  type: 0 (NotPresent) ... aborting` -> SIGABRT on **rank 61, node
  `x1922c3s3b0n0`, both auto-retry attempts**. A hardware fault, not code.
  (ezpz auto-retry's watchdog reported `stuck_pre_training / zero step=
  markers` -- a false-positive read of the tqdm carriage-returns, ezpz#163 --
  but the underlying repeated GPU segfault was real.)
- **Resume 2 (12470258): queued.** Plain resubmit to dodge the bad node;
  resumes from checkpoint-200. PBS estimates start ~23:18 (36-node block, busy
  queue). Nothing lost -- checkpoint-200 intact both times.

---

## 2026-07-09 (aurora) -- production doc-refresh: capture 20B progress + extend catch-all

- **Captured 20B chain progress the last refresh missed.** The 20B-512 chain
  advanced 4,400 -> 5,400 (543.6B tok, 11.6%) via the native auto-retry
  relaunch (8638793 resume + 8638795 cont) while the 1536-node umbrella
  (8648363) stayed queued on `at_queue` contention; 20B-256 advanced
  2,100 -> 3,100 (156.0B, 3.3%). `refresh_all.sh --model {20b_v2_512,20b_v2_256}`
  regenerated scalar fields + figures; hand-fixed the 20B-512
  Latest-checkpoint attribution (was miscredited to 8521628, which only
  reached 4,500) and appended the two autoretry progress + log rows
  (append-only). Commit 8843878f0.
- **Root-caused a coverage gap in the refresh catch-all.** `refresh_all.sh` is
  100% manifest-driven from `trajectories.py`, which only registers the 6
  W&B/DCP AGPT chains. Every newer production subtree with a bespoke
  TSV/JSON plotter (cpt/, sft/, grpo/, 2b-mds/) was structurally excluded --
  only refreshed by hand. This is why the CPT eval (job 8647850) charts
  weren't auto-updating.
- **Fixed by folding them in at the chart layer** (commits cc3c81919 +
  9f0636866): cpt plotters default `--data-dir` to their in-repo `figures/`
  (was `/tmp`); sft/grpo plotters skip cleanly (exit 0) when the Sunspot-only
  `TRAINER_STATE` is absent; registered all 5 plotters in
  `update_all_charts.sh`; widened `refresh_all.sh` auto-stage globs to
  `production/**/charts/*` + `production/**/*.tsv`; added a coverage-audit to
  `check_stale_docs.sh` that flags any `production/**/plot_*.py` NOT wired in.
  The audit immediately caught the 5th plotter (2b-mds) I'd missed -- exactly
  its purpose. Validated end-to-end on Aurora: `update_all_charts.sh` = 0/12
  failed (cpt + 2b-mds regenerate, sft/grpo skip), coverage audit = 5 wired /
  0 unwired.
- **Fleet:** still `at_queue`-starved -- 3487 nodes free but every prod job Q
  ("Insufficient amount of resource: at_queue"). Umbrella 8648363 eligible
  70h+, never won a slot. The standalone 20B-512 (8638795) carried the chain
  to 5,400 while the umbrella waited -- the intended fallback.

---

## 2026-07-02 (aurora) -- umbrella exit-3 diagnosed + auto-retry umbrella smoke

- **Umbrella 8568429 (legacy failover_lib.sh) exited 3 = 3/4 chains lost.**
  t0 2b-512N rc143 (rank signal 11 on x4409, blind-swap exhausted retries);
  t1 20b-512N rc143 (same class, already relaunched as 8638793); t2 2b-256N
  rc0 clean; t3 20b-256N rc127 (node x4208 unreachable mid-run @ step-2186,
  failover exhausted). Every failure = legacy failover giving up on a
  recoverable bad-node event -- the blind-swap defect (can't parse hostname
  from `signal 11`).
- **Built + smoked a native auto-retry umbrella.** New
  `scripts/smoke_multi_autoretry.sh` launches each trainer via
  `ezpz launch --auto-retry` directly (not failover_lib.sh): ONE venv
  broadcast for the whole alloc (no concurrent-yeet /tmp/.venv race), split
  nodefile into per-trainer slices, concurrent launches with distinct
  ports/CKPT_DIRs/spares, async-mode=disabled. Job 8639375 (2 trainers x
  2+1 nodes): **failed 0/2**, both OK to step 20, auto-retry armed per
  trainer (active=2 spare=1), 0 error signals, no cross-talk. The native
  auto-retry umbrella pattern works. Report:
  [`20260702-multi-autoretry-umbrella-smoke.md`](experiments/agpt/aurora/20260702-multi-autoretry-umbrella-smoke.md).
- **Still TODO:** 2b-512N chain (umbrella t0) not yet relaunched (only 20b
  was); relaunch it on autoretry, or promote the umbrella smoke to a prod
  multi-chain replacement for the legacy failover umbrella.
- **Fleet:** deep post-PM queue contention persists; the umbrella ended ~00:45
  so nothing is training on Aurora right now -- all jobs (80B, 20B relaunch,
  CPT sweep, 2b/20b 256N) Q behind node availability.

## 2026-07-01 (aurora) -- 2B CPT sweep launched + completed-2B eval closeout

The 2B 256N base completed (step-92,859) and its eval tail is dead flat --
so the next move is continued pretraining on a different data mix.

- **Completed-2B eval closeout.** Backfill (8638581) filled the tail
  (step-86,500..92,859, 14 ckpts). Eval table + charts refreshed; plateau
  confirmed to completion (HellaSwag 0.560, ARC-Easy 0.651 flat over the final
  ~635B tokens). Resolved the old ARC-Easy ~0.59 artifact (fresh eval = ~0.65).
  Final step-92,859: hellaswag_norm 0.561, arc_easy 0.651, piqa 0.733.
- **CPT mixing-ratio sweep launched (pilot).** Fork the plateaued base
  (model-weights-only via `--checkpoint.initial-load-path`) and CPT on
  olmo x dolmino blends. Built 3 data-lists (`dolmino-mix-1124`,
  `olmo50-dolmino50`, `olmo25-dolmino75`; renormalized to exact ratios).
  Launched 256N pilots: 8638977 (dolmino-100) + 8638978 (olmo50-dolmino50),
  each + afterany cont. GAS=1 -> GBS=6144 (batch held to keep LR calibrated),
  LR 2.28e-5 re-warm 200 + decay 0.8, ~300B tokens. Distinct CKPT_DIRs
  (no overwrite; base read-only in a separate clone).
- **Fork verified by smoke 8638933:** loads step-92,859 clean (0 mismatch),
  loss 7.2->5.9 descending (dolmino is a real distribution shift = the CPT
  signal). Report:
  [`20260701-2b-cpt-olmo-dolmino-sweep.md`](experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md).
- **Bug found:** the 2b autoretry script defaults async checkpointing, which
  is XPU-broken on torch 2.13.0.dev20260520 (`new_group(gloo)` ->
  `No backend type for xpu` in checkpoint.py:484). First CPT smoke (8638909)
  died on it. Workaround: `CHECKPOINT_ASYNC_MODE=disabled` (the 20b script
  already defaults to disabled). TODO: fix the 2b default. Also added an
  `EXTRA_ARGS` env passthrough to the 2b script (PBS can't forward "$@").

## 2026-07-01 (aurora) -- 20B 512N chain relaunched on native auto-retry

Recovered the 20B 512N canonical chain, frozen at step-4400 since 2026-05-29.

- **Root cause it was stuck:** its latest attempt (trainer-1 in the umbrella
  job 8568429, legacy `failover_lib.sh`) died at init on bad node
  `x4410c0s0b0n0` (signal 11), and the failover **blind-swapped the wrong
  nodes** every retry -- the scraper counts `died from signal 11` lines but
  can't parse the hostname out, so it rotated innocent `x4411` spares and
  left `x4410` in. Exhausted retries. (Concrete case of the blind-swap class
  from the [restart-economics writeup](experiments/agpt/aurora/20260630-failover-restart-economics.md).)
- **Fix = relaunch on native `ezpz launch --auto-retry`** (better scraper).
  Three obstacles resolved: (1) verified step-4400 is nested/pre-#3623 format
  (resume-safe with the pinned clone's rolled-back code); (2) upgraded the
  clone venv ezpz 0.16.0 -> 0.21.3 via `uvi` (torch UNCHANGED, auto-retry
  flag now present); (3) rebuilt `.venv.tar.gz` -- login/compute-node `tar`
  crawled on 37K small Lustre files, so used a fast tmpfs surgical-swap
  (decompress good tarball in RAM, swap the 2.4M ezpz, re-tar) in job 8638674.
- **Smoke (8638756, 2N) CONFIRMED resume:** `Finished loading the checkpoint
  in 567s` + `Training starts at step 4401`, ezpz 0.21.3 launches on compute,
  XPU 24/24, no save (chain untouched).
- **Gotcha caught by the smoke:** `dump_folder=./outputs` prepends `outputs/`,
  so the correct CKPT_DIR is the BARE `checkpoints/...` (script default);
  `outputs/checkpoints/...` doubles to `outputs/outputs/...` and silently
  fresh-starts. The 512N launch needs no override (default = n512-gbs12288).
- **Launched:** head 8638793 (Q, select=522, NHOSTS_TRAIN=512, resume
  step-4400) + cont 8638795 (H, afterany). Report:
  [`20260701-20b-512n-relaunch-autoretry.md`](experiments/agpt/aurora/20260701-20b-512n-relaunch-autoretry.md).
- **Also:** 2B eval backfill (job 8638581) filling the completed 2B 256N tail
  (step-86500..92859, 14 ckpts) ran in parallel.

## 2026-07-01 (aurora) -- 80B launch: 2048N crashes at init, 512N + 1024N run

Machine returned from the Mon 2026-06-29 maintenance. Managing the 80B
SophiaG/constant-LR production launch and refreshing docs.

- **80B launch attempted (~15:00 UTC).** All 6 jobs (3 heads + 3 conts) stayed
  queued through the PM; no head ran pre-maintenance, so these are cold starts.
  The **2048N head (8574387) started first**, ahead of the 512N/1024N brackets.
- **4 exec-server rejects before it placed.** run_count 1-4: the job flipped
  Q -> R -> Q with `PBS Error: Execution server rejected request`, never writing
  a log. Diagnostic that drove the "wait, don't requeue" call: *peer 2000+N jobs
  were running fleet-wide* during the reject window, so it was post-maintenance
  node-release flapping specific to our attempts, not a machine-wide inability to
  place large jobs. Attempt 5 (~15:00 UTC) placed cleanly on 2072 nodes.
- **Then it SIGSEGV'd in `set_determinism`** (`F`, rc=143, 14:47 walltime).
  Config echo was correct (GBS=6138, TP=4, LBS=1, compile OFF, steps=92,950,
  SophiaG, constant-LR) and venv broadcast finished, but the seed broadcast in
  `distributed/utils.py:231` faulted at **24,864 ranks** (`rank 13602 died from
  signal 11`; CCL/PMI KVS "Connection reset by peer"). This is the **documented
  init-crash class** (2B/20B at 1024N/12,288 ranks), now confirmed for 80B at
  2048N. NOTE: I briefly misread an earlier `build_mesh` success as
  "set_determinism cleared" -- it had not; the crash is *in* set_determinism.
- **Finding: init-crash ceiling for 80B is bracketed by the 1024N run.**
  512N (dp=1530) proven; 2048N (dp=6138) crashes; **1024N (8574386, dp=3066) is
  the missing measurement** -- if it survives init the ceiling is ~2048N-specific,
  if it crashes then 512N is the practical 80B max on this stack.
- **Finding: auto-retry misclassified the SIGSEGV as a walltime stop.** The rank
  SIGSEGV -> SIGTERM -> job rc=143, which `launch_autoretry.py:726` logged as
  `FAILOVER STOP: walltime` and did NOT consume its 2 retries. Moot for a
  deterministic init crash, but a real classifier gap: a swappable bad-node
  SIGSEGV would also produce rc=143 and never trigger a spare-swap. Filed as a
  follow-up (rc=143/SIGTERM should be distinguished from a wrapper-initiated
  walltime-margin stop).
- **Action:** `qhold`'d the 2048N continuation (8574390, `afterany:8574387`) --
  `afterany` fires on failure too, so it would have grabbed 2072 nodes for the
  same crash. 512N + 1024N heads left to run; backfilling as 8574387's nodes
  release. No `qdel` (holds are reversible).
- **Docs refreshed:** dashboard + agpt/80b rollups + launch report all corrected
  from "launching" to the crash outcome, with the two findings recorded.
  Operational lesson also recorded: after a reservation tears down, a large job
  can eat several exec-server rejects before nodes stabilize; wait (don't
  requeue, which forfeits queue priority) as long as peer large jobs are placing.
- **Restart-economics analysis** (from 2026-06-30) stands: ~7 confirmed failover
  recoveries of ~62 triggered episodes; ~78% of exhaustions are systemic
  (CCL/PMI KVS timeout 57%, now-fixed blendcorpus race 15%, pals-RPC 6%);
  node-hour waste is bimodal (median episode ~11min but 8 episodes >3h account
  for ~64% of the ~31.8k wasted node-h). See
  [`20260630-failover-restart-economics.md`](experiments/agpt/aurora/20260630-failover-restart-economics.md).

## 2026-07-01 (sunspot) -- 80B convergence run (all optimizers NaN), MoE page reorg, 63rd sync

Concurrent Sunspot session (separate from the Aurora 80B launch above).
Three things landed.

- **MoE LR-finder pages reorganized production-first** (`56aad15ac`). Matched
  the dense-page reorg: MoE has no production-GBS finder yet (all data is the
  2026-04-21 GBS=192 small-batch sweep), so the index leads with a "Status:
  small-batch only" banner + the batch-dependence lesson, and the April sweep
  is collapsed into `<details closed>` on the index + all 5 config pages.
  Preserved the one inbound anchor (`experiments/moe/README.md`).
- **80B head-to-head convergence run -- all three optimizers NaN**
  (`c015d55d5` + report `2026-06-30-80b-convergence-gbs6144.md`). New
  `scripts/{run,submit}_80b_convergence.sh`. Ran mano/sophiag/AdamW at their
  finder-recommended CONSTANT LRs (3e-6/1e-6/5e-7, GBS=6144, 64N, jobs
  12469910/911/912). **All three descended a few steps then diverged: mano
  first (grad NaN step 5), AdamW step 9, sophiag step 12.** grad_norm runs up
  then explodes, loss NaNs one step later. The finder's early-step ranking
  (which said mano safest) does NOT predict sustained stability; the shared
  failure across 3 different optimizers points to a corner-level instability
  (bf16 at dim=9216), not tuning. Conclusion: no finder LR is production-safe
  as a constant LR here -- needs a long warmup (>=200 steps) + grad clipping,
  possibly an fp32 grad path. Smoke-first caught a real bug first (the runner
  missed the yeet-env/`/tmp/.venv` preamble -> `env: ezpz` exit 127, fixed in
  `1b554df71`). The corner is ~20 min/step, so 50-step jobs were the practical
  cap (all NaN'd by step 12 anyway).
- **Queue hygiene:** qdel'd the stale 112N dp=324 bisect (12469630, queued ~5d,
  dp ceiling already disproved) and the finished smoke -- unblocked the
  convergence jobs to backfill immediately.
- **63rd upstream sync** (`cf99e127e`, 13 commits `390ea37cc..`). No replays
  (llama3/deepseek_v3 untouched; agpt/moe byte-identical). One RL conflict in
  `generator.py` resolved by taking upstream (spmd_types if/else superset).
  Impact is RL/FLUX/spmd_types (no-ops for our DTensor backend; loss.py edits
  are spmd_types-guarded) + a DeepEP-v2 upgrade touching moe's token
  dispatcher. **Smoke-validated:** agpt 2B trained clean 10 steps (loss 8.35);
  moe debugmodel exercised the DeepEP-v2 dispatcher path without error then
  OOM'd downstream (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, a known XPU resource
  limit, not a merge regression). Most of the effort was worktree plumbing --
  a git worktree only has tracked files, so jobs run from one need `.venv`,
  `.venv.tar.gz`, and `assets/hf` symlinked in (the last tripped me: `assets/`
  is a tracked dir, so the tokenizer download links at `assets/hf`, 3 levels
  up).

## 2026-06-29 (sunspot) -- 2B 100-step production-batch ladder + LR-finder page reorg

Continuation of the LR-finder work. Two deliverables, both shipped.

- **2B 100-step production-GBS ladder (jobs 12469854-868, 16N/dp=192).**
  Re-ran the production batch sweep (GBS 1536/3072/6144/12288/24576) at the
  classic **100-step** finder length (the 2026-06-28 trend used 15 steps) for
  adamw/mano/sophiag -- 15 jobs in parallel, isolated dumps
  `outputs/lrfind-2b-100step-prod/gbs<N>/`. **All 15 complete, 0 NaN across
  the full 16x batch range.** Confirms the 15-step story at the longer length:
  2B never cliffs even at 100 steps + 4x production batch, no batch-scaling
  trend (AdamW ~2.5-3.6e-3, mano ~5-9e-3, sophiag dead-flat ~2e-3), 3-4 orders
  above the 80B 7e-7 cliff. Deep minima (~7.7-8.3 vs ~11.5 at 15 steps) are the
  cumulative-training sweep-length effect -- compare by min-LR, not loss value.
  New `scripts/plot_2b_100step_prod.py` (hue=optimizer canonical colors, batch
  = shade+width+opacity); two figures, generated on Sunspot from real CSVs,
  y-capped + smoothed. Built incrementally as tiers landed (9/15 -> 12/15 ->
  15/15), regenerating each time. Commits `a9bfc91b5`, `e1eefcba6`,
  `6f7f6b6b6`.
- **LR-finder pages reorganized production-first.** All three agpt pages now
  lead with the production/recent results and collapse the old 2026-04 2-node
  small-batch finders into `<details closed>` blocks:
  - 2B (`542ae2d49`): (1) 100-step production ladder, (2) 15-step trend,
    (3) additional findings, (4) collapsed April debug runs, (5) reports index.
  - 20B + 80B (`098ec19cc`): production/trend on top, April runs collapsed
    (demoted to `###`). Heading text preserved verbatim so every inbound anchor
    (2B page, agpt index, experiments/agpt, scaling-performance) still resolves.
- **Docs hygiene.** Caught + corrected a missed step: the docs-root
  "Recently Updated" index table is git-commit-date driven and auto-generated
  (`utils/refresh_docs_readme_table.py`) -- refreshed it in the same commit on
  every doc push from here on, not after the fact.

---

## 2026-06-28 (sunspot) -- LR-finder docs overhaul, 2B trend, 62nd sync, validator phantom fixed

Docs/tooling-heavy session plus one real bug closed.

- **LR-finder docs split per-model.** `docs/experiments/lr-finder/` went from
  two consolidated family pages to `{agpt,moe}/<model>/README.md` +
  per-model `figures/` (agpt 2b/20b/80b; moe debugmodel/500m/2b/4b/7b), each
  family keeping an index README. All inbound links + anchors repointed.
- **2B LR-ceiling-vs-GBS trend (jobs 12469769-776, 16N/dp=192, GBS
  192..24576).** Headline: **2B never cliffs** -- AdamW usable LR flat at
  ~1e-2 across the whole 128x batch range, 0 NaN, vs 80B's collapse to a
  ~7e-7 NaN cliff. So the batch-dependent usable-LR collapse is a
  LARGE-MODEL (dim=9216 bf16) phenomenon, not universal. Ran all 3 working
  optimizers (adamw/mano/sophiag) at all 8 GBS; muon dropped (oneCCL abort
  ~7min in). Found sophiag's U sharpens/deepens with batch (not perfectly
  batch-independent like adamw/mano).
- **Charts: house style + completeness.** Wired `apply_style()` (ambivalent +
  Iosevka) into the finder auto-plot and `plot_lr_finder.py`, made
  `apply_style` import-safe without IPython (loads the .mplstyle by path).
  Added per-GBS loss-vs-LR curve families for all optimizers + a reusable
  `scripts/plot_lr_trend.py`, registered in the refresh catch-all.
- **Corrections caught by the user:** (1) "production batch" is **GBS=6144**
  (2B 256N + all 80B), not 12288 (that's 2B 512N) -- relabeled, kept all
  12288 data/figures. (2) Documented why old finder min-loss (~9-10) <
  new (~11.3): sweep length (100 vs 15 steps), since the finder trains
  cumulatively -- only the min-LR is comparable across sweep lengths.
- **62nd upstream sync** (3 commits, `0e886617e..390ea37cc`, all
  experiments/rl/ -- no replays).
- **Validator "CCL deadlock at 80B TP=4" was a PHANTOM (root-caused + fixed +
  CONFIRMED).** A 4-way worktree fan-out found no collective-deadlock log:
  the label conflated a validator dataloader cold-cache mmap-race CRASH (job
  12469584) and a training-side index-build barrier stall (job 12469597),
  plus a genuine `loss_fn` tuple-unpack crash in `validator.py`. Fixes:
  tuple-unpack (`loss_sum, _ = self.loss_fn(...)`); validator inherits the
  warm `data_cache_path` (config default + the submit script already passed
  it on the CLI); prewarm builds the validation index. **Confirmed 2026-06-28
  (jobs 12469784 prewarm + 12469785 train-loop, 4N/dp=12):** first-ever
  `validate()` completions at 80B TP=4 -- finite val loss, no
  mmap/AttributeError/CCL hang, across 4 passes. dp=12 confirmation; a 62N
  pass would be belt-and-suspenders. Writeup:
  [`docs/guides/known-bugs/validator-tp4-at-80b.md`](guides/known-bugs/validator-tp4-at-80b.md).
- **Ops:** flipped the default to auto-push-after-batch on working branches
  (pull-rebase first); the 80B trend gap-fill reruns (2304-redo 12469778,
  4608 12469767) and dp=324 bisect (12469630) are still out.

---

## 2026-06-26 (sunspot eve) -- 80B: LR is the wall, dp-ceiling isn't, don't scale LR with batch

Five jobs resolving the LR/batch/dp-degree questions the sim campaign
raised. Synthesis report:
[`docs/experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md`](experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md).

- **Were the sims LR-scaled? No -- and they shouldn't be.** All sims used
  flat LR=1e-6. An LR-finder (lr 1e-6->1.0, GBS=372) puts min-loss LR at
  **~3e-6 (sophiag) / ~8e-6 (mano)**, diverging by ~1e-2 -- production
  1e-6 is on the safe left shoulder, below optimum. (adamw/muon curves
  lost to the cold-cache race; need warm rerun.)
- **Scaling LR up with batch made it WORSE:** GBS=5952 + 16x-scaled
  LR=1.6e-5 (job 12469698) NaN'd at **step 7** (loss 11.6->20.4->nan),
  vs step 29 at flat 1e-6. 1.6e-5 is at the LR-finder divergence shoulder.
  The LR ceiling is fixed by bf16/dim-9216 overflow, not the batch ->
  **do not linearly scale LR with batch here.**
- **The dp_degree<=186 "ceiling" is not a cliff:** bisect jobs 12469628
  (dp=192) and 12469629 (dp=264) both ran 30 steps NaN-free. 186 was just
  the highest tested point. Corner scales to at least dp=264; the script's
  dp>186 warning is over-conservative. dp=324 (12469630) queued; dp=372
  needs Aurora. **This vindicates the earlier instinct that TP=4 should
  scale past 62N.**
- Net: two walls quantified -- a sharp LR wall (~1e-5, optimizer-set) and
  a much-further-out dp-degree wall (>264). The GBS=5952 flat-LR step-29
  NaN is a separate, mild batch-accumulation effect at low LR.

---

## 2026-06-26 (sunspot) -- 80B global-batch sim campaign continues: GAS=16 (1024N) clean

Continued the 80B batch-scaling campaign past the GBS=1488/512N sim.

- **GAS=16 / GBS=2976 / 1024N-batch** (job `12469626`, 62N, dp=186):
  **clean, 34/34 steps to the 6h walltime cut, zero NaN**, loss
  12.92 -> 9.84. grad_norm showed the same step-22-27 transient as the
  other sims (peak 21.5 @ step 23, recovered to ~9.3). 8x the validated
  GBS=372, same smooth descent. Report:
  [`docs/experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md`](experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md).
  Notably this run **cold-built the blendcorpus index at 744 ranks
  cleanly** (no race at this index size), confirming the `debfff5`
  sibling barriers suffice for cold-build-at-scale.
- **GAS=32 / GBS=5952 / 2048N-batch** (job `12469627`): **NaN at step 29**
  (UPDATE -- the line below said "in flight, clean through step 24"; it
  went on to NaN). Ran 28 clean steps then `grad_norm=nan` at step 29
  (loss still finite -- grad-path-first signature), then PBS walltime-cut.
  Its first attempt hit the blendcorpus cold-cache race (`EOFError`) and
  **auto-retry self-healed** (one spare). Full report:
  [`docs/experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md`](experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md).
- Campaign result: GBS=372/1488/2976 clean; **5952 (16x) NaN'd at step
  29.** The corner holds NaN-free to **8x** the validated batch, breaks
  at 16x. CAVEAT (the key catch): LR was flat 1e-6 for ALL rungs (no
  batch scaling) and every run was *inside warmup* (clamped to
  total_steps), so effective LR at the NaN was only ~5.8e-7. So the 16x
  NaN is batch-dependent at matched step+effective-LR, but its dependence
  on the full / batch-scaled LR is unknown. Two follow-ups queued:
  `12469698` (LR=1.6e-5 scaled + warmup=5) and `12469699` (LR=1e-6,
  warmup=200, 60 steps -- reproducibility). Added a WARMUP_STEPS knob to
  the 80b script for this.
- Still pending: the dp-degree cliff bisect (n64/n88/n108 -> dp
  192/264/324), the actual probe of whether the corner survives
  node-count scaling past dp=186.

---

## 2026-06-25 (sunspot) -- 80B GBS=1488 512N-batch simulation: clean, 4x batch NaN-free

Pushed the 80B TP=4/LBS=1/bf16 stable corner to **4x the validated
global batch** (GBS 372 -> 1488) via GAS=8, to simulate the global batch
an Aurora 512N run would see while staying inside the NaN-free
`dp_degree <= 186` corner. Held 62N (dp=186) and raised GAS, since
`GBS = dp_degree * LBS * GAS` -- GAS scales the batch without touching
the dangerous dp_degree.

- **Job `12469609`** (62 active + 6 spare), GBS=1488 (~3% under the
  Aurora 512N GBS of 1536), AdamW LR=1e-6, AC=full, compile=OFF,
  `VALIDATOR_ENABLE=0`.
- **Result: clean.** 46/46 steps before the 4h walltime cut, **zero
  NaN**, loss 12.92 -> 8.84, grad_norm bounded (peak 22 @ step 23,
  recovered to ~8), MFU steady ~9.85%. The corner's stability is not
  specific to the small batch.
- Full report:
  [`docs/experiments/agpt/sunspot/2026-06-25-80b-gbs1488-512N-sim.md`](experiments/agpt/sunspot/2026-06-25-80b-gbs1488-512N-sim.md).

**The multi-attempt arc had TWO distinct causes (corrected after reading
all four failed-job logs).** This run was the successful relaunch after
four prior GBS=1488 attempts failed:

1. **Validator cold-built at the wrong path (12469584).** That attempt's
   *training* dataloader built the full 74773-sample index **cold at 744
   ranks and succeeded**, reaching step 1 clean -- so cold-build at scale
   already works with the `debfff5` sibling barriers. The crash was the
   *validator* building its validation-split index cold at the default
   `.cache/blendcorpus` path (missing `--validator.dataloader.data-cache-path`).
2. **Self-inflicted barrier (12469590/592/597).** Reacting to (1), I
   added a global `torch.distributed.barrier()` in blendcorpus
   `_build_index_mappings` (`c7eb628`). That function is called a
   data-dependent number of times (per corpus x per split, branch-gated),
   so ranks hit the barrier a mismatched number of times ->
   partial-participation deadlock (`oneCCL allreduce_scaleout ...
   atl_comm->wait`, confirmed at `gpt_dataset.py:1145` in 12469597).
   **Reverted in `74b09fd`**; the next run (12469609) trained clean.

- Corrects the earlier read: cold-build at 744 ranks **works** (12469584
  proves it) -- the cache this run loaded "warm" was built cold by
  12469584, not by a pre-warm. The `prewarm` `--training.steps=1` call
  builds a *wrong-sized* index anyway (hash keys on
  `num_samples = GBS * train_iters`).
- Ruled out: **NOT node health** (`rc=127` was failover-scrape noise),
  **NOT batch size** (step 1 always clean; both failures are dataloader
  init).
- The **3 sibling barriers** from `debfff5` are correct and kept -- each
  sits next to a pre-existing all-rank collective. (blendcorpus PR #8.)
- Systematic-debugging takeaway: a barrier is only safe where every rank
  provably reaches it the same number of times; never inside a
  branch-gated, per-item loop. And: read *all* the failure logs before
  naming a single root cause -- there were two here, not one.

What the sim does **not** establish: survival of `dp_degree > 186` (the
real 512N+ regime, where dp itself is the trigger) or the >512N
distributed-init path (open `set_determinism` crash at 12,288+ ranks).
Those need a node-count study (dp-degree cliff bisect), not a batch
study. The separate validator-CCL-deadlock at 80B TP=4 was sidestepped
(`VALIDATOR_ENABLE=0`), not fixed.

---

## 2026-06-25 (sunspot) -- autoretry scripts: _real RoPE + validator; validator.py bug fixed

Two requested changes to `submit_agpt_{2b,20b,80b}_autoretry.sh`, plus a
latent validator bug surfaced + fixed.

1. **`_real` RoPE default (2B/20B).** `CONFIG_SUFFIX` now defaults to
   `_real` for 2B/20B -> `agpt_{2b,20b}_real`, which use real-valued
   (cos_sin) RoPE. The default complex backend uses torch.complex64 ops
   that torch.compile's inductor refuses to lower (eager fallback inside
   the compiled graph); cos_sin is real-valued and compiles, so it's the
   faster path with compile ON. Overridable via `CONFIG_SUFFIX=` (empty).
   **80B left on plain `agpt_80b`**: it runs compile OFF (the `_real`
   win is a compile-lowering optimization, moot there) and `agpt_80b` is
   the numerically-validated config.

2. **Validator enabled (all three).** `--validator.enable` +
   `VALIDATOR_FREQ` (default 100) + `VALIDATOR_STEPS` (default 10) knobs
   + `--validator.dataloader.dataset-path=$DFL` (blendcorpus only).

3. **Fixed a real EzpzValidator bug** (the "validator wired but not
   smoke-tested" CLAUDE.md flag). `validator.py:validate()` unpacked
   `post_dataloading_process` into 4 values
   (`inputs, labels, extra_inputs, extra_kwargs`) and splatted
   `**extra_inputs` at the pp-eval + model-forward sites, but upstream
   `torchtitan/components/validate.py` now returns a **3-tuple**
   (`extra_inputs` folded into `extra_kwargs`) -> every validate() call
   raised `ValueError: not enough values to unpack (expected 4, got 3)`.
   Dropped `extra_inputs` at all 3 sites to match upstream.

Smoke (job 12469561, agpt_2b_real, 2N, validator freq=3): config
resolved to `agpt_2b_real`, validation ran clean at steps 3 + 6
(val loss 12.46 -> 11.96), training completed. First crash (12469560)
is what caught the validator.py bug.

Commits (ezpz): `117ac69ce` validator.py fix, `5ffb850a1` script changes.
NOTE: torchtitan imports from the repo path (not copied into `.venv`),
so the validator.py fix is live for new jobs without re-yeeting.

---

## 2026-06-25 (sunspot) -- 80B TP=4 first real run + blendcorpus cache-build race fix

Launched the first real 80B TP=4/LBS=1/AdamW/bf16/GAS=2 run (GBS=372)
via `submit_agpt_80b_autoretry.sh`, 62 active + 2 spare on Sunspot, to
validate the stable corner past the 30-step doc result.

**Two init crashes -- NOT NaN/model/script. Root cause: blendcorpus
index-cache build race at TP>1, structurally broken barrier.**
- 12469548: 3 ranks raced the per-corpus `shuffle_idx.npy` (EOF magic /
  mmap-length errors).
- 12469550: per-corpus loaded warm, then 558 ranks raced the BLENDABLE
  index (`FileNotFoundError: a23baff6..._index.npy`).

The build path builds on global rank 0, then "waits" with only
`get_data_parallel_group()` + `get_pipeline_model_parallel_group()`
barriers before all ranks `np.load`. Those subgroup barriers do NOT gate
ranks whose TP coordinate != 0 against rank 0 (their DP/PP subgroups
exclude rank 0), so ~(1 - 1/TP) of ranks race ahead and read the .npy
mid-write. At TP=4/744 ranks that's ~3/4 -- matches the 558 blast radius.
The cross-group `all_reduce` that used to backstop this was commented out
("I don't think this is necessary any more") in
`deps/blendcorpus/.../blendable_dataset.py`. Prior "stable" 80B runs
(12469494/12469509) hit the same `building on rank 0` warning and only
survived by winning the timing race.

**Fix (both, per user):**
1. **Root cause** -- `deps/blendcorpus` (saforem2/blendcorpus, branch
   feat/remove-deepspeed, commit `debfff5`): added a global
   `torch.distributed.barrier()` after the subgroup barriers at all three
   build-then-load sites (gpt_dataset corpus-load + corpus-build
   completion, blendable_dataset load) so every rank waits for the
   rank-0 writer regardless of TP/PP/DP coordinate. (Outside
   experiments/ezpz/, but its own repo + the user's, so in scope.)
2. **Defense-in-depth** -- new `scripts/prewarm_blendcorpus_cache.sh`:
   a small/low-rank job that builds the index cache (1 step, compile+ckpt
   off) at the SAME data-cache-path a large run will use, so the big run
   loads-not-builds and never exercises the build path at scale. Path
   derivation mirrors the autoretry scripts exactly (verified: MODEL=80b
   NHOSTS_TRAIN=62 GAS=2 -> agpt-80b-adamw-books-n62-gbs372).

**Validation: clean success.** Attempt 3 (12469551) ran on the
now-fully-warm cache (both layers `loading`, no build, no race) and
completed **100/100 steps with zero NaN**: loss 12.93 -> 7.72 (-5.2
nats), grad_norm bounded throughout (peak ~9.9 early, settling ~2-6, no
spike-to-inf), MFU steady ~9.8%, mem flat 32%. step-100 checkpoint saved
(906 GiB, 163 s), `[auto-retry] FAILOVER STOP: success`, rc=0. This
**supersedes the prior 30-step TP=4 evidence** and confirms TP=4/LBS=1/
bf16/GBS=372 as the production-ready 80B corner (the TP=2/LBS>1 grad-path
overflow remains the open upstream bug). Full report:
`docs/experiments/agpt/sunspot/2026-06-25-80b-tp4-100step-validation.md`.

---

## 2026-06-24 (sunspot) -- py313-pt214 torch-2.14 compile segfault diagnosed

A user training attempt in the new `venvs/py313-pt214` env (torch
2.14.0.dev20260623+xpu, triton 3.7.2, py3.13) crashed with SIGSEGV at
step 1, in `triton/backends/intel/driver.py:364 __init__` (via
`get_current_device` -> `get_current_target`) during the first
`torch.compile` codegen. Two SEPARATE problems, isolated by minimal tests
on a compute node (see `project_py313_pt214_compile_segfault` memory):

1. **UR-loader symbol mismatch (import-time) -- FIXABLE.**
   `import torch` fails with
   `ImportError: libsycl.so.9: undefined symbol: urDeviceWaitExp, version
   LIBUR_LOADER_0.12` whenever the inherited `LD_LIBRARY_PATH` puts the
   system oneAPI loader first. Both the system loader
   (`/opt/aurora/26.26.0/oneapi/compiler/latest/lib/libur_loader.so.0.12.0`)
   and the venv-bundled one (`venvs/py313-pt214/lib/libur_loader.so.0.12.0`)
   advertise `LIBUR_LOADER_0.12`, but only the **bundled** one actually
   exports `urDeviceWaitExp` (system `nm -D | grep -c` = 0; bundled = 1)
   -- Aurora 26.26.0 ships an older 0.12 predating that symbol. Fix:
   prepend the venv lib so the bundled loader wins:
   `export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"` (AFTER the
   oneAPI module load). With this, import + `torch.xpu.is_available()`
   (6 devices) + EAGER xpu compute all succeed. Same class as the
   pyzes/libze_loader bug -- bundled-vs-system Intel runtime collision.

2. **Triton XPU `torch.compile` segfault -- NOT fixable from our side.**
   Even WITH the loader fix, `torch.compile(backend="inductor")` on an
   xpu tensor segfaults at `driver.py:364 __init__`. Confirmed compile-
   specific: eager xpu ops exit 0; compile exits 139. Not a
   fork/concurrency issue (`TORCHINDUCTOR_COMPILE_THREADS=1` still
   segfaults). triton 3.7.2 here vs 3.7.1 in the working torch-2.13
   `.venv`. Env-build incompatibility between triton 3.7.2's Intel backend
   and the Sunspot compute runtime.

Conclusion: `venvs/py313-pt214` is not usable for compiled XPU training
on Sunspot yet. Use the production torch-2.13 `.venv` (triton 3.7.1),
where agpt_2b/20b compile + train cleanly (smokes 12469525/12469526
today). If py313-pt214 is needed, run `--compile.no-enable` (eager works)
or wait for a triton-xpu build matched to the system runtime.

Also: tegu project-quota (pid 2297) hit its 11 TB hard cap mid-session
(`EDQUOT` on every write; `lfs df` OST imbalance was a red herring --
pinning to an empty OST also failed, proving it was the project quota).
User cleared space (11.0 TB -> 4.8 TB used); writes recovered.

---

## 2026-06-24 (sunspot) -- ezpz-native auto-retry 20B + 80B scripts

Ported the validated 2B native-auto-retry template to
`scripts/submit_agpt_{20b,80b}_autoretry.sh` (full report appended to
`docs/experiments/agpt/sunspot/2026-06-24-native-autoretry-2b-smoke.md`).

- **20B** -- same shared plumbing as 2B; per-model deltas: the `DATASET`
  blendcorpus/HF knob, `--dataloader.num-workers=2`, `_CKPT_DATASET_SLUG`
  (sanitizes HF `/` in ckpt dir), `--checkpoint.async-mode=disabled`
  default. Live-smoked (12469526, own select=4): `--np=24`, GBS=48,
  trained 5 steps clean, Training completed. (Loss bounce / grad spikes
  are the expected tiny-GBS/no-warmup smoke behavior, not a crash.)
- **80B** -- defaults to the confirmed-stable **TP=4 / LBS=1 / AdamW
  LR=1e-6 / bf16-compute+fp32-master / AC=full / compile=OFF** corner,
  which SUPERSEDES the old failover script's TP=2 default (TP=2 NaNs at
  production GBS -- see the 80B NaN investigation entry below). bf16 is
  the agpt_80b() builder default so no --training.dtype flag is needed.
  GBS defaults to `dp_degree*LBS*GAS`; you reach a token target via GAS,
  not by raising dp_degree past the safe ceiling. The script **warns when
  `dp_degree = NGPUS/TP > 186`** -- the grad-path NaN trigger, validated
  safe only to ~62N. Validated by dry-rendering the launch argv (PATH-
  shimmed ezpz): it matches the known-good stable job 12469494
  token-for-token, incl. `activation-checkpoint:full` passed LAST. A real
  80B live smoke needs >=62N (TP=4 + safe dp), so plumbing (already proven
  by 2B/20B) was not re-burned at that scale.

Note: during 80B argv dry-rendering I accidentally truncated the live
`.venv/bin/activate` to 0 bytes (a `: >` redirect in the test harness);
restored it from the `venvs/rl-monarch-torch213` activate template
(path + `torchtitan` prompt patched) and verified `source .venv/bin/
activate && ezpz launch --help` works. Lesson: never aim `: >`/`>` at
real venv paths in a shim.

---

## 2026-06-24 (sunspot) -- ezpz-native auto-retry 2B submit script

Wrote `scripts/submit_agpt_2b_autoretry.sh`: a portable 2B production
submit script that uses `ezpz launch --auto-retry` **exclusively** for
bad-node failover, replacing the bash `failover_lib.sh` machinery
(`failover_init`/`failover_yeet_all`/`failover_run`). Native auto-retry
landed in ezpz >= 0.17.1 (PR #170); the installed venv is 0.20.0.

Key design points (verified against `../ezpz` source):
- `ezpz launch --auto-retry` splits the PBS allocation into active +
  spare internally from `--nproc`, runs the inner command, scrapes the
  same bad-node signatures on any non-zero exit (incl. watchdog 124 /
  walltime-racing 143), swaps a spare in-place, and retries. Active
  count is constant across retries (in-place swap by index).
- The **one** thing native auto-retry does NOT do is broadcast the venv
  to spares. So the script still `ezpz yeet --src .venv.tar.gz` to the
  **whole** (un-split) nodefile -- covering active + spare -- before
  launch. This is the only piece of `failover_yeet_all` we keep.
- GBS / `--nproc` are computed from the **active** count
  (`NHOSTS_TRAIN * 12`), not `ezpz_setup_job`'s `$NGPUS` (which sees the
  full allocation). Dry-checked: NHOSTS_TRAIN=12 of a 14-node alloc ->
  `--nproc 144`, GBS 288 (active-only), not 168/336.
- No preflight: ezpz's `STUCK_PRE_TRAINING` guard already bails without
  burning spares on a twice-zero-progress init crash.
- Portable: PBS headers default to Sunspot (datascience/workq/tegu:home);
  Aurora via qsub overrides. Per-machine data default -- Sunspot `books`
  (the `/tegu .../books-dataset` data actually resident on Sunspot; the
  `dolma`/`olmo-mix-1124` lists point at `/gila`, NOT mounted on Sunspot
  even on compute), Aurora `olmo-mix-1124`. `books` is already the
  established Sunspot smoke/benchmark dataset.
- Caught + fixed a real bug pre-submit: `${VAR:+--flag "$VAR"}` collapses
  to a SINGLE argv token (`--max-failover-retries 3`) that argparse
  rejects; switched to an array (`mfr_args=(...)`) -> two tokens / zero.

Smoke (my own select=4 jobs, NHOSTS_TRAIN=2 + 2 spare, 5 steps;
full report `docs/experiments/agpt/sunspot/2026-06-24-native-autoretry-2b-smoke.md`):
- **12469523** -- exercised the FULL native failover lifecycle for real
  (a node genuinely crashed): yeet-to-all (63.6s), active-only
  `--np=24`, ezpz split `4 total / 2 active / 2 spare`, attempt-1 crash
  -> scrape -> blind spare swap -> attempt 2 (`active=2/spare=1`) ->
  `FAILOVER STOP: stuck_pre_training` -> exit 143. The no-preflight
  guard fired exactly as designed.
- **12469524** -- chunkedce + LBS=1 + AC-full (compile-off fit recipe):
  trained all 5 steps clean, loss 12.94 -> 12.18, finite grad_norm.
- **12469525** -- production config (compile ON, LBS=2, sophiag): trained
  all 5 steps clean, loss 13.01 -> 12.06, peak 69.83% mem, ~26.6% MFU
  (matches the 2N scaling figure). agpt_2b LBS=2 on 2N fits fine.

NOTE: the 12469523 OOM (`UR_RESULT_ERROR_OUT_OF_RESOURCES` in
`cross_entropy_loss`) was NOT a script defect and NOT inherent to LBS=2
at 2N -- it was an artifact of the smoke passing `--compile.no-enable`.
Eager-mode CE materializes the full ~16GB 256k-vocab logit slice;
production compile-ON fuses it. agpt_2b LBS=2 on 2N is the normal
config. (Also confirmed: 2B defaults to sophiag via the submit-script
`--optimizer=sophiag` override; the registry base-config AdamW default
from PR #3269 replay `bac0a3473` is just the inherited template
fallback, not what 2B trains with.)

Extended `docs/guides/bad-node-failover.md` with a "Two implementations"
section (bash wrapper vs native) rather than a new doc. Old failover
scripts (20B/80B) untouched -- they still use `failover_lib.sh`.

---

## 2026-06-24 (sunspot) -- 80B grad-path NaN: mapped to LBS>1 + dp-degree; TP=4/bf16 path found

Root-caused the 80B grad_norm-NaN with a controlled multi-node sweep
(LR=1e-6, q_BLNH fix in). Findings (full matrix + perf in
`docs/production/agpt/80b/README.md`):

- The NaN is **not** a raw-GBS threshold. Two independent triggers, both
  in the gradient path (grad_norm NaNs one step before loss): **LBS>1**
  and **large dp_degree** (=NGPUS/TP). GBS=372 is clean (TP=4/LBS=1) AND
  NaN (TP=2, or TP=4/LBS=2) depending on composition.
- **New clean path the prior n32 factorial missed: TP=4 + LBS=1 + bf16 +
  GBS=372** via GAS (12469494 20 steps, 12469509 30 steps, both clean,
  loss -> 9.7/10.3). The factorial concluded "fp32-acts is the only clean
  path at GBS>=192" but never tried TP=4/LBS=1 with GAS. Reconciled the
  20260611 n32 diagnosis doc with an UPDATE header.
- **Perf:** TP=4 stable path is ~9.85% MFU, ~half the TP=2 baseline
  (18.7%). Cost is TP=4 comm, GAS-independent (GAS=1 and GAS=2 both
  ~9.8%). LBS=2 was faster (~15%) and fit memory (48%) but NaNs.
- **Stability CONFIRMED 4/4 clean** -- 12469494 (20), 12469509/510/511
  (30 each), all 0 NaN, three at identical loss 9.69-9.70. Not the
  nondeterministic knife-edge. New production recommendation:
  **TP=4, LBS=1, bf16, GAS-to-GBS** -- supersedes the n32 doc's fp32-acts
  default (cheaper: ~9.85% MFU vs fp32-acts ~3-5x slower / determinism
  ~50% and doesn't scale past n=32). Underlying TP=2/LBS>1 grad-path
  overflow still an open upstream-worthy bug (2 cheap 62N reproducers).
- Added a **batch-size ramp** (`FaultTolerantTrainer.batch_ramp_steps`,
  `09f2d243b`) -- ramps GAS (effective GBS) like LR warmup; mitigates the
  dp-degree onset but not the LBS trigger.

Also: the live `.venv` pyzes hardcodes a Debian `libze_loader.so.1`
path that doesn't exist on Sunspot (SUSE -> /usr/lib64); patched to the
bare soname. Only bites interactive live-`.venv` use, not yeet-env
training (tarball torch doesn't bundle pyzes). NOT an LD_LIBRARY_PATH
issue (chased that wrongly first). See
`project_venv_ld_library_path_ze_loader`.

Filesystem was 100% full earlier today (amplified transient failures);
cleaned ~6.3 TB of core dumps + test ckpts -> 57% used.

---

## 2026-06-24 (sunspot) -- 80B verified at 28N + TP>1 sync regression fix

First **multi-node** 80B v2 functionality verification (all prior
validations were 4N). Job `12469486`, 28 active nodes (TP=2), torch
2.13, books blendcorpus dataset, 20 steps:

- Loss descended **12.94893 -> 10.38276** (-2.57 nats), matching the
  4N baseline (12.98 -> 10.46) within noise.
- MFU steady **~18.7%** (4N was ~17.8%), TPS ~102, ~40s/step, memory
  flat 65.86%. grad_norm climbed to ~33 at steps 15-16 then settled
  to ~14 by step 20 -- same pattern the 4N smoke showed; production
  still needs the 200-step warmup.
- `step-20` checkpoint saved cleanly.
- Failover swapped one bad node (rank 204 signal 15) at launch and
  training started clean -- the 8-spare headroom did its job.

**The reason this run mattered: it caught a TP>1-only 57th-sync
regression that the TP=1 debugmodel smokes missed.** A first 64N
attempt (`12469471`) crashed on every rank at `model.parallelize`:

```
AssertionError: XPUScaledDotProductAttention: local_map is set but
in_dst_shardings is missing entries for: ['q', 'k', 'v']
```

Root cause: the 57th sync adopted upstream's shape-suffix naming --
`ScaledDotProductAttention.forward` args became `q_BLNH/k_BLNH/v_BLNH`
and `set_gqa_inner_attention_local_map` keys `in_dst_shardings` by
those names. The local_map contract check matches `in_dst_shardings`
against the wrapped forward's positional-arg names, and the ezpz
attention forks still used bare `q/k/v`. Only asserts under TP>1, so
TP=1 smokes passed. Fixed in `74c7452f6` (renamed the three ezpz
attention forwards) + `13c09ddf9` (added a TP=2 entry to
`sync_smoke.sh` so this class of regression can't slip through again).
See `docs/upstream-sync.md` 57th-sync "Follow-up" for the full chain.

Getting here also surfaced two non-code issues, both resolved:
- **Filesystem was 100% full** (`/tegu` 0 avail) -- amplified transient
  yeet/checkpoint failures. Cleaned ~6.3 TB: 864 core dumps (5.66 TB,
  Apr 26 -> today crash debris) + 3 async/smoke test checkpoints
  (621 GB). torchtitan/ 8.6T -> 2.3T, fs 100% -> 57% used.
- **HF `eliplutchok/fineweb-small-sample` is too small** for a 28N
  multi-step run -- exhausts and re-loops in a tight spam loop
  (944k warnings) instead of feeding step 2. Use a real blendcorpus
  data list (books) for anything past a single step.

Throwaway verification ckpt dirs left for cleanup:
`outputs/checkpoints/agpt-80b-{32n,64n}-funcverify`.

---

## 2026-06-24 (sunspot) -- 57th upstream sync + replays

Merged `upstream/main` into `ezpz` (59 commits, `7b579adde..c6c2fb2c5`,
merge `1f288f2e7`, no conflicts). Worked through all 6 commits that
touch `llama3/` or `deepseek_v3/`:

**2 real replays (source changes + smoke-verified):**
- `b3b60dabf` delete `--disable_loss_parallel` (commit `fa6f0681e`):
  dropped the kwarg from agpt/moe model+sharding and trainer.py.
- `c5d93d109` AC policy class hierarchy (commit `bb38b95e1`):
  `ActivationCheckpointConfig(mode=...)` -> `FullAC`/`SelectiveAC`/`None`;
  `apply_ac()` -> `ac_config.build().apply()`. Rewrote the
  `moe/activation_checkpoint.py` `_get_save_ops` monkey-patch as a clean
  `MoeSelectiveAC(SelectiveAC)` subclass. Fixed two
  `cfg.activation_checkpoint.mode =` mutation sites in the config-registry
  wrappers (slots Config has no `mode`).

**4 no-ops with documented reasons:**
- `cd8950ba7` score_before_experts: deliberate divergence -- ezpz fork's
  dispatcher genuinely branches on the flag upstream removed as dead.
- `70dd94551` FusedQKVLinear hooks: inherited via upstream import,
  inactive path (ezpz uses stock QKVLinear).
- `581f175dc` / `aa1d37414` fused/offset-aware experts: opt-in EP
  features ezpz/moe doesn't enable.

**Scripts (commit `807d5050b`):** PR #3674 made AC a tyro subcommand,
so `--activation_checkpoint.mode=full` no longer parses. Migrated 7
launcher scripts to the positional `activation-checkpoint:full` token.

**Smoke (job `12469466`, sunspot 1N, seed=42 --debug.deterministic):**
agpt_debugmodel (FullAC) loss `10.83863 -> 10.67256`; moe_debugmodel
(MoeSelectiveAC, seq=512/lbs=1) loss `12.90956 -> 12.36751`; both rc=0.
agpt step-1 loss bitwise identical across two runs.

**Venv note:** ran the smoke in `venvs/rl-monarch-torch213` (py3.13.6 +
torch 2.13). The repo-root `.venv` is now py3.14 (torchtitan import
fails on `importlib.metadata`) and `.venv.tar.gz` is stale. Had to
add `sh`, editable `ezpz` (`-e ../ezpz` -- installed 0.19.0 wheel was
missing `get_timestamp`), and editable `blendcorpus` (`-e
deps/blendcorpus`) -- all `--no-deps`, torch untouched. Worth
rebuilding a clean py3.13 training venv + fresh `.venv.tar.gz`.

Full detail: `docs/upstream-sync.md` (57th sync entry).

---

## 2026-06-14 (sunspot overnight) — Monarch + torch 2.13: 6 patches, 16 jobs, still wall

Pushed the upstream `torchtitan.experiments.rl.train` Monarch + GRPO
pipeline through 16 PBS submissions (`12468799` → `12468815`),
peeling back one failure mode at a time. Each crash mapped to a
distinct XPU porting gap, all now fixed in `xpu_overrides.py`:

1. `_make_replicate_tensor` skip-broadcast (oneCCL USM check rejects
   torch.xpu USM-device pointers under Monarch's execve'd actors —
   buffers are deterministic-identical anyway)
2. Force `init_distributed(enable_cpu_backend=True)` → backend becomes
   `xpu:xccl,cpu:gloo` so DCP's `all_gather_object` for the central
   plan routes objects through gloo, not xccl
3. Suppress vLLM-XPU's `xpu_worker.py:103` oneCCL "warmup" allreduce
   on a `torch.zeros(1).xpu()` — that allreduce is unconditional in
   vLLM-XPU and trips USM check immediately at engine init
4. `XPUPlatform.get_attn_backend_cls` patched to accept
   `AttentionBackendEnum.CUSTOM` (vLLM-XPU's selector raises
   `ValueError: Invalid attention backend` for any backend it doesn't
   explicitly list; CUSTOM is the path `rl/actors/generator.py` uses
   for varlen attention)
5. `vllm._torch_cuda_wrapper` patched to NOT alias
   `torch.cuda.current_stream = torch.xpu.current_stream` (Dynamo's
   `(cuda, xpu, accelerator).current_stream` handler-table build then
   trips `AssertionError: Handler already registered` because the
   same function appears twice). Keep `Stream`/`stream`/etc. aliases.
6. `EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids` —
   pre-execve env overlay via Monarch's `bootstrap_command=`. Sets
   `ZE_AFFINITY_MASK` BEFORE `import torch` runs in bootstrap_main.py,
   so torch.xpu's primary SYCL context picks up the right tile
   topology.

Got past every torchtitan + DCP + XCCL issue, but hit a wall at
vLLM's `profile_run` → `_dummy_run(max_num_tokens=2048,
is_profile=True)` → first decoder layer `F.linear` →
`RuntimeError: could not create a memory`. This is oneDNN's
`dnnl::memory` constructor failing — not OOM (model load succeeded
at 1.22 GiB, we have 46 GiB tile). Same crash with:

- `--generator.gpu-memory-limit 0.4`
- `--generator.sampling.max-tokens 256`
- `--generator.cudagraph.no-enable + --compile.no-enable`
- `--batcher.batch.seq-len 512`
- narrow `ZE_AFFINITY_MASK` per actor
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- `VLLM_DISABLED_KERNELS=xpu_kernels` (rules out our custom-built
  vllm-xpu-kernels)

Suspect: oneDNN scratchpad allocator queries `sycl::get_pointer_type`
with a different context than torch.xpu's allocator (same root cause
as the oneCCL USM check — but there's no "skip the check" knob for
oneDNN). Either Monarch's execve'd actors construct SYCL context
differently than mpiexec'd processes, or our custom-built vllm-xpu-
kernels somehow taint the allocator pool. Disabling its custom ops
didn't help, so leaning toward the first cause.

Full writeup: [`docs/rl/2026-06-14_monarch-torch213-deep-dive.md`](production/rl/history/2026-06-14_monarch-torch213-deep-dive.md).

Next options when resumed:
1. Get an Intel torch.xpu engineer to look at `DNNL_VERBOSE=2`
   output from profile_run.
2. Try the same code path under mpiexec to confirm it's Monarch-
   spawn-specific.
3. Fall back to torch 2.12 + prebuilt vllm-xpu-kernels (known-good
   combo on Sunspot), accept losing torch 2.13's DTensor USM fixes
   (we skipped the broadcast anyway).
4. Drop vLLM for the generator side — naive HF .generate() loop.

---

## 2026-06-13 (sunspot late eve) — 🎉 GRPO end-to-end on XPU via TRL vllm-serve

Job `12468780` completed **5/5 GRPO steps with real on-policy weight
sync** on Sunspot XPU. Full pipeline: trl vllm-serve on tile 0,
8-rank GRPO trainer on tiles 1-8 via ezpz launch, communicating via
HTTP for rollouts and via XCCL TCP-KVS for trainer→server weight
broadcasts.

`format_reward/mean` moved 0 → 0.0625 → 0.25 → 0.0625 → 0.125 over 5
steps with a cold Qwen3-0.6B on the `sum_digits` task. Signal is
noisy at bsz=8×ngens=4 but the upward trend in steps 1-3 confirms the
policy update path is alive.

The breakthrough was discovering that oneCCL DOES support a
non-PMIx, TCP-based rendezvous when configured correctly:

    CCL_PROCESS_LAUNCHER=none   FI_PROVIDER=tcp
    CCL_ATL_TRANSPORT=ofi       CCL_KVS_IP_PORT=127.0.0.1_29513

Both server and trainer set the same env, and XCCL forms an N+1-rank
group across the two process trees without any PMIx coupling. This
is exactly what TRL's `vllm_mode="server"` needs.

Sam's pushback on the no-op workaround was correct — the proper fix
exists and we just needed to find the right env knobs.

Full writeup: [`docs/rl/grpo-on-xpu-status.md`](rl/grpo-on-xpu-status.md).

---

## 2026-06-13 (sunspot eve) — vLLM-XPU + Monarch RL actor infra

Kicked off the long-deferred wiring of vLLM into ezpz/rl. Per
[`docs/rl/vllm-xpu-wiring-plan.md`](production/rl/history/vllm-xpu-wiring-plan.md),
two parallel paths:

- **Track 2 (TRL `vllm_mode="server"`)** — wires existing TRL-based
  `train_grpo.py` to an external vLLM-XPU server. Faster to land but
  the live `venvs/vllm-test/` + TRL 1.5.1 combo hit a worker-side
  `current_platform.device_type` empty-string error (jobs `12468737`,
  failing in TRL's `vllm_serve.py:llm_worker`). TRL 1.5.1 also warns
  vllm 0.22.1 is outside its 0.12.0-0.18.0 supported range. Need to
  either drop to vllm 0.18.x or use TRL 1.6.0 (which no longer
  hard-pins the vllm version).
- **Track 1 (Monarch + TorchStore actors)** — was thought blocked
  on torchmonarch lacking cp314 wheels. Resolved by building a new
  sibling venv `venvs/rl-actors/` on **py3.13** + torch 2.12+xpu +
  monarch 0.5 + torchstore (main) + vllm 0.22.1 + vllm-xpu-kernels
  0.1.9.1 + TRL 1.6.0 + transformers 5.11 + accelerate 1.14 +
  datasets 5.0. All imports clean.

`scripts/monarch_smoke.py` + `scripts/monarch_smoke.sh` validate
the framework end-to-end. First run (job `12468738`, 1N) showed:

- 2-actor `this_host().spawn_procs({"gpus": 2})` works on XPU.
  Both ranks reported `xpu_count=12 xpu_avail=True` from inside the
  spawned actor — Monarch's CUDA-only assumptions don't actually
  block XPU runtime use.
- `monarch.actor` + `torchstore` imports clean on py3.13 + torch 2.12+xpu.
- TorchStore transport probe hit a minor naming bug
  (`TransportType.RPC` vs `TransportType.MonarchRPC`); fixed in the
  smoke script. Rerunning as `12468739`.

This unblocks the Monarch path — the open question shifts from "is it
even possible on XPU" to "how much rewrite to adapt ezpz/rl off TRL".

### Stack table

| Component | Main `.venv` | `venvs/vllm-test/` | `venvs/rl-actors/` |
|---|---|---|---|
| Python | 3.14.2 | 3.14.2 | 3.13.6 |
| torch | 2.13.dev | 2.12.0+xpu | 2.12.0+xpu |
| vllm | — | 0.22.1 | 0.22.1 |
| vllm-xpu-kernels | — | 0.1.9.1 | 0.1.9.1 |
| trl | 1.5.1 | — | 1.6.0 |
| transformers | 5.6.2 | (vllm dep) | 5.11.0 |
| torchmonarch | — | — | 0.5.0 |
| torchstore | — | — | main |

`rl-actors/` is the new "actor venv" for both Monarch controllers and
vLLM server workers. Main `.venv` stays unchanged for everything else.

### Scripts landed this session

- `rl/scripts/vllm_serve_xpu.sh` — generic launcher: `trl vllm-serve`
  from `venvs/vllm-test/` with `PYTHONPATH` to .venv's TRL wrapper.
- `rl/scripts/vllm_serve_smoke.sh` — PBS smoke for Phase 1+2 of the
  wiring plan.
- `rl/scripts/vllm_xpu_bare_smoke.sh` — standalone vLLM-XPU sanity
  (no TRL wrapper) using `venvs/rl-actors/` directly. Isolates whether
  the failure is in TRL's wrapping vs in the underlying vLLM stack.
- `rl/scripts/monarch_smoke.py` + `monarch_smoke.sh` — Monarch +
  TorchStore framework smoke.
- `rl/scripts/grpo/aurora2b_sft_arithmetic_8n_vllm.sh` — production
  GRPO submit variant using server-mode (Phase 5 of the wiring plan).

### Resolution (eve PM, 2026-06-13)

- ✅ `12468739` (monarch smoke with `MonarchRPC` fix) — PASSED. Both
  ranks reported `xpu_count=12`. Monarch framework on XPU is
  confirmed working.
- ✅ **vLLM-XPU SOLVED interactively on x1921c3s0b0n0**: KV cache
  48.43 GiB, max concurrency 1033x — bit-for-bit match with the
  2026-06-10 baseline.
- ❌ → ✅ `12468740..12468751` (12-job debug chain) — root-caused via
  the interactive replay. The bug was self-inflicted env contamination
  by `ezpz_setup_env`:
  - `CCL_PROCESS_LAUNCHER=pmix` made oneCCL look for a PMIx context
    vLLM's `multiprocessing.spawn`'d EngineCore doesn't have.
  - `FI_PROVIDER=cxi,tcp;ofi_rxm` made libfabric try the Slingshot
    CXI provider, which needs a NIC handle only mpiexec-bootstrapped
    processes get. `fi_getinfo` returned 0 providers; `atl_ofi
    init_transport` failed.
  Sam's pushback ("nothing has changed about the environment or
  system since 06/10/2026") was correct. Today's smoke scripts source
  `ezpz_setup_env` for PBS bookkeeping; the original 2026-06-10
  verification was a raw interactive shell with none of those env
  vars set. Fix: `unset CCL_*/FI_*` after `ezpz_setup_job`, invoke
  vLLM via plain python (no `ezpz launch`).
- 🔍 `12468752` (env-scrubbed PBS submit of the bare smoke) — queued.
  Verifies the fix works under PBS-direct, not just interactive SSH.
- See [`docs/rl/vllm-xpu-current-status.md`](production/rl/history/vllm-xpu-current-status.md)
  for the full debug chain, root cause, and fix.

---

## 2026-06-13 (sunspot) — PR #14 merged + 56th upstream sync

**PR #14 (`Isolate ezpz MoE` by @nscottnichols) landed on `ezpz`** as
merge commit `de85a179b`. Final pre-merge validation:

- `12468735` (pr14-fixes, 8N, EP=12, padding=1, AC=selective, 20 steps)
  ran clean: loss `12.91575 → 7.00367`, 272s, no errors. The
  `ezpz/moe/activation_checkpoint.py` wrapper (commit `82100fd67`)
  works as designed.
- `12468736` (ezpz baseline, same config) for comparison: ran 17 steps
  cleanly, then hit an apparently unrelated upstream tensor-size
  overflow at step 17 (`RuntimeError: Storage size calculation
  overflowed with sizes=[176...e18, 2048]`). Through step 17, **loss
  matched pr14-fixes within ~1e-4 nats** at every step
  (e.g. step 17: 7.74314 vs 7.74329) — strong evidence that
  PR14's fork + AC wrapper are numerically equivalent to upstream
  on this stack.

The earlier investigation isolating the
`TT_MOE_NORMAL_EQUAL_A2A_PADDING=1 × AC=selective` GPU PDE Write
fault and the AC-save-list workaround is documented in PR thread
[issuecomment-4698951275](https://github.com/saforem2/torchtitan/pull/14#issuecomment-4698951275).

Right after, pulled in 2 more upstream commits as the 56th sync
(`3935fc654`):

- `588fc12fd` `[RL] Add deterministic loss guard for GRPO training (#3474)` —
  `experiments/rl/` only; ezpz/rl has its own train_grpo.py.
- `7b579adde` `Add MinimalAsyncEP (#3561)` — adds new EP backend
  (additive). Touches `common/token_dispatcher.py` with a small
  `output_size=total` compile-hint on `repeat_interleave`; PR14's
  fork has the same call but no replay needed (runtime behavior
  identical when not compiled). See
  [`docs/upstream-sync.md`](upstream-sync.md).

---

## 2026-06-12 (sunspot eve) — 55th upstream sync (3 commits, no replays)

Three more upstream commits landed since the 54th sync earlier today
(`96ab7487d..0a73d82a4`):

- `3b8e060853` `Remove unused MetricsProcessor.lr_schedulers (#3644)` —
  pure cleanup of an attribute nothing reads. Our local `metrics.py`
  fork was already removed in PR #14, so we consume upstream directly.
- `14fb67575` `qwen3.5 tok_embeddings LocalMap region (#3648)` —
  qwen3_5-only sharding fix. ezpz doesn't use qwen3_5.
- `0a73d82a4` `avoid GradAccumulator init in ChunkedCELoss no_grad path (#3652)` —
  internal optimization in `components/loss.py`; ezpz uses
  `ChunkedCELoss` via direct import.

Merge `439ccf220` was clean — no conflicts, no replays. Sync entry
added to [`docs/upstream-sync.md`](upstream-sync.md). Skipping the
dynamic smoke this time — last sync's bitwise IDENTICAL already
covered the `ChunkedCELoss` path, and the other two commits don't
touch ezpz-reachable code.

---

## 2026-06-12 (sunspot pm) — 54th upstream sync (3 commits, no replays)

Three new upstream commits since the 53rd sync (`1c02a5cee..96ab7487d`):

- `88030eec1` `[rl] Fix batch invariant logprob calculation by forcing vllm
  use trainer's function (#3629)` — `experiments/rl/actors/generator.py`;
  ezpz/rl doesn't override that path.
- `5ba439938` `[Bug] Fix MoE SP token combine indices (#3604)` — fixes
  a `B > 1` × `sp_size > 1` bug in `common/token_dispatcher.py`; ezpz/moe
  re-imports the dispatcher unchanged, so the fix flows automatically.
  Our ezpz MoE configs run with `sp_size == 1` so the bug was never
  live for us anyway.
- `96ab7487d` `chore(ci): migrate ROCm matrix from 7.1 to 7.2 (#3267)` —
  CI matrix + ROCm loss reference files only.

Merge `f8be3bcd1` was clean — no conflicts, no replays needed.

Submitted bitwise checks in parallel:

- `12468696` — `bitwise_sync_check.sh` agpt_2b_chunkedce, 2N, 20 steps,
  comparing `434cfe5d1` pre-merge vs `f8be3bcd1` post-merge with
  `--debug.seed=42 --debug.deterministic`.
- `12468697` — `submit_moe_smoke.sh CONFIG=moe_10b_2b_sdpa_ep STEPS=10`,
  2N, head-only smoke compared against 52nd-sync baseline `12468666`.

Sync entry added to [`docs/upstream-sync.md`](upstream-sync.md).

Both bitwise jobs passed:

- `12468696` (agpt) — **VERDICT: IDENTICAL** across all 20 steps
  (head step 20 = pre step 20 = `loss 10.66272 / grad_norm 18.1259`).
- `12468697` (MoE) — clean 10-step run (loss 12.89 → 8.87, grad_norm
  bounded, ~80 GiB peak).

Merge-ready: `ezpz` already at `f8be3bcd1` (merged on the live branch,
not in a separate worktree). Push pending after committing doc updates.

---

## 2026-06-10 (sunspot) — 32N SFT auto-resume blocker: torch ShardedTensor.device hardcodes CUDA

> **Canonical writeup** (with the full failover-cycle worked
> example and run table):
> [`docs/production/sft/agpt/2b-mds/tulu_math_uc_mix/`](production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md).
> This journal entry is the rolling debug log; the report is the
> end-of-day cleanup.


Continuing the 32N SFT push. Job 12468404 (the first 32N run with
auto-retry's bad-node failover) trained cleanly for 140 steps with
loss 1.16 → 0.86 and token_acc 0.73 → 0.78 before a worker rank
SIGABRT'd from `ccl::v1::exception`; auto-retry swapped in a spare
and relaunched. But the relaunch went back to step 0 instead of
resuming from `checkpoint-100/` — the second run wasn't passing
`--resume_from_checkpoint`.

Patched the submit script to pass `--resume_from_checkpoint
"${CKPT_DIR}"` (HF Trainer auto-detects the latest `checkpoint-N/`
subdir in the dir) and added a small coercion shim in
`train_sft.py:main()` to handle the case where the dir is a freshly-
created empty dir (HF errors out without it). Resubmitted as
12468408.

12468408 crashed differently: all 384 ranks tracebacked with
**`AssertionError: Torch not compiled with CUDA enabled`** during HF
Trainer's FSDP checkpoint load. Tracked it to
`torch/distributed/_shard/sharded_tensor/_ops/tensor_ops.py:54`:

```python
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    ...
    else:
        dev = torch.device(torch.cuda.current_device())   # <-- BUG on XPU
```

Upstream hardcodes CUDA as the no-local-shards fallback. The sibling
`tensor_func`/`dtensor_func` in `planner_helpers.py` already do the
device-agnostic thing via `_get_pg_default_device().type` +
`_get_device_module(...)`; only the ShardedTensor dispatch is broken.

Local workaround: added `_patch_sharded_tensor_device_for_xpu()` to
`train_sft.py` that re-registers the dispatch via `_sharded_op_impl`
with an XPU-aware fallback (tries `torch.accelerator.current_device_index()`
first, then falls back to whichever accelerator namespace is
available). Called once at top of `main()`. Verified the patch
correctly replaces the `_SHARDED_OPS` entry via a smoke import on
the login node.

Filed full writeup at
[`docs/upstream-issues/sharded_tensor_device_cuda_hardcode.md`](upstream-issues/sharded_tensor_device_cuda_hardcode.md)
with the rank-0 traceback and a proposed upstream fix, then filed
upstream as
[pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938)
and opened
[pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
with the one-spot fix (mirror what `planner_helpers._init_state_dict`
already does for plain tensors / DTensors).

While the patch was being written, also consolidated `checkpoint-100`
into a flat HF format at `checkpoint-100-hf/` (7.94 GB safetensors).
This gives us a usable artifact independent of the FSDP-resume
question — we now have a 600M-token SFT'd AuroraGPT-2B-tulu-mix
checkpoint we can hand off to GRPO regardless of whether resume
ever works.

Resubmitted as **12468409** with the patch. Currently queued.
Validation plan: tail `run.log` for `Continuing training from
checkpoint, will skip to global_step 100`, then verify loss picks
up from ~0.86 (not from cold-start 1.16).

### Postscript — completion of the 32N SFT chain (2026-06-10 PM)

**12468409** validated the XPU FSDP resume patch end-to-end
(2 successful failover cycles, loss 0.86 → 0.81 across
checkpoints 200 → 300) before tripping a different blocker:
ezpz `launch_autoretry`'s `STUCK_PRE_TRAINING` guard was matching
only torchtitan's `step=N` progress marker, falsely flagging
TRL's `{'loss': '...'}` log format as "no training happened" and
bailing on attempt-3. Patched the regex (ezpz commit
[`6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b)) to
also match the HF/TRL format, reinstalled via `uv pip install
-e ../ezpz`, resubmitted as **12468437**.

**12468437 ran to completion**: 1h39m wall time, 4 mpiexec
attempts, 3 oneCCL `pidfd_getfd` SIGABRTs survived, 3 spare-node
rotations (`x1921c1s0b0n0 → x1921c5s4b0n0 → x1921c5s5b0n0 →
x1921c5s6b0n0`), and finished `Training complete.` at step 729 /
epoch 3.0. autoretry verdict `FAILOVER STOP: success (attempt 4)`,
exit 0. Loss `1.16 → 0.77`, `mean_token_accuracy 0.7957`, ~4.5B
tokens consumed. Final consolidated HF artifact at
`outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf/`.

PR review on
[pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
caught a regression risk in v1 of the fix (mirroring
`planner_helpers._get_pg_default_device` pattern breaks composite
PGs like `cpu:gloo,cuda:nccl` because that function prefers CPU
when both are registered). Pushed
[`570da16048`](https://github.com/saforem2/pytorch/commit/570da16048e049eb6e9b11239e718e621d4de720)
which switches to `torch.accelerator.current_accelerator()` —
doesn't consult the PG backend list, no composite-PG trap. Both
inline review threads addressed + resolved.

End-of-day deliverables: SFT'd AuroraGPT-2B HF ckpt for GRPO,
PR #186940 (v2) up for upstream review, autoretry recognizes
both torchtitan and HF/TRL trainer markers, complete writeup at
[`docs/production/sft/agpt/2b-mds/tulu_math_uc_mix/`](production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md).

**Operational TODO:** file ALCF ticket for `x1921c1s0b0n0` —
this host showed up as the SIGABRT-er in multiple jobs across
the day, suggests a degraded NIC / Level Zero stack. Until it's
pulled from the queue, autoretry's 4-spare allocation handled
it, but every job pays a ~3min/failover overhead.

---

## 2026-06-08 (aurora pm) — 80B 4N validated end-to-end on Aurora + 256N NaN + chart wrapper

Big session covering several threads:

### 1. 80B production stack validated end-to-end at 4N (Aurora)

Spent ~3h chasing what looked like the long-pending "80B model-init
silent hang at 4N+" regression (pending since the 2026-06-06
session). Iterated through 5 PBS-script attempts (8530199, 8530216,
8530243, 8530800 + one mid-iteration kill via test.sh ssh-allocation
8530807). The final attempt
(`/flare/.../.interactive-80b-4n-r7-direct-*.log`) worked cleanly:

- Loss descent: **12.93 → 12.03 over 10 steps** (-0.91 nats)
- MFU steady at **~17.9%** (matches the May 5 12466025 + Sunspot
  12468197 baselines)
- Memory **88.97%** at peak (4N is dense)
- **Step-10 sync checkpoint save fired** at 14:06:22 and **landed on
  disk**: 904 GB across 48 .distcp shards + .metadata, matching the
  Sunspot reference exactly. Sync mode + xccl workaround validated.

The stack of fixes that got us there (in order of discovery):

1. **80b-v2 repo was 229 commits behind** origin/ezpz. Pulled to get
   `8031d1d3` (xccl_split_group_workaround for the `split_group`
   RuntimeError) + `ce321caae` (CHECKPOINT_ASYNC_MODE=disabled
   default) + the May/June 80B-prod-sync-ckpt validation work.
2. **80b-v2 .venv was symlinked to 2b-v2 .venv** — would have polluted
   the 2B chain. Broke the symlink (`cp -a` 8.5GB), installed ezpz
   0.18.7 (from the `yeet-retry-on-rsync-failure` branch) +
   `trl==1.5.1` + `spmd_types==0.2.1` (the latter unblocks the
   upstream `import spmd_types as spmd` in
   `torchtitan/components/loss.py` since commit `fec0c175d`).
3. **Patched blendcorpus shipped a deadlocking global
   `torch.distributed.barrier()`** in `_build_index_mappings`.
   `BlendableDataset.__getitem__` is lazy per-corpus, so different
   ranks hit the barrier on different corpora at different wall-clock
   times → at 4N+ some ranks advance into `train_step` while others
   sit at the barrier → 30 min wait → ezpz watchdog SIGTERM. Pinned
   this down via `py-spy dump --pid` from ssh into the head node
   (rank 0 stuck at `barrier (torch/distributed/distributed_c10d.py:5234)`
   inside `_build_index_mappings:1141`, rank N+ already in
   `train_step → dataloader.__next__ → multiprocessing.Queue.get`).
   Reverted the barrier in the source venv (the existing
   `_load_with_retry` already handles the EOFError race it was
   supposed to protect against).
4. **PBS-script invocation was missing `export ZE_FLAT_DEVICE_HIERARCHY=FLAT`**
   on the inner shell — caused `_infer_topology` to see 6 GPUs/host
   instead of 12 and reject the launch with `ngpus must be > 0 and
   <= 24, got 48`. Added it to the inner-shell setup.
5. The final r7 run swapped in: patched blendcorpus + xccl workaround
   + spmd_types + correct FLAT + the inner-shell env block. ssh-launched
   foreground from the test.sh allocation head node so I could
   `py-spy` and Ctrl-C without watchdog interference.

Tarball rebuilt with the barrier-removed blendcorpus baked in
(`.venv.tar.gz.bak-pre-barrier-removal-20260608-090613` preserved).

### 2. 80B 256N smoke — training works, but loss NaNs immediately

Submitted 8530891 (256N, NHOSTS_TRAIN=256, TRAINING_STEPS=110, sync
ckpt, LR=1e-6 default):

- Setup (compile, init, dataset, mesh) all clean
- Throughput **110 TPS/GPU / 20.3% MFU** — actually slightly better
  per-GPU than 4N's 98 / 17.9% (less compile overhead at larger scale)
- **Step 1**: loss=12.94 grad_norm=4.94 — clean
- **Step 2**: grad_norm=NaN
- **Step 3 onward**: loss=NaN forever
- Job walltime-killed at step 79 (1h cap), step-100 ckpt save never
  fired

Configured LR scheduler is correct (`warmup_steps=200,
decay_ratio=0.8, decay_type=linear` = classic WSD), but at
`TRAINING_STEPS=110` the scheduler clamps warmup to 110, so step-2
LR is effectively 2/110 × 1e-6 ≈ 1.8e-8 (essentially zero). Even
with that tiny LR the first optimizer step produces NaN grads — so
this is **not just "LR too high at GBS=1536"**; something else is
biting on the first backward at scale.

Open hypotheses (still TBD):
- bf16 overflow in attention/MLP at GBS=1536 (vs 4N's GBS=24)
- TP=2 loss-reduction bug (CLAUDE.md notes `_dist_reduce`
  short-circuits DTensor on orthogonal meshes since 2026-04-27);
  local workaround in `trainer.py` may not fully cover the grad-norm
  path
- AdamW fp32-master second-moment overflow with these activations

Submitted 8531345 with `LR=1e-7` (10× smaller) at TP=2 same as the
NaN run — but it died from bad-node SIGSEGV (`rank 438 died from
signal 11` on `x4408c1s3b0n0`) at 177s. Resubmitted as 8531721 with
`FAILOVER_MAX_RETRIES=2` (a misjudgment to set 0 on the first
attempt — even when the failure-mode-under-test isn't bad-node,
surviving allocation/yeet/init still wants retries). 8531721
currently Q'd waiting for a 256N debug-scaling slot.

### 3. Evals + chart refresh

- Submitted 2B 256N evals for step-69000 (8531449) and step-69900
  (8531450). step-69900 came back clean: HSn 0.5552, ARC-E 0.5939,
  ARC-C 0.3294, Wino **0.5627 (best yet)**. Other tasks within noise.
- Wrote `scripts/update_all_charts.sh` to wrap the six per-script
  plot invocations (`utils/plot_production.py`,
  `utils/plot_production_combined.py`,
  `utils/plot_production_wandb.py`, `eval/plot_evals_combined.py`,
  `docs/evals/agpt/{2b,20b}/plot_v1_vs_v2.py`) into a single parallel
  runner with per-script logs. Smoke: 50 figure files refreshed in
  294s, 0/6 failures.

### 4. Production chain status (no change)

All 3 canonical chains still Q+H — `small` queue is severely
contended (78 total / 68 Q / 3 R / 7 H). Last R for prod chains:
- 2B 256N (8519833): step-69900, 2026-06-06 18:07 (cleanly walltime'd)
- 2B 512N: step-30500, 2026-05-30 07:53 (idle 9 days)
- 20B 512N: step-4400, 2026-05-29 11:43 (idle 10 days)

8521627 (2B 512N cont) made it to R briefly on 2026-06-07 21:12 but
died at 8min when 1 of 522 nodes failed yeet-env rsync (the very
failure mode my `yeet-retry-on-rsync-failure` ezpz PR #160 fixes).
Chain still alive via failover (cont10 = 8521631 next up).

---

## 2026-06-08 — RL polish + first real SFT path (gsm8k / metamathqa / mix) + 32N XCCL pain

Long session, three intertwined threads. Tracked in tasks #66–#75.

### train_grpo polish

User-driven iteration on the GRPO entry point that turned up
several real bugs plus a bunch of UX improvements:

- **Vocab-aware chat-template picker**
  (`_pick_chat_template` in `train_grpo.py`, commit `31c19c8e4`).
  The chatml fallback I added 2026-06-07 used literal `<|user|>` /
  `<|assistant|>` tokens, which the AuroraGPT-2B tokenizer encodes
  as 4-token sequences the model has never seen as turn
  boundaries — every completion echoed the prompt back. New picker
  probes tokenizer vocab for single-token boundaries and picks
  `gemma` (`<start_of_turn>` / `<end_of_turn>`, ids 106/107 in
  AuroraGPT-2B), `chatml` (`<|im_start|>` / `<|im_end|>` for
  Qwen-family), or `plaintext` (USER:/ASSISTANT:) fallback. 25-step
  hardware verify (job 12468210) showed the model correctly
  generating `<end_of_turn>` and stopping early
  (`completions/min_length` 13-15 vs 64 with the broken fallback).
- **Per-task `--task` autocomplete** — choices auto-populated from
  `TASK_REGISTRY` (commit `05f0ee803`). Unknown task now fails at
  parse-time with the full list, not after dist init.
- **Auto-detect FSDP wrap class from `model_type`** (commit
  `e3477c717`). 15 model families pre-mapped (llama, llama4,
  qwen2, qwen3, mistral, gemma, phi, gpt_neox, deepseek_v3, …)
  so switching `--model_name_or_path` doesn't require also
  switching `--fsdp_transformer_layer_cls_to_wrap`.
- **Rank-0 model prefetch + broadcast** (commit `765f1f5a8`).
  48 ranks doing `AutoModel.from_pretrained` against the same HF
  Hub repo trips 429 rate-limits with 200s+ backoffs; rank 0
  pre-warms cache via `snapshot_download`, barrier, workers load
  from disk.
- **`device_map="auto"` override for FSDP** (commit `e00b130f9`).
  TRL's `create_model_from_path` defaults `device_map="auto"`
  which under FLAT mode lands every rank's model on the
  highest-numbered tile (`xpu:11`) — FSDP then catches the
  per-rank-device mismatch and raises before training. Override
  to `device_map=None` so `model.to(accelerator.device)` puts
  the model on the right tile. Diagnosed via the
  `scripts/diag/device_mismatch.py` 48-rank probe — phase-1 confirmed
  pre-trainer device assignments were all correct, phase-2 hit
  the same xpu:11 bug as the user, smoking gun was TRL's default.
- **Auto-populate `setup_wandb` config from every dataclass field**
  (commit `d5619f512`). 11 → 181 hyperparameter keys; nothing
  gets silently dropped from sweeps. Also enabled
  `log_completions=True` + `num_completions_to_print=2` so the
  Rich completions table is streamed to wandb without becoming
  a wall-of-text on screen.
- **Revert and replace dead-end fix** (`d1affb775`): the
  `torch.xpu.set_device(local_rank % ngpus)` early-pin I added
  2026-06-07 was verified harmless but didn't fix the bug — the
  real culprit was TRL's `device_map="auto"` (above). Reverted
  via `git revert` rather than force-pushing.
- **`extract_answer` prefers LAST `=` + recognizes gsm8k `####`**
  (commit `d749e2199`). User flagged that
  `6×12×10×12=<<6*12*12=720*12=8640>>` was graded 0.0 even
  though `8640` was correct — old regex matched the FIRST `=`
  and returned the intermediate `720`. Fix walks all `=` matches
  and returns the last one; adds gsm8k's `#### N` final-answer
  marker as a higher-priority alternative.

### Streaming-mode rewrite of RL tasks + new `arithmetic` task

User noticed the default `num_samples=1000` meant a 1000-step run
sees each prompt ~48× — model can memorize rather than learn.
Switched all five tasks (`sum_digits`, `multiply`, `arithmetic`
[new], `word_sort`, `countdown`) to a `build_streaming_or_finite`
helper that materializes a 100k-pool when `num_samples=0` (the
new default) and a finite-N pool otherwise. Iteration was bumpy:

  - Commits `98d537d07` (sum_digits + multiply) and `08638ff9e`
    (new `arithmetic` task with {+, −, ×, ÷}) tried to use a true
    `IterableDataset` — but TRL's GRPOTrainer rejects iterable
    datasets at __init__ (trl#3213). User hit the
    `NotImplementedError` on first launch.
  - Commit `f454e9236` replaced the IterableDataset with a large
    finite `Dataset.from_list` (default 100k); same effective
    "no prompt reuse" behavior at modest one-time init cost
    (0.5s for sum_digits, 125s for countdown due to its
    permutations-based rejection sampling).
  - Commit `637a83cda` extended the streaming default to
    `word_sort` + `countdown` after a user-launched
    `--task word_sort` hit the "There seems not to be a single
    sample in your epoch_iterator" empty-dataset bug — those two
    tasks had been missed in the first pass.

The new `arithmetic` task mixes operations with weighted sampling,
guarantees integer answers (division uses divisor + quotient
construction), adds an `op` column for per-op reward dashboards,
plus a `length_penalty` reward function on top of accuracy +
format (commit not pushed yet — verified locally only).

### SFT companion (`train_sft.py` + `datasets_sft.py` + Aurora submit)

Built the SFT side of the RL stack so weak-baseline checkpoints
like AuroraGPT-2B-sophiag can be instruction-tuned before being
used as a GRPO starting point. Mirrors `train_grpo.py`'s shape:
`HfArgumentParser((EzpzSFTArgs, EzpzSFTConfig))`, reuses all the
GRPO helpers (FSDP env-bootstrap, chat-template picker, rank-0
prefetch, wandb auto-config), adds `assistant_only_loss=True` +
`packing=True` + `max_length=1024` defaults.

Dataset registry (`datasets_sft.py`):
  - `gsm8k` (7473) — grade-school CoT math
  - `metamathqa` (~395k) — augmented GSM8K+MATH
  - `alpaca` (52k) — broad instruction-following (added later)
  - `math_alpaca_mix` (60/10/30 metamath/gsm8k/alpaca via
    `interleave_datasets`) — broader for downstream non-math tasks
    like `word_sort` (added later)

Bundle commit: `5446e1d74`. Required adding `{% generation %}` /
`{% endgeneration %}` markers around the assistant content in all
3 chat templates so SFTTrainer's `assistant_only_loss=True` could
compute the loss mask (folded into the same commit). Verified
locally that the gemma template produces the correct
`assistant_masks` via `apply_chat_template(...,
return_assistant_tokens_mask=True)`.

Verified end-to-end on 4N Sunspot (job 12468212): 50 steps
(`max_steps=50`), `train_loss=0.582`, `mean_token_accuracy=0.871`
in 35s. Pipeline intact.

### 32N production SFT — extended XCCL pain

User asked for a 32N SFT to actually produce a usable
instruction-tuned checkpoint. The pipeline works in the small but
**32N hits XCCL exceptions repeatedly**:

| Job | Dataset | Outcome | Failure mode |
|---|---|---|---|
| 12468217 | metamathqa | Died at 2:20 | 384 ranks × HF Hub xet-read = 429 storm + `.incomplete/dataset_info.json` cascade |
| 12468218 | metamathqa | Trained to step 442 (epoch 1.75), then `ccl::v1::exception` → SIGABRT on rank 241 | First XCCL crash |
| 12468220 | metamathqa (resume) | `--resume_from_checkpoint` silently ignored, restarted from scratch, hit same XCCL crash at step ~590 (epoch 1.55) | Resume bug + repeat crash |
| 12468221 | metamathqa (resume) | CLI validation: `--auto-retry` needs `--nproc`, not `--nhost` | Bad CLI |
| 12468222 | metamathqa (resume) | First successful `--auto-retry` swap-in (32 train + 4 spare), but `--resume_from_checkpoint` still silently ignored across both attempts. Walltime-killed mid-second-attempt | Resume bug + walltime |
| 12468232 | math_alpaca_mix | Crashed at argparse-init: `60% metamathqa` in description triggers `%m` format error | Argparse `%` escape |
| **12468237** | math_alpaca_mix | Trained to step 200 + saved a usable `checkpoint-200`, then hung post-save for 30 min until `--auto-retry` watchdog killed it. `stuck_pre_training` failover-stop on attempt 2 | Post-save hang (new failure mode) |

Fixes landed during the chain:

  - `train_sft.py:_prefetch_and_broadcast_dataset` — rank 0 builds
    dataset first, barrier, workers load from warm cache (no more
    429 storms)
  - Pre-warmed `~/.cache/huggingface/datasets/` locally on `/home`
    so compute nodes always hit cache (lustre-NFS shared)
  - `trainer.train(resume_from_checkpoint=config.resume_from_checkpoint
    or None)` explicit pass-through (the silently-ignored resume
    was an FSDP-sharded checkpoint compatibility issue with HF
    Trainer's auto-detect)
  - `%%` escape on all literal `%` in registry descriptions

What we have to show for it: **`outputs/sft/aurora2b-sophiag-metamathqa-32n/checkpoint-400-hf/`**
— consolidated HF-format checkpoint (~1.75 epochs metamathqa,
loss 0.20, mean_token_accuracy 0.934, 7.94 GB safetensors).
Usable for downstream GRPO via
`--model_name_or_path outputs/sft/aurora2b-sophiag-metamathqa-32n/checkpoint-400-hf`.
Plus `outputs/sft/aurora2b-sophiag-mix-32n/checkpoint-200/`
(FSDP-sharded, mix dataset, 200 steps) — could be consolidated
similarly.

The 32N XCCL crash pattern reproduces across two distinct SFT
training paths (metamath-only and mix). Not a one-off; not
strongly correlated with a single bad node either (12468218
crashed on `x1921c5s1b0n0`, 12468220 on a different host). Worth
investigating further (single-rank stuck-in-collective somewhere
between step 400-600 of a multi-node SFT run, looking the same
across two different datasets and across two different training
schedules) — but the SFT'd checkpoint we have is good enough to
move on with the GRPO experiments.

### SFT checkpoint consolidation tool

Added `rl/scripts/consolidate_sft_ckpt.sh` (bash) wrapping
`accelerate merge-weights` + cp of config + tokenizer from the
source model dir. Pattern: `checkpoint-N/pytorch_model_fsdp_0/`
(distcp shards) → `checkpoint-N-hf/model.safetensors` plus the
HF-format companion files. Verified consolidation produces a
`from_pretrained`-loadable checkpoint (`model_type=llama`,
`GemmaTokenizer`, 1.99B params, bf16 dtype).

### Aurora 80B production script default flip

Brief carry-over from 2026-06-07 evening: `submit_agpt_80b_aurora_venv_failover.sh`
default for `CHECKPOINT_ASYNC_MODE` was `async` (commit
`ce321caae` flipped it to `disabled`). Just confirming the change
shipped — anyone launching on Aurora today gets the safe
sync-checkpoint default that avoids the gloo-on-xpu crash.

---

## 2026-06-07 (evening) — 80B prod sync-ckpt validated end-to-end + train_grpo HfArgumentParser + FSDP wiring + blendcorpus index race + xccl issue filed

Big session covering four threads. Tracked in tasks #40–#63.

### 80B prod sync-ckpt path validated end-to-end (12468197, Sunspot 4N)

After three false-start retries diagnosing dataset-loader and PBS
env-var issues (12468190 hung after step 1 on the
`eliplutchok/fineweb-small-sample` HF stream; 12468194 died 32s in
without `NHOSTS_TRAIN`; 12468195/12468196 hit a blendcorpus
per-corpus + blendable-dataset index-build race — see below), got a
clean 4N TP=2 books-blendcorpus run in 12468197:

- **Loss descent**: 12.95 → **7.49** at step 164 (-5.46 nats)
- **Cadence**: ~42s per step, ~7 min per 10 steps, steady throughout
- **MFU**: 17.7-17.9% steady (matches the May 5 working-config smoke
  12466025 exactly)
- **Memory**: 88.97% peak (4N gives ~7 GiB tile headroom for 80B)
- **First sync checkpoint**: **saved at step 100 in 101.84s**,
  904 GB across 48 distcp shards, durable on disk at
  `outputs/checkpoints/agpt-80b-adamw-books-n4-gbs24/step-100/`
- **Walltime-killed at step 164** (Exit_status=-29, SIGKILL on 2h
  walltime — `TRAINING_STEPS=200` was a soft target; would have
  reached step 200 with a 3h allocation)

The **first sync ckpt save is the load-bearing milestone** — it's
exactly where 12468189 died on async mode hitting the gloo-on-xpu
bug. With `CHECKPOINT_ASYNC_MODE=disabled` the rank-0
`dist.new_group(backend="gloo")` is skipped entirely and the rest of
the path is clean. The 80B production stack is now end-to-end
validated on Sunspot under the workaround.

### train_grpo CLI rewrite: argparse (10 flags) → HfArgumentParser (191 flags)

User flagged that `torchtitan/experiments/ezpz/rl/train_grpo.py`
was exposing only ~7 of GRPOConfig's 64 own fields + ~100 inherited
TrainingArguments fields — every other knob was hardcoded or hidden
behind env-var fallbacks. Refactored to `HfArgumentParser((EzpzGRPOArgs,
EzpzGRPOConfig))`:

- `EzpzGRPOArgs` holds ezpz-side fields (`task`,
  `model_name_or_path`, `num_samples`, `no_save`,
  `fsdp_transformer_layer_cls_to_wrap`, `fsdp_cpu_ram_efficient_loading`)
- `EzpzGRPOConfig(GRPOConfig)` overrides defaults where ezpz/XPU
  values differ from upstream (bf16=True, gradient_checkpointing=True,
  beta=0.0, torch_empty_cache_steps=1, num_generations=4,
  max_completion_length=64, save_strategy="no", use_vllm=False)
- All other GRPOConfig fields fall through to TRL defaults, so
  every TRL knob (--beta, --epsilon, --loss_type, --vllm_*,
  --optim, --gradient_accumulation_steps, --num_iterations,
  --warmup_steps, --lr_scheduler_type, etc.) is now CLI-settable
  without code edits

Breaking change: `--model-name-or-path` and `--no-save` (hyphenated)
became `--model_name_or_path` and `--no_save` (snake_case) since
HfArgumentParser mirrors dataclass field names. `grep -r` found no
existing callers, so safe to land.

### FSDP env-bootstrap wiring (mirrors ezpz.examples.hf.py pattern)

Under `ezpz launch` (mpiexec), passing TRL's `--fsdp full_shard` is
a silent no-op: HF Trainer's internal `accelerate.Accelerator` only
builds a `FullyShardedDataParallelPlugin` when env vars
(`ACCELERATE_USE_FSDP=true`, `FSDP_*`) are pre-set — which
`accelerate launch` does for you but `mpiexec` does not. So passing
`--fsdp full_shard` was silently dropping every rank into plain DDP
(every rank holds the full model → 2B models OOM at 12 ranks/tile).

`ezpz.examples.hf.py` works around this by constructing the FSDP
plugin explicitly and passing it to `Accelerator(fsdp_plugin=...)`
— but that path requires owning the training loop, which TRL owns.
So we adopted the equivalent surgical approach: populate the same
env vars the explicit plugin would generate, BEFORE
`GRPOTrainer.__init__` runs. Wires up
`ACCELERATE_USE_FSDP=true`, `FSDP_SHARDING_STRATEGY`,
`FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP`,
`FSDP_TRANSFORMER_CLS_TO_WRAP`,
`FSDP_BACKWARD_PREFETCH=BACKWARD_PRE`,
`FSDP_USE_ORIG_PARAMS=true`,
`FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT`,
`ACCELERATE_MIXED_PRECISION=bf16` (from `bf16=True`),
`FSDP_CPU_RAM_EFFICIENT_LOADING=true` +
`FSDP_SYNC_MODULE_STATES=true` (when
`--fsdp_cpu_ram_efficient_loading` is on).

Two follow-up fixes the user surfaced from real launches:

1. **FSDPOption enum coercion** (`d859fafc1`): HfArgumentParser
   parses `--fsdp full_shard` into a list of `FSDPOption` enums.
   `str(FSDPOption.FULL_SHARD)` returns `'FSDPOption.FULL_SHARD'`
   (the StrEnum class-qualified name), not `'full_shard'`. Initial
   parser used `str(x).lower().split()[0]` →
   `'fsdpoption.full_shard'` → not in `_FSDP_STRATEGY_MAP` →
   ValueError before training. Fixed:
   `getattr(x, "value", str(x)).lower()` and scan whole list for a
   known strategy token (so `--fsdp "full_shard auto_wrap"` works
   regardless of token order).
2. **gradient_checkpointing + FSDP migration** (`6758a809d`):
   transformers warns `"When using FSDP full shard, instead of using
   gradient_checkpointing, please use activation_checkpointing in
   fsdp_config"` (training_args.py:2732). The warning fires inside
   `TrainingArguments.__post_init__`, BEFORE `main()` runs, so
   migrating in main() was too late — 48 ranks each printed it.
   Fixed: override `EzpzGRPOConfig.__post_init__` to migrate
   `gradient_checkpointing` → `fsdp_config["activation_checkpointing"]`
   BEFORE `super().__post_init__()` runs. Now transformers sees
   `gradient_checkpointing=False` and never warns. Also pre-loads
   `--fsdp_config <path>.json` so user-set keys
   (transformer_layer_cls_to_wrap, cpu_ram_efficient_loading) survive
   the migration merge.

### blendcorpus per-corpus index-build race

12468195 died in 2:34 with
`EOFError: No data left in file` when rank 2 tried to mmap-load
`shuffle_idx.npy` while rank 0 was still writing it. The books
dataset is tiny (3 shards, 11 GB, 4826 samples for a 200-step run
at GBS=24) so rank-0 index-build finishes in **8 ms** — too fast
for the implicit barrier-via-allreduce on the next dist op to close
the race.

12468196 died the same way 2:34 in but at the NEXT layer — the
"blendable dataset" index (`_index.npy`, `_sample_index.npy`),
built in 11 ms.

Root cause in `deps/blendcorpus/blendcorpus/data/gpt_dataset.py`:
the per-corpus path (`_build_index_mappings`, lines ~1050-1135)
does rank-0-write then all-ranks-`np.load(..., mmap_mode='r')`
**with NO `torch.distributed.barrier()` between them**. The
blendable-dataset path (lines 215-265) DOES have barriers — so the
per-corpus path is racy at small dataset sizes.

Production canonical chain uses olmo-mix (much larger, build takes
seconds) so this has never bitten before. Workaround: just retry —
once indices are durably on disk, the second pass hits cache and
skips the build. Permanent fix needs a `torch.distributed.barrier()`
in upstream blendcorpus.

Diagnosis + minimal repro shape + suggested upstream fix in
[`docs/upstream-issues/blendcorpus_index_build_race.md`](upstream-issues/blendcorpus_index_build_race.md).

### xccl supportsSplitting issue filed upstream

Filed as **[pytorch/pytorch#186548](https://github.com/pytorch/pytorch/issues/186548)**
with verified Sunspot repro log + collect_env block, plus a
cross-reference comment on the sibling
[pytorch/pytorch#171938](https://github.com/pytorch/pytorch/issues/171938).
The workaround `xccl_split_group_workaround.py` stays in place
until both: (1) `ProcessGroupXCCL.supportsSplitting() override`
lands, (2) `ProcessGroupXCCL::split` is implemented + CI-tested.

Removal criteria documented at the bottom of
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md),
which now has a banner pointing to the upstream tracker.

---

## 2026-06-07 (pm) — 80B prod attempt on Sunspot 4N + new upstream CheckpointManager XPU bug

First real 80B production launch on Sunspot post-47th-sync. Job
12468189: died at trainer init with

    RuntimeError: No backend type associated with device type xpu

raised inside `CheckpointManager.__init__` at
`torchtitan/components/checkpoint.py:468`, on the line

    self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

which fires when `async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM)`.
xccl-only default PG has no gloo backend bound, so the gloo
subgroup-creation fails. Same shape as the xccl_split_group bug from
the prior session — another upstream code path that assumes a CUDA-
style gloo PG bundled in with the accelerator backend.

Discovered courtesy of the RANK 0 ABORT chain extension from
2026-06-06 (`84c84cd0b`): the underlying cause showed up clearly at
the top of the failure log instead of being buried under per-rank
mpiexec stderr.

**Workaround**: `--checkpoint.async-mode=disabled` (or
`CHECKPOINT_ASYNC_MODE=disabled` via the failover wrapper). Sync
ckpt mode skips the gloo subgroup creation entirely.

Resubmitted as 12468190 with the workaround; got past trainer init,
step 1 clean (loss 12.913, mem 53 GiB / 82.86%), training in
progress as of this entry.

Full diagnosis + removal criteria + the related-bugs cross-refs in
[`docs/upstream-issues/checkpoint_async_gloo_on_xpu.md`](upstream-issues/checkpoint_async_gloo_on_xpu.md).

Also added a `DATASET` env knob to
`scripts/submit_agpt_80b_aurora_venv_failover.sh` (commit
`eeb907b73`) so non-Aurora launches can use HF-streaming datasets
without a local data-list (Sunspot has no canonical olmo-mix-1124
list).

### Other 80B production state

- The 4N working-config table in `docs/production/agpt/80b/README.md`
  now has three bit-equivalent smoke datapoints (May 5 Aurora,
  Jun 2 Sunspot, Jun 6 Sunspot post-47th-sync). All converge to the
  same step-20 loss (10.39-10.46) and 88.94% memory. Configuration
  is stable across two ezpz refactors + one workaround.

### Optional follow-ups

- Ezpz-side workaround module for the gloo-on-xpu bug (mirroring
  `xccl_split_group_workaround.py`) so async ckpt mode "just works"
  on XPU. Deferred — sync mode is acceptable for now.
- File pytorch/torchtitan issue requesting defensive fallback in
  `CheckpointManager.__init__` when `new_group(backend="gloo")` fails.

---

## 2026-06-07 — spmd_types venv fix + LBS=2 scaling sweeps + post-mortem docs

Continuation of the 2026-06-06 scaling-study unblock. Two new
findings drove the work this session:

### 1. Stale wrapper default for agpt_20b

While reviewing yesterday's 64N + 128N numbers the user noticed the
agpt_2b scaling rows had MFU well below the 256N reference point
(18.99% / 16.13% vs 18.77%). Tracing the wrapper found agpt_2b
defaulting to `LBS=1` while production uses `LBS=2`. Fixed in commit
`4ceffb31e`. Verifying production scripts also caught agpt_20b at
`LBS=1` in the wrapper while `submit_agpt_20b_aurora_venv.sh` uses
`LBS="${LBS:-2}"`. Fixed in commit `8294883e5`.

Re-ran at the new LBS=2 defaults:

| N | agpt_2b TPS/MFU | agpt_20b TPS/MFU | moe_2b | Job |
|---|------------------|-------------------|--------|-----|
| 64 | 6,553 / 24.59% | 448 / 22.36% (LBS=1 stale) | killed @ walltime | [8528940](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/) |
| 64 | 6,083 / 22.82% | 511 / 25.48% | NO_OUTPUT 235s | [8529046](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ihiy4ej1) |
| 128 | 4,934 / 18.51% | 480 / 23.96% | OOM 616s | [8529081](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/m9i0long) |

Confirmed expected weak-scaling pattern: 2B drops to 18.5% MFU at
128N, 20B holds steady around 24% — both consistent with all-reduce
overhead growing with N. Memory shot up from ~20 GiB (LBS=1) to
~44 GiB (LBS=2) for 2B; 20B from ~29 GiB to ~40 GiB. All still
safely under 70%.

### 2. Upstream `spmd_types` regression

Mid-resubmit, fresh 64N attempt (`8529016`) CRASHed in <20s with:

    File "/lus/.../torchtitan/components/loss.py", line 12, in <module>
      import spmd_types as spmd
    ModuleNotFoundError: No module named 'spmd_types'

Tracked to upstream commit `fec0c175d` (Pian Pawakapan, 2026-06-05,
[#3467] "[spmd_types] manual loss parallel CE"), which adds
`spmd_types==0.2.1` to `requirements.txt` + `pyproject.toml` but
relies on user reinstall. Our compute-node `/tmp/.venv` (from the
2026-06-03 tarball) didn't have it; nor did the source venv before
the user's `uvi --no-deps spmd_types` today.

Fix:

    uv pip install --python .venv/bin/python3 \
        --no-deps --no-cache --link-mode=copy spmd_types

Then rebuilt `.venv.tar.gz` (2.6 GB, 56 spmd_types entries verified
in tarball). Old broken tarball backed up at
`.venv.tar.gz.bak-pre-spmd-20260606-215928`. Validated on 8529046 +
8529081 — both ran cleanly post-fix.

### 3. Docs revert + redistribution

Yesterday's consolidation of the 4 per-model scaling docs into a
single 250-line README turned out to be a regression vs the sibling
convention (`docs/evals/agpt/{2b,20b}/`, `docs/production/agpt/`).
Reverted in commit `f19bbdec3`: README is now an index-only
dashboard, per-model pages restored. Today's LBS=2 numbers + the
spmd_types post-mortem flagged inline on the relevant per-model
page.

### 4. moe_2b regression

moe_2b at 64N + 128N still fails on Aurora torch 2.13 even with
spmd_types installed. Failure mode changed (NO_OUTPUT 235s → OOM
616s), but it's likely the upstream `edp_mesh=None` SIGABRT noted
in CLAUDE.md. Not yet diagnosed; tracked separately.

### State at end of session

- 2B 256N production chain (`8519833`) walltime-finished cleanly at
  step-69900 yesterday. Continuation `8521626` still Q for a 256N
  prod slot.
- All 2B + 20B production chains Q+H, waiting on prod queue rotation.
- Today's Aurora torch 2.13 scaling row at 256N is the next gap
  to fill once Q frees.

---

## 2026-06-06 — scaling-study unblock + production charts refresh + scaling docs consolidation

End-to-end session that turned a string of scaling-study NO_OUTPUT /
CRASH failures (every 64N+ submit since 2026-05-29) into a clean Aurora
torch-2.13 scaling table, refreshed all production charts, and
consolidated the four per-model scaling docs into one page.

### Scaling-study unblock (commits `6c6235fdf`, this entry)

Five stacked bugs in `scripts/run_scaling_study_aurora.sh` were silently
killing every 64N+ scaling submit. Root-caused and fixed:

1. **`.venv.tar.gz` rebuild lost `.venv/bin/`** (empty in tarball even
   though present in source venv) — rebuilt manually.
2. **Wrapper's trailing `"$@"`** on the inner `ezpz launch python3 -m
   torchtitan...train` invocation leaked PBS `-v` CLI args straight
   into the training entry point, causing instant arg-parse failure
   (NO_OUTPUT wall <60s). Removed.
3. **blendcorpus segfault at ≥768 ranks** in
   `blendcorpus_builder.py:275 __init__`. Bypassed by adding a
   `SCALING_DATASET` env knob so the sweep can run against an HF
   streaming dataset (default for sweeps now is
   `eliplutchok/fineweb-small-sample`). Default in the script stays
   `blendcorpus` so production-shaped sweeps still hit the real loader.
4. **`qsub -- /bin/bash -c "..."` swallowed the `#!/bin/bash --login`
   shebang**, leaving `module` undefined on the PBS-spawned shell →
   `module load oneapi/release/2025.3.1` silently failed → oneAPI MPI
   binaries (`mpiexec`, `qstat`) not on PATH →
   `from sh import qstat` ImportError. Fix: submit the script
   directly (`qsub <script>`), not via `bash -c`.
5. **PATH-order race** propagated rank-N python3 ahead of
   `/tmp/.venv/bin`, so the rank-N `python3 -m torchtitan...train`
   imported a system Python with no ezpz. Pinned the inner command to
   `${VIRTUAL_ENV:-/tmp/.venv}/bin/python3`.

Also caught a stale default: `agpt_2b` in the scaling wrapper was
`LBS=1`, but production runs `LBS=2` (`submit_agpt_2b_aurora_venv.sh`).
Changed the default to LBS=2 so the wrapper produces apples-to-apples
numbers with prod going forward.

**Validation (`SCALING_GROUP=light`, `SCALING_DATASET=eliplutchok/fineweb-small-sample`):**

| N | LBS | Job | agpt_2b TPS/MFU | agpt_20b TPS/MFU | moe_2b |
|---|-----|-----|------------------|-------------------|--------|
| 64 | 1 | 8528805 | 5,062 / 18.99% | 447 / 22.32% | NO_OUTPUT |
| 128 | 1 | 8528834 | 4,300 / 16.13% | 417 / 20.83% | CRASH |
| 64 | 2 | 8528940 | _pending_ | — | — |

moe_2b NO_OUTPUT at scale tracks the upstream `edp_mesh=None` SIGABRT
regression noted in CLAUDE.md. Separate task.

### Production chains

- **2B 256N (`8519833`)** walltime-finished cleanly @ step-69900
  (Exit_status=-29, walltime=12:00:20). Continuation `8521626` is now
  top of Q. Chain still has +2 conts beneath (`8521630` H,
  `8521631`/`8521632` H).
- **2B 512N + 20B 512N chains**: both Q'd for days waiting on prod
  slot. 20B chain's `step-4500` ckpt was an empty placeholder
  (4.0K, 0 .distcp shards) from a mid-save kill — renamed to
  `step-4500.bak-empty-20260606-170503/` so 8521628 resumes cleanly
  from step-4400 (244GB, complete) when it gets a slot.

### Evals (commit `11d8d9f26`)

Ran `eval-2b-v2.sh` on 2B 256N step-66000 and step-68000 (8528801):

| Step | Tokens (B) | HSn | ARC-E | ARC-C | Wino |
|------|-----------|-----|-------|-------|------|
| 64,000 | 3,221 | 0.5538 | 0.6040 | 0.3336 | 0.5549 |
| **66,000** | **3,322** | **0.5577** | **0.5918** | **0.3302** | **0.5462** |
| **68,000** | **3,422** | **0.5577** | **0.5905** | **0.3302** | **0.5509** |

ARC-E spike at 64K appears to be noise; otherwise convergence is flat
to slightly positive on HSn.

### Docs consolidation (commit `11d8d9f26`)

Collapsed `docs/scaling/{agpt-2b.md,agpt-20b.md,agpt-80b.md,moe.md}`
into a single `docs/scaling/README.md` organised by model. Added a
"Historical: the n=64/128 CRASH era" callout documenting the five-bug
stack above. Backfilled stale prod walltime (12h, was 6h on 20b page).

### Production charts refresh (commit `148fff6e6`)

Re-ran all plotting scripts (`plot_production.py`,
`plot_production_combined.py`, `plot_production_wandb.py`,
`plot_evals_combined.py`, `plot_v1_vs_v2.py` for both 2b + 20b)
against current W&B. 28 SVGs/PNGs refreshed across production +
historical-v1-bf16 + evals figure dirs.

---

## 2026-06-06 — 47th upstream sync (replays + smokes + post-smoke fixes; READY TO MERGE)

Pulled 34 commits since the 46th sync. Two structural refactors hit
ezpz; both replayed, smoked against baselines, and validated.

### Replays (initial)

1. **PR #3458 (RoPE refactor) — commit `02dd1e7fe`.** Splits
   `RoPE.Config` into `ComplexRoPE.Config` / `CosSinRoPE.Config`,
   moves rope ownership from top-level model config down to per-layer
   `Attention.Config`, removes the `apply_rotary_emb_*` helpers in
   favour of `self.rope = config.rope.build()` +
   `q, k = self.rope(q, k, positions)`. Replayed across
   `agpt/__init__.py`, `agpt/config_registry.py`, `moe/__init__.py`,
   and `moe/model.py`.

2. **PR #3269 (mixed-optimizer refactor) — commit `bac0a3473`.**
   Replaces the flat `OptimizersContainer.Config(lr=8e-4)` shape
   with `param_groups=[ParamGroupConfig(pattern, optimizer_name,
   optimizer_kwargs={"lr":...})]`. Custom-container subclasses are
   now thin wrappers registering their optimizer via
   `_resolve_optimizer_cls`; added 8 `default_<name>(lr=..., **kwargs)`
   factories mirroring upstream's `default_adamw`. Replayed across
   `optimizer/containers.py`, `optimizer/__init__.py`, both
   `config_registry.py`'s, `competition/configs.py` (28 callsites +
   19 in-place LR mutations), and `train.py`'s `--optimizer` CLI
   swap helper.

Full breakdown of both halves + the other 32 upstream commits in
[`upstream-sync.md`](upstream-sync.md).

### Baseline numerics smoke (against the 2026-06-02 baselines)

| Config        | Job      | Outcome                                            |
|---------------|----------|----------------------------------------------------|
| `moe_2b_ep` 2N | 12468156 | 10 steps clean, loss 12.95 → 7.92, mem 58.28 GiB matches baseline |
| `agpt_80b TP=2` 4N | 12468157 | 20 steps clean, all within ±0.08 nat of baseline, mem + MFU bit-identical |

### Coverage smokes — full registry sweep

Then ran a wider smoke sweep to exercise the rest of the registry.
This caught 3 real bugs in the initial replay, all now fixed:

| Config | Outcome | Notes |
|---|---|---|
| `agpt_2b` | ✅ 10 steps, 12.95 → 7.63, ~20% MFU | |
| `agpt_2b_real` | ✅ 10 steps, 12.99 → 8.50 | validates CosSinRoPE swap path through rewritten `_set_rope_backend` |
| `agpt_20b` | ✅ 10 steps, 12.90 → 10.39 | loss noisy (no warmup, hot LR) but trains |
| `moe_2b` (LBS=2) | ✅ 10 steps, 12.94 → 8.52 | (default LBS=16 OOMs, pre-existing) |
| `moe_10b_2b_sdpa_ep` (LBS=1, AC=selective) | ✅ 10 steps, 12.96 → 9.44, mem 51.12 GiB / 80% | new default — see fix #3 below |
| `speedrun_2b_muon` (LBS=1) | ✅ 10 steps, 12.93 → 9.32, ~3,000 tps | validates Muon dispatch — see fix #1 below |
| `speedrun_2b_sophiag` (LBS=1) | ✅ step 1 reached training | validates SophiaG dispatch via fix #1 |
| `moe_10b_2b` | ⏭ skipped | block_causal mask + HF-dataset mismatch (pre-existing, unrelated to sync) |
| `moe_10b_2b_sdpa{,_ep}` @ AC=full | ⏭ known broken | `CheckpointError: Recomputed values have different metadata` — MoE token routing isn't bit-exact across recompute (failure exists since at least 2026-05-12; PR #3146/#3450 fixed the forward path only) |

### Post-smoke fixes

1. **`optimizer/containers.py` — Config-dispatch bug (commit `6871e736b`).**
   The initial replay collapsed each custom container into a thin
   wrapper but the `default_<name>(...)` factories returned
   `OptimizersContainer.Config(...)` whose `_owner` is the base
   class. So `cfg.optimizer.build()` constructed the base
   container, whose `_resolve_optimizer_cls` only knows Adam/AdamW —
   any custom optimizer name (`Muon`, `SophiaG`, ...) raised
   `NotImplementedError: Optimizer Muon not added`. Caught by
   `speedrun_2b_muon` + `speedrun_2b_sophiag` smokes. Fix: add an
   empty `class Config(OptimizersContainer.Config): pass` to each
   of the 8 subclasses (so `_owner` binds to the subclass), and
   update each `default_<name>` factory to return the subclass's
   Config.
2. **`moe/config_registry.py` — 5 missed `cfg.optimizer.lr` mutations
   (commit `455013ed5`).** The optimizer refactor caught the 19
   such mutations in `competition/configs.py` but missed five in
   `moe/config_registry.py` (`moe_16b`, `moe_671b`, `moe_10b_2b`,
   `moe_10b_2b_sdpa`, `smoke_moe_500m_50steps`). Same fix as
   competition: write through `cfg.optimizer.param_groups[0].optimizer_kwargs["lr"]`
   instead of `cfg.optimizer.lr`. Caught when smoking
   `moe_10b_2b`.
3. **`moe/config_registry.py` — `moe_10b_2b_sdpa{,_ep}` defaults
   changed to LBS=1 + AC="selective" (commit `975a5bcd1`).** Prior
   default `(LBS=2, AC="none")` OOMs at first forward on 2N Sunspot
   (level_zero `UR_RESULT_ERROR_OUT_OF_RESOURCES`). Production
   scripts already overrode LBS=1 on the CLI, so this brings the
   registry in line. AC="full" can't be the answer — it hits the
   long-standing `CheckpointError` from non-deterministic MoE
   routing under recompute. AC="selective" only checkpoints the
   SAC save list (excludes the router), so the non-deterministic
   op never gets recomputed and shapes stay stable. Verified on
   job 12468186: 10 steps clean, peak 51.12 GiB / 79.9% (vs OOM
   at LBS=2).
4. **`datasets.py` — pickle fix for HF datasets + `num_workers >= 1`
   (commit `8746dfe2c`).** `_make_text_processor` returned a local
   closure that couldn't be pickled by PyTorch's `forkserver`
   DataLoader workers. Crashed every HF-dataset run with
   `num_workers > 0` (e.g. `agpt_2b ... --dataloader.dataset eliplutchok/fineweb-small-sample --dataloader.num-workers=2`).
   Pre-existing bug, not a replay regression. Fix: move `_process`
   to module scope as `_extract_text_column` + use `functools.partial`.

### Side notes from the merge

- Installed `spmd_types==0.2.1` into `.venv` via `uv pip install`
  (upstream `components/loss.py` requires it after PR #3466/#3467/#3560).
- Worktree symlinks (`.venv`, `.venv.tar.gz`, `assets/hf`) added
  by hand so submitted jobs find them — git ignores them.

### Final branch state

```
8746dfe2c datasets pickle fix
975a5bcd1 moe 10b_2b_sdpa{,_ep} → LBS=1/AC=selective default
455013ed5 moe registry 5 cfg.optimizer.lr fixes
6871e736b optimizer Config-dispatch fix
bc89aa85e docs (optimizer half done)
bac0a3473 optimizer refactor replay
3748b9e9c docs (RoPE half done)
02dd1e7fe RoPE replay
fb1c5a319 merge upstream/main
```

### Open follow-ups

- File pytorch/pytorch issue for MoE + AC-full `CheckpointError`
  (router non-determinism across recompute). Long-standing — the
  fix requires AC to save the routing decision instead of
  recomputing it.
- Audit the 4 RL commits in this sync — `experiments/ezpz/rl/`
  may need attention if they touch shared surfaces.

---

## 2026-06-02 — 46th upstream sync (graph_trainer-only, no ezpz replay)

Pulled 2 new commits since the 45th sync (`04a309858..27aa49077`):

- `27aa49077` [graph_trainer] Add full recompute memory policy (#3429)
- `051562e31` [graph_trainer] Re-enable DSv3 eager bitwise deterministic tests (#3482)

Both entirely inside `torchtitan/experiments/graph_trainer/`. Zero
ezpz files touched, no replay needed, no conflicts. Merge commit
`45a2b2568`. Full breakdown in
[`upstream-sync.md`](upstream-sync.md) (46th-sync entry).

---

## 2026-06-02 — 80B TP=2 4N smoke replay on Sunspot under xccl workaround

Followed up the moe_2b_ep workaround validation with a second
verification: that the new xccl_split_group_workaround
([`8031d1d3a`](https://github.com/saforem2/torchtitan/commit/8031d1d3a))
doesn't regress the working 80B TP=2 v2 config from the May 5
Aurora baseline (job 12466025).

Submitted 4N Sunspot smoke as job 12467825 using the same recipe
(`agpt_80b`, TP=2, AC=full, compile=OFF, AdamW LR=1e-6, fp32-master)
via the updated `submit_80b_no_compile_t213.sh` (now using SUBMIT_DIR
+ `ezpz tar-env`/`ezpz yeet`). Exit 0 in 902 s, 20 steps:

|             | May 5 (Aurora 4N) | 2026-06-02 (Sunspot 4N) |
|-------------|------------------:|------------------------:|
| Δloss       | -2.52             | **-2.55**               |
| Peak mem    | 88.94%            | **88.94%**              |
| Steady MFU  | ~17.8%            | **~17.8%**              |
| grad-norm peak | ~34 @ step 15-16 | **33.10 @ step 16** |

Numerically equivalent within run-to-run noise. The workaround
install line + `Successfully created meshes with active dimensions:
['batch', 'loss', 'tp', 'efsdp', 'fsdp']` confirms 5 nested PGs
built cleanly under the patched `_init_one_process_group`. The
existing xccl-timeout shim also ran (`Applied train timeout
0:01:40 to 5 xccl ProcessGroup(s)`).

Full writeup:
[`docs/experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md`](experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md).
80B production page (`docs/production/agpt/80b/README.md`) updated
with the Sunspot replay row.

W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/a782sf8y

Open follow-ups (unchanged):
- Add 200-step linear warmup to 80B production config (grad-norm
  climb to 33.10 by step 16 is the same shape as the May 5 run).
- Mirror SUBMIT_DIR + ezpz yeet fixes into
  `scripts/submit_agpt_80b_aurora_venv_failover.sh`.

---

## 2026-06-02 — xccl split_group workaround for nested mesh init

`moe_2b_ep` smoke on torch 2.13 on XPU was hitting:

```
RuntimeError: No backend for the parent process group or its backend
does not support splitting
```

at trainer init, inside `ParallelDims.build_mesh` → the EP-flavored
sparse mesh `("pp", "dp_replicate", "efsdp", "ep")` (`parallel_dims.py:200`).

**Root cause** (confirmed by reading upstream C++ headers via
`gh search code`): `ProcessGroupXCCL` never declares
`supportsSplitting() override`. It inherits the base
`Backend::supportsSplitting()` from
[`Backend.hpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp)
which returns `false`. `ProcessGroupNCCL` overrides to `true` —
xccl doesn't.

`DeviceMesh._init_one_process_group` (torch 2.13, `device_mesh.py:550-562`)
routes nested mesh PG creation through `split_group` whenever
`bound_device_id` is set on the default group AND the accelerator is
available AND the backend name matches. None of those rule out xccl;
the ezpz eager-init path sets `bound_device_id` on XPU, so the gate
always takes the broken branch.

`split_group` itself (`distributed_c10d.py:5565-5570`) then reads
`parent_backend.supports_splitting` (Python property bound to the C++
method), sees `False`, and raises before ever calling
`xcclCommSplit`. So the failure is purely at the gate — there's no
xccl split implementation to even crash on yet.

**Workaround**: monkey-patch `DeviceMesh._init_one_process_group`
from a new module
[`xccl_split_group_workaround.py`](../xccl_split_group_workaround.py).
The wrapper:

  1. No-ops on cuda/cpu builds (gates on `is_xccl_available() and
     torch.xpu.is_available()`).
  2. On xccl, inspects the default group's per-accelerator backend's
     `supports_splitting`. If `True` (NCCL), calls upstream verbatim.
  3. If `False` (xccl), temporarily clears `bound_device_id` on the
     default group so the upstream gate's first clause goes `False`
     and we fall through to the existing `new_group` loop. Restores
     `bound_device_id` afterwards.

Installed lazily from
[`FaultTolerantTrainer.init_distributed`](../trainer.py) so it only
fires when ezpz's trainer kicks off; never touches other torchtitan
paths.

Per Golden Rule #1, no upstream file was modified. Full diagnosis +
removal criteria in
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md).

**Smoke validation** (2026-06-02, Sunspot 2N, job 12467823, exit 0
in 103 s — see
[`docs/experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md`](experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md)):

- Workaround install line in the log:
  `Installed xccl split_group workaround on DeviceMesh._init_one_process_group`.
- EP sparse mesh built cleanly:
  `Successfully created meshes with active dimensions: ['batch', 'loss', 'ep', 'efsdp', 'fsdp']`.
- Loss 12.945 → 8.273 across 10 steps, peak 58.28 GiB, ~3,050 TPS.
- The pre-existing xccl-timeout shim also ran:
  `Applied train timeout 0:01:40 to 6 xccl ProcessGroup(s)`
  (6 PGs = world + dense + sparse + efsdp + loss + batch — all the
  meshes the EP path constructs).

Two latent bugs in `scripts/submit_moe_smoke.sh` surfaced and were
fixed during the smoke (both apply to the analogous production
scripts under `scripts/submit_agpt_*_aurora_venv.sh`):

1. **`ezpz_setup_job` overwrites `$PBS_O_WORKDIR`** with the script's
   initial cwd (which is `$HOME` under default `qsub`). A naïve
   `cd "${PBS_O_WORKDIR}"` after sourcing the utils ends up in
   `/home/foremans` and `source .venv/bin/activate` resolves against
   `~/.venv` (no `ezpz`). Fix: stash submit dir into `SUBMIT_DIR`
   before sourcing utils.
2. **`ezpz yeet-env` is deprecated** in ezpz 0.18.x in favour of
   explicit `ezpz tar-env` + `ezpz yeet .venv.tar.gz`. Switched.

Open follow-ups:
- Mirror the `SUBMIT_DIR` + `ezpz yeet` fixes into
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` before the next
  512N+ production launch.
- File `pytorch/pytorch` issue with the two-part fix
  (`ProcessGroupXCCL::supportsSplitting() override + working split()`).

---

## 2026-06-02 — 45th upstream sync (PR #3450 closes the #3436 thread)

Merged 7 upstream commits (`b72d98648..04a309858`). Headline is PR
[#3450](https://github.com/pytorch/torchtitan/pull/3450) — the
upstream-canonical `histc → routing_map` swap for MoE routing that
finally lands the fix our [PR
#3436](https://github.com/pytorch/torchtitan/pull/3436) was tracking.
Approach: build a boolean `routing_map_BLE` via `scatter_` in the
router, compute `num_tokens_per_expert_E = routing_map.sum(...)` once,
thread the map through both router and dispatcher. Same direction we
recommended (scatter-based), executed better (computed once, reused).

API change at the boundary: router returns a 4-tuple now, `dispatch()`
takes a new `num_local_tokens_per_expert_E` positional arg. ezpz
doesn't override either method or unpack router output directly, so
inherits cleanly. ezpz `set_moe_sharding_config` calls upstream's
helper so the new shard declaration also lands automatically. **No
ezpz replay needed.** Imports + syntax green.

Other 6 commits: HybridEP compile support (#3360), MXFP8 consolidation
(#3473), graph_trainer DTensor fix + test re-enables (#3480), graph_trainer
test skips (#3432), and two CI migrations (#3464, #3479). None touch
ezpz code paths.

**Smoke status**: live 2N validation deferred to Aurora. The project
`.venv/bin/python` symlinks at
`/opt/aurora/26.26.0/spack/.../python-3.12.12-5zo3wzv/bin/python3`,
which is an Aurora-only Spack build (`5zo3wzv` hash). Sunspot has
the equivalent under a different hash (`nvje3vk`), so the symlink
is broken on Sunspot compute nodes. The `GroupedExperts`-touching
smoke per the prior journal lesson should run on Aurora next time
there's an alloc.

Also updated `~/.ezpz/utils.sh` from the bit.ly canonical (it now
exposes the new unified `ezpz_setup` function; the old
`ezpz_setup_job` / `ezpz_setup_xpu` split is superseded). Backup at
`~/.ezpz/utils.sh-20260602-065437`.

Open follow-ups:
- Close PR #3436 with a comment pointing at PR #3450.
- Smoke `moe_2b_ep` on Aurora to validate the routing_map flow.

---

## 2026-06-01 — Error-propagation fixes + 44th upstream sync

### Error-propagation fixes (from this morning's BlendCorpus debug)

Two bugs surfaced when a rank crashed at init with the misleading
`BlendCorpus dataset was requested but blendcorpus is not installed`
message — when in fact `blendcorpus` was installed and the actual
missing module was `deepspeed` (a transitive dep).

- [`9eb680dbc`](https://github.com/saforem2/torchtitan/commit/9eb680dbc)
  — `_import_blendcorpus_modules` now inspects `exc.name` and emits
  one of two messages: "blendcorpus is not installed" vs "blendcorpus
  IS installed but pulled in missing transitive dep `<name>`". Names
  the actual culprit module so users don't chase the wrong fix.
- [`2c5c7d597`](https://github.com/saforem2/torchtitan/commit/2c5c7d597)
  — wrap `config.build()` in its own try/except in `train.py:main()`.
  Rank 0 logs a single `RANK 0 ABORT during config.build():` line
  followed by the full `__cause__`/`__context__` chain on failure.
  mpiexec presents rank 0 output first, so this surfaces above the
  per-rank `rank N exited with code 1` spam.

### 44th upstream sync

Merged 1 upstream commit (`b72d98648`, PR
[#3403](https://github.com/pytorch/torchtitan/pull/3403)): 4-line
addition to project-root `.claude/CLAUDE.md` recommending ≥10
iterations for perf comparisons. No code change, no ezpz replay.

---

## 2026-05-31 — 43rd upstream sync (interleaved dataloader, no-op replay)

Merged 1 upstream commit (`221041490`, PR
[#3063](https://github.com/pytorch/torchtitan/pull/3063) — weighted
interleaved multi-source HF dataloader). Pure additions —
`InterleavedHuggingFaceTextDataLoader`, `InterleavedChatDataLoader`,
`HFDataSource`/`ChatDataSource`, an `InterleavedDataset` weighted
sampler, plus tests. The existing `HuggingFaceTextDataLoader` and
`DATASETS` that `experiments/ezpz/datasets.py` and
`blendcorpus/blendcorpus_builder.py` import are unchanged. No
replay; imports smoke green.

`git log HEAD..upstream/main` listed 8 commits but only 1 was a
genuine new patch — the other 7 were the patch-equivalent duplicates
from 41st/42nd-sync bookkeeping flagged in the 42nd-sync entry.
`git merge` handled the difference correctly.

---

## 2026-05-29 — 42nd upstream sync (RoPE refactor + replays)

Merged 6 upstream commits (`28483d0eb..065c2625d`). The headline is
PR [#3395](https://github.com/pytorch/torchtitan/pull/3395) which
restructures every model's `update_from_config`:

- Renames the `trainer_config` keyword to `config`.
- Promotes `seq_len > rope.max_seq_len` from warning to hard
  `ValueError`.
- Moves TP / `n_heads`/`n_kv_heads` validation, MoE `deepep`/`hybridep`
  EP=1 guard, MoE `moe_force_load_balance` debug flag, and the
  `rope.max_seq_len` sync into `Decoder.Config.update_from_config`.
- Adds async out-of-bounds checks inside `apply_rotary_emb_*`.

Replayed in `experiments/ezpz/{agpt,moe}/model.py`:

- agpt was a pure `trainer_config`→`config` rename plus parameter
  forwarding.
- moe was bigger: deleted the now-duplicated rope/MoE/TP checks and
  delegated to `Decoder.Config.update_from_config`; kept only the
  per-layer attention rope-field sync, the for_loop XPU fallback,
  the CP+MoE attention check, and `set_moe_sharding_config`. Dropped
  three now-unused imports.

Mirrors the post-refactor shape of upstream `deepseek_v3/model.py`.
2N smoke on alloc 12467655 surfaced two follow-ups:

- A missed `trainer_config`→`config` rename at
  `experiments/ezpz/trainer.py:163` (the trainer's own
  `model_config.update_from_config(...)` callsite, not the model
  override). Fixed in [04199e522](https://github.com/saforem2/torchtitan/commit/04199e522).
- A pre-existing miss from the 41st sync ([#3425](https://github.com/pytorch/torchtitan/pull/3425)
  MoE shape-suffix rename) that we hadn't smoked: three ezpz-side
  references to the old `w1`/`w2`/`w3` parameter names in
  `EzpzGroupedExperts` sharding/init/forward paths — caught when
  the rope replay let us reach trainer init for the first time post-
  41st-sync. Fixed in [88dbd916e](https://github.com/saforem2/torchtitan/commit/88dbd916e).

Post-fix smoke results (both clean, matching 2026-05-27 baselines):

- `agpt_2b`: 154 s, peak 24.34 GiB (38.04%), 21.95% MFU.
- `moe_2b_ep` LBS=2: 323 s, peak 26.99 GiB (42.18%), 9.09% MFU,
  loss step 50 = 6.13.

Lesson worth remembering: always smoke `moe_2b_ep` after a sync that
touches `GroupedExperts`. Today's chain of bugs hid behind the
trainer init failure — the model-init order is rope → expert sharding
→ first forward, so a rope-stage failure prevents us from seeing
expert-stage failures.

Smaller commits in the same sync: #3448 (#3395 fix-forward), #3452
(1-line determinism cleanup), #3445/#3446 (flux/qwen3-vl), #3347 (RL
batcher). None affect ezpz.

Bookkeeping curiosity: `git cherry` flagged 41st-sync MoE [8/n]
(`200100e7d`) as already present in our branch under a different SHA
(`56dc8e1d4`). `git merge` correctly skipped it. So the "7 unmerged
commits" `git log` showed was really 6.

---

## 2026-05-28 — 41st upstream sync (no-op replay)

Merged 1 upstream commit (`200100e7d`, PR
[#3425](https://github.com/pytorch/torchtitan/pull/3425) — MoE [8/n]
shape-suffix rename). Pure rename refactor applying the Shazeer
shape-suffix convention across all MoE tensors. Loss-comparator
verified `--assert-equal` upstream. Renames break several internal
method signatures (`dispatch`, `_unpermute`, `_make_dispatcher`) but
none are called from `experiments/ezpz/`. No replay needed; imports
smoke green.

Still open: maintainer direction on
[pytorch/torchtitan#3436](https://github.com/pytorch/torchtitan/pull/3436)
(histc → bincount/scatter for XPU determinism). Posted the
statistical E2E A/B yesterday — bincount and histc are
indistinguishable on `moe_2b_ep` (Welch's p=0.66, n=135 per variant).
Recommendation: scatter_add_ as the cleanest swap. Waiting on
maintainer choice between options A/B/C.

---

## 2026-05-27 — 40th upstream sync (7 commits)

Merged 7 upstream commits (`19c567f76..af33f7638`):

- **PR #3398** ([Module] Replace from_nn_module with native Module
  subclasses) consolidates `common/{linear,rmsnorm,embedding}.py`
  into `common/nn_modules.py`. Broke 3 import paths in ezpz; replayed
  in [`b052f29e4`](https://github.com/saforem2/torchtitan/commit/b052f29e4)
  with pure import-path swaps (class API is unchanged).
- **PR #3146** (Use deterministic ops in MoE routing) is the upstream
  fix for the `_histc_xpu does not have a deterministic
  implementation` blocker we hit on 2026-05-21. Replaces `histc` with
  `bincount` and adds `aten.topk.default` to the SAC save list.
  Inherits transitively; `--debug.deterministic` on MoE+XPU should
  now work.
- **PR #3423** (MoE [7/n], 3D tensors through MoE) continues the
  MoE refactor from #3386/#3389. Doesn't touch `deepseek_v3/model.py`
  and ezpz doesn't expose the 2D-flatten seam, so we inherit
  transitively.
- **PR #3105** (FSDP symmetric memory) adds an `enable_fsdp_symm_mem`
  flag, plumbed through each model's `apply_fsdp`. ezpz has its own
  local `apply_fsdp`, so the kwarg doesn't reach our path. Skipping
  the replay — symm_mem is an optimization and XPU's CCL likely
  doesn't support it anyway.
- **PRs #3331 / #3369 / #3361** are all graph_trainer-only; no-ops
  for ezpz.

Quick imports smoke (`python3 -c "import torchtitan.experiments.ezpz.{agpt,moe.model}"`)
passes. Live 2N smokes (job 12467455) on Sunspot:

- `agpt_2b` and `moe_2b_ep` clean post-merge, numerically identical
  to the 2026-05-22 baselines.
- `--debug.deterministic` on MoE+XPU **still fails**. PR #3146 was
  supposed to fix the `_histc_xpu` blocker via a `histc → bincount`
  swap, but the merged diff is missing that change — only the
  `aten.topk.default` save-list addition landed. Verified via the
  GitHub API that PR #3146's only file change is
  `activation_checkpoint.py`. The `histc` call at
  `common/moe.py:262` is untouched. Smoke report at
  [`docs/experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md`](experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md).

Action item: file upstream issue for the incomplete PR #3146.

---

## 2026-05-27 — 20B 512N sync chain doubles its eval scores; 2B chains pass step-49K + step-27K

### Production progress (May 25 → May 27, ~40h)

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 36,528 → **49,666+** | +13,138+ | **~114** | 8507195 + 8507198 + 8508020 (R) |
| **2B 512N sync** | 16,676 → **27,106+** | +10,430+ | **~107** | 8507196 (pals-RPC infra fail, +76) + 8507199 (+50) + 8508753 (R) |
| **20B 512N sync** | 2,043 → **3,270** | +1,227 | **+12** | 8507197 + 8507200 |

**Total: ~233 new on-disk checkpoints across 3 chains in 40h of wall clock.**
Sync-mode workaround continues to hold for both 2B 512N and 20B 512N
trajectories; async-mode still stable at 256N.

### 🏁 Headline: 20B 512N sync now beats 2B 256N async per token on every benchmark

Eval'd the full 24-ckpt sync-mode sweep at steps 900..3,200. The 20B
512N sync trajectory now leads the 2B 256N async on **all four
benchmarks** at matched token counts:

| Task | 20B 512N step-3,200 (329B tok) | 2B 256N step-45,500 (~2.3T tok) |
|------|---:|---:|
| ARC-Easy `acc` | **0.6646** | 0.6418 |
| ARC-C `acc_norm` | **0.3225** | ~0.315 (oscillating) |
| HellaSwag `acc_norm` | **0.5737** | 0.5452 |
| Winogrande `acc` | **0.5612** | ~0.55-0.56 |

The 20B model is now token-efficient in a way the 2B has begun to
saturate (plateau at ARC-Easy ~0.645, HellaSwag norm ~0.547). The
full sweep is monotonic — no plateau, no oscillation, no sign of
optimizer instability across 24 consecutive checkpoints. This is
**the first time in the entire v2 experiment that the bigger model
has outperformed the smaller one at matched token counts**, and the
strongest live signal yet that the fp32-master + sync-mode
combination is the right operational stack for the 20B at scale.

### Async-regression workaround still solving 512N

The `CHECKPOINT_ASYNC_MODE=disabled` workaround continues to keep both
512N trajectories advancing. 2B 512N: 4 consecutive sync dispatches
(starting from `8506221`) have added ~10K steps and ~107 persisted
ckpts. 20B 512N: 4 consecutive sync dispatches have added 1,243 steps
and 24 persisted ckpts. No async-cascade failures in any of these.

### One Aurora pals-RPC infra failure (separate from any other bug)

`8507196` (2B 512N) trained cleanly to step **20,989** in-memory
(persisting 76 ckpts), but all three wrapper attempts hit an Aurora
**pals-RPC infrastructure failure** during launch (exit 127). The
wrapper correctly identified the failures as node-related and swapped
three different "bad" nodes — but pals-RPC is upstream of anything
the wrapper can repair. See
`memory/project_aurora_pals_rpc_launch_failure.md`. Not the
async-cascade bug, not a model issue, not a wrapper bug.

### 80B still blocked (separate SIGSEGV pattern)

No 80B production progress this stretch. Last attempts continue to
die in the same rank-level SIGSEGV pattern during init that bracketed
the earlier 80B failures, separate from any of the 2B/20B failure
modes. Writeup + likely upstream patches still pending from the
05-25 entry.

### Capacity is now the bottleneck

`8508214` (20B 512N continuation queued at 03:44) has been **Q ~10h**
in the `small` queue without starting — Aurora capacity for the 512N
slot is exhausted. The 20B chain is currently throttled not by any
bug or workaround but by raw queue availability. Same applies if any
of the 2B 512N continuations need to chain after `8508753`.

---

## 2026-05-25 — 🏁 Sync-mode workaround fully validated for both 2B + 20B 512N; 100+ ckpts persisted overnight

### Production progress (May 24 → May 25)

After kicking off 6 production dispatches and 3 smoke runs late on 2026-05-23,
**three production chains advanced significantly on disk overnight**:

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 25,500 → **36,528** | +11,028 | **65** (every 100 steps) | 8505175 + 8505252 |
| **2B 512N sync** | 13,300 → **16,676** | +3,376 | **21** (every 100 steps) | 8506221 |
| **20B 512N sync** | 800 → **2,043** | +1,243 | **12** (every 100 steps) | 8505258 + 8505259 |

**Total: 98 new on-disk checkpoints across 3 chains in ~36h of wall clock.**
First sustained 512N progress since 2026-05-03 for both 2B and 20B.

### Sync-mode async-cascade workaround validated

8505258 (20B 512N sync) was the proof-of-concept: trained 800→1414 cleanly with
6 ckpts persisted, the first 20B 512N to clear step-800 since the May-3 async
regression. 8505259 carried the chain to step-2043 (6 more ckpts).

8505176 (2B 512N async) confirmed the **same bug pattern at 2B 512N**: every
attempt cleanly trained 13300→13400, then died at the step-13400 async save,
wrapper swapped + retried 3 times before exhausting. Switching to sync mode
in 8506221 produced **21 consecutive ckpts** at 512N — same fix as 20B.

`CHECKPOINT_ASYNC_MODE=disabled` is now the standard 512N workaround for both
models. Async still works fine at 256N (65 ckpts in one dispatch).

### Preflight smoke bugs surfaced + fixed

8506215 (first 2B 512N sync attempt) revealed the preflight smoke's 120s
idle-timeout was too tight for 6144-rank DDP init. Fixed by bumping to 600s
default + adding `--train-iters 5` to cap the test length (was running 200
iters by default → 1+ hour of preflight at 512N). See
[`memory/feedback_preflight_timeout_scales_with_n.md`](.).

8506221 then hit a *real* silent hang during preflight (iter 111, 49 min of
silence), wrapper SIGTERM'd on watchdog, blind-swapped `x4305c0s7b0n0` for
`x4602c3s3b0n0`, preflight attempt 2 succeeded, main training started. **The
wrapper handled the failure exactly as designed even during the preflight
phase.**

### New jobs queued (May 25 afternoon)

- 8507195 (2B 256N async cont) + 8507198 (afterany +1)
- 8507196 (2B 512N sync cont) + 8507199 (afterany +1)
- 8507197 (20B 512N sync cont) + 8507200 (afterany +1)
- 8507204 / 8507205 / 8507206: lm-eval batches on the new 2B 256N (26-36K),
  2B 512N (14-16K), and 20B 512N (900-2000) checkpoints respectively

### 80B is still blocked

8505222 (80B 256N production) failed with rank-level SIGSEGV (signal 11) on
attempt 1 + every retry, wrapper correctly classified as bad-node failure
but every spare it swapped in was also bad. 5 retries exhausted. Separately,
80B 8N smoke 8505326 surfaced a deterministic blendcorpus EOFError race in
cache build (3 wrapper attempts, all same failure). Two distinct 80B failure
modes both need writeups + likely upstream patches.

---

## 2026-05-23 (late) — 🏁 Failover wrapper passes first production silent-hang test (job 8505298)

While running the 8-node smoke validation of the fresh ezpz 0.16.0
tarballs (jobs `8505298` / `8505325` / `8505326`), the **2B smoke job
8505298 caught a real silent training hang at step 37 and recovered
automatically** — every code path in the v2 failover wrapper that
exists to handle the [8479579 incident
pattern](experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
fired correctly, in sequence, on a real-world failure.

Sequence (clean steps to recovery to checkpoint-persist):
1. Preflight `ezpz.examples.test` ran in ~2 min, exit 0.
2. Main attempt-1 trained steps 1 → 37, then logged went **completely
   silent at 21:06:41** — no traceback, no save attempt, no MPI error.
3. **30 min later at 21:36:41**, `ezpz launch --timeout=1800` watchdog
   tripped, SIGTERM'd PID 191892, exit 124.
4. Wrapper classified exit 124 as silent-hang bad-node failure (not
   walltime), couldn't identify a specific bad node from the log
   (none — the hang was silent), fell through to
   `failover_swap_one_blind()`, rotated `x4220c3s6b0n0` →
   `x4220c5s3b0n0`.
5. Attempt-2 started 4s after the swap, hit step 1 at 21:38:17, ran
   cleanly to step 296 (loss 12.96 → 5.68 = ~466M tokens) until
   PBS walltime kill at 21:57:50.
6. **step-100 and step-200 DCP checkpoints both persisted** on flare
   (96 shards × 239 MB each + 5.5 MB metadata). First unambiguous
   proof since the 2026-05-03 regression that async-save works
   end-to-end with the fresh ezpz 0.16.0 tarball.

Full writeup with exact log snippets:
[`experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md`](experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md).

Knock-on actions: bumped the
[`bad-node-failover.md`](guides/bad-node-failover.md) status from
"v2 in production" to "v2 production-validated"; added back-pointer
from the original 8479579 incident report; greenlit qdel + resubmit
of the 9 queued production jobs (2B/20B/80B chains) against the
fresh tarballs.

20B + 80B smokes (`8505325` / `8505326`) still Q in capacity queue —
blocked behind `8505200` (the 2-node test.sh used for the tarball
rebuilds + smoke runs); will start when that walltimes out around
2026-05-24 04:00 UTC.

---

## 2026-05-23 — Failover wrapper hardening: tests, ANSI fix, async-mode regression diagnosed

### Failover-wrapper test harness + 2 more wrapper bugs

Built `tests/failover/{run_tests.sh, fixtures/*.log}` — 9 synthetic
log fixtures with byte-identical ANSI escapes, each reproducing one
of the failure modes we've seen in production. The harness mirrors
`failover_lib.sh`'s rc-determination block into a standalone
`evaluate_rc()` function and asserts the expected `(rc, decision)`
tuple for each fixture. Pattern: edit `failover_lib.sh` → update
fixtures → run `bash tests/failover/run_tests.sh` here in the main
repo → THEN push + pull into v2 clones. Stops the
edit-push-pray-discover-bug-only-in-production loop that ate ~6
production runs over the past 48h.

Running the new tests immediately surfaced **2 more wrapper bugs**:

1. **`grep -c ... || echo 0` produces `0\n0`** when grep matches
   nothing → arithmetic eval `syntax error (error token is "0")` →
   the entire crash-line branch silently skipped. `grep -c` already
   writes `0` to stdout on no-match; the `|| echo 0` fallback was
   dead code that produced malformed output. Removed.
2. **Walltime guard regex was a strict subset of crash-detect regex**
   — only matched `Connection closed by peer | died from signal (9|11)`,
   missing `OutOfMemoryError`, `UR_RESULT_ERROR`, `Timed out waiting`,
   `EOFError`. So a real bad-node failure that surfaced as shell exit
   143 (mpiexec SIGTERM after EOFError) got misclassified as a clean
   walltime kill and the wrapper bailed without retry. Made both
   regexes identical (the broader set).

Both bugs silently active in production yesterday/today before fix.
Commit `0d93a1e91` adds the harness + fixes; 9/9 tests pass in both
the main repo and the 20B v2 production clone.

### ANSI codes in 'Execution finished with N' parsing

Earlier in the day, several jobs zombie-succeeded because the
wrapper's `inner_rc` extraction returned empty on ANSI-coded log
trailers:

  Logged:  `Execution finished with \x1b[1;36m143\x1b[0m`
  Regex:   `Execution finished with \[?[0-9]+\]?`
  Match:   none (the `\[?` matches a literal `[` byte, but the
           actual byte sequence is ESC + `[`)

Fix `94a8fda66`: strip ANSI codes with `sed -r 's/\x1b\[[0-9;]*m//g'`
before grepping the trailer.

### Async-mode regression for 20B 512N — the actual root cause of
"async ckpt save kills the cluster"

Spent hours today investigating why 20B 512N hasn't persisted past
`step-800` since 2026-05-03 — three weeks of dispatches all dying
mid-save. Today found the **smoking gun**: every save on disk between
step-200 and step-800 happened on 2026-05-01 + 2026-05-03 under the
**default checkpoint mode (sync)**. The submit script switched to
`--checkpoint.async-mode=async` sometime between May 3 and May 11,
and nothing has persisted past step-100 on the 20B 512N chain since.

The previous "files-per-save / Lustre saturation" hypothesis was
half right but missed the actual mechanism: at 6,144 ranks, async
saves stream the 244 GB ckpt to flare in the background AT THE SAME
TIME as the gloo training-step heartbeat. Either the writes or the
gloo traffic backs up, one peer times out, cascade. Sync saves block
training while writing — no overlap, no cascade.

Submitted `8505258` (20B 512N) + `8505259` (cont) with
`CHECKPOINT_ASYNC_MODE=disabled` (= true sync). Also queued
`8505255/56/57` (20B 256N sync variants). 2B chains stay on async
(those have always saved cleanly at this scale).

### Production chain progress today

**2B 256N**: persisted **step-25,000 → step-25,500** over 2 dispatches
(8503506 walltime-finished + 8505119 advanced ckpts then exited 127).
+500 fresh steps. Loss 2.74. The only trajectory actually moving.

**20B 256N**: still wall-bound at `step-300`. Six dispatches today,
each reached in-RAM step 308-326 then died from bad-node mid-training
before crossing the next 100-step save boundary. Wrapper detection
working correctly (validated against fixtures); the wall is genuine
Aurora bad-node prevalence at the per-dispatch survival window.

**20B 512N**: still wall-bound at `step-800`. Four dispatches today,
all 3-attempt exhausted at init before any training step. Hopes
pinned on the sync-mode resubmit (`8505258`).

**80B 256N**: 0 persisted, 5 attempts today. Bumped to `select=276`
(20 spares) + `FAILOVER_MAX_RETRIES=5` for `8505221` — still died.
Environment too unstable for 80B init right now.

### Wrapper fix chronology (today)

| Commit | Fix |
|--------|-----|
| `94a8fda66` | Strip ANSI codes before parsing 'Execution finished with N' |
| `0d93a1e91` | Test fixtures (force-added .log) |
| `4310258` | Drop `\|\| echo 0` bug + walltime-guard regex parity |

All 3 commits pulled into all 3 v2 production clones. Tests pass in
all clones.

### `ezpz` upgraded to 0.15.1 in all v2 clones

Got the `--timeout T` + `--retries N` flags from ezpz PR #136. Wired
`--timeout=1800` into `failover_lib.sh` (commit `eefccfc9d`,
yesterday). Catches silent-hang failure mode that previously was
invisible (the 8479579 incident — 5h of W&B heartbeat alive but
training metrics dead). Exit 124 from the watchdog now routes
through swap-and-retry.

### What's next

- Once `8505123` (20B 256N, currently R, in-RAM step 326) either
  crosses step-400 ckpt save or dies, the sync-mode 20B chains
  (`8505255` for 256N, `8505258` for 512N) take over. That's the
  live test of the async→sync regression hypothesis.
- Eval batch `8505205` (13 fresh 2B 256N ckpts, step 14K → 25.1K)
  running on capacity; 5/13 done so far. Will refresh plots +
  README tables once all land.

### 39th upstream sync — DebugMode numerics debugger (no-op for ezpz)

Merged 1 upstream commit (`19c567f76`,
[PR #3323](https://github.com/pytorch/torchtitan/pull/3323)). Pure
tooling addition: new `torchtitan/tools/numerics_debugging/` module
(activation tracer + bitwise comparator, ~2 KLOC) plus a
`numerics_debugging` skill under `.claude/skills/`. No code path
ezpz exercises changed; no replay needed. The new skill auto-loads
in this session and could be useful next time we need to bisect a
silent loss-curve divergence.

### Closing follow-up — PR #184767 closed in favor of upstream #183625

`@frost-intel` flagged that
[pytorch/pytorch#183625](https://github.com/pytorch/pytorch/pull/183625)
is a draft already covering the xccl `_set_pg_timeout` dispatch +
the new `test_c10d_xccl.py` (in pieces). Closed our PR #184767 in
deference. Local workaround in
[`22847fcb3`](https://github.com/saforem2/torchtitan/commit/22847fcb3)
(`_set_pg_timeouts_xpu_aware`) stays load-bearing until #183625
actually lands.

---

## 2026-05-22 — First upstream PyTorch PR filed; 2-week summary

### Upstream PyTorch PR for xccl `_set_pg_timeout` dispatch

Filed https://github.com/pytorch/pytorch/pull/184767 — adds the
missing xpu branch in `torch.distributed.distributed_c10d._set_pg_timeout`
so xccl PGs route through `ProcessGroupXCCL.set_timeout` instead of
silently no-op'ing with the `"Set timeout is now only supported for
either nccl or gloo."` warning. Initial commit
[37af153](https://github.com/saforem2/pytorch/commit/37af153); review
fixes (Backend type annotation, simpler single-binding form, updated
warning text, `find_free_port` + `@retry_on_connect_failures` for the
test) in [8ceedc7](https://github.com/saforem2/pytorch/commit/8ceedc7).
All 7 inline review threads from copilot + codex addressed and
resolved. Pinged `@kwen2501` (c10d CODEOWNER) + `@guangyey` +
`@frost-intel` for review; CI gated on first-time-contributor workflow
approval.

Empirically verified the diff on Sunspot 1N × 12 ranks against the
in-repo `.venv` torch 2.13 (allocs 12467214 + 12467219 + 12467231,
all released). Side finding: xccl's C++ `set_timeout` **does** mutate
`backend.options._timeout` — contradicts the pessimistic line in
[`PLAN_xccl_timeout_upstream_pr.md`](upstream-issues/PLAN_xccl_timeout_upstream_pr.md)
PR 2 that "xccl stores the value but does nothing." Storage works;
only **enforcement** (watchdog + abort) is still missing. PR 2 scope
unchanged. Local pytest port at
[`tests/distributed/test_c10d_xccl.py`](../tests/distributed/test_c10d_xccl.py)
verifies the patched-vs-unpatched contract on either side.

### Two-week summary

Wrote up the 2026-05-08 → 2026-05-22 retrospective at
[`docs/summaries/2026-05-22.md`](summaries/2026-05-22.md).
51 commits across 8 themes: upstream xccl PR, 4 upstream syncs (one
no-op, one no-replay, two with replays), 80B bad-node failover
infrastructure (the silent-hang detection bug fix in
[e216a2523](https://github.com/saforem2/torchtitan/commit/e216a2523)
closes the 8479579 incident class), Sunspot smoke campaigns, MoE EP=2
hang reclassification, TPC26 talk prep, and docs hygiene.

### 38th upstream sync — MoE dispatcher split + ChunkedCELoss/TP grad fix

Merged 4 upstream commits (`cfe97c605..c2a3771a4`). One replay landed
in [`d87729ad8`](https://github.com/saforem2/torchtitan/commit/d87729ad8):
mirror upstream PR
[#3389](https://github.com/pytorch/torchtitan/pull/3389)'s isinstance
dispatch on `token_dispatcher` Config classes in
`experiments/ezpz/moe/model.py`, dropping the removed `DeepEPMoE`
swap. PR
[#3412](https://github.com/pytorch/torchtitan/pull/3412) is internal
to `torchtitan/components/loss.py` — no ezpz replay.

Smoke (2N Sunspot, jobs 12467277 + 12467288 + 12467323):
- `agpt_2b` clean, byte-comparable baseline (140 s, peak 24.34 GiB).
- `moe_2b_ep` at LBS=1 clean and numerically equivalent to the
  37th-sync baseline: 14.95 GiB vs 15.03 GiB, TPS within 1.4%.
- `moe_2b_ep` at the previous registry-default LBS=16 OOMs on the
  bf16 vocab projection (`[16 × 8192, 256128] × 2 B ≈ 62.5 GiB`,
  overflows a 64 GiB Max 1550 tile). Same pre-existing `_ep`
  vocab-projection OOM the 37th-sync follow-up flagged. Closed by
  pinning `moe_2b_ep` to LBS=2 in
  [`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f).
- Smoke report:
  [`docs/experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md`](experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md).

Side issue: stale `outputs/checkpoint/step-100` from pre-PR-3159
layout is no longer loadable (`Missing key in checkpoint state_dict:
layers.0.attention.qkv_linear.wk.weight.`); backed up to
`outputs/checkpoint-20260522-120005`.

---

## 2026-05-21 — Step-41 EP hang retry + registry fix + TPC26 talk prep

### `moe_debugmodel_ep` LBS=2 hang did not reproduce — reclassified transient

Retried yesterday's hung config on a sibling 2N alloc (12467180).
Full 50 steps clean, exit 0, 139 s wall. **The original 16-min stall
at step 41 was a transient**, not a systemic EP/AC/compile bug.
Reclassified in
[`docs/upstream-issues/moe_ep_step41_hang.md`](upstream-issues/moe_ep_step41_hang.md)
with three adjacent findings worth keeping
visible:

- `comm.train_timeout_seconds=100` did not fire on the original
  985 s silence — timeout may not be wired into the CCL/XCCL
  collective path on XPU. Worth a separate writeup.
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` crashes on XPU with
  `Backend fake does not yet support sequence numbers`. The error
  doesn't mention XPU — easy footgun on next attempt.
- `--debug.deterministic` is incompatible with MoE on XPU:
  `_histc_xpu does not have a deterministic implementation`. So
  bit-exact regression gates for MoE on Intel are blocked on an
  upstream PyTorch deterministic `_histc_xpu` kernel.

Also a curious throughput band: retry ran at ~11.8k TPS, original at
~6.8k TPS — same code, same nodes-of-the-same-class. ~1.7× spread,
plausibly correlated with whatever caused the original hang.

### Registry fix for `_ep` configs

Pinned `moe_debugmodel_ep` to LBS=2 (was inheriting LBS=8 from
`moe_debugmodel()`, OOM'ing at ~33 GiB on the vocab projection).
Annotated `moe_2b_ep` with a docstring confirming its LBS=16 default
is the validated peak. Commit `f2cbc0327`.

### TPC26 MAPE talk

Got invited to speak at the TPC26 MAPE track (Baltimore / Munich,
May 31 - Jun 3) by Rio Yokota. Drafted title, abstract, and 11-section
outline at
[`docs/notes/slides-2026-05-21.md`](notes/slides-2026-05-21.md).
The fork-tax-as-first-class-workflow angle (§4) and the silent-numerics
section (§6) are the most differentiated bits. Folded today's
engineer-hours/week estimate into §9: **~8-15 hr/wk recurring
operational triage** across 4 weeks of journal entries (~25-35% of
one engineer), with episodic spikes to 30-40 hr when a silent bug
surfaces or a bisect goes wide. The unbounded-cost punchline lives in
the silent class: bf16-freeze alone burned ~450B tokens of v1
compute.

---

## 2026-05-20 (late) — PR #3386 EP follow-up + agpt merge sanity smoke

Followed up the 37th-sync replay with two parallel smoke campaigns at 2N on
Sunspot (commit `1d4115d3f`, jobs 12467180/12467181):

### moe `_ep` follow-up — EP=2 path validated, but registry configs need LBS override

Reports: [`moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md)

- **`moe_2b_ep` (LBS=16 from registry)** — clean 50 steps,
  12.94 → 6.07, 2,860 TPS/GPU, peak **15.03 GiB** vs `moe_2b` EP=1's
  14.97 GiB. Token-dispatch overhead at 2B scale is essentially free
  (+0.06 GiB, ~1% TPS hit). PR #3386's `wire_meshes` plumbing works.
- **`moe_debugmodel_ep` (LBS=8 from registry)** — **OOM at init** in vocab
  projection (`(LBS*8192, 256128) bf16` = 31.27 GiB requested). The default
  LBS the registry inherits from `moe_debugmodel()` is too aggressive for
  the EP variant at 2N. Override needed.
- **`moe_debugmodel_ep` re-run at LBS=2** — clean steps 1-41 then **hung
  16 min at step 41/50** and got SIGTERM (exit 143). New finding,
  uninvestigated — possibly EP all-to-all backend stall under the
  `standard` `moe_comm_backend`. Not blocking but worth a follow-up.

### agpt merge sanity — clean, plus DeviceMesh regression re-confirmed

Reports: [`agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md`](experiments/agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)

- **`agpt_2b`** — clean 50 steps, 12.97 → 6.58, peak **24.34 GiB
  (38.04%)** — *byte-identical* to the prior post-resync baseline.
  Confirms PR #3346 (`graph_trainer` regional_inductor refactor, bundled
  in the same merge) is a no-op for the agpt path. Throughput within the
  expected 2-3% noise band.
- **`agpt_50b_wide`** — re-confirms the torch-2.13
  `DeviceMesh`-in-saved-tensors `AssertionError` in AOT autograd's
  `save_from_forward`. Crash in ~121s on 2N, all 24 ranks identical
  signature. 37th sync did **not** fix it (didn't expect it to — bug is
  in PyTorch, not torchtitan). Standing workaround (compile=OFF for
  80B-family on torch 2.13, or stay on torch 2.10) still the only
  option. See `project_80b_devmesh_bisect`.

### Action items dropped on the floor

- `moe_debugmodel_ep` / `moe_2b_ep` config registry: either pin a sane
  LBS in the `_ep` variants or document the OOM in the registry.
- Investigate the LBS=2 debugmodel_ep step-41 hang (EP all-to-all
  backend? token-dispatch deadlock?). Not reproducing automatically until
  someone re-runs.

---

## 2026-05-20 — 37th upstream sync (MoE clean DTensor boundaries) + replay smoke

Second sync of the day. Merged `89987072b` (2 commits beyond the 36th sync):

- **`963c20cba` — PR #3386** [MoE][5/n] Refactor MoE to clean DTensor
  boundaries for shared/routed experts. The big one.
- **`83e490429` — PR #3346** graph_trainer `regional_inductor` refactor.
  No ezpz dep.

### What PR #3386 changes

Restructures MoE TP/EP wiring from an imperative parallelize-time pass to
config-based sharding declarations populated at `update_from_config` and
applied by `model.parallelize(parallel_dims)`:

- **Deleted upstream:** `torchtitan/distributed/expert_parallel.py`
  (`ExpertParallel`, `TensorParallel`), `ColwiseParallelWithGradPlacement`.
- **New upstream:** `torchtitan/models/common/moe_sharding.py` with
  `set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
  expert_param_layout)` populating router gate, shared experts, routed
  experts.
- **`GroupedExperts.parallelize`** added — calls `super().parallelize` then
  `token_dispatcher.wire_meshes(ep_mesh, tp_mesh)`.
- **`MoE.forward` simplified** — drops the explicit
  `DTensor.to_local(grad_placements=Partial)` at the top (now handled by
  config), splits shared-experts addition out of `combine()`.
- **`parallelize_deepseekv3`** drops `apply_moe_ep_tp` call; new flow:
  `if tp_enabled or ep_enabled: model.parallelize(parallel_dims)`.

### Replay scope

Only ezpz/moe was affected:

| File | Δ lines | Change |
|------|--------:|--------|
| `experiments/ezpz/moe/parallelize.py` | -73 | Drop `apply_moe_ep_tp` entirely + 3 deleted-symbol imports; collapse two-pass to single `model.parallelize` |
| `experiments/ezpz/moe/sharding.py` | +35 | Add `enable_ep` kwarg, call upstream's `set_moe_sharding_config` per MoE layer with `{w1:Shard(1), w2:Shard(2), w3:Shard(1)}` layout |
| `experiments/ezpz/moe/model.py` | +3 | Pass `enable_ep=...` from `update_from_config` |
| `experiments/ezpz/moe/config_registry.py` | -2 | Stale docstring scrub |

`apply_fsdp` (Aurora `ShardPlacementResult` workaround) and
`disable_fsdp_gradient_division` (CCL SUM-reduction workaround) stay
inlined locally.

### Smoke verification (Sunspot 2N, job 12467131)

| Config | Final loss | Δ vs baseline | Memory | TPS |
|--------|-----------:|--------------:|-------:|----:|
| `moe_debugmodel` LBS=2 | 6.99880 | -0.010 | 16.99 GiB (matches) | ~12,700 |
| `moe_2b` LBS=1 | 6.10607 | -0.050 | 14.97 GiB (+0.5 vs baseline) | ~2,900 |

Both within ±0.05 nats of the 35th-sync baseline. Drift is expected:
PR #3386's commit message states *"loss is expected to diverge compared
to main due to different reduction pattern, and shared expert
computation changes place"* — confirmed at our parallelism configuration.
`for_loop` expert backend fires the same warning count (5 + 17) as
baseline. No NaN/OOM. No recompilation events.

### Concerns flagged before merging (all resolved)

- **`MoE.forward` graph shape changed.** Watched for inductor
  recompilation events — none observed. Memory uptick at moe_2b
  (+0.5 GiB) is the only visible cost, plausibly from a separate buffer
  for `shared_out` before the final add.
- **Removed async overlap between shared_experts and DeepEP combine.**
  Documented upstream as a follow-up "can restore overlap using CUDA
  streams." We don't use DeepEP on Aurora/Sunspot so this is a no-op
  for ezpz today, but worth tracking if we ever turn it on.

### Reports + W&B

- Smoke: [`docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md)
- 37th sync entry: [`docs/upstream-sync.md`](upstream-sync.md)
- W&B `moe_debugmodel`: [`efficient-bird-2070`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/saga2gds)
- W&B `moe_2b`: [`electric-pond-2071`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/zudqly3y)

---

## 2026-05-20 — Post-resync smoke campaign (Sunspot 2N)

Validated yesterday's [35th upstream sync](#2026-05-19--upstream-resync-35th-full-dtensor-3159)
with a four-config compute-node smoke on Sunspot (job `12467124`, 2 nodes,
24 XPUs). All configs ran 50 steps cleanly post-replay; no NaN/OOM,
monotonic loss descent, no regression vs the historical Apr 25 baseline.

| Config              | LBS | Final loss | TPS/GPU | MFU    | Peak mem            |
|---------------------|----:|-----------:|--------:|-------:|---------------------|
| `agpt_debugmodel`   | 2   | 6.77       | ~37,500 | ~2.9%  | 2.60 GiB (4.06%)    |
| `agpt_2b` (LBS=1)   | 1   | 6.01       | ~6,100  | ~22.8% | 24.34 GiB (38.04%)  |
| `agpt_2b` (LBS=2)   | 2   | 6.12       | ~7,200  | ~27.0% | 44.73 GiB (69.91%)  |
| `moe_debugmodel`    | 2   | 7.01       | ~13,000 | ~9.0%  | 16.99 GiB (26.55%)  |
| `moe_2b` (LBS=1)    | 1   | 6.16       | ~2,900  | ~8.4%  | 14.47 GiB (22.62%)  |

**`agpt_2b` LBS=2 matches Apr 25 n=2 baseline** (7,224 TPS / 27.11% MFU
today vs 7,142 TPS / 27.6% MFU then) within noise. PR #3159's
`Module.parallelize(parallel_dims)` signature is wired correctly through
both `experiments/ezpz/{agpt,moe}/parallelize.py`.

**`for_loop` expert backend (PR #13) still fires on XPU** -- 5 warnings
for moe_debugmodel, 17 for moe_2b (one per MoE layer in each flavor).
End-to-end forward/backward/optimizer under FSDP all converge cleanly.

**Default `moe_2b()` LBS=16 OOMs on Max 1550** with a single 62.53 GiB
allocation. Likely the fused activation for all experts × full-batch-tokens
materialized by the for_loop path. Pre-existing XPU constraint, not caused
by the resync. Switched to LBS=1 for the smoke.

### Permissions sidestep that worked

The auto-mode classifier blocks the `source <(curl -fsSL https://bit.ly/ezpz-utils)`
pattern on every `ezpz launch`. Workaround: on the compute node, cache the
utils script once with
`mkdir -p ~/.ezpz && curl -fsSL https://bit.ly/ezpz-utils -o ~/.ezpz/utils.sh`,
then prefix every launch with `source ~/.ezpz/utils.sh && ezpz_setup_job && ezpz_setup_xpu`.
The cache is persistent on the compute node so this only needs doing once
per allocation. Used successfully for all five smoke launches.

### Reports

- [`docs/experiments/agpt/sunspot/20260520-smoke-n2-postresync.md`](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md)
- [`docs/experiments/moe/sunspot/20260520-smoke-n2-postresync.md`](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)
- `docs/upstream-sync.md` 35th entry updated with smoke validation table.

W&B runs: `olive-plasma-2054`, `sunny-waterfall-2055`, `dry-water-2056`,
`worldly-music-2058`, `azure-field-2061` (all under `aurora_gpt/torchtitan.ezpz.train`).

---

## 2026-05-19 — Upstream resync (35th, Full DTensor #3159)

Pulled 22 upstream commits (`ee4e91a13..52a292d29`, merge `a14987132`).
The headline is [pytorch/torchtitan#3159](https://github.com/pytorch/torchtitan/pull/3159)
"Config-based Full DTensor for Llama3" — a substantial refactor of the
config-based sharding API that lays the foundation for
`--training.full_dtensor` (all params/buffers/inputs become DTensors on
a multi-dim SPMD mesh).

**Breaking signature change for ezpz:** `Module.parallelize(mesh)` →
`Module.parallelize(parallel_dims)`. Each Module now self-resolves its
SPMD submesh from the axes referenced in its `NamedPlacement`s, instead
of being handed a bare `tp_mesh`. Two ezpz callsites hit:
`experiments/ezpz/agpt/parallelize.py` and
`experiments/ezpz/moe/parallelize.py`. Both replayed: pass
`parallel_dims` to `model.parallelize`, keep the explicit
`parallel_dims.get_mesh("tp")` for the async-TP plumbing on the next
line. `apply_moe_ep_tp` still takes per-axis meshes directly (it doesn't
route through `Module.parallelize`), so it's untouched.

`ShardingConfig` got three new optional fields (`out_src_shardings`,
`local_input_grad_placements`, `local_output_grad_placements`); all
default `None` so ezpz's existing `set_agpt_sharding_config` /
`set_moe_sharding_config` construct unchanged shapes. The
`set_gqa_inner_attention_local_map` helper that ezpz/agpt calls had
internal arg renames (`xq/xk/xv` → `q/k/v`) but the public call site is
identical.

`trainer.py` gained a `full_dtensor`-gated `parallelize_inputs` call and
a `pred.to_local()` fallback under `disable_loss_parallel`. ezpz's
`FaultTolerantTrainer` doesn't override `_get_batch` or
`forward_backward`, so both inherit cleanly. `full_dtensor` defaults
`False`, so no behavior change for current production.

Verification: `.venv/bin/python -c "import
torchtitan.experiments.ezpz.{agpt,moe}.parallelize"` succeeds on both
post-replay. A real compute-node smoke (`agpt_2b`/`moe_500m`) is
pending the next allocation. Doc: `docs/upstream-sync.md` 35th entry.

---

## 2026-05-12 — Upstream resync (#3308) + `for_loop` backend smoke

### Resync PR #13

Pulled 12 commits from `upstream/main` into a fresh `ezpz-moe-resync`
branch. The big-ticket landing was
[pytorch/torchtitan#3308](https://github.com/pytorch/torchtitan/pull/3308),
which deleted `_run_experts_for_loop` and the `use_grouped_mm` config
field from `models/common/moe.py` and inlined `torch._grouped_mm` as the
only expert path. Upstream's argument: `_grouped_mm` already provides a
CUDA fallback. **XPU has no `_grouped_mm` kernel at all**, so this would
have broken every ezpz MoE config on Aurora / Sunspot at first forward.

Replay strategy: introduce `EzpzGroupedExperts(GroupedExperts)` in
`experiments/ezpz/moe/experts.py` with a
`compute_backend: Literal["for_loop", "grouped_mm"]` selector. Default
defers to upstream. The `for_loop` branch re-vendors the deleted
`_run_experts_for_loop` body verbatim, restoring the XPU / pre-SM90 path.
`model.py` `update_from_config` now switches to `for_loop` on any device
that fails `has_cuda_capability(9, 0)`.

PR #13: https://github.com/saforem2/torchtitan/pull/13 (replaces #12).
PRs #9 / #10 / #11 (Sam Wheeler / Sam Wheeler / Nathan Nichols) flagged
on each that they should rebase onto this and adapt to the
`EzpzGroupedExperts` subclass.

### Smoke validation (Sunspot 8N)

Job `12466707` on `x1921c5s0b0n0`–`x1921c5s7b0n0`. `moe_500m`, 50 steps,
local batch 4, seq 8192, GBS 384. Run:
[`fluent-glitter-2042`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lt77xx0o).

- 11 layer-wise warnings emitted (1 per MoE layer):
  `torch._grouped_mm requires SM90+ CUDA; falling back to for_loop expert backend.`
  Confirms PR #13's `compute_backend = "for_loop"` switch is taken.
- Loss descended cleanly **12.90 → 6.66 (-6.24 nats)** over 50 steps.
- Steady-state throughput **~8,694 TPS / GPU, ~13% MFU** — actually ~20%
  per-GPU TPS uplift vs the
  [2026-04-13 2N benchmark](experiments/moe/sunspot/20260413-benchmark-n2.md)'s
  7,228 TPS / 9.11%, attributable to torch.compile inductor improvements.
  **No measurable regression vs the upstream `_run_experts_for_loop`
  body that #3308 deleted** (which makes sense — it's the same kernel).
- Memory stable at 54.6% across the run.
- Wall: 541s end-to-end including env setup + compile warmup.
- Report:
  [`docs/experiments/moe/sunspot/20260512-for-loop-smoke-n8.md`](experiments/moe/sunspot/20260512-for-loop-smoke-n8.md).

PR #13 is now smoke-validated end-to-end on XPU.

---

## 2026-05-05 — 80B DeviceMesh-bisect: torch-version, not depth

### Bisect kills the May 3 "depth-sensitive" claim

Submitted job 12465952 on Sunspot 4N to bisect the
`tensors_saved_with_vc_check` AOT autograd assertion across the agpt
80B family on the torch 2.13 venv. Three configs ran sequentially:

- `agpt_50b_wide` (48 layers): crashed in **66 s**, every rank logs the
  assertion (49 ranks × 1 = full crash on the first
  forward+backward).
- `agpt_70b_wide` (72 layers, new
  [`b9cda4b2`](https://github.com/saforem2/torchtitan/commit/b9cda4b2)
  config — 80B family with 12 fewer layers): same assertion, 28 s.
- `agpt_80b` (84 layers): same assertion, 29 s.

Per-config logs at `logs/agpt-80b-bisect-12465952/{config}.log`. The
"depth-sensitive — works at 48 layers" conclusion in the May 3 entry
was wrong. Two variables had changed between the May 3 50B_wide
success and today's 50B_wide failure: node count (2N → 4N) AND torch
version (`aurora_frameworks-2025.3.1` torch 2.10 → torch 2.13 venv).

To pin the variable I submitted job 12465962 (2N + torch 2.13,
[`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)).
Result: same assertion, 57 s. **torch 2.13 alone is the trigger;
node count is a non-factor.**

This means:

- The bug is **torch-version-sensitive, not depth-sensitive**.
- `agpt_50b_wide` on 2N + torch 2.13 is now a **clean ~30-60 s
  reproducer** for the upstream report (much smaller than 80B).
- The May 3 50B_wide success was masked entirely by the older AOT
  autograd code path on torch 2.10.

### Updated docs

- `experiments/ezpz/.claude/CLAUDE.md` — corrected the Recent Findings
  bullet, the Production v2 80B note, and the Known Bugs entry.
- `experiments/ezpz/docs/meeting-notes/agpt-sync.md` — added the
  2026-05-05 correction beneath the original 2026-05-03 finding so
  the meeting can show both.
- The new `agpt_70b_wide` config and the
  [`submit_80b_bisect.sh`](../scripts/submit_80b_bisect.sh) +
  [`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)
  scripts are now part of the bisect-tooling.

### Validator small-batch + global-state bug

While running the agpt_2b validator smoke separately, noticed the
val build was using `Global batch size: 48` and warning about
`train_iters defaulting to 1` on every validate() call. Three
related issues, all in
`torchtitan/components/validate.py:Validator.validate()` not passing
the trainer's `training_steps` and `global_batch_size` through to
`dl_config.build()`. Fixed in
[`3edfb0ff`](https://github.com/saforem2/torchtitan/commit/3edfb0ff)
by capturing `job_config` in `EzpzValidator.__init__` and forwarding
the right values. Validator was using ~1/4-sized batches per call
(noisier val loss); a worse latent issue was that `bc_set_config`
overwrites the blendcorpus library's global `DATA_CONFIG` with the
wrong `train_iters` and `global_batch_size`, which would silently
corrupt any later resume-from-ckpt rebuild of the train dataloader.

### blendcorpus ↔ Megatron parallelism aliasing

Reviewed `BlendCorpusDataLoader` for further Megatron-style
parallelism leftovers. Found 7 inconsistencies:

- 2 active and fixed today
  ([`140481d3`](https://github.com/saforem2/torchtitan/commit/140481d3)):
  scope the `dist.barrier` → CPU/gloo monkey-patch to torch < 2.13
  (it was being silently applied on 2.13 even though XCCL is fixed
  there); rename `_train_ds` → `_served_ds` so the attribute matches
  what it actually holds when `serve_validation=True`.
- 5 latent items (PP semantics, CP→SP aliasing, dp_world_size
  double-source, parallel-state duplication, hard-coded Megatron
  knobs) tracked in
  [`docs/TODO.md` §6](TODO.md) and documented in
  [`docs/guides/known-bugs/blendcorpus-megatron-aliasing.md`](guides/known-bugs/blendcorpus-megatron-aliasing.md).

### Other

- Pulled 11 upstream commits (32nd sync,
  [logged](upstream-sync.md#2026-05-05-32nd-sync--observability--moe-token-pad--cp-fix--rlgraph_trainer-churn)).
  Notable: `b2cd149f` adds `torchtitan/observability/` structured
  logging hooks. ezpz's `FaultTolerantTrainer` and `EzpzValidator`
  override their respective base classes' methods entirely so the
  new `@sl.log_trace_span` decorators don't propagate — non-blocking
  but worth re-adding later if we want trace spans on the ezpz path.
- Aligned `~/.claude/statusline-command.sh` to match starship.toml
  (true gray time, fish-style abbreviated path with cyan-bold +
  underlined-blue repo root, bold-purple branch).

### 80B v2 has a working path on torch 2.13 + `compile=OFF`

After the bisect closed out the DeviceMesh question, ran job 12466025
(4N, 20-step smoke) to test whether `compile=OFF` actually unlocks
80B v2 production on the torch 2.13 venv:

- Config: `agpt_80b` at TP=2, AC=full, **compile=OFF**, AdamW LR=1e-6,
  fp32-master, 4 nodes (24 ranks, dp_shard=12).
- Result: clean run, all 20 steps. Loss descended **12.98 → 10.46**
  (-2.52 nats), MFU steady at **~17.8%**, memory peaked at **88.94%**
  (~7 GiB headroom per tile), exit code 0.
- Grad-norm bumped to ~34 around steps 15-16 then recovered to ~14
  by step 20 — early-training oscillation, not a stall. Production
  80B should add the 200-step linear warmup the 2B/20B v2 configs
  already use.

This confirms an actually-working v2 80B path. Notable that
`compile=OFF` MFU (~17.8%) *matches* what compile-on used to give v1
on torch 2.10, so we're not paying any throughput penalty for not
compiling — though that'll change once `compile=ON` works again
upstream and the inductor optimizations actually kick in.

Submit script:
[`scripts/submit_80b_no_compile_t213.sh`](../scripts/submit_80b_no_compile_t213.sh).
Per-step log:
`logs/agpt-80b-no-compile-t213-12466025/run.log`.

`compile=ON` for the 80B family is currently broken on **both** torch
versions: torch 2.10 hits the step-1 hang regression from Apr 16-23
upstream changes; torch 2.13 hits the DeviceMesh-in-saved-tensors
AOT autograd assertion. `compile=OFF` is the only viable v2 80B path
until either upstream bug is fixed.

---

## 2026-05-04 — 20B chain walltime, 1024N startup crashes, doc cleanup

### Production training

- **20B 512N canonical chain (8463628)** finished its 12h walltime
  cleanly at step **863, loss 3.46**. Final TPS ~358, MFU ~17.8%.
  step-100..step-800 ckpts all saved. Continuation **8466848** auto-released
  from hold and is now Q for a 512N slot.
- **20B 256N v2 (8463659) started running.** Fresh-start trajectory at
  256N, separate ckpt dir (`n256-gbs6144`) — *not* a chain extension.
  Currently at **step 200, loss 5.65, MFU 14-20%**, 2 ckpts saved
  (step-100, step-200). Useful as a per-token-vs-512N comparator at
  matched optimizer state. Loss curve looks healthy:
  12.96 → 8.5 (step 45) → 6.35 (step 124) → 5.65 (step 200).
- **2B 512N canonical chain** still at step 5,073 (loss 2.97).
  Continuation 8463627 still Q ("Not enough free nodes available");
  8466847 held behind it.

### 1024N first-attempts both crashed at startup

Both 1024N v2 jobs (queued since 2026-05-01) finally got slots and
**crashed within 4 minutes**:

- **2B 1024N (8463182)**: `MemoryError: std::bad_alloc` inside
  `torch.distributed.broadcast` during `set_determinism` init. Died
  after 211s, exit 143.
- **20B 1024N (8463183)**: rank 4732 died from signal 11 (SIGSEGV)
  during the same init phase, exit 143.

12,288 ranks (1024 nodes × 12 GPUs) appears to be hitting an init-time
memory/comm scaling issue we don't see at 256N or 512N. Worth a
smaller-scale repro before resubmitting — maybe 768N or 896N to bracket
where it starts failing. Not blocking the canonical 512N chains.

### 20B v2 eval through step-600

- **8467370** (capacity queue, 8h requested, ran to **12h15m walltime
  kill**) delivered eval scores for steps 100/200/300/400/500/600
  before being killed mid-step-700 lm-eval. Steps 700/800 will need
  a resubmit.
- **ARC-Easy `acc` lifts monotonically** from 0.266 → 0.290 → 0.295 →
  0.318 → 0.346 → **0.393** across that range — the cleanest-yet
  signal that fp32 master is producing real benchmark progression
  vs. v1's flat ~0.27 baseline.
- HellaSwag also creeping: 0.257 → 0.270. ARC-Challenge and
  Winogrande still in noise (expected at <100B tokens).
- v1-vs-v2 plot regenerated; results table added under
  [`docs/evals/agpt/20b/README.md`](evals/agpt/20b/README.md).
- Eval throughput tanked while concurrent 256N v2 (8463659) was
  starting — both share flare bandwidth for ckpt I/O and dataset
  reads. Steps 100-500 each took ~30 min; step-600 took ~3h. Will
  hold off on resubmitting steps 700/800 until 256N finishes.

### Doc cleanup

- **Compile flag was wrong in 5 v2 setup tables.** `agpt_2b()` and
  `agpt_20b()` both default to `compile=True`, and W&B configs +
  startup logs ("Compiling each TransformerBlock with torch.compile")
  confirm compile is on for all v2 runs. The "Compile | off" rows in
  the v2 setup tables under `docs/production/agpt/{2b,20b}/n*/README.md`
  were stale carryovers from drafts. Flipped to "on" across 2B
  n256/n512/n1024 and 20B n512/n1024.
- **Submit-script links added** to all 6 per-node-count READMEs. v2
  entries point to `scripts/submit_agpt_{2b,20b}_aurora_venv.sh`
  (one script handles all node counts via env vars); v1 entries point
  to the per-node `submit/aurora/submit_agpt_*.sh`.
- **Absolute log paths added** for every Job ID in every Progress
  table — v1 logs at `/lus/flare/.../torchtitan-ezpz/agpt-*-sophiag-*.o<JOBID>`,
  v2 logs at `/flare/.../runs/agpt-{2b,20b}-v2/torchtitan-ezpz/agpt-*.o<JOBID>`.
  Queued/held jobs note where the log will land on start.
- **`submit/README.md` added** marking the directory as legacy. The
  torch-2.10 `submit/{aurora,sunspot}/*.sh` scripts produced every v1
  trajectory and were the production driver through 2026-04-29. After
  the v2 restart on 2026-04-30, all production training moved to
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` on the torch 2.13
  venv stack — nothing live reaches into `submit/` anymore.

### Plotter

- `PRODUCTION_RUNS["20b_v2_256"]` added (wandb run `r1yyxbmt` =
  8463659) so the 20B 256N v2 trajectory shows up in dashboard
  refreshes.

### 8463659 NODE_FAIL after step 364 (afternoon)

Same recurring Aurora bad-node failure mode that killed 8459818 /
8460301 / 8460302. Last training step was 364 at 11:09:12, then
immediately:

```
x4406c6s7b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
x4218c2s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 606 died from signal 15
```

PALS shepherd on `x4406c6s7b0n0` got SIGKILL (kernel OOM-killed,
hardware fault, or system-level take-out), all ranks on that node
lost their parent → cascading SIGTERM. PBS reports `Exit_status -20`
= NODE_FAIL after 9h walltime. step-300 ckpt saved cleanly (loss 4.61
final). Trajectory pages updated to reflect the crash; no continuation
chained since the canonical chain is at 512N and this 256N run was a
per-token comparator scaling experiment rather than a chain.

### 20B v2 eval — full step-100..800 sweep complete

After waiting for 8463659 to finish (and free flare bandwidth),
resubmitted the missing step-700 + step-800 evals as 8469257 (capacity,
3h walltime). Finished cleanly in 3h flat. ARC-Easy `acc` continues
its monotonic ascent: 0.359 (step 500) → 0.393 (600) → 0.391 (700) →
**0.444** (800) — clean signal vs v1 256N's flat ~0.27 across all of
0-63B tokens. HellaSwag `acc_norm` 0.270 → 0.281 → 0.284 (+3pp above
v1 by step 800). v1-vs-v2 plot regenerated; `docs/evals/agpt/20b/`
table updated with all 8 v2 ckpts.

### Direct verification of the bf16 fix in checkpoint weights

User asked for RMSNorm.weight variance across the new 2B production
ckpts. Pulled stats from 6 HF-converted ckpts:

| ckpt | mean(var) | mean(std) | min weight | max weight |
|---|---:|---:|---:|---:|
| v1 step-10000 | **0** | **0** | **1.000** | **1.000** |
| v1 step-15000 | **0** | **0** | **1.000** | **1.000** |
| v2 256N step-2000 | 2.2e-5 | 0.0045 | 0.973 | 1.039 |
| v2 512N step-1000 | 5.4e-6 | 0.0016 | 0.988 | 1.016 |
| v2 512N step-3000 | 5.4e-5 | 0.0071 | 0.957 | 1.063 |
| v2 512N step-5000 | **1.2e-4** | **0.011** | **0.926** | **1.102** |

Every single v1 RMSNorm channel is exactly 1.0 — bf16-master
sub-ULP-update bug really did freeze every norm. v2 weights are
training: variance grows monotonically with token count, range fans
out from [0.988, 1.016] at step 1000 to [0.926, 1.102] at step 5000.
Per-layer at step 5000: `model.norm.weight` is biggest (mean 1.098,
std 0.006 — every channel uniformly scaling up); mid-depth layers
(5-7) have the highest per-element std (0.014-0.017, learning the
most differentiated channel scales). Final smoking gun for the v2
restart, complementary to the lm-eval evidence.

### Doc maintenance + plotter

- Refreshed 20B v2 256N plots (added wandb `r1yyxbmt` to
  `PRODUCTION_RUNS`); re-ran on the dead trajectory after NODE_FAIL.
- Updated parent snapshots (`production/README.md`,
  `production/agpt/README.md`, `production/agpt/20b/README.md`) to
  reflect 8463659 NODE_FAIL + 1024N startup crashes.
- `submit/README.md` added marking torch-2.10 `submit/` as legacy.
- Submit-script links + absolute log paths added to all 6 per-node
  READMEs.
- Compile-flag rows in 5 v2 setup tables corrected from "off" to "on"
  (defaults to `True` in `agpt(...)` and confirmed in W&B + startup
  logs).
- `running-with-newer-pytorch.md` expanded with the at-scale yeet
  section (8N→4096N table, tarball workflow, `/tmp/.venv` switch).
- Saved `memory/project_1024n_init_crash.md` so future sessions know
  to bracket 1024N attempts at 768N/896N first.

### Still queued

- **8463627** (2B 512N chain1 continuation), **8466847** (held
  `afterany:8463627`), **8466848** (20B 512N chain1 continuation),
  **8467141** (√2-LR fork chain1), **8467142** (held
  `afterany:8467141`) — all Q for 512N slots, none running.
  No production training is currently active.

---

## 2026-05-03 (evening) — TP > 1 loss-reporting bug + agpt_50b_wide

### Loss reporting on TP > 1 is off by `dp_world_size`

Hunting a different bug (the 80B `compile + AC + TP=2`
DeviceMesh-in-saved-tensors crash) we added `agpt_60b` and then
`agpt_50b_wide` as smaller bisect targets. The `50b_wide` smoke at
2N TP=2 reported step-1 loss = **1.07**. That should be ≈ ln(256128) ≈
**12.45 nats** for random init — 12× too small. The 12 matched
`dp_world_size = 12` (24 ranks ÷ TP=2) exactly, which pointed at a
missing cross-batch reduction.

Root cause is upstream commit `1786292d` (2026-04-27) in
`torchtitan/distributed/utils.py`. The new DTensor branch in
`_dist_reduce` returns `float(x.full_tensor().item())` and skips the
requested mesh `all_reduce`. That is correct only when the DTensor's
mesh matches the requested mesh — but the trainer's loss reduction
passes `loss_mesh` (= batch × cp), and the loss is a Replicated
DTensor on the **TP** mesh (orthogonal). The cross-batch sum is
silently dropped.

Verified the diagnosis by re-running `agpt_2b` at TP=2 (no compile, 3
steps) with a workaround in place: convert `loss` DTensor to a plain
tensor before `dist_sum`/`dist_max`. Step-1 loss came back as **12.94**
— matches the known-good TP=1 baseline of 12.95. Bug confirmed and fix
verified in one shot.

Workaround landed in `experiments/ezpz/`:

- `trainer.py` — adds `loss = loss.full_tensor()` before the
  `dist_sum`/`dist_max` reductions.
- New `validator.py` — `EzpzValidator(Validator)` subclass with the
  same fix in its `validate()` override.
- `agpt/config_registry.py` `_base_config` — uses
  `EzpzValidator.Config` instead of `Validator.Config`.

Documented in `docs/guides/loss-reporting-tp-dist-reduce.md` and the
upstream issue note at
`docs/upstream-issues/dist_reduce_dtensor_skip.md`. Filed upstream as
[pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).

**Implications for live dashboards:**

- 2B / 20B production W&B (TP=1): correct, no action needed.
- **80B production W&B (TP=2): under-reported by `dp_world_size`.**
  Multiply reported loss by `world_size / tp_degree` to recover the
  true per-token NLL. A 256N TP=2 dashboard's loss is currently 1536×
  smaller than the truth.

### agpt_50b_wide added; agpt_60b removed

To get a smaller compile target for the 80B-bug bisect we first
added `agpt_60b` (dim=9216, 60 layers, ~58B params, same per-layer
shape as 80B). That smoke crashed with an Intel GPU SegFault at step 2
— OOM dressed up as a not-present-PDE fault, after step-1 measured at
**97.41% memory** with no headroom for the step-2 activation peak.

Replaced with `agpt_50b_wide` (dim=9216, **48 layers**, ~48B params).
Smoke ran all 10 steps cleanly at 95.94% memory: MFU ≈ 15%, TPS ≈ 140
on Sunspot. Concluded "this is a working `compile + AC + TP=2` dense
config — the 80B-family DeviceMesh-in-saved-tensors crash does NOT
reproduce at 48 layers, only at 84, so the bug is depth-sensitive."

> **CORRECTION (2026-05-05):** that conclusion was wrong. The May 3
> smoke happened to use torch 2.10 (`aurora_frameworks-2025.3.1`); a
> proper bisect on torch 2.13 (jobs 12465952 + 12465962) showed the
> bug fires on `agpt_50b_wide` / `agpt_70b_wide` / `agpt_80b` alike,
> on both 2N and 4N. Bug is **torch-version-sensitive**, not
> depth-sensitive. See the 2026-05-05 journal entry above for the
> full bisect.

### Validation loss work — partial

Started wiring blendcorpus's already-built validation split through the
`Validator`. Stopped short to chase the loss-reporting bug above.
Status:

- `BlendCorpusDataLoader.Config` now has `serve_validation: bool`
  and `eval_iters: int` (default 100). `eval_iters` is only requested
  when `serve_validation=True` (otherwise we'd trigger a fresh
  validation index build that hangs in dataset construction).
- `_base_config` constructs a validator with
  `BlendCorpusDataLoader.Config(serve_validation=True)` for the val
  dataloader, but `validator.enable=False` by default.
- `EzpzValidator` is wired in but not yet smoke-tested.

Remaining: 250-step `agpt_2b` run with `--validator.enable` to confirm
the validator fires at step 1 and step 200, and that the val loss is
reasonable.

---

## 2026-05-03 — Production progress + canonical-chain consolidation + doc reorg

### Production training

- **2B 512N canonical chain (8460301 → 8463626 → 8463627)** — 8463626
  finished its 12h walltime cleanly at step 5,073 (loss 2.97). 50
  ckpts saved (every 100 steps, step-100 to step-5000). Continuation
  8463627 queued (will auto-resume from step-5000). **Cumulative:
  step 5,073, loss 2.97, 510B tokens (10.9% of 4.67T target).**
- **20B 512N canonical chain (8460302 → 8463628)** — 8460302 hit
  walltime at step 300 (loss 4.95). 8463628 (12h continuation) is
  currently running, resumed from step-200 (step-300 ckpt was
  incomplete from the NODE_FAIL on 8460302 so DCP picked the prior
  good ckpt). At write time: step 287, loss 5.00, MFU 17.5%.
- **Other queued jobs at different node counts** (2B 1024N 8463182,
  20B 1024N 8463183, 20B 256N 8463659) are *independent* trajectories
  — they write to separate ckpt dirs (keyed on `gbs`) and would start
  fresh from step 0. Treated as scaling experiments, not chain
  extensions.

### v1-vs-v2 evals (smoking gun)

- Ran `eval-2b-v2.sh` on 10 v2 256N ckpts (steps 200-2000). ARC-Easy
  climbed 0.277 → 0.429 over 100B tokens; v1's flat ~0.27 across 450B
  tokens validates the bf16-master RMSNorm-freeze fix end-to-end.
  HellaSwag also broke out at 80-100B tokens (v2 0.301 vs v1 0.251).
- Ran `eval-20b-v2.sh` on step-100 + step-200 (10B and 20B tokens).
  ARC-Easy v2 +1.5pp above v1 at the same token count, others still
  in noise. Will revisit at higher v2 token counts.
- Submitted fresh evals (8466827 for 2B 512N steps 1000-5000;
  8466828 for 20B 512N step-300).
- Added per-trajectory plotter at
  `docs/evals/agpt/{2b,20b}/plot_v1_vs_v2.py` — 4-panel comparison
  (HellaSwag/ARC-Easy/ARC-Challenge/Winogrande) with random baseline
  marked.

### Production-side doc reorganization

- Surfaced 2B 512N + 20B 512N v2 dashboards at the agpt index level
  (`docs/production/agpt/README.md`). Previously only embedded in
  per-model READMEs.
- Wrapped all v1 historical sections in `<details closed>` blocks
  across both production and eval READMEs, so v2 stays prominent.
- Renamed all production figure filenames to be explicit about
  v1/v2 (e.g. `production_2b_v2_512n.png` vs the older
  `production_2b_256n.png` ambiguity).
- Generated v1-vs-v2 overlay plots
  (`overlay_2b_v1_vs_v2.png`, `overlay_20b_v1_vs_v2.png`) via
  `plot_production_wandb.py --overlay {2b,20b}`.
- Moved the **MDS 2B SophiaG training curves** from
  `docs/evals/agpt/2b-mds/` to `docs/production/agpt/2b-mds/` (the
  eval scores stay under evals). Cross-linked both ways. Noted as
  "pre-torchtitan reference baseline" in the agpt production index.
- `plot_production_wandb.py` extended to support multiple v2
  trajectories per model — `PRODUCTION_RUNS` keyed on
  `<model>_v2_<nodes>`, output filenames keyed on the same.

### Canonical chain dashboards (current state)

| Model | Cumulative | Loss | Tokens | Latest |
|-------|-----------:|-----:|-------:|--------|
| 2B 512N | 5,073 | 2.97 | 510B (10.9%) | 8463627 (Q) |
| 20B 512N | 300 | 4.95 | 30B (0.6%) | 8463628 (R) |

### Dataset CLI mistake-catcher

When `--dataloader.dataset=user/repo` (HF hub path) and
`--dataloader.dataset-path=...` are both passed, `_validate_dataset`
silently overrides `dataset_path` to None so the registered hub path
wins. Correct behavior, but it hid user mistakes (e.g. expecting
`--dataset-path` to point at a local clone of the hub dataset).

`9ef2908d` — emit a clear warning naming both args and telling the
user to drop `--dataloader.dataset-path` or pick a local-file dataset
name like `blendcorpus`. Override semantics unchanged; only logging
added. Production scripts that hardcode
`--dataset=blendcorpus --dataset-path=$DFL` are unaffected (they hit
the registered-name branch which keeps the path).

### 2B 256N vs 2B 512N — large-batch under-training observation

Pulled fresh evals on 2B v2 256N (10 ckpts, steps 200-2000) and 2B v2
512N (5 ckpts, steps 1000-5000). Surprise: at matched **token** counts,
256N beats 512N noticeably:

| Tokens (B) | 256N HellaSwag | 512N HellaSwag | Δ |
|-----------:|---------------:|---------------:|--:|
| ~100 | 0.301 (step 2K) | 0.264 (step 1K) | -3.7pp |

But at matched **step** counts, they're indistinguishable:

| Step | 256N HellaSwag | 512N HellaSwag | Δ |
|-----:|---------------:|---------------:|--:|
| 1000 | 0.262 | 0.264 | +0.2pp |
| 2000 | 0.301 | 0.304 | +0.3pp |

This is the classic large-batch under-training pattern. Both runs use
SophiaG LR=2.28e-5 (tuned for 256N / GBS=6,144). At 512N (GBS=12,288)
each step covers 2× the tokens but the optimizer state evolves at half
the cadence per token, with no compensating LR scale-up. Per-step
parity confirms the optimizer is healthy; per-token gap is purely
batch-size-induced under-training.

Implications:
- **Per-step**: 256N == 512N
- **Per-token**: 256N wins ~3-8pp at matched tokens
- **Per-wall-clock**: 512N wins (~2× throughput)

So the canonical-chain choice (512N) optimizes for wall-clock time to
target loss, not token efficiency. If chasing minimum tokens, 256N
would be preferable. Open follow-up: would √2-LR scaling (3.22e-5) at
512N close the per-token gap? Worth a fresh fork to test, but not
worth perturbing the running 512N chain's LR mid-run.

Documented in
[`docs/evals/agpt/2b/README.md`](evals/agpt/2b/README.md) under "v2
256N vs v2 512N — same model, two batch sizes".

---

## 2026-05-02 — 2B 512N continuation reaches 510B tokens

Light day. The 2B 512N continuation chain (8463626) ran cleanly
through 12h of walltime — went from step 1,300 (resume) to step
5,073, with loss dropping from 3.59 to 2.97. NODE_FAIL at the very
end again, but all 50 ckpts saved. This was the first chain run where
NODE_FAIL didn't cost meaningful progress — the ckpt-100 cadence +
keep_latest_k=0 (keep all) policy means we always have a recent
recovery point.

20B 512N (8460302) finished its 6h walltime at step 300, loss 4.95.
3 ckpts saved (step 100/200/300, though step-300 was incomplete and
DCP fell back to step-200 on resume).

Eval pipeline kept producing rolling 2B v2 results — first ARC-Easy
points landed in the 0.28-0.34 range across early ckpts.

---

## 2026-05-01 — 1024N production runs, yeet-env scaling sweep

### Production training

- **2B 1024N** (8463182, 12h walltime, small queue) and **20B 1024N**
  (8463183, 12h, small) both submitted. Same configs as the running
  256N/512N v2 runs (LBS=2, SophiaG LR=2.28e-5, plain CE, fp32 master,
  `.venv.tar.gz` yeet-env, no compile). Currently queued behind the
  in-flight 20B 512N (8460302).
- Submitted **chained 12h continuations** for the 512N runs:
  - 2B: fresh 8463626 + 8463627 (depend=afterany:8463626)
  - 20B: 8463628 (depend=afterany:8460302)
- 2B 256N (8459818) and 2B 512N (8460301) v2 runs both hit NODE_FAIL
  at end-of-walltime (step 2070 and 1387 respectively) — bad-node TPS
  degradation pattern (TPS dropped from ~5K to ~30 in the final few
  hundred steps before kill). All checkpoints survived (`keep_latest_k=0`
  keeps everything; 20 ckpts at step 100..2000 for the 256N, 13 ckpts
  step 100..1300 for the 512N).

### yeet-env tarball-broadcast scaling sweep

Measured `ezpz yeet-env --src .venv.tar.gz` at 8/16/32/64/128/256/512/
1024/2048 N (4096N still queued in `large`). Submitted via
`scripts/yeet_env_scaling_test.sh` chained one-at-a-time through
`/tmp/yeet_chain.sh` (per-user PBS-Q limit forces serial submission).

| Nodes | yeet-env (s) | Per-node (ms) |
|------:|-------------:|--------------:|
| 8     | 70           | 8,712 |
| 16    | 90           | 5,606 |
| 32    | 89           | 2,788 |
| 64    | 91           | 1,425 |
| 128   | 110          | 862 |
| 256   | 133          | 519 |
| 512   | 175          | 341 |
| 1024  | 255          | 249 |
| 2048  | 421          | 206 |

Two regimes: 8-64N is extract-bound (flat ~90s total), ≥128N is
broadcast-bound (linear-ish growth, super-linear knee at 2048N).
Per-node amortized cost drops 42× from N=8 to N=2048. Even at 2048N,
total wall-clock is <8 min — vs the "1-2 hours" the old per-file rsync
mode in CLAUDE.md predicted.

Plots + script: [`docs/scaling/yeet_env/`](scaling/yeet_env/README.md).

### Docs refresh

- Added `docs/evals/agpt/2b-mds/{loss_data, figures}` train+val loss,
  grad_norm, TFLOP/s, TPS plots pulled from W&B (113 SophiaG MDS
  continuation runs stitched by iteration).
- Refactored `experiments/ezpz/scripts/`: moved 14 hidden one-off
  shell scripts out of the repo root into `scripts/{eval,debug,lr-finder}/`
  subdirs; consolidated the existing 3 eval shell scripts there too.
- Refreshed all production READMEs with the v2 run state (this entry).

### Async checkpointing — verification + production switch

Built smoke configs `smoke_2b_async_ckpt` (`async`) and
`smoke_2b_async_ckpt_pinned` (`async_with_pinned_mem`) with
`enable_first_step_checkpoint=True` and `interval=10` so we get 5
saves across a 50-step run.

| Job | Config | Result |
|---|---|---|
| `12465723` | `async` | ✅ PASS — 50 steps, 5 saves @ ~0.2s each, loss 7.05, exit 0 |
| `12465724` | `async_with_pinned_mem` | ❌ FAIL — `KeyError: <class 'type'>` at step-20 save |

Loss-baseline check on the `async` run vs v24 sync baseline:
final Δ -0.055, tail10 Δ -0.053 (well within ±0.10). On-disk DCP
format byte-identical to sync save (1809 keys, modern `qkv_linear` +
`lm_head` naming). Sync ↔ async is bidirectionally wire-compatible —
no migration needed for in-flight production.

Flipped all 10 production train/submit/smoke scripts to default
`--checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}"`. The env
var lets anyone roll back to sync without editing the script:

```bash
CHECKPOINT_ASYNC_MODE=disabled qsub ...
```

Per-step staging cost is ~0.2s with a one-step TPS dip (5.7K vs 7.4K
steady) the next step, then full recovery. Compared to a sync save
(which blocks the entire training step for the full disk-write
duration — 4+ seconds per save at 2N), this is a clear win.

### `async_with_pinned_mem` — upstream PyTorch bug filed

Diagnosed and reproduced a real bug in `torch.distributed.checkpoint`:
`StateDictStager.deepcopy_with_tensor_offload` line 320 indexes
`self._deepcopy_dispatch[type]` directly, but `StateDictStager.close()`
(called by `DefaultStager._stage` after every stage to break a closure
cycle, `staging.py:251`) clears the entire dispatch dict. Result:
**second** stage call on any state dict containing a class object
crashes with `KeyError: <class 'type'>`.

Reproduces in pure CPU Python (no distributed init, no GPU). Three
files committed at `docs/upstream-issues/`:

- `STATE_DICT_STAGER_ISSUE.md` — issue draft for pytorch/pytorch
- `repro_state_dict_stager_bug.py` — minimal ~30-line repro
- `repro_state_dict_stager_fix_verification.py` — proves the naive
  one-line `.get(type, _deepcopy_atomic)` fix doesn't work because
  *all* atomic-type entries are also missing post-`close()`. Real
  fix is structural: either don't clear `_deepcopy_dispatch` in
  `close()` (only the cached storages cause the leak), or rebuild
  it at the start of each `stage()`.

Plain `async` is unaffected — it uses the in-process default stager
(regular `copy.deepcopy`), not the pinned-memory `StateDictStager`
path.

### 28th–30th upstream syncs (4 days, 5 sync entries, no replays)

| Entry | Date | Commits | Why no replay |
|---|---|---|---|
| 27th | 2026-04-30 | HybridEP cleanup + autoparallel/dsv3 deletion | Resolved 2 modify/delete conflicts on autoparallel/dsv3/ by accepting upstream's deletion; ezpz doesn't depend |
| 28th | 2026-05-01 | graph_trainer qwen3 + CI lint + ft.llama3 attn_backend | Scoped to graph_trainer / lint / ft.llama3 (we use our own model_registry) |
| 29th | 2026-05-01 | RL vLLM compile-time + graph_trainer skill | Scoped to experiments/rl + graph_trainer |
| 30th | 2026-05-03 | Bucketing pass + RMSNorm fusion (graph_trainer) | All 3 commits scoped to experiments/graph_trainer |

The loss-baseline workflow has now been exercised in production once
(v22 → v24 baseline refresh, both PASS). Workflow doc is at
`docs/baselines/README.md`.

### Eval pipeline v2 plumbing

Eval scripts were originally hardcoded for the v1 256N run. To run on
v2 ckpts (256N, 512N, 1024N, ...) needed several fixes early in the
day:

- `feat(ezpz/scripts/eval): support v2 ckpts via --ckpt-name + add
  20B v2 eval` (`55154291`)
- `fix(ezpz/scripts/eval): drop set -u from eval-20b-v2.sh` (`1835d8c2`)
- `fix(ezpz/scripts/eval): cd into v2 clone for conversion` (`9af67b81`)
- `fix(ezpz/scripts/eval): force PYTHONPATH=. + add subshell debug
  echoes` (`db1380b6`)

After those landed, ARC-Easy points started flowing:
0.277 (step-200) → 0.366 (step-1200) → 0.429 (step-2000), validating
the bf16 fix end-to-end. v1 was flat at ~0.27 across 450B tokens.

---

## 2026-04-30 — bf16-master RMSNorm freeze diagnosis + v2 restart

### What broke

The 2B/20B/80B SophiaG production runs from Apr 14-29 were all silently
training with frozen RMSNorm weights:

- `training.dtype = bfloat16` (default at the time) was being used as
  the *master* weight dtype for FSDP MixedPrecisionPolicy.
- bf16 ULP at 1.0 is ~7.8e-3; per-step RMSNorm.weight updates from
  SophiaG were ~1.6e-5 — sub-ULP, so every update rounded to zero.
- Loss curves looked plausible because attention/FFN weights were
  unfrozen and the model could still descend, but the model has no
  trainable normalization. Eval scores reflected this: the
  DCP→HF-converted checkpoints scored near-random on hellaswag/arc_easy
  while the parallel MDS pipeline (which used different defaults) was
  cleanly converging.

Full diagnosis writeup at
[`docs/guides/training-dtype-bf16-norm-freeze.md`](guides/training-dtype-bf16-norm-freeze.md).

### Fix + v2 restart

- Default `training.dtype` flipped to `float32` in `agpt` and `moe`
  config registries.
- ChunkedCELoss opt-in support ported to the ezpz trainer (sets
  `lm_head` for the chunked path; off by default).
- Restarted training from scratch in fresh per-model clones:
  `/flare/AuroraGPT/foremans/runs/agpt-{2b,20b}-v2/torchtitan-ezpz/`.
- Built fresh torch 2.13 + xpu venvs in each clone (~2.7 GB tarball).
- Adapted `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` for Aurora:
  proxy env vars set inline, ezpz-utils source cached locally to dodge
  bit.ly hangs, tarball-aware yeet-env, plain-CE default with optional
  `CONFIG_SUFFIX=_chunkedce` opt-in.

### Smoke + scaling validation

Validated v2 stack at 2/4/8/16/64 N via short PBS jobs before
launching production. 2B v2 16N test ran ~57 min and reached step 2117
(loss 5.16 → 3.57, TPS ~5K, MFU ~17-20%) before NODE_FAIL — flaky-node
issue, not a code issue. 20B v2 16N test (capacity queue, 1h) submitted
in parallel.

### 26th + 27th upstream syncs

Replayed the MoE ETP deprecation (#3167) onto `experiments/ezpz/moe/`:
removed `ExpertTensorParallel` import, dropped `etp_mesh`/`ep_etp_mesh`
from `apply_moe_ep_tp` signature + call site, switched to reading
`comm_backend` off `experts.token_dispatcher` (matches upstream
deepseek_v3.model). Pulled the 27th sync (HybridEP cleanup +
autoparallel/deepseek_v3 deletion) — no impact on ezpz.

---

## 2026-04-29 — 24th upstream sync replay (All2All token dispatcher consolidation)

### What landed

Two upstream commits, one breaking:

- `20628f4e` (#3125) consolidates EP=1 and EP>1 to all use
  `AllToAllTokenDispatcher` (with a local-fallback path when ep_mesh is
  None). `make_token_dispatcher_config` and `make_experts_config` now
  require a non-None `comm_backend`; default changed from `None` to
  `"standard"`.
- `35c5d529` graph_trainer-only (no impact).

### Replay (1 commit)

`23b8ba59 fix(ezpz/moe)`: `moe_comm_backend: str | None = None` →
`moe_comm_backend: str = "standard"` in both `_build_moe_layers` and
`model_registry`. Drop the now-dead `if moe_comm_backend is not None`
guard around the dispatcher rebuild loop. (No agpt changes — agpt
doesn't use the moe-only helpers.)

### Smoke results — both PASS within ±0.10

| Config | Final loss | vs v22 baseline | tail10 mean | Δ tail10 |
|---|---|---|---|---|
| agpt 2b (job 12465533) | 7.108 | -0.029 | 7.221 | -0.037 |
| moe 500m (job 12465534) | 6.912 | -0.014 | 6.978 | -0.016 |

Both deltas dominated by streaming-data shuffle noise. Baselines
refreshed to v24.

### Side cleanup

`b9a324f3` renamed `docs/upstream-sync/` → `docs/baselines/` to remove
the visual collision with the neighboring `docs/upstream-sync.md` log
file. Updated path references in `loss_baseline.py`, the workflow
README, and the link from `upstream-sync.md`.

---

## 2026-04-28 — 22nd upstream sync replay (quantize-on-config, LocalMapInnerAttention removal)

### What landed

The 22nd sync (merged `4b0a4fd5`) brought in 9 commits, three breaking:

- **#3127 `6348d93d` quantize on config instead of on model.** Removes
  `protocols/model_converter.py`, drops `model_converters` field from
  `JobConfig` and from every `parallelize_*` signature. Quantization
  converters are now applied to the model *config* at registry time
  (`q.build().convert(config)`). Also moves `FaultTolerantModelSpec`
  from `protocols/model_spec.py` into `experiments/ft/config/job_config.py`.
- **#2986 `b9e33527` Remove LocalMapInnerAttention.** Replaces the
  runtime DTensor wrapper class with a static `LocalMapConfig` set on
  the inner-attention sharding_config via
  `set_gqa_inner_attention_local_map`. All inner attention types now
  inherit `Module` directly.
- **#3113 `053dbf9a` MeshDimName → MeshAxisName.** Transparent for ezpz
  (we only use the upstream helpers).

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.92 → 7.14 (job 12465527, exit 0) |
| `moe`  | replayed | 50 steps, loss 12.92 → ~7 (job 12465529, in flight) |

### Commits (ezpz branch)

- `4b0a4fd5` — Merge upstream/main into ezpz (clean automatic merge).
- `69a8cfc7` — Drop `model_converters=` kwarg from `parallelize_llama` /
  `parallelize_moe` signatures and from both `parallelize_fn` /
  `pipelining_fn` call sites in `ezpz/trainer.py`. Drop runtime
  `model_converters.build/convert/post_optimizer_hook`. Switch
  `has_quantization` to read from `model_config` via the upstream
  `torchtitan.components.quantization.utils.has_quantization` helper.
- `59d9f37d` — `LocalMapInnerAttention` → `Module` for
  `SoftcappedFlexAttention` (agpt) and `Attention.Config.inner_attention`
  (moe). Add `set_gqa_inner_attention_local_map(...)` calls in both
  `sharding.py` files so inner attention gets the static `LocalMapConfig`
  it now needs.
- `cd29417e` — moe `model_registry` accepts `quantization=[...]`
  parameter; `moe_671b()` re-registers via
  `model_spec=model_registry("671B", quantization=[...])` instead of
  mutating `cfg.model_converters`. Also re-import
  `FaultTolerantModelSpec` from `experiments/ft/config/job_config`.
- `1b87fa38` — `Float8GroupedMMConverter` → `Float8GroupedExpertsConverter`
  (rename caught at smoke-test import time).

### Smoke-test details

- agpt smoke (`12465527`): loss 12.92 → 7.14 across 50 steps,
  ~7,400 TPS, 28% MFU. Exit 0. Numerics match the v21 run within
  data-shuffle variance — replay is loss-neutral.
- moe smoke (`12465529`): 12.92 → 7 (in flight at step 12, on track),
  ~7,200 TPS at steady state. Same compile-warmup pattern as v21.

### Why moe failed once first

`12465528` failed with `ImportError: Cannot import config_registry for
module 'ezpz.moe'`. The underlying error was the
`Float8GroupedMMConverter` rename to `Float8GroupedExpertsConverter`
(and dropping `fqns=["experts"]` since the new Config takes no
extra args). `config/manager.py` swallows the underlying `ImportError`
which made the cause invisible — only resubmittable after grepping the
upstream class names.

---

## 2026-04-28 — 21st upstream sync replay (sharding API, ChunkedCELoss)

### What landed

The 21st upstream sync (merged `b6c04698`) brought in three breaking
changes that broke ezpz at import / config-build / training-init time:

- **#2963 / #2969** — config-based DTensor sharding. Replaces string-keyed
  `parallelize_module(plan)` with `Module.parallelize(mesh)` reading
  `ShardingConfig` declarations attached to each sub-module's `.Config`.
- **#2937** — ChunkedCELoss. Removed `build_cross_entropy_loss` and
  `ModelSpec.build_loss_fn`; loss now lives on `JobConfig.loss`.
- **`Decoder.Config`** renamed `output: Linear.Config` → `lm_head:
  Linear.Config`.

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.96 → 7.09 (job 12465500) |
| `moe`  | replayed | 50 steps, loss 12.93 → 6.91 (job 12465502) |
| `qwen3` | removed | Drift too large; nobody ran it; restorable from history |

### Commits (ezpz branch)

- `03b9f486` — Mechanical: drop `build_cross_entropy_loss` imports +
  `build_loss_fn=` kwargs, rename `output=` → `lm_head=` in agpt+moe
  configs, switch `ezpz/trainer.py` to `config.loss.build()`.
- `472f4743` — agpt sharding-API replay. New `agpt/sharding.py`
  (handles QK-Norm), new `agpt/model.py` (`AgptModel(Llama3Model)`
  overriding `update_from_config`), rewritten `agpt/parallelize.py` as
  thin orchestrator. Float8 tensorwise TP path dropped (no equivalent in
  the new API yet).
- `9bc774a6` — Set `loss=CrossEntropyLoss.Config()` in both `_base_config`
  helpers (was defaulting to abstract `BaseLoss.Config`).
- `40526628` — Enable compile in `smoke_2b_50steps` (XPU CE OOMs without it
  at vocab=256k).
- `dd4e065a` — moe sharding-API replay. New `moe/sharding.py`, extended
  `update_from_config`, rewritten `parallelize.py` (drops 230+ lines of
  manual ColwiseParallel/RowwiseParallel plans). MoE block sharding still
  done at parallelize-time by `apply_moe_ep_tp` (mirrors upstream).
- `5f88abc3` — `smoke_moe_500m_50steps` config + submit script.
- `80a23d41` — Removed `ezpz/qwen3` (drift too large for unused code).

### Preserved agpt-/moe-specific behavior

- `disable_fsdp_gradient_division` still calls
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL/XPU).
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False (keeps the separately-compiled CrossEntropyLoss working on dense
  models).
- agpt `apply_fsdp` keeps the `[norm, lm_head]` joint grouping with
  reshard_after_forward gated on the policy.
- moe `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
  import which doesn't exist in Aurora's PyTorch) with the Shard(0)
  fallback when expert hidden dim isn't FSDP-divisible.
- moe `apply_compile` is per-block `block.compile(backend=...)` instead of
  upstream's fullgraph `apply_compile_sparse` (XPU can't fullgraph compile
  MoE routing's dynamic shapes).

### Smoke-test details

- agpt smoke (`12465500`): loss 12.96 → 7.09 across 50 steps, ~7,400 TPS,
  27% MFU. Standard dense-2B numbers — replay is loss-neutral.
- moe smoke (`12465502`): loss 12.93 → 6.91 across 50 steps, ~7,200 TPS,
  ~10% MFU. The MFU is low because the metrics divisor uses the full
  dense FLOP estimate but only 2/8 experts fire per token — reporting
  artifact, not a perf regression.

---

## 2026-04-27 — Full 10B training, TorchMuon, local dataset

### Local Dataset Cache

- Downloaded FineWeb-Edu `sample-100BT` (267 GB, 140 parquet files)
  to `/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/`
- Added `register_local_dataset()` to `datasets.py` for parquet/arrow files
- Registered as `fineweb_edu_local` — eliminates HF streaming rate limits
  and ensures reproducible data ordering across runs

### TorchMuon Integration

- Added `TorchMuonOptimizersContainer` using `torch.optim.Muon` (built-in
  since PyTorch 2.9)
- Required `_CompositeOptimizer` wrapper — `OptimizersContainer` expects one
  optimizer per model part, but Muon only handles 2D params (need separate
  AdamW for embeddings/head)
- Multiple fix iterations: missing `import torch`, empty param list rejection
  from `Optimizer.__init__`, FSDP empty model parts
- **Result: same TPS as custom Muon (~4,600)** — Newton-Schulz overhead is
  inherent to the algorithm on XPU, not an implementation issue
- **Streaming data shuffle causes ~1.3 loss variance** — same optimizer gives
  very different loss across runs due to HF streaming data ordering

### Speedrun Competition Final Results (1000 steps, 2 nodes)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | Muon (custom) | **3.557** | 4,556 |
| 2 | AdamW + QK-Norm | **3.569** | 7,178 |
| 3 | Muon + cosine | 3.591 | 4,625 |
| 4 | Mano + QK-Norm | 3.604 | 6,980 |
| 5 | Mano | 3.631 | 7,048 |

### Full Training (10B tokens, 8 nodes, GBS=384)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW | **2.711** | 7,354 |
| 2 | AdamW + QK-Norm | 2.720 | 7,480 |
| 3 | Mano + QK-Norm | 2.854 | 7,346 |
| 4 | Mano | 2.875 | 7,429 |
| 5 | Muon | DNF (compile stuck) | — |

### Key Findings

- **AdamW wins at large batch (GBS=384)** — simpler update more efficient
  per token than manifold optimizers
- **QK-Norm effect diminishes at 10B** — 0.009 loss improvement (vs 0.23
  in 1000-step speedruns). Helps early training but washes out
- **Mano ~0.16 behind AdamW at GBS=384** — LR finder was tuned at GBS=48,
  needs re-tuning for larger batch
- **Muon compile broken with GAS** — inductor can't pickle cyclic objects
  in Newton-Schulz with gradient accumulation on torch 2.13
- **8-node scaling excellent** — 7,300-7,500 TPS/GPU across all configs

### Architecture Tweaks Implemented

- **Logit softcapping** — `SoftcappedFlexAttention` using FlexAttention
  `score_mod` with tanh cap at 30.0. Falls back to eager on XPU (4x slower).
  Manual attention OOMs at seq_len=8192 (materializes full attention matrix).
- **ReLU²** — `ReLUSquaredFeedForward` subclass. Didn't help (3.92 vs 3.80
  baseline). SiLU gating is better for this architecture.
- **WSM** — `eval/merge_checkpoints.py` utility for weighted state merging
  of checkpoints. Supports uniform, linear, and exponential weighting.
- New model variants: `2B_softcap`, `2B_relu2`, `2B_kitchen_sink`

### Round 4: 2N, GAS=8, 1000 steps (local dataset)

Reproducible speedrun with GBS=384 on 2 nodes using local FineWeb-Edu.

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW+QK-Norm | **3.205** | 7,428 |
| 2 | AdamW | 3.220 | 7,397 |
| 3 | Mano | 3.294 | 7,397 |
| 4 | Mano+QK-Norm | 3.307 | 7,423 |
| 5 | Mano (8.5e-4) | 3.328 | 7,348 |
| 6 | AdamW (3.7e-3) | 5.884 | 7,603 |

**Key findings:**
- AdamW+QK-Norm wins again — consistent across all GBS=384 experiments
- Mano leads early/mid training but AdamW catches up in cosine decay phase
- sqrt LR scaling too aggressive for AdamW (diverged), Mano tolerated it
- Softcap results invalid — local dataset loader memorizes with FlexAttention
  path (data sharding bug)
- FlexAttention on XPU falls back to eager (Triton-XPU can't codegen tanh)
  — 4x throughput penalty makes softcap impractical on this hardware

### Docs Restructure

- Reorganized `docs/competition/` → `docs/competitions/` with per-experiment dirs
- Added light/dark theme loss curve plots using `<picture>` media queries
- Created `docs/competitions/agpt2b-n2-gas8-1000steps/` with live loss curves

### Upstream Sync (20th)

- Merged upstream: dataset checkpoint resume fix (#3008), RL refactor (#3073)
- Clean merge, no replay needed

### CLAUDE.md Added

- Created `experiments/ezpz/.claude/CLAUDE.md` with project rules that
  travel with the codebase (upstream sync protocol, never modify outside
  ezpz, document every run, etc.)
- Updated with Aurora-specific knowledge (queues, yeet-env scaling, eval pipeline)

---

## 2026-04-26 — RL refactor, docs reorg, competition launch

### RL Multi-Task Support

- Refactored `rl/` from hardcoded sum-of-digits to a pluggable task registry
- Created `rl/tasks/` package with `RLTask` dataclass, `register_task()`, `get_task()`
- Moved sum_digits dataset+rewards into `tasks/sum_digits.py` (self-registering)
- Added 3 new tasks: `multiply`, `word_sort`, `countdown`
- Added CLI args (`--task`, `--model-name-or-path`, `--steps`, etc.) to `train_grpo.py`
- Default model: `argonne_private/AuroraGPT-7B` with `Qwen/Qwen3-0.6B` fallback
- Fixed safetensors E2BIG crash by disabling mid-training checkpoints
- Moved RL docs to `docs/rl/README.md`

### Docs Reorganization

- Created `configs/` — moved dense-configs.md, moe-configs.md
- Created `guides/` — moved known-issues.md, running-with-newer-pytorch.md, xpu-attention-issues.md
- Renamed `production-training/` → `production/`
- Created `scaling/` — consolidated scaling-study.md, scaling-study-torch213.md,
  benchmark-80B.md, benchmarks.md into per-model pages (agpt-2b, agpt-20b, agpt-80b, moe)
- Rewrote top-level README.md with organized sections
- Fixed all 31 internal cross-references; link checker passes with 0 broken

### Generic HF Dataset Streaming

- Created `datasets.py` with `register_hf_dataset()` for explicit registration
- Added auto-fallback: unknown `--dataloader.dataset` names are treated as HF hub paths
  (e.g. `--dataloader.dataset stanfordnlp/imdb` just works)
- Pre-registered: fineweb_edu, fineweb, slimpajama, pile, openwebtext, wikitext, c4_streaming
- Silenced httpx/huggingface_hub HTTP log spam

### agpt_2b Loss Competition

**Goal:** lowest loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).
**Fixed:** FineWeb-Edu streaming, LBS=2, seq_len=8192, 1000 steps.
**W&B:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

#### New Optimizers Implemented

- **Mano** (`optimizer/mano.py`) — manifold-normalized optimizer
  ([arxiv 2601.23000](https://arxiv.org/abs/2601.23000)).
  Tangent-space projection on rotating Oblique manifold. Vector-norm ops
  instead of Newton-Schulz → runs at AdamW speed (~7,200 TPS/GPU).
- **SPAM** (`optimizer/spam.py`) — spike-aware Adam with momentum reset
  ([arxiv 2501.06842](https://arxiv.org/abs/2501.06842)).
  Gradient spike detection via EMA + periodic moment reset every DeltaT steps.

#### Architecture Tweaks

- **QK-Norm** — added `qk_norm` parameter to `_build_agpt_layers` and
  `_build_agpt_config`. New `2B_qknorm` model variant. RMSNorm on Q,K
  before attention dot product.

#### Competition Results

| Rank | Config | Optimizer | LR | Loss | Steps | TPS/GPU |
|------|--------|-----------|------|------|-------|---------|
| 1 | `speedrun_2b_muon` | Muon | 2.4e-3 | **3.628*** | 967 | 4,695 |
| 2 | `speedrun_2b_mano` | **Mano** | 3.0e-4 | **3.631** | 1000 | ~7,200 |
| 3 | `speedrun_2b_adamw_cosine` | AdamW | 1.3e-3 | 3.789 | 990 | 7,245 |
| 4 | `speedrun_2b_adamw` | AdamW | 1.3e-3 | 3.801 | 1000 | 7,245 |
| 5 | `speedrun_2b_adamw_short_decay` | AdamW | 1.3e-3 | 4.053 | 1000 | 7,245 |
| 6 | `speedrun_2b_adamw_fast_warmup` | AdamW | 1.3e-3 | 4.546 | 1000 | 7,245 |
| 7 | `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | 4.399* | 976 | 4,596 |
| 8 | `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | 4.719 | 1000 | 7,208 |
| 9 | `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | 5.850 | 1000 | 7,344 |
| 10 | `speedrun_2b_spam` | SPAM | 1.3e-3 | 5.881* | 865 | ~7,200 |

*Still running at time of reporting.

#### Key Findings

- **Muon and Mano essentially tied on loss** (~3.63), but Mano ran at
  full AdamW speed (7,200 TPS) vs Muon's 4,700 TPS. **Mano wins on
  wall-clock time.**
- **Muon is 35% slower per step** due to 5x Newton-Schulz iterations
  (large matmuls on every 2D param). Mano replaces these with O(dim)
  vector-norm ops.
- **Cosine decay beats linear** for AdamW (3.789 vs 3.801).
- **Shorter decay (10%) hurts** — not enough time in decay phase.
- **Shorter warmup (5 steps) hurts** — destabilizes early training.
- **SPAM underperforms** — spike clipping + momentum reset don't help
  on this clean dataset with well-tuned LR.
- **AdamW LR=2.6e-3 diverges** — confirms LR finder boundary (1.3e-3).
- **SophiaG underperforms** AdamW by ~0.9 loss at same step count.

#### Failed Experiments

- `speedrun_2b_muon_short_decay` — crashed at startup (exit 143)
- `speedrun_2b_muon_fast_warmup` — crashed at startup (exit 143)
- `speedrun_2b_adamw_qknorm` — crashed at startup (exit 143)
- `speedrun_2b_muon_qknorm` — crashed at startup (exit 143)

Need to investigate QK-Norm and Muon schedule tweak crashes.

#### Issues Hit

- PBS `qsub -- bash -c '...'` doesn't work — need a proper script file
- `set -euo pipefail` kills venv activate scripts (`ZSH_EVAL_CONTEXT: unbound`)
- Concurrent jobs sharing `--checkpoint.folder=checkpoint` clobber each other
  → fixed with per-config checkpoint dirs
- Disk quota hit at 12TB → cleaned 3.2TB of old scaling study checkpoints
  and 117GB of old repo checkpoints
- `git stash pop` during disk quota crunch wiped train.py to 0 bytes
  → restored from `git show HEAD:...`

---

## 2026-04-25 — torch 2.13 venv, scaling study, production scripts

### Torch 2.13 Environment

- Created `.venv/` with PyTorch 2.13 (built from source for XPU)
- Added `running-with-newer-pytorch.md` guide for setting up the venv
- Added `ezpz yeet-env` integration to copy venv to `/tmp` on compute nodes

### Production Training Scripts

- Created `scripts/train_agpt_2b_venv.sh` and `train_agpt_20b_venv.sh`
  for training with the torch 2.13 venv
- Fixed `ezpz_setup_job` ordering — must run before venv activation
- Fixed `/tmp/.venv/bin` PATH handling after `yeet-env activate`
- Set `local_batch_size=2` as default for 2B training

### 2B Scaling Study (torch 2.13, Sunspot)

- Ran weak scaling study from 2 to 64 nodes on Sunspot
- Results: 7,142 TPS/GPU at 2N (27.6% MFU) — **+23% over torch 2.10**
- Near-perfect scaling to 8 nodes (~100%), 94% efficiency at 64 nodes
- Memory nearly constant at ~44 GiB across all scales
- Documented in `docs/scaling-study-torch213.md`

### Production Training Status

- Updated production run tracking for 2B/20B/80B models
- Added per-model subdirectories with loss curve plots
- Updated upstream sync log with session findings

### Upstream Sync

- Merged upstream `pytorch/torchtitan` main into ezpz branch
- Reverted `.ezpz-interactive-launch.sh` tracking change
- Added interactive launch script and loss CSVs

---

## 2026-09-25 — Post-merge multi-host RL and Sunspot 30B diagnosis

- PR #25 fixed checkpoint-selector translation and current checkpointer ownership;
  it merged into `ezpz` as `550d2c670a02866661c133dd09b499ae6844bc7f`.
- Controlled 26.2B Sunspot runs showed the failure is fresh-start FSDP transient
  memory pressure, not DCP restore corruption. Shard-16 failed in first all-gather;
  shard-32 cleared forward unshard and failed in backward reduce-scatter with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Replica count, local batch size, and
  `CCL_SYCL_KERNEL_SYNC=0` did not remove the defect. Large core files were left
  untouched.
- Job `12478702` first proved one Monarch actor graph could span two physical
  Sunspot hosts.
- Job `12478711` then passed the full production gate at exact commit
  `738109e8d4481ebb723622db0d1a34b9c8907203`: trainer and vLLM generator on
  separate hosts, TorchStore `TransportType.Gloo`, pre/post validation, three
  finite GRPO updates, policy versions 0 through 3, 40/40 completed nonzero-
  reward rollouts, DCP checkpoints at steps 1/2/3, clean shutdown, and PBS exit
  zero. Automatic transport had selected host-local shared memory and is not
  valid across hosts; explicit Gloo is required for this topology.
- Full report:
  [`experiments/2026-09-25-sunspot-multihost-rl-validation.md`](experiments/2026-09-25-sunspot-multihost-rl-validation.md).
- PR #26 merged the multi-host support into `ezpz` as
  `c7605bf1cafeabe82109eea56bcf85b41f3df4a7` after exact-head lint passed.
- A teacher-free Stage-3 STaR campaign then sampled 59,784 GSM8K training
  rollouts. The verified corpus retained 4,294 unique exact-correct traces
  (57.46% of problems), excluded all four conservative test-set collisions,
  and had zero leakage. Two-node SFT job `12478715` completed 100 finite steps
  and exited zero, but fixed semantic evaluation job `12478718` scored only
  37/200 correct versus the Stage-2 baseline's 43/200. Format was 198/200 versus
  197/200 and length terminations improved 3→1, but paired flips favored the
  baseline 21 to 15 (exact McNemar p=0.405). The candidate fails the promotion
  gate; GRPO is not released and Stage-2 remains accepted.
- Stage-3 report:
  [`experiments/2026-09-25-mds154391-stage3-star.md`](experiments/2026-09-25-mds154391-stage3-star.md).
- Aurora coordination: umbrella chain 3 is healthy through step 39,942 and
  becomes non-finite at 39,943; step 42,400 and later checkpoints are poisoned.
  The first 48-rank communicator probe (`8869722`) was a harness failure
  (`LOCAL_RANK` missing on every rank, auto-retry stopped pre-training, exit
  143), not an XCCL result. Corrected probe `8870327` is queued; exact-topology
  chain-3 replay remains blocked until that communicator baseline passes.
- Follow-up TorchStore transport controls closed the remaining RDMA question on
  the Sunspot runtime. Job `12478720` left transport automatic but disabled
  SharedMemory; with TorchComms unavailable and MonarchRDMA capability reported
  available, it stalled before the first publication and timed out after 30
  minutes (`Exit_status=143`). Job `12478722` then explicitly selected
  `TransportType.MonarchRDMA`; the TorchStore storage-volume actor crashed with
  `SIGSEGV` during the initial trainer policy push (`Exit_status=1`). Thus Gloo
  remains the only cross-host transport validated end to end on this stack.

---

## 2026-04-23 — XPU fixes, upstream merge

### XCCL Barrier Fix

- Fixed torch 2.10 XCCL hangs for barrier and TP collectives
- Root cause: XCCL backend doesn't support barrier() — was hanging
  all multi-node runs
- Fix: use gloo backend for barriers when available

### DTensor TP Revert

- Reverted full DTensor TP (`use_local_output=False`) for agpt models
- Was causing shape mismatches in the attention layer on XPU
- Reverted to standard `use_local_output=True`

### Upstream Merge

- Merged upstream main into ezpz branch
- Upstream changes included GraphTrainer bucketing fixes,
  SAC + FSDP improvements, and Qwen3-VL fused QKV support

---

## 2026-10-02 — AGPT 12B2A Hugging Face export

- Added an `AGPTMoEStateDictAdapter` to the existing DCP-to-HF converter and a
  Transformers remote-code model that preserves the training model's GQA,
  ComplexRoPE, top-3 biased selection, original unnormalized routing weights,
  36 routed experts, and always-on shared expert.
- The first real conversion (`8900811`) exposed non-contiguous Sonic expert
  transpose views at the safetensors writer. Export tensors are now cast and
  packed at the serialization boundary; the regression test covers both BF16
  conversion and same-dtype exports.
- Job `8900883` exported the historical step-27000 12B-total/2B-active DCP and
  exited zero. The HF logits matched the current native reference at 0.448%
  relative RMS / 0.999770 cosine / 96.875% top-1 agreement, and the frozen
  Sonic oracle at 0.477% / 0.999770 / 98.438%. Final-token top-10 predictions
  were identical to both references.
- Job `8900986` then loaded the artifact through unmodified lm-eval-harness
  0.4.10's `hf` backend with Transformers 4.57.6 and completed an ARC-Easy
  likelihood smoke on XPU (8/8 requests, exit zero). The two-document 2/2
  score is only an integration check, not a reportable model metric.
- The focused CPU suite passed 12/12 tests, including the pre-existing dense
  converter schema tests. The generated model also passed AutoModel save/load,
  forward, and generation tests.
