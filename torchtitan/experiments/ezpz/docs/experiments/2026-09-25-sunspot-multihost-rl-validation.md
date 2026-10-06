# Sunspot multi-host Monarch, TorchStore, and vLLM validation

Date: 2026-09-25

> [!NOTE]
> **Superseded transport status (2026-10-01).** This page remains the immutable
> record of the first successful two-host run, which explicitly forced Gloo.
> Automatic locality resolution was subsequently root-caused and fixed:
> inherited `HOSTNAME` collapsed distinct hosts, while unqualified XPU
> MonarchRDMA/XCCL capability probes preempted Gloo. Final exact-head automatic
> job `12479179` completed the same full gate with PBS exit 0. The current
> topology-aware policy is same-host SharedMemory and cross-host Gloo; see the
> [current Monarch production runbook](../production/rl/monarch.md).

## Result

Merged `ezpz` plus the multi-host validation additions at commit
`738109e8d4481ebb723622db0d1a34b9c8907203` completed a real two-host RL
actor graph on Sunspot. PBS job `12478711` placed the trainer on host index 0
and the vLLM generator on host index 1, transferred policy weights with
TorchStore's Gloo transport, completed three finite GRPO optimizer updates,
ran pre- and post-training generation, wrote three DCP checkpoints and a
40-row rollout artifact, shut down cleanly, and exited zero.

This closes the gap left by one-node job `12478621`: policy publication and
vLLM consumption now work when trainer and generator reside on different
physical hosts in one Monarch actor graph.

## Upstream `f359667` exact-head confirmation

Job `12479032` repeated the complete two-host acceptance gate after the
`f359667` integration, on commit `a893208c8f94c41c16367e1b8cf572db0642a8a3`.
It used the same explicit TorchStore Gloo topology and produced:

```text
RL_MULTIHOST_VERDICT: ok rows=40 versions=[0, 1, 2, 3]
grads=[0.35, 0.34, 0.28]
losses=[0.037, 0.0059, 0.24]
pushes=4 pulls=4
```

All 40 rollouts completed without truncation, checkpoints were written at
steps 1, 2, and 3, and PBS exited 0. The first exact-head attempt `12479031`
failed during cross-host actor attachment because the second host's cold import
consumed almost all of the 120-second attach window. Raising only that timeout
to 300 seconds allowed the otherwise identical gate to complete.

## Exact contract

- Commit: `738109e8d4481ebb723622db0d1a34b9c8907203`
- Job: `12478711`
- Runtime: `/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214`
- Worktree: `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-multihost-rl-738109e8d`
- Topology: two PBS hosts, one trainer XPU actor on host 0, one vLLM generator
  XPU actor on host 1
- TorchStore transport: `TransportType.Gloo`
- Starting model:
  `/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/stage2/agpt2b-mds154391-step600-gsm8k-r1cot-2n-r2/final`
- Training: three GRPO updates, four prompts per update, four samples per prompt,
  zero target off-policy steps, LoRA rank 8, fp32 generation
- Output:
  `/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/grpo/agpt2b-multihost-sync-12478711`

## Acceptance evidence

```text
MULTIHOST_RL_PLACEMENT trainer_host_index=0 generator_host_index=1 trainer_world_size=1 generator_world_size=1
Forcing TorchStore weight transport to TransportType.Gloo
Validation | Step: 0  validation/response_length/mean: 117.9
Train | Step: 1  loss/mean: 0.14  rollout_reward/_mean: 0.50  trainer/grad_norm/mean: 0.32
Train | Step: 2  loss/mean: 0.15  rollout_reward/_mean: 0.48  trainer/grad_norm/mean: 0.34
Train | Step: 3  loss/mean: 0.13  rollout_reward/_mean: 0.46  trainer/grad_norm/mean: 0.33
Validation | Step: 3  validation/response_length/mean: 120.4
MULTIHOST_TORCHSTORE_VLLM_OK
MULTIHOST_RL_DONE rc=0
PBS Exit_status = 0
```

`rollout_samples.jsonl` contains 40/40 completed rows, all with nonzero reward.
Observed policy versions were 0, 1, 2, and 3. DCP checkpoints were written at
steps 1, 2, and 3; each data shard is approximately 7.96 GB and has matching
metadata.

## Controlled failures

The failed jobs were necessary to isolate the production requirements:

- `12478706`: stale import of upstream's removed
  `breakable_cuda_graph_env`; failed before actor attachment.
- `12478707`: full RL startup exceeded Monarch's default attach config-push
  timeout.
- `12478708`: attachment passed, but integer host slices removed the `hosts`
  dimension and caused `KeyError: 'hosts'` in role provisioning.
- `12478709`: role placement and vLLM initialization passed, but automatic
  TorchStore selection misclassified the remote storage volume as local and
  entered the SharedMemory path; the generator rejected it with
  `Shared memory storage not found`. This is a current locality-resolution bug,
  not evidence that TorchStore's intended automatic cross-host order prefers
  SharedMemory.
- `12478710`: forced Gloo completed cross-host vLLM pre-validation, but the
  non-controller rank's default five-minute `FileStore.get()` timeout killed the
  otherwise healthy MPI world before training completed.
- `12478720`: automatic selection with SharedMemory disabled remained at the
  first TorchStore operation for 30 minutes, completed no publication or
  rollout, and terminated with `Exit_status=143`. TorchComms was unavailable;
  MonarchRDMA was the expected available automatic candidate, but no
  `[ts-transport] resolved=...` line was emitted before the stall.
- `12478722`: explicit `TransportType.MonarchRDMA` confirmed backend selection,
  then the TorchStore storage-volume actor died from `SIGSEGV` during the
  initial trainer publication. No rollout or optimizer step completed; PBS
  recorded `Exit_status=1`.

At the time of this run, the implementation used Monarch's scheduler-SPMD
`host_mesh_from_store`, dimension-preserving host slices, a 120-second attach
timeout, a 30-minute coordination-store timeout, and explicit Gloo as the
validated workaround. The 2026-10-01 fix recorded above supersedes only that
transport-selection workaround; this job's actor/update/checkpoint evidence
remains valid.

## Scope of the claim

This proves cross-host actor routing, vLLM generation, TorchStore policy
synchronization, optimizer execution, checkpoint writing, and clean shutdown for
the two-host AGPT-2B validation topology. It does not claim that forced XCCL
or MonarchRDMA TorchStore transfer works; those controls remain failed. It also
does not establish a model-quality improvement, which requires paired semantic
evaluation.
