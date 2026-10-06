# Production RL with Monarch, TorchStore, and vLLM on XPU

**Current production runbook. Last updated: 2026-10-01.**

> [!IMPORTANT]
> This page is the canonical operator guide for the current upstream-style
> TorchTitan RL path. Older TRL server-mode, oneAPI 2025.3, Torch 2.12/2.13,
> external-fork, and pre-`next-eval` instructions are retained only as historical
> reproductions. Do not copy their environment setup into a new production run.

## Current status

| Machine / queue | Status | Validated contract |
|---|---|---|
| Sunspot `workq` | **Validated end to end** | Two physical hosts; one trainer actor on host 0; one vLLM generator actor on host 1; Monarch scheduler-SPMD mesh; TorchStore Gloo transport; three finite GRPO updates; pre/post generation; checkpoints; clean shutdown. Job `12478711`, PBS exit 0. |
| Aurora `next-eval` | **Runtime contract established; multi-host RL gate still required** | oneAPI 2026.1 + an image-independent, versioned `.venv.next-eval` archive; no frameworks Conda environment; 12 flat XPU tiles per node. Do not call Aurora production-ready until the same actor, weight-sync, optimizer, artifact, and shutdown gates pass there. |

The merged implementation landed through PR #26. The authoritative Sunspot
hardware report is
[Sunspot multi-host Monarch, TorchStore, and vLLM validation](../../experiments/2026-09-25-sunspot-multihost-rl-validation.md).

## Architecture

The controller builds one Monarch actor graph over scheduler-launched host
workers:

```text
PBS allocation
  └─ ezpz launch: one rank per host
       └─ host_mesh_from_store(): one Monarch host mesh
            ├─ host 0 slice → trainer ProcMesh → TrainerActor
            └─ host 1 slice → generator ProcMesh → vLLM GeneratorActor

TrainerActor ── TorchStore/Gloo policy publication ──> GeneratorActor/vLLM
      │                                                    │
      └──────────── gradients + optimizer updates <─ rollouts/rewards
```

The production entry point is
`torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py`.
It supplies separate trainer/generator `HostMeshes` to the existing
`train_upstream.spawn_proc_mesh()` path; model, controller, reward, and
checkpoint logic remain the normal TorchTitan RL implementation.

## Automatic transport locality and the former forced-Gloo workaround

TorchStore's intended automatic (`TransportType.Unset`) preference order is:

```text
same-host SharedMemory
→ TorchComms / MonarchRDMA
→ XCCL when available
→ Gloo
→ MonarchRPC
```

The successful one-host automatic runs (`12478538`, `12478621`) logged
`TransportType.Unset`. Because trainer and generator were colocated, the
expected first choice was SharedMemory; those jobs do **not** establish that
automatic selection chose XCCL.

The explicit-XCCL control `12478537` held the model and one-host actor topology
fixed but hung during the first publication after flatten/cast. Thus forced
XCCL is not a passing result on this stack.

For the two-host topology, automatic-selection job `12478709` incorrectly
entered the SharedMemory path even though the generator was remote. The remote
generator correctly rejected the inaccessible shared-memory volume:

```text
Shared memory storage not found. This may indicate the storage volume is on a different host.
```

### Root cause: `HOSTNAME` is inherited, not per-host

This was a locality-classification defect, and it is now diagnosed. TorchStore
decides co-location by string-comparing
`os.environ.get("HOSTNAME", socket.gethostname())` on the client and on the
storage volume (`torchstore.utils.get_local_hostname`, and
`StorageVolume.get_id`). `HOSTNAME` is an ordinary exported shell variable, so
MPI/scheduler launchers propagate the *submitting shell's* value to every
remote rank.

Two-host Sunspot probe `12479166` measured this directly — one process per
node, stdlib only:

```text
LAUNCH HOSTNAME=x1922c6s3b0n0 real=x1922c6s3b0n0
PROBE_ROW {"env_hostname": "x1922c6s3b0n0", "real_hostname": "x1922c6s3b0n0", "torchstore_resolved": "x1922c6s3b0n0"}
PROBE_ROW {"env_hostname": "x1922c6s3b0n0", "real_hostname": "x1922c6s5b0n0", "torchstore_resolved": "x1922c6s3b0n0"}
```

Both ranks resolve to `x1922c6s3b0n0`, so `is_local_to_volume()` returns true
for a volume that is physically on another node and automatic selection picks
SharedMemory. The same mechanism has an inverse failure mode: Sunspot's batch
shell exports `HOSTNAME` as the FQDN while `socket.gethostname()` is short, so
a genuinely local volume can look remote and silently take a slower transport.

**Fix.** `torchtitan/torchstore_compat.py` gains `repair_hostname_env()`, which
aligns `HOSTNAME` with the real hostname of the running process. It is called
in the RL actor bootstrap (before any TorchStore import) and in
`_torchstore_strategy_from_env()`. It is a no-op when `HOSTNAME` is absent or
already correct, and it never invents a value. Contracts, including the
measured two-host values, are locked in
`tests/rl/unit_tests/cpu/test_torchstore_locality.py`.

Reusable diagnostic:
`torchtitan/experiments/ezpz/rl/scripts/grpo/torchstore_locality_probe.pbs`.

The actor-local repair passed on two physical hosts in job `12479169`: each
actor corrected the inherited launcher FQDN to its own short physical hostname,
the two resolved names were distinct, and the job emitted
`TORCHSTORE_LOCALITY_OK` with PBS exit 0.

Two follow-up controls tested the remaining automatic/RDMA question directly:

| Job | Selection | Result |
|---|---|---|
| `12478720` | `Unset` with SharedMemory disabled | The log remained at TorchStore initialization for 30 minutes with no completed publication, generation, or optimizer step; the coordination timeout terminated the job (`Exit_status=143`). In this runtime TorchComms availability was false and MonarchRDMA availability was true, so MonarchRDMA was the expected automatic candidate, but TorchStore did not emit its resolved-transport log before the stall. |
| `12478722` | Explicit `TransportType.MonarchRDMA` | Backend selection was confirmed in the log. The TorchStore storage-volume actor crashed with `SIGSEGV` during the initial trainer `push_model_state_dict`; no rollout or optimizer step completed (`Exit_status=1`). |

TorchComms was not testable in this environment: the `torchcomms` package was
absent and both TorchComms availability probes returned false. MonarchRDMA was
available at capability-probe level, but the explicit end-to-end control failed.

The original forced-Gloo fallback was:

```bash
export TORCHTITAN_TORCHSTORE_TRANSPORT=gloo
```

That global pin is no longer required. The repaired XPU automatic policy is
topology-aware:

```text
same host  → SharedMemory
cross host → Gloo
```

MonarchRDMA and XCCL remain explicit experimental controls, but are excluded
from automatic XPU selection on this runtime. Capability probes had reported
both as candidates despite neither being a qualified TorchStore weight-transfer
backend here: explicit MonarchRDMA `12478722` and automatic `12479170` crashed
with SIGSEGV on the initial pull; explicit XCCL `12478537` hung there, and
automatic controls `12479171`/`12479174` reproduced that boundary after RDMA
was excluded.

Matched explicit-Gloo control `12479172` and final exact-head automatic job
`12479177` both completed the full two-host gate. `12479177` used
`TransportType.Unset`,
completed 40/40 rollouts across policy versions 0–3, four trainer
publication/generator pulls, three finite nonzero-gradient updates, DCP
checkpoints at steps 1–3, clean shutdown, and PBS exit 0. Final update metrics:

```text
step 1: loss=0.0480, grad_norm=0.38
step 2: loss=0.0058, grad_norm=0.27
step 3: loss=0.0500, grad_norm=0.39
RL_MULTIHOST_VERDICT: ok rows=40 versions=[0, 1, 2, 3]
  grads=[0.38, 0.27, 0.39] losses=[0.048, 0.0058, 0.05]
  pushes=4 pulls=4
```

The committed two-host launcher now defaults to `TORCHSTORE_TRANSPORT=auto`;
operators can still request `gloo`, `xccl`, or `monarch_rdma` explicitly for a
controlled comparison.

### Runtime used for revalidation

The protected `rl-monarch-torch214` runtime had 41 RECORD-listed files missing
across 30 packages, including `click.core`, Torch FX unification, SymPy, Triton,
and vLLM modules. It was not modified. A separate clone at
`/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001`
was repaired exclusively from exact RECORD-hash-matching uv cache objects, with
`py-cpuinfo==9.0.0` reinstalled `--no-deps` for its missing console script and
the previously qualified BlendCorpus commit `50502b0c9de3` installed
`--no-deps`. The resulting full RECORD existence audit reported zero missing
files, and the Torch/ezpz/Monarch/TorchStore import closure passed before jobs
`12479169`–`12479179` ran.

## Required evidence before promotion

A scheduler state or model load is not a pass. Require all of the following:

1. exact checkout SHA asserted by the batch script;
2. clean tracked worktree;
3. trainer and generator placement on distinct host identities;
4. `TORCHSTORE_TRANSPORT=auto` requested and the XPU topology policy installed
   (`SharedMemory` only when local, otherwise `Gloo`); explicit transport
   controls must log the requested backend;
5. vLLM pre-training generation completes;
6. initial trainer policy publication and generator pull complete;
7. at least three finite optimizer updates with finite loss and gradient norm;
8. policy versions advance in retained rollout rows;
9. post-training generation completes;
10. non-empty rollout JSONL and DCP checkpoint metadata/data shards exist;
11. actor/process-mesh shutdown is clean;
12. scheduler exit is zero.

The passing Sunspot reference produced 40/40 completed nonzero-reward rollouts,
policy versions 0 through 3, and DCP checkpoints at steps 1, 2, and 3.

## Inspectable rollout examples

The authoritative raw artifact for job `12478711` is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-mds154391-broad-grain-sft900/grpo/
  agpt2b-multihost-sync-12478711/rollout_samples.jsonl
```

It contains 40 rows: 16 at policy version 0 and eight each at versions 1, 2,
and 3. Every row has `status="completed"`; all 40 have nonzero total reward.
The renderer stores the reasoning body in `reasoning_content` and the answer
tail in `content`, so the examples below reconstruct the raw sampled response
as `<think>{reasoning_content}</think>{content}`.

### Exact-correct updated-policy rollout

Artifact row 16, policy version 1, training sample, reward `0.8`:

```text
User: A cake of 400 grams is divided into eight equal parts. Nathalie eats
one-eighth of the cake, and Pierre eats double what Nathalie ate. How much did
Pierre eat, in grams?

Assistant:
<think>
Nathalie eats 1/8 x 400g = 50g of cake.
Pierre eats 2 x 50g = 100g of cake.
</think>
<answer>\boxed{100}</answer><end_of_turn>
```

Reward components:

```text
ThinkFormatReward       1.0
AnswerExtractableReward 1.0
AnswerCloseReward       0.0
AnswerCorrectReward     1.0
```

This proves the remote vLLM actor consumed an updated policy version and
returned a bounded, correctly formatted, exact-answer completion.

### Well-formed partial-credit rollout

Artifact row 24, policy version 2, training sample, reward
`0.31666666666666665`:

```text
User: John can play 200 beats per minute. If he plays 2 hours a day for 3 days
how many beats does he play?

Assistant:
<think>
He played for 60*2=120 minutes
That means he played 120*200=24000 beats
</think>
<answer>\boxed{24000}</answer><end_of_turn>
```

The response is bounded and format-valid, but it omits the three-day factor.
The scorer therefore records:

```text
ThinkFormatReward       1.0
AnswerExtractableReward 1.0
AnswerCloseReward       0.33333333333333337
AnswerCorrectReward     0.0
```

This is useful negative evidence: the pipeline distinguishes successful
generation/formatting and partial numeric closeness from exact task correctness.

### Post-training validation can still be wrong

Artifact row 32, policy version 3, validation sample, reward `0.25`, emits a
well-formed but incorrect answer (`148`) after flawed traffic arithmetic. All
eight policy-version-3 validation rows were completed and bounded, but none had
`AnswerCorrectReward=1.0`. Therefore the job proves multi-host runtime and
policy-version advancement, **not** semantic improvement. Never promote a model
from these rollout snippets or aggregate reward alone; run a fixed paired
held-out evaluation.

## Sunspot: validated two-host reference

Use a clean detached worktree at an immutable pushed SHA. The merged validation
launcher is:

```bash
qsub \
  -v EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs
```

The launcher owns these machine-specific settings:

```bash
V=/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCHTITAN_TORCHSTORE_TRANSPORT=gloo
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_KVS_IP_PORT="${head_node}_${port}"
```

It launches two scheduler ranks (`-n 2 -ppn 1`), then the entry point spawns one
trainer XPU actor on the first host and one generator XPU actor on the second.
The committed PBS script is a bounded production validation, not a long quality
campaign; change training budget only after the unmodified gate passes.

## Aurora `next-eval`: current runtime contract

Aurora `next-eval` uses the TEST compute image, not the ordinary production
image. New RL work must use all of the following:

```text
queue: next-eval
account: AuroraGPT
filesystems: home:flare
runtime module: oneapi/release/2026.1.0 (+ hdf5, pti-gpu as needed)
Python/PyTorch: image-independent, versioned .venv.next-eval archive
base frameworks Conda environment: forbidden
XPU hierarchy: ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

A representative allocation header is:

```bash
#PBS -A AuroraGPT
#PBS -q next-eval
#PBS -l select=2
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
```

Inside the allocation:

```bash
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
  source /etc/bash.bashrc.local
fi
module load oneapi/release/2026.1.0 hdf5 pti-gpu

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PATH="/opt/pbs/bin:${PATH}"

source <(curl -fsSL https://ezpz.cool/utils.sh)
ezpz_setup_job

VENV_ROOT=<shared directory containing the validated archive>
ARCHIVE="${VENV_ROOT}/.venv.next-eval.tar.gz"
ezpz yeet --src "${ARCHIVE}"
source /tmp/.venv.next-eval/bin/activate
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PBS_O_WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
```

The archive name and extraction directory must be versioned and asserted by the
wrapper; do not overwrite a production archive in place. If the current archive
extracts under a different node-local basename, derive and assert that path
instead of assuming `/tmp/.venv.next-eval`.

### Aurora fail-closed preflight

Before allocating a full RL run, execute this on every selected host using the
node-local interpreter that will launch the actors:

```bash
python3 - <<'PY'
import importlib.metadata as metadata
import inspect
import torch

import monarch
import torchstore
import vllm
from spmd_types import SpmdType
from torchtitan.experiments.ezpz.rl import train_upstream
from torchtitan.rl.controller import Controller

assert torch.__version__.startswith("2.15.") and "+xpu" in torch.__version__
assert metadata.version("spmd-types") == "0.2.5"
assert torch.xpu.is_available()
assert torch.xpu.device_count() == 12
assert callable(train_upstream.spawn_proc_mesh)
print(torch.__version__, monarch.__file__, torchstore.__file__, vllm.__file__, SpmdType, Controller)
PY
```

Also reject pip MPI runtimes such as `impi-rt`; site MPICH/PMIx must remain
authoritative. Import success on the login node is not a compute-node preflight.

### Aurora promotion sequence

Aurora has not yet passed the final multi-host RL gate. Promote in this order:

1. two-host actor-placement preflight with distinct host identities;
2. same two-host AGPT validation shape as Sunspot;
3. require explicit Gloo, pre/post vLLM generation, three finite updates,
   policy-version advancement, checkpoints, rollout artifacts, clean shutdown,
   and PBS exit 0;
4. only then increase actor counts, model size, or training budget.

Do not claim Aurora success from the generic communicator probe, scheduler
state, a parser check, or model construction.

## Operational invariants

- Use one scheduler allocation and one Monarch actor graph. Two independent MPI
  controllers are not an equivalent multi-host test.
- Keep checkpoints, datasets, repository, and outputs on the shared filesystem;
  only the runtime venv is node-local.
- Namespace output, TorchStore/FileStore rendezvous, and ports by PBS job ID.
- Retain the 120-second Monarch attach timeout and 30-minute coordination-store
  timeout: full RL/vLLM imports exceeded the defaults during healthy startup.
- Use dimension-preserving host slices (`slice(0, 1)`, not integer `0`) so role
  meshes retain the named `hosts` dimension.
- Scrub standalone vLLM subprocesses of machine-specific CCL/FI variables when
  the launcher does so; do not reintroduce inherited transport pollution.
- Never infer model-quality improvement from a mechanically successful GRPO
  run. Evaluate a fixed held-out set with retained raw generations.

## Failure signatures

| Symptom | Meaning | Action |
|---|---|---|
| attach config-push timeout | full RL imports exceeded Monarch's default attach window, or reverse channel is unroutable | retain the tested 120 s timeout; first confirm the lightweight two-host actor preflight |
| `KeyError: 'hosts'` | integer host slice dropped the named dimension | use `hosts.slice(hosts=slice(i, i + 1))` |
| `Shared memory storage not found` | automatic selection misclassified the remote volume as local and chose SharedMemory | use the validated Gloo workaround; separately fix/retest locality metadata before relying on `Unset` |
| `Unset` + SharedMemory disabled freezes before publication | automatic network selection did not complete; in the tested runtime MonarchRDMA was the expected available candidate | inspect the resolved-transport log and backend availability; do not call capability detection a transfer pass |
| forced MonarchRDMA kills the storage-volume actor | current MonarchRDMA path crashed during initial trainer publication | preserve the core/log; use validated Gloo pending an RDMA-specific fix |
| non-controller rank times out after 300 s | wrapper's FileStore coordination timeout is shorter than healthy RL startup/training | retain the tested 30-minute timeout |
| forced XCCL hangs after flatten/cast | TorchStore XCCL transport remains unvalidated on this actor bootstrap | return to Gloo; treat XCCL as a separate experiment |
| vLLM engine fails after inheriting CCL/FI settings | standalone EngineCore inherited launcher transport state | use the committed XPU environment scrub; compare against the validated launcher |

## Historical and deprecated paths

- [`trl.md`](trl.md): legacy TRL `GRPOTrainer` + external vLLM-server path. Kept
  for reproduction; not the current production recommendation.
- [`2026-07-06_multinode-grpo-root-cause.md`](2026-07-06_multinode-grpo-root-cause.md):
  historical diagnosis of the legacy TRL multi-trainer-node path.
- [`monarch.md`](monarch.md): July-era one-host/two-tile Monarch result and reward
  studies. Useful history, superseded as an operator guide by this page.
- [`history/`](history/README.md): pre-current runtime bring-up records; never use
  their environment snippets as new-run instructions.
- [Aurora `prod` newer-PyTorch guide](../../guides/running-with-newer-pytorch.md):
  current oneAPI 2026.1 environment and image-independent archive handling.
