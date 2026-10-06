---
name: alcf-job-preflight
description: Use BEFORE submitting any PBS/Slurm job on Sunspot, Aurora, Polaris or Perlmutter, and before writing a job script from scratch. Covers the environment settings a hand-written script omits (ZE_FLAT_DEVICE_HIERARCHY, --login, activation-checkpoint syntax, shared-filesystem paths, XCCL split-group), plus the login-node dry-run that catches API errors without a queue round-trip.
---

# ALCF job preflight

Nine jobs were burned in one session answering two questions. Eight failures
were the script, not the system. Every one is checkable before `qsub`.

## Rule 0: copy the house script, do not write one

`torchtitan/experiments/ezpz/scripts/sync_smoke_{aurora,sunspot}.sh` already
encodes everything below. Start from it. A scratch-written script has
reproduced the same five omissions repeatedly.

## The settings a hand-written script forgets

| setting | symptom when missing |
|---|---|
| `#!/bin/bash --login` | `module: command not found` -- a PBS job has no `module` function otherwise |
| `export ZE_FLAT_DEVICE_HIERARCHY=FLAT` | `device index out of range [0, 6)`. Aurora/Sunspot = 6 GPUs x 2 tiles: torch sees **6** under COMPOSITE (the job default) and **12** under FLAT. `--ppn 12` needs FLAT. |
| `activation-checkpoint:none` | `Unrecognized options: --activation-checkpoint.mode`. It is a tyro SUBCOMMAND, not `--flag=value`. Same for `:full`, `:selective`. |
| `--training.max-context-length=512` and `--training.num-tokens-per-microbatch-per-dp-rank=512` | OOM at the vocab projection. Full-size defaults are 16384; `16384 x 256128 x 2B ~= 8.4 GiB` of bf16 logits on one tile. |
| scripts/data on a SHARED filesystem | `can't open file '/tmp/x.py'`. `/tmp` EXISTS on compute nodes and is writable, but it is `tmpfs` -- node-local. PBS ships the job script itself, so the job STARTS and only dies when it opens a second file by path. Stage to `/lus/tegu`, `/lus/flare`, `/eagle`, `/pscratch`. |
| `maybe_install_xccl_split_group_workaround()` | `No backend for the parent process group or its backend does not support splitting`. XCCL cannot split a PG. `train.py` installs this before building any mesh; a standalone script that builds a mesh must call it too. |
| Polaris: `module load cray-pals`, `mpiexec --no-transfer`, `libfabric` | `mpiexec: command not found` (rc=127); then PALS stages the interpreter without `libpython3.12.so.1.0`; then `libfabric.so.1: cannot open shared object file`. |
| Polaris: scope the `mpi-compat` dir to ranks via `mpiexec --env`, never `export` it | `*** stack smashing detected ***` in EVERY coreutil (`mkdir`, `whoami`, `head`). Presents as an unrelated filesystem failure. |

## Submit through the wrapper, not qsub directly

```bash
.agents/skills/alcf-job-preflight/submit.sh <worktree> <venv> <machine> <job.sh>
```

`preflight.sh` exits non-zero, but that only helps if something acts on it.
Running `bash preflight.sh ...; qsub job.sh` in one chain ignores the exit
code and submits anyway -- which happened: preflight printed "PREFLIGHT FAIL,
do not submit" against a 4-commit-stale worktree and the job was queued in the
same command. `submit.sh` runs the check and refuses to submit when it fails.

A check whose result you can ignore by accident is not a gate.

(If a job is already queued with a stale worktree, you may not need to qdel it:
PBS snapshots the SCRIPT at submit time but the worktree is read at RUNTIME, so
updating the checkout while the job sits in the queue can rescue it.)

## Before every submission

**1. Dry-run the APIs on the login node.** Presence checks are not enough --
a job died on `setup_torch(backend="ccl")` while every import passed. Bind the
arguments:

```python
import inspect
inspect.signature(ezpz.setup_torch).bind()          # raises if the kwarg is wrong
inspect.signature(ParallelDims.from_config).bind(cfg.parallelism, world_size=24)
```

**2. Preflight the assets**, in the worktree the job will actually use:
tokenizer (`assets/hf/gemma-7b`), machine data-list
(`data-lists/<machine>/books.txt`), `vendor/` if needed, and `python -c "import
blendcorpus, mpi4py, ezpz, torch"`.

**3. Check the worktree is current.** A worktree pinned to an older tip will
be missing files a recent commit moved or added.
For deliberate feature-branch validation, set `PREFLIGHT_SOURCE_SHA` to the
reviewed commit instead of comparing with the tracking branch. The PBS script
must independently assert that SHA and reject uncommitted source changes.

## Eval-job traps specifically

Four that cost jobs on 2026-09-17, none caught by the checks above. All
produced a clean exit.

| trap | symptom | check before `qsub` |
|---|---|---|
| **`PBS_O_WORKDIR` is wherever you ran `qsub`** | every relative path resolves under `$HOME`; `cp` fails for config and tokenizer, lm-eval then raises `Unrecognized model`, and the script prints `Eval done` and exits **0** having written nothing | `cd` into the repo before `qsub`, or pass an absolute `PBS_O_WORKDIR`. Confirm with `qstat -xf <job> \| grep Output_Path` -- if it points at `$HOME`, the job ran from there |
| **Conversion clone lacks `agpt/state_dict_adapter.py`** | with `MODEL_FLAVOR=*_real` the guard refuses (good, exit 2); the pinned run-clones predate `ComplexRoPE`/`CosSinRoPE` so the adapter cannot simply be copied in | convert from the main clone and symlink the chain's ckpt dir into its `outputs/checkpoints`. The guard is **cos_sin-only** -- a `complex` flavor runs fine from a pinned clone |
| **A smoke run wrote the same ckpt dir name** | eval finds only `step-50` under a production chain's name and evaluates nothing | `ls -1d <ckpt_dir>/step-*` and eyeball the range BEFORE submitting. Link the real chain under a distinct suffix if the name is taken |
| **Seats in one umbrella can differ in RoPE flavor** | the wrong flavor "loads fine and only fails as gibberish" | resolve per chain with `scripts/eval/rope_flavor_for_step.py`, then confirm against the `TRAINERS` table in the submit script. On 2026-09-17 the two 2B stage-2 seats were `_real` (n512) and `complex` (n256) deliberately |

## Reading results

- **When `rc != 0`, print the tail UNCONDITIONALLY.** A grep for known failure
  patterns that matches nothing reads exactly like success. A tyro argparse
  error inside a Unicode box matched none of `ur_die|abort|Error:` and the
  report said "no failure modes found" on a dead job.
- Treat the pattern list as *classification*, never *detection*.
- A check that cannot distinguish "broken" from "informative" is not a check.
  A comparison that returns MISMATCH when one side threw `KeyError` reports 24
  ranks disagreeing when the real answer is "I read the wrong object".
- Never `tail -3` a build log; capture it to a file and grep after.

## Machine facts

- **Aurora/Sunspot**: XPU. 6 devices COMPOSITE / 12 FLAT. `mpi4py` must be
  compiled on a compute node against the local MPICH (`MPICC=$(command -v
  mpicc)`); a PyPI wheel gives `PMIX_Init returned -25`.
- **Polaris**: CUDA. Default PrgEnv is **nvhpc**, so `cc` is `nvc` and rejects
  CPython's GCC flags -- mpi4py then reports the misleading "Cannot compile MPI
  programs". Use `module swap PrgEnv-nvhpc PrgEnv-gnu`.
- **Perlmutter**: CUDA, Slurm. `cc` already wraps gcc. Use
  `--gpus-per-node=4 --gpu-bind=none`; `--gpus-per-task=1` gives each rank one
  device numbered 0 and rank 1 dies selecting `cuda:1`. Account `m3957_g` has
  no node-hour balance; use `amsc013_g`. No tokenized corpus on the machine.
- `uv` and `qsub` are not on a non-interactive `PATH`: use
  `$HOME/.local/bin/uv` and `/opt/pbs/bin/qsub`.
- Point `UV_CACHE_DIR` off `$HOME` on Polaris (`os error 122` = HOME quota).
