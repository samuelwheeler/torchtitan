#!/bin/bash
# alcf-job-preflight: run this BEFORE qsub/sbatch. Exits non-zero if anything
# the job needs is missing, so a bad submission never reaches the queue.
#
#   usage: preflight.sh <worktree> <venv> [machine]
#
# Catches the failure classes that cost nine jobs in one session: node-local
# /tmp, missing ZE_FLAT_DEVICE_HIERARCHY, stale worktree, absent tokenizer or
# data-list, and API signatures that changed under us.
set -o pipefail
W="${1:?usage: preflight.sh <worktree> <venv> [machine]}"
V="${2:?usage: preflight.sh <worktree> <venv> [machine]}"
M="${3:-$(hostname -s | sed 's/[0-9-].*//')}"
fail=0
ok()   { printf "  OK    %s\n" "$1"; }
bad()  { printf "  BAD   %s\n" "$1"; fail=1; }
# Informational only -- never sets `fail`. For facts worth printing that are
# not gates (see the #181519 line below).
note() { printf "  note  %s\n" "$1"; }

cd "$W" 2>/dev/null || { bad "worktree $W does not exist"; exit 1; }

# --- worktree currency: a pinned worktree misses files recent commits moved ---
if git rev-parse --git-dir >/dev/null 2>&1; then
  h=$(git rev-parse --short HEAD 2>/dev/null)
  if [ -n "${PREFLIGHT_SOURCE_SHA:-}" ]; then
    u=$(git rev-parse --short "$PREFLIGHT_SOURCE_SHA" 2>/dev/null)
  else
    u=$(git rev-parse --short '@{u}' 2>/dev/null || echo "$h")
  fi
  [ "$h" = "$u" ] && ok "worktree current ($h)" || bad "worktree $h != upstream $u -- fetch/checkout first"
fi

# --- assets ---
tokenizer_path="${PREFLIGHT_TOKENIZER_PATH:-assets/hf/gemma-7b/tokenizer.json}"
[ -s "$tokenizer_path" ] && ok "tokenizer $tokenizer_path" || bad "tokenizer $tokenizer_path"
dl="torchtitan/experiments/ezpz/data-lists"
if [ -d "$dl" ]; then
  found=$(ls -1 "$dl" 2>/dev/null | tr '\n' ' ')
  case "$found" in *"$M"*) ok "data-list for $M";; *) bad "no data-list for '$M' (have: $found)";; esac
fi

# --- python deps, in the venv the JOB will use ---
for m in torch mpi4py ezpz blendcorpus; do
  "$V/bin/python" -c "import $m" 2>/dev/null && ok "py:$m" || bad "py:$m not importable in $V"
done

# --- torch floor: the symbol HEAD actually imports, by SYMBOL not prose ---
#
# This used to hard-FAIL on pytorch #181519 (_resolve_spmd_types_for_storage)
# and call it "this torch dies at FSDP wrapping". That verdict fails EVERY venv
# we own -- both 2.13 builds score 0, including the production venv the live
# 2B/20B chains are training on right now. A check that red-lights a
# known-good production runtime is not a floor, it is a blocker for work that
# would have succeeded.
#
# What HEAD actually requires is DataParallelMeshDims, imported at
# torchtitan/distributed/fsdp.py:16. That is the real floor (it is what the
# torch-2.10 Polaris venv cannot satisfy), and it is present on both 2.13
# builds. Verified by import, not by grep: `from torch.distributed.fsdp import
# DataParallelMeshDims` succeeds and torchtitan.distributed.fsdp loads.
#
# #181519 is kept as an informational line, because it is still the thing to
# check when FSDP wrapping misbehaves on a 2.15 nightly -- just not a gate.
if "$V/bin/python" -c "from torch.distributed.fsdp import DataParallelMeshDims" 2>/dev/null; then
  ok "torch floor: DataParallelMeshDims importable"
else
  bad "torch floor: DataParallelMeshDims MISSING -- HEAD imports it at torchtitan/distributed/fsdp.py:16"
fi
f=$(ls "$V"/lib/python3*/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py 2>/dev/null | head -1)
if [ -n "$f" ]; then
  n=$(grep -c _resolve_spmd_types_for_storage "$f")
  [ "$n" -gt 0 ] && note "#181519 present ($(wc -l < "$f") lines)" \
                 || note "#181519 absent ($(wc -l < "$f") lines) -- normal for 2.13, not a blocker"
fi

# --- XPU geometry: the one that keeps biting ---
case "$M" in
  aurora*|sunspot*|x[0-9]*)
    if [ "${ZE_FLAT_DEVICE_HIERARCHY:-}" = "FLAT" ]; then ok "ZE_FLAT_DEVICE_HIERARCHY=FLAT"
    else bad "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset} -- need FLAT for 12 ranks/node"; fi;;
esac

# --- nothing the job reads may live on node-local tmpfs ---
for p in "$@"; do
  case "$p" in /tmp/*) bad "$p is on node-local tmpfs -- invisible to compute nodes";; esac
done

[ $fail -eq 0 ] && echo "  ---> PREFLIGHT PASS" || echo "  ---> PREFLIGHT FAIL, do not submit"
exit $fail
