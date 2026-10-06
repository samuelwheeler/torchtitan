#!/bin/bash --login
#PBS -A datascience
#PBS -N sync-agpt2b-multihost
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
set -o pipefail

WT="${SYNC_RL_WORKTREE:?set SYNC_RL_WORKTREE}"
V="${SYNC_RL_VENV:?set SYNC_RL_VENV}"
CKPT="${SYNC_RL_CHECKPOINT:-/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/stage2/agpt2b-mds154391-step600-gsm8k-r1cot-2n-r2/final}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA}"
EXPECTED_MODEL_SHA="${EXPECTED_MODEL_SHA:-72dbc06a391ee936ae9f3c0acd375e88c98c4a296d2ffd3a403faefeab56c88e}"
JOB="${PBS_JOBID%%.*}"
OUT="/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-sync-multihost-${JOB}"
STORE="$WT/.multihost-rl-${JOB}.store"
PORT=$((35000 + JOB % 20000))
LOG="$OUT/controller.log"

cd "$WT" || exit 11
source "$WT/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job
ezpz_load_modules
export VIRTUAL_ENV="$V"
export PATH="$V/bin:/opt/pbs/bin:$PATH"
hash -r
[[ "$(command -v python)" == "$V/bin/python" ]] || exit 13
[[ "$ZE_FLAT_DEVICE_HIERARCHY" == FLAT ]] || exit 14
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || exit 15
[[ "$(sha256sum "$CKPT/model.safetensors" | cut -d' ' -f1)" == "$EXPECTED_MODEL_SHA" ]] || exit 16
[[ -s "$CKPT/chat_template.jinja" ]] || exit 16
[[ ! -e "$OUT" ]] || exit 17
mkdir -p "$OUT"

export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1
export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCHSTORE_TRANSPORT="${TORCHSTORE_TRANSPORT:-auto}"
case "$TORCHSTORE_TRANSPORT" in
    auto) unset TORCHTITAN_TORCHSTORE_TRANSPORT TORCHSTORE_SHARED_MEMORY_ENABLED ;;
    gloo|xccl|monarch_rdma)
        export TORCHTITAN_TORCHSTORE_TRANSPORT="$TORCHSTORE_TRANSPORT"
        ;;
    *) printf 'FATAL: unsupported TORCHSTORE_TRANSPORT=%s\n' "$TORCHSTORE_TRANSPORT"; exit 18 ;;
esac
unset TORCHSTORE_GLOO_ENABLED TORCHSTORE_XCCL_ENABLED CCL_OP_SYNC CCL_OFI_PROVIDER
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp
head_node=$(head -1 "$PBS_NODEFILE")
export CCL_KVS_IP_PORT="${head_node}_$((29500 + JOB % 1000))"

"$V/bin/python" -m py_compile \
    torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py
"${V}/bin/python" - <<'PY' || exit 18
import os
from torchtitan.experiments.ezpz.rl.reason_agpt.config_registry import (
    rl_grpo_lora_agpt_2b_gsm8k_b2smoke,
)
config = rl_grpo_lora_agpt_2b_gsm8k_b2smoke()
assert config.async_loop.num_samples_per_prompt == 4
assert config.generator.model_dtype == "float32"
assert tuple(config.renderer.extra_stop_token_ids) == (1, 107)
requested = os.environ.get("TORCHTITAN_TORCHSTORE_TRANSPORT", "auto")
assert requested in {"auto", "gloo", "xccl", "monarch_rdma"}
print(f"RL_MULTIHOST_TRANSPORT_REQUESTED={requested}")
PY

printf 'RL_MULTIHOST_START job=%s commit=%s transport=%s out=%s\n' \
    "$JOB" "$EXPECTED_SHA" "$TORCHSTORE_TRANSPORT" "$OUT" | tee "$LOG"
"$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
    --nproc 2 --nproc_per_node 1 --cpu-bind none --timeout 1800 -- \
    "$V/bin/python" -u \
    torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py \
    --multihost-store "$STORE" --multihost-port "$PORT" --multihost-world-size 2 \
    --module torchtitan.experiments.ezpz.rl.reason_agpt \
    --config rl_grpo_lora_agpt_2b_gsm8k_b2smoke \
    --hf_assets_path="$CKPT" \
    --dump_folder="$OUT" \
    --async-loop.num-training-steps=3 \
    --async-loop.num-prompts-per-train-step=4 \
    --async-loop.num-samples-per-prompt=4 \
    --async-loop.target-offpolicy-steps=0 \
    --async-loop.validation.num-samples=8 \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
    --generator.sampling.max-tokens=700 \
    --generator.parallelism.data-parallel-degree=1 \
    --generator.parallelism.tensor-parallel-degree=1 \
    --trainer.parallelism.data-parallel-shard-degree=1 \
    --trainer.parallelism.tensor-parallel-degree=1 \
    --trainer.checkpointer.interval=1 \
    --metrics.no-enable-wandb \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
rm -f "$STORE"
test "$rc" -eq 0 || {
    printf 'RL_MULTIHOST_VERDICT: failed rc=%s\n' "$rc" | tee -a "$LOG"
    exit "$rc"
}

"$V/bin/python" - "$OUT" <<'PY' | tee -a "$LOG"
import glob
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "rollout_samples.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) == 40, len(rows)
assert all(row.get("status") == "completed" for row in rows), "incomplete rollout"
versions = {turn["max_policy_version"] for row in rows for turn in row.get("turns", [])}
assert {0, 1, 2, 3}.issubset(versions), versions
assert all((root / f"checkpoint/step-{step}/.metadata").stat().st_size > 0 for step in (1, 2, 3))

metric_lines = [line for line in (root / "controller.log").read_text(errors="replace").splitlines() if "Train | Step:" in line]
assert len(metric_lines) == 3, len(metric_lines)
grads = [float(line.split("trainer/grad_norm/mean:", 1)[1].split()[0]) for line in metric_lines]
losses = [float(line.split("loss/mean:", 1)[1].split()[0]) for line in metric_lines]
assert all(math.isfinite(value) for value in grads + losses)
assert all(value > 0 for value in grads), grads

events = []
for path in glob.glob(str(root / "structured_logs/*.jsonl")):
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line).get("log_type_name"))
        except json.JSONDecodeError:
            pass
assert events.count("push_model_state_dict_end") >= 4
assert events.count("pull_model_state_dict_end") >= 4
assert events.count("optimizer_step_end") >= 3
print(
    "RL_MULTIHOST_VERDICT: ok "
    f"rows={len(rows)} versions={sorted(versions)} grads={grads} losses={losses} "
    f"pushes={events.count('push_model_state_dict_end')} "
    f"pulls={events.count('pull_model_state_dict_end')}"
)
PY
