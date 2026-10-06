#!/usr/bin/env bash
# Static contracts for batch launchers that cannot run off-machine.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
sonic="$repo_root/torchtitan/experiments/ezpz/submit/aurora/submit_agpt_moe_full_sonic_2n_1100.pbs"
production="$repo_root/torchtitan/experiments/ezpz/submit/aurora/submit_agpt_dense_moe_256n_50k.pbs"
resumable="$repo_root/torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh"
dcp_sync="$repo_root/torchtitan/experiments/ezpz/scripts/sync_dcp_resume_sunspot.sh"
rl_sync="$repo_root/torchtitan/experiments/ezpz/rl/scripts/grpo/sync_agpt2b_weight_sync_sunspot.sh"
rl_multihost="$repo_root/torchtitan/experiments/ezpz/rl/scripts/grpo/sync_agpt2b_multihost_weight_sync_sunspot.sh"
fsdp_probe="$repo_root/torchtitan/experiments/ezpz/scripts/probe_30b_fsdp2_xccl_sunspot.sh"
fsdp_probe_py="$repo_root/torchtitan/experiments/ezpz/tests/probe_fsdp2_storage_collectives.py"
collective_probe_py="$repo_root/torchtitan/experiments/ezpz/tests/probe_collective_op.py"
full_model_probe="$repo_root/torchtitan/experiments/ezpz/scripts/probe_30b_full_model_sunspot.sh"
umbrella="$repo_root/torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh"
docs="$repo_root/torchtitan/experiments/ezpz/docs/guides/aurora-moe-training.md"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    local file=$1 pattern=$2 message=$3
    grep -Eq -- "$pattern" "$file" || fail "$message"
}

assert_not_contains() {
    local file=$1 pattern=$2 message=$3
    if grep -Eq -- "$pattern" "$file"; then
        fail "$message"
    fi
}

# Broadcast archives are relocated to a caller-selected /tmp directory. The
# activation script may retain its build-time VIRTUAL_ENV, so the umbrella must
# execute staged entry points by absolute path instead of sourcing it and
# accidentally falling back to $HOME/bin.
assert_not_contains "$umbrella" 'source "\$venvdst/bin/activate"' \
    'umbrella must not source a non-relocatable staged activation script'
assert_contains "$umbrella" '"\$venvdst/bin/ezpz" launch' \
    'umbrella must execute staged ezpz by absolute path'
assert_contains "$umbrella" '"\$venvdst/bin/python" -m torchtitan.experiments.ezpz.train' \
    'umbrella payload must execute staged Python by absolute path'

# Keep the public ezpz invocation; only fix status handling around its pipeline.
assert_contains "$resumable" '^[[:space:]]*ezpz launch ' \
    'resumable SFT must continue to use ezpz launch'
assert_contains "$resumable" 'rc=\$\{PIPESTATUS\[0\]\}' \
    'resumable SFT must capture the ezpz launch status'
assert_contains "$resumable" 'exit "\$\{rc\}"' \
    'resumable SFT must return the ezpz launch status'
assert_not_contains "$resumable" '\|[[:space:]]*tee.*\|\|[[:space:]]*true' \
    'resumable SFT must not swallow launcher/timeout failures'

# The sync DCP gate must compare a resumed trajectory with an uninterrupted
# control and preserve full checkpoint state at the save boundary.
assert_contains "$dcp_sync" 'run_phase control 4' \
    'sync DCP gate must run the four-step uninterrupted control'
assert_contains "$dcp_sync" 'run_phase save 4' \
    'sync DCP gate must keep the save phase schedule identical to the control'
assert_contains "$dcp_sync" 'run_phase resume 4' \
    'sync DCP gate must destroy/rebuild and resume through step four'
assert_contains "$dcp_sync" '--checkpoint.no-last-save-model-only' \
    'sync DCP save must include optimizer, scheduler, and dataloader state'
assert_contains "$dcp_sync" 'DCP_SYNC_VERDICT: ok' \
    'sync DCP gate must emit a machine-readable success verdict'
assert_contains "$dcp_sync" 'ansi_escape' \
    'sync DCP metric parser must strip terminal color sequences'

# Exact-head RL gate must preserve the validated one-controller Monarch path
# while proving both weight transfers and a real optimizer update.
assert_contains "$rl_sync" 'EXPECTED_SHA' \
    'sync RL gate must assert the immutable source SHA'
assert_contains "$rl_sync" 'EXPECTED_MODEL_SHA' \
    'sync RL gate must assert the immutable starting checkpoint'
assert_contains "$rl_sync" 'train_upstream' \
    'sync RL gate must exercise the real upstream RL bridge'
assert_contains "$rl_sync" '--config rl_grpo_lora_agpt_2b_easy' \
    'sync RL gate must use the bounded one-turn task with reward variance'
assert_contains "$rl_sync" 'RL_SYNC_VERDICT: ok' \
    'sync RL gate must emit a machine-readable semantic success verdict'
assert_contains "$rl_sync" 'trainer/grad_norm/mean' \
    'sync RL gate must reject a zero-gradient optimizer no-op'
assert_contains "$rl_sync" '--generator.sampling.max-tokens=256' \
    'sync RL gate must retain enough generation budget for bounded completion'
assert_not_contains "$rl_sync" 'pip install' \
    'sync RL gate must not mutate the protected runtime'

assert_contains "$rl_multihost" 'TORCHSTORE_TRANSPORT:-auto' \
    'multi-host sync gate must exercise automatic weight transport by default'
assert_contains "$rl_multihost" 'RL_MULTIHOST_TRANSPORT_REQUESTED=' \
    'multi-host sync gate must log the requested transport policy'
assert_contains "$rl_multihost" 'multihost_train_upstream.py' \
    'multi-host sync gate must use the scheduler-SPMD entry point'
assert_contains "$rl_multihost" 'RL_MULTIHOST_VERDICT: ok' \
    'multi-host sync gate must emit a machine-readable semantic verdict'
assert_contains "$rl_multihost" 'trainer/grad_norm/mean' \
    'multi-host sync gate must require nonzero gradient evidence'
assert_not_contains "$rl_multihost" 'pip install' \
    'multi-host sync gate must not mutate the isolated runtime'

# The 30B failure discriminator must preserve shard degree 16 while separating
# raw transport, pure-FSDP storage, and HSDP mesh-interaction failures.
assert_contains "$fsdp_probe" 'mib="12 73.5"' \
    '30B XCCL all-gather must use exact BF16 1/16 production shards'
assert_contains "$fsdp_probe" 'mib="384 2352"' \
    '30B XCCL reduce-scatter must use exact FP32 production tensors'
assert_contains "$fsdp_probe" 'mib="24 147"' \
    '30B XCCL all-reduce must use exact FP32 1/16 production shards'
assert_contains "$fsdp_probe" 'group_stride=16' \
    'HSDP all-reduce control must use stride-16 replica groups'
assert_contains "$fsdp_probe" 'nproc=48' \
    'HSDP all-reduce control must launch the full four-node reduced topology'
assert_contains "$fsdp_probe" '--nproc .*--nproc_per_node 12' \
    '48-rank raw arm must use the regular four-node ezpz topology'
assert_contains "$fsdp_probe" 'EZPZ_PROBE_DTYPE=' \
    '30B raw controls must pass the production collective dtype explicitly'
assert_contains "$collective_probe_py" 'float32.*torch.float32' \
    'raw collective probe must support production FP32 gradient traffic'
assert_contains "$fsdp_probe" '--dp-replicate 1 --dp-shard 16' \
    '30B probe must test pure FSDP at exact shard degree 16'
assert_contains "$fsdp_probe" 'run_fsdp pure-fsdp 16 12' \
    'pure-FSDP arm must preserve Sunspot production ranks-per-node placement'
assert_contains "$fsdp_probe" 'timeout 600 mpiexec --envall --line-buffer --np=16 --ppn=12' \
    'irregular 16-rank arm must bypass ezpz even-occupancy inference explicitly'
assert_contains "$fsdp_probe" '--hostfile=' \
    'irregular 16-rank arm must preserve scheduler host ordering'
assert_contains "$fsdp_probe" '--dp-replicate 3 --dp-shard 16' \
    '30B probe must test HSDP at exact shard degree 16'
assert_contains "$fsdp_probe" 'run_fsdp hsdp 48 12' \
    'HSDP arm must use the exact four-node 48-rank topology'
assert_contains "$fsdp_probe_py" 'FSDP2_XCCL_PROBE_PASS' \
    '30B storage probe must emit a machine-readable success marker'
assert_contains "$fsdp_probe_py" 'num_embeddings=100352' \
    '30B storage probe must use the exact OLMo embedding vocabulary'
assert_contains "$fsdp_probe_py" 'embedding_dim=6144' \
    '30B storage probe must use the exact 30B embedding width'
assert_contains "$fsdp_probe_py" 'maybe_install_xccl_split_group_workaround' \
    'standalone 30B storage probe must install the XCCL mesh workaround'
assert_contains "$fsdp_probe_py" 'set_gradient_divide_factor' \
    'standalone probe must match production FSDP gradient scaling'
assert_contains "$fsdp_probe_py" 'set_force_sum_reduction_for_comms' \
    'standalone probe must avoid unsupported oneCCL AVG reduction'
assert_contains "$fsdp_probe_py" 'MASTER_ADDR' \
    'standalone 30B storage probe must configure env rendezvous under PALS'
assert_contains "$fsdp_probe_py" 'global gradient' \
    'sparse embedding gradients must be validated over the full shard group'
assert_contains "$fsdp_probe_py" 'global parameter update' \
    'probe acceptance must require a globally nonzero optimizer update'
assert_contains "$fsdp_probe_py" 'shard_index = rank % args.dp_shard' \
    'probe tokens must exercise every embedding storage shard'
assert_contains "$fsdp_probe" 'SKIP_RAW_CONTROLS' \
    'FSDP-only retries must preserve completed raw controls without rerunning them'
assert_contains "$fsdp_probe" 'exit 96' \
    'raw collective arm must fail closed when its success marker is absent'
assert_contains "$fsdp_probe" 'exit 97' \
    'FSDP arm must fail closed when its success marker is absent'
assert_not_contains "$fsdp_probe" 'pip install' \
    '30B probe must not mutate the isolated runtime'
assert_not_contains "$fsdp_probe_py" 'pip install' \
    '30B probe payload must not mutate the isolated runtime'

# The next escalation after a green synthetic probe is a four-node full-model
# canary, not a blind 64-node LR sweep.
assert_contains "$full_model_probe" 'CANARY_MODEL_CONFIG:-agpt_30b_olmo2tok_smoke' \
    '30B full-model canary must use the canonical OLMo-tokenizer model'
assert_contains "$full_model_probe" 'CANARY_NPROC:-48' \
    '30B full-model canary must retain a 48-rank launch path'
assert_contains "$full_model_probe" 'CANARY_DP_REPLICATE:-3' \
    '30B full-model canary must retain the HSDP replicate axis'
assert_contains "$full_model_probe" 'CANARY_DP_SHARD:-16' \
    '30B full-model canary must preserve production shard degree 16'
assert_contains "$full_model_probe" 'NPROC % 12 != 0' \
    '30B full-model canary must detect irregular Sunspot occupancy'
assert_contains "$full_model_probe" 'WORLD_SIZE="\$NPROC" timeout "\$LAUNCH_TIMEOUT" mpiexec' \
    '30B full-model canary must launch irregular shard degrees with PALS'
assert_contains "$full_model_probe" 'CANARY_TOKENS_PER_MICROBATCH:-4096' \
    '30B full-model canary must parameterize historical local batch geometry'
assert_contains "$full_model_probe" '--training.steps 3' \
    '30B full-model canary must complete three optimizer updates'
assert_contains "$full_model_probe" 'FULL_MODEL_CANARY_PASS' \
    '30B full-model canary must emit a machine-readable success marker'
assert_not_contains "$full_model_probe" '--diagnostics' \
    'full-model canary must not add per-parameter DTensor diagnostic collectives'
assert_not_contains "$full_model_probe" '--compile.no-enable' \
    '30B canary must preserve the production compile path and memory behavior'
assert_not_contains "$full_model_probe" 'pip install' \
    '30B full-model canary must not mutate the isolated runtime'

# A resumable timeout remains a nonzero batch result for schedulers and callers.
assert_contains "$production" 'resumable=1' \
    'production launcher must identify checkpointed timeout as resumable'
assert_contains "$production" 'elif test "\$\{rc\}" -eq 124' \
    'production launcher must distinguish timeout status 124'
python3 - "$production" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
branch = text.split('elif test "${rc}" -eq 124; then', 1)[1].split('else', 1)[0]
assert 'exit "${rc}"' in branch, 'resumable timeout branch must propagate rc=124'
warmup = text.split('warmup_rc=${PIPESTATUS[0]}', 1)[1]
guard = 'if test "${warmup_rc}" -ne 0; then'
assert guard in warmup, 'production warm-up must explicitly guard nonzero status'
guarded = warmup.split(guard, 1)[1].split('fi', 1)[0]
assert 'exit "${warmup_rc}"' in guarded, 'production warm-up must propagate status'
PY

# The two-node Sonic launcher documents these relocatable artifact overrides.
for variable in \
    TT_ROOT TT_VENV_TAR TT_MOE_KERNEL_CACHE TT_DATA_LIST \
    TT_DATA_LIST_SHA256 TT_TOKENIZER TT_TOKENIZER_SHA256; do
    assert_contains "$sonic" "${variable}" \
        "two-node Sonic launcher must honor ${variable}"
done
assert_contains "$sonic" '--dataloader.dataset-path="\$[0-9]"' \
    'two-node Sonic launcher must pass TT_DATA_LIST to training'
assert_contains "$sonic" '--hf-assets-path="\$[0-9]"' \
    'two-node Sonic launcher must pass the TT_TOKENIZER directory to training'
assert_contains "$sonic" 'missing .*TT_' \
    'two-node Sonic launcher must emit clear missing-override diagnostics'
assert_contains "$sonic" 'warmup_rc=\$\{PIPESTATUS\[0\]\}' \
    'two-node Sonic warm-up must capture timeout/launcher status'
assert_contains "$sonic" 'rc=\$\{PIPESTATUS\[0\]\}' \
    'two-node Sonic training must capture timeout/launcher status'
python3 - "$sonic" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
for assignment in ('warmup_rc=${PIPESTATUS[0]}', 'rc=${PIPESTATUS[0]}'):
    tail = text.split(assignment, 1)[1]
    status = assignment.split('=', 1)[0]
    guard = f'if test "${{{status}}}" -ne 0; then'
    assert guard in tail, f'{status} must have an explicit nonzero guard'
    guarded = tail.split(guard, 1)[1].split('fi', 1)[0]
    assert f'exit "${{{status}}}"' in guarded, f'{status} guard must propagate status'
PY

assert_contains "$docs" 'retains all checkpoints' \
    'checkpoint retention documentation must say all checkpoints are retained'
assert_not_contains "$docs" 'retains the newest two checkpoints' \
    'checkpoint retention documentation must not claim only two are retained'

printf 'launcher contract tests: PASS\n'
