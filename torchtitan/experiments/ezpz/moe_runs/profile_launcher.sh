#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <wandb_name> [extra torchtitan args ...]"
  exit 1
fi

WAND_NAME="$1"
shift || true
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_RUNS_ROOT="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

copy_into_dir() {
  local src="$1"
  local dst_dir="$2"

  if [[ -f "${src}" ]]; then
    mkdir -p "${dst_dir}"
    cp -f "${src}" "${dst_dir}/"
  fi
}

# Lmod scripts used by ezpz env setup can reference unset vars (e.g. ZSH_EVAL_CONTEXT),
# so temporarily disable nounset during environment initialization.
set +u
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_env
set -u

EZPZ_BIN="${EZPZ_BIN_OVERRIDE:-}"
if [[ -z "${EZPZ_BIN}" ]]; then
  EZPZ_BIN="$(command -v ezpz || true)"
fi
if [[ -z "${EZPZ_BIN}" ]]; then
  EZPZ_BIN="$(find "${REPO_ROOT}/venvs" -path '*/bin/ezpz' -type f | head -n 1 || true)"
fi
if [[ -z "${EZPZ_BIN}" || ! -x "${EZPZ_BIN}" ]]; then
  echo "Unable to locate ezpz after ezpz_setup_env."
  exit 1
fi

export ZE_FLAT_DEVICE_HIERARCHY="FLAT"

RUN_SLUG="$(echo "${WAND_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${REPO_ROOT}/outputs/moe_runs/${RUN_SLUG}}"
CKPT_ROOT="${RUN_ROOT}/checkpoints"
RUN_LOG_ROOT="${RUN_ROOT}/logs"
RUN_METADATA_ROOT="${RUN_ROOT}/metadata"
RUN_CONFIG_ROOT="${RUN_ROOT}/configs"
mkdir -p "${RUN_ROOT}" "${CKPT_ROOT}" "${RUN_LOG_ROOT}" "${RUN_METADATA_ROOT}" "${RUN_CONFIG_ROOT}"
export RUN_ROOT_OVERRIDE="${RUN_ROOT}"

export WANDB_PROJECT="${WANDB_PROJECT:-torchtitan.moe_runs}"
export WANDB_TEAM="${WANDB_TEAM:-moe_experiments}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WAND_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_SLUG}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-deepseek_v3_moe_ep12}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

export TT_CONFIG_JSON="${TT_CONFIG_JSON:-${MOE_RUNS_ROOT}/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_smoke.json}"

if [[ -z "${HF_ASSETS_PATH:-}" ]]; then
  CANDIDATES=(
    "${REPO_ROOT}/assets/hf/gemma-7b"
    "${REPO_ROOT}/assets/hf/deepseek-moe-16b-base"
    "${REPO_ROOT}/assets/hf/DeepSeek-V3.1-Base"
  )
  for p in "${CANDIDATES[@]}"; do
    if [[ -d "${p}" ]]; then
      HF_ASSETS_PATH="${p}"
      break
    fi
  done
fi

if [[ -z "${HF_ASSETS_PATH:-}" || ! -d "${HF_ASSETS_PATH}" ]]; then
  echo "No valid tokenizer/assets path found."
  echo "Set HF_ASSETS_PATH or download tokenizer assets first, for example:"
  echo "  python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer"
  exit 1
fi

if [[ ! -f "${HF_ASSETS_PATH}/tokenizer.json" && ! -f "${HF_ASSETS_PATH}/tokenizer.model" ]]; then
  echo "HF_ASSETS_PATH=${HF_ASSETS_PATH} does not contain tokenizer.json or tokenizer.model"
  echo "Please point HF_ASSETS_PATH at a valid tokenizer assets directory."
  exit 1
fi
export HF_ASSETS_PATH

copy_into_dir "${TT_CONFIG_JSON}" "${RUN_CONFIG_ROOT}"
copy_into_dir "${SCRIPT_DIR}/launch_deepseek_v3_moe_ep12.sh" "${RUN_METADATA_ROOT}/launchers"
copy_into_dir "${SCRIPT_DIR}/run_torchtitan_ezpz_train.sh" "${RUN_METADATA_ROOT}/launchers"
if [[ -n "${RUN_SUBMITTER_PATH:-}" ]]; then
  copy_into_dir "${RUN_SUBMITTER_PATH}" "${RUN_METADATA_ROOT}/submitters"
fi

{
  printf 'WANDB_NAME=%q\n' "${WAND_NAME}"
  printf 'RUN_SLUG=%q\n' "${RUN_SLUG}"
  printf 'RUN_ROOT=%q\n' "${RUN_ROOT}"
  printf 'TT_CONFIG_JSON=%q\n' "${TT_CONFIG_JSON}"
  printf 'HF_ASSETS_PATH=%q\n' "${HF_ASSETS_PATH}"
  printf 'RUN_SUBMITTER_PATH=%q\n' "${RUN_SUBMITTER_PATH:-}"
  printf 'PBS_JOBID=%q\n' "${PBS_JOBID:-}"
} > "${RUN_METADATA_ROOT}/run_context.env"

echo "Repo root: ${REPO_ROOT}"
echo "Using TT_CONFIG_JSON=${TT_CONFIG_JSON}"
echo "Using HF_ASSETS_PATH=${HF_ASSETS_PATH}"
echo "Run slug: ${RUN_SLUG}"
echo "Dump folder: ${RUN_ROOT}"
echo "Checkpoint folder: ${CKPT_ROOT}"
echo "Run logs: ${RUN_LOG_ROOT}"

TRAIN_ENTRY=(
  python3 -m torchtitan.experiments.ezpz.train
)

LAUNCH_CMD=(
  "${EZPZ_BIN}" launch
)
LAUNCH_CMD+=(
  "${TRAIN_ENTRY[@]}"
  --module deepseek_v3
  --config deepseek_v3_10b_2b_ep12_from_json
  --dump-folder "${RUN_ROOT}"
  --hf-assets-path "${HF_ASSETS_PATH}"
  --checkpoint.enable
  --checkpoint.folder checkpoints
  --checkpoint.load_step -1
  --metrics.enable_wandb
  --debug.print-config
  --debug.save-config-file "configs/effective_config.json"
  "${EXTRA_ARGS[@]}"
)

set +e
"${LAUNCH_CMD[@]}"
launch_rc=$?
set -e

exit "${launch_rc}"
