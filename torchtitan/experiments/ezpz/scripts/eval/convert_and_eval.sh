#!/bin/bash
# convert_and_eval.sh — Convert DCP checkpoint to HF and run lm-eval
#
# Usage:
#   bash convert_and_eval.sh --model 2b --step 5000
#   bash convert_and_eval.sh --model 20b --step 1000 --tasks "hellaswag,mmlu"
#   bash convert_and_eval.sh --model 2b --step 5000 --convert-only
#   bash convert_and_eval.sh --model 2b --step 5000 --eval-only
#   bash convert_and_eval.sh --model 12b2a --step 27000 \
#     --dcp-dir /path/to/step-27000 --tokenizer-dir /path/to/tokenizer

set -euo pipefail

# ---- Defaults ----
MODEL=""
STEP=""
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
CONVERT_ONLY=false
EVAL_ONLY=false
EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
CONVERT_PYTHON="${CONVERT_PYTHON:-python3}"
CONVERT_PYTHONPATH="${CONVERT_PYTHONPATH:-${PYTHONPATH:-}}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-}"
DEVICE="${DEVICE:-xpu:0}"
MAX_LENGTH=""
SEED="0,1234,1234,1234"
LIMIT=""
# `--ckpt-name` overrides the auto-derived "agpt-${MODEL}-sophiag-olmo-mix-1124-n256-gbs3072"
# checkpoint name. Use this for v2 runs (e.g. n256-gbs6144 or n512-gbs12288).
CKPT_NAME_OVERRIDE=""
# `--repo-root` overrides the path to the clone whose outputs/ holds
# the DCP checkpoints. Defaults to this script's enclosing repo.
REPO_ROOT_OVERRIDE=""
# `--label` is appended to the output dir so v1 vs v2 results don't collide.
EVAL_LABEL=""
DCP_DIR_OVERRIDE=""
TOKENIZER_DIR_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --convert-only) CONVERT_ONLY=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --export-dtype) EXPORT_DTYPE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --ckpt-name) CKPT_NAME_OVERRIDE="$2"; shift 2 ;;
        --repo-root) REPO_ROOT_OVERRIDE="$2"; shift 2 ;;
        --label) EVAL_LABEL="$2"; shift 2 ;;
        --dcp-dir) DCP_DIR_OVERRIDE="$2"; shift 2 ;;
        --tokenizer-dir) TOKENIZER_DIR_OVERRIDE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$STEP" ]]; then
    echo "Usage: $0 --model {2b|20b|2b_50k|12b2a} --step STEP [--dcp-dir DIR] [--tokenizer-dir DIR]"
    exit 1
fi

# ---- Paths ----
SCRIPT_REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-$SCRIPT_REPO_ROOT}"
EVAL_DIR="${SCRIPT_REPO_ROOT}/torchtitan/experiments/ezpz/eval"
CKPT_BASE="${REPO_ROOT}/outputs/checkpoints"
TOKENIZER_DIR="${TOKENIZER_DIR_OVERRIDE:-${SCRIPT_REPO_ROOT}/assets/hf/gemma-7b}"

# Checkpoint input (DCP format). Default name is the v1 layout
# (n256-gbs3072) — v2 callers MUST pass --ckpt-name.
CKPT_NAME="${CKPT_NAME_OVERRIDE:-agpt-${MODEL}-sophiag-olmo-mix-1124-n256-gbs3072}"
DCP_DIR="${DCP_DIR_OVERRIDE:-${CKPT_BASE}/${CKPT_NAME}/step-${STEP}}"

# Output dirs. The label suffix prevents v1/v2 result collisions when
# they share a step number (e.g. both have a step-100).
LABEL_SUFFIX="${EVAL_LABEL:+-${EVAL_LABEL}}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-${SCRIPT_REPO_ROOT}/outputs/evals/agpt-${MODEL}${LABEL_SUFFIX}/step-${STEP}}"
HF_DIR="${OUTPUT_ROOT}/hf"
RESULTS_DIR="${OUTPUT_ROOT}/results"

# HF config for this model size
# The data-matched dense and MoE models share the HF evaluation backend.
MODEL_NAME="experiments.ezpz.agpt"
MODEL_FLAVOR="${MODEL}"
HF_CONFIG="${EVAL_DIR}/configs/agpt_${MODEL}_config.json"
EVAL_BACKEND="vllm"
if [[ "$MODEL" == "2b" ]]; then
    TP=1
elif [[ "$MODEL" == "20b" ]]; then
    TP=12
elif [[ "$MODEL" == "2b_50k" || "$MODEL" == "12b2a" ]]; then
    TP=1
    if [[ "$MODEL" == "12b2a" ]]; then
        MODEL_NAME="experiments.ezpz.moe"
        MODEL_FLAVOR="AGPT_2B_50K_MOE_sdpa_aurora_full_sonic"
    fi
    HF_CONFIG=""
    EVAL_BACKEND="hf"
    if [[ "$EVAL_ONLY" != true && -z "$TOKENIZER_DIR_OVERRIDE" ]]; then
        echo "ERROR: ${MODEL} conversion requires --tokenizer-dir with the training tokenizer"
        exit 1
    fi
else
    echo "ERROR: unsupported model: ${MODEL}"
    exit 1
fi

echo "============================================"
echo "AGPT Evaluation Pipeline"
echo "============================================"
echo "Model:        agpt_${MODEL}"
echo "Step:         ${STEP}"
echo "DCP dir:      ${DCP_DIR}"
echo "HF output:    ${HF_DIR}"
echo "Results:      ${RESULTS_DIR}"
echo "Tasks:        ${TASKS}"
echo "TP:           ${TP}"
echo "Backend:      ${EVAL_BACKEND}"
echo "Export dtype:  ${EXPORT_DTYPE}"
echo "============================================"

# ---- Validate conversion inputs ----
if [[ "$EVAL_ONLY" != true ]]; then
    if [[ ! -d "$DCP_DIR" ]]; then
        echo "ERROR: DCP checkpoint not found: ${DCP_DIR}"
        exit 1
    fi
    if [[ -n "$HF_CONFIG" && ! -f "$HF_CONFIG" ]]; then
        echo "ERROR: HF config not found: ${HF_CONFIG}"
        exit 1
    fi
    REQUIRED_TOKENIZER_FILES=(tokenizer.model)
    if [[ -n "$HF_CONFIG" ]]; then
        REQUIRED_TOKENIZER_FILES+=(
            tokenizer.json tokenizer_config.json special_tokens_map.json
        )
    fi
    for name in "${REQUIRED_TOKENIZER_FILES[@]}"; do
        if [[ ! -f "${TOKENIZER_DIR}/${name}" ]]; then
            echo "ERROR: tokenizer asset not found: ${TOKENIZER_DIR}/${name}"
            exit 1
        fi
    done
fi

# ---- Step 1: Convert DCP → HuggingFace ----
if [[ "$EVAL_ONLY" != true ]]; then
    echo ""
    echo "[1/3] Converting DCP checkpoint to HuggingFace format..."
    mkdir -p "${HF_DIR}"

    PYTHONPATH="${SCRIPT_REPO_ROOT}${CONVERT_PYTHONPATH:+:${CONVERT_PYTHONPATH}}" \
        "${CONVERT_PYTHON}" -m torchtitan.experiments.ezpz.eval.convert_to_hf \
        "${DCP_DIR}" \
        "${HF_DIR}" \
        --hf_assets_path "${TOKENIZER_DIR}" \
        --model_name "${MODEL_NAME}" \
        --model_flavor "${MODEL_FLAVOR}" \
        --export_dtype "${EXPORT_DTYPE}"

    echo "[1/3] Conversion complete."

    # ---- Step 2: Copy config + tokenizer into HF dir ----
    echo ""
    echo "[2/3] Copying config.json and tokenizer files..."
    if [[ -n "$HF_CONFIG" ]]; then
        cp "${HF_CONFIG}" "${HF_DIR}/config.json"
        cp "${TOKENIZER_DIR}/tokenizer.json" "${HF_DIR}/"
        cp "${TOKENIZER_DIR}/tokenizer.model" "${HF_DIR}/"
        cp "${TOKENIZER_DIR}/tokenizer_config.json" "${HF_DIR}/"
        cp "${TOKENIZER_DIR}/special_tokens_map.json" "${HF_DIR}/"
    fi

    echo "[2/3] Assets copied."
else
    echo ""
    echo "[1-2/3] Skipping conversion (--eval-only)"
    if [[ ! -d "$HF_DIR" ]]; then
        echo "ERROR: HF checkpoint not found at ${HF_DIR}. Run without --eval-only first."
        exit 1
    fi
fi

# ---- Step 3: Run lm-eval ----
if [[ "$CONVERT_ONLY" != true ]]; then
    echo ""
    echo "[3/3] Running lm-eval with ${EVAL_BACKEND} backend..."
    mkdir -p "${RESULTS_DIR}"
    EVAL_COMMAND=(lm_eval)
    if [[ -n "$LM_EVAL_PYTHON" ]]; then
        EVAL_COMMAND=("${LM_EVAL_PYTHON}" -m lm_eval)
    fi
    LIMIT_ARGS=()
    if [[ -n "$LIMIT" ]]; then
        LIMIT_ARGS=(--limit "${LIMIT}")
    fi

    if [[ "$EVAL_BACKEND" == "hf" ]]; then
        MODEL_ARGS="pretrained=${HF_DIR},dtype=${EXPORT_DTYPE},trust_remote_code=True"
        if [[ -n "$MAX_LENGTH" ]]; then
            MODEL_ARGS+=",max_length=${MAX_LENGTH}"
        fi
        "${EVAL_COMMAND[@]}" \
            --model hf \
            --model_args "${MODEL_ARGS}" \
            --device "${DEVICE}" \
            --tasks "${TASKS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_fewshot "${NUM_FEWSHOT}" \
            --seed "${SEED}" \
            --output_path "${RESULTS_DIR}" \
            "${LIMIT_ARGS[@]}" \
            --log_samples
    else
        MODEL_ARGS="pretrained=${HF_DIR},tensor_parallel_size=${TP},dtype=auto,gpu_memory_utilization=0.8,max_model_len=${MAX_LENGTH:-4096}"
        "${EVAL_COMMAND[@]}" \
            --model vllm \
            --model_args "${MODEL_ARGS}" \
            --tasks "${TASKS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_fewshot "${NUM_FEWSHOT}" \
            --seed "${SEED}" \
            --output_path "${RESULTS_DIR}" \
            "${LIMIT_ARGS[@]}" \
            --log_samples
    fi

    echo "[3/3] Evaluation complete."
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo "View with: cat ${RESULTS_DIR}/results.json | python3 -m json.tool"
else
    echo ""
    echo "[3/3] Skipping evaluation (--convert-only)"
fi

echo ""
echo "Done."
