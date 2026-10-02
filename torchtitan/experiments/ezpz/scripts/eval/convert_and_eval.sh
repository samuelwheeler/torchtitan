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
    echo "Usage: $0 --model {2b|20b|12b2a} --step STEP [--dcp-dir DIR] [--tokenizer-dir DIR]"
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
# Dense models use vLLM; the custom AGPT MoE uses the existing HF lm-eval
# backend so its remote-code model preserves routing and shared-expert semantics.
MODEL_NAME="experiments.ezpz.agpt"
MODEL_FLAVOR="${MODEL}"
HF_CONFIG="${EVAL_DIR}/configs/agpt_${MODEL}_config.json"
EVAL_BACKEND="vllm"
if [[ "$MODEL" == "2b" ]]; then
    TP=1
elif [[ "$MODEL" == "20b" ]]; then
    TP=12
elif [[ "$MODEL" == "12b2a" ]]; then
    TP=1
    MODEL_NAME="experiments.ezpz.moe"
    MODEL_FLAVOR="AGPT_2B_50K_MOE_sdpa_aurora_full_sonic"
    HF_CONFIG=""
    EVAL_BACKEND="hf"
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
    if [[ "$MODEL" != "12b2a" ]]; then
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

    python3 "${EVAL_DIR}/convert_to_hf.py" \
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
    if [[ "$MODEL" != "12b2a" ]]; then
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

    if [[ "$EVAL_BACKEND" == "hf" ]]; then
        lm_eval \
            --model hf \
            --model_args "pretrained=${HF_DIR},dtype=${EXPORT_DTYPE},trust_remote_code=True" \
            --device xpu:0 \
            --tasks "${TASKS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_fewshot "${NUM_FEWSHOT}" \
            --output_path "${RESULTS_DIR}" \
            --log_samples
    else
        lm_eval \
            --model vllm \
            --model_args "pretrained=${HF_DIR},tensor_parallel_size=${TP},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096" \
            --tasks "${TASKS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_fewshot "${NUM_FEWSHOT}" \
            --output_path "${RESULTS_DIR}" \
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
