#!/bin/bash
set -eo pipefail

# Anole runner for lmms-eval. The adapter (lmms_eval/models/anole.py) wraps
# anole/chameleon/inference/chameleon.py's ChameleonInferenceModel and runs
# one synchronized batched generate() per chunk (no per-sample fallback).
#
# Caveats baked in:
#   - batch_size is passed via the --batch_size CLI flag, NOT --model_args.
#     The framework auto-injects batch_size into create_from_arg_string()
#     and would collide with a duplicate kwarg if it appeared in both.
#   - The Chameleon worker subprocesses are bound to the active
#     CUDA_VISIBLE_DEVICES at construction time and are not thread-safe; we
#     run on a single GPU (no accelerate launch).
#
# Example:
#   TASKS=chartqa LIMIT=8 BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=2 ./run_anole.sh

# --- ckpt resolution -------------------------------------------------------
# Logical name (used for output dir + reporting).
CKPT_NAME=${CKPT_NAME:-GAIR/Anole-7b}
# Filesystem path to the chameleon-native checkpoint dir. Must contain:
#   models/7b/consolidated.pth
#   tokenizer/{text_tokenizer.json,vqgan.yaml,vqgan.ckpt}
# GAIR/Anole-7b's weights are the same as GAIR/Anole-7b-v0.1; the v0.1 repo
# ships them in the chameleon-native layout that ChameleonInferenceModel
# expects. We default to the locally-downloaded copy.
CKPT_PATH=${CKPT_PATH:-/home/yoonjeon.kim/dLLM-EVAL/anole/ckpts/Anole-7b-v0.1}

for asset in \
  "${CKPT_PATH}/models/7b/consolidated.pth" \
  "${CKPT_PATH}/tokenizer/text_tokenizer.json" \
  "${CKPT_PATH}/tokenizer/vqgan.yaml" \
  "${CKPT_PATH}/tokenizer/vqgan.ckpt"; do
    if [[ ! -e "$asset" ]]; then
        echo "Error: missing required Anole asset: ${asset}" >&2
        echo "Hint: download GAIR/Anole-7b-v0.1 to ${CKPT_PATH} (chameleon-native layout)." >&2
        exit 1
    fi
done

# --- run knobs -------------------------------------------------------------
# BATCH_SIZE controls the single-pass batched generate() inside the adapter.
# Anole runs the whole chunk as one synchronized stream — image/text modality
# switches fire only when *all* batch rows reach the boundary together — so
# this is a true batched generation, not per-sample looping.
BATCH_SIZE=${BATCH_SIZE:-1}
TASKS=${TASKS:-chartqa}
LIMIT=${LIMIT:-}
TEMPERATURE=${TEMPERATURE:-0}            # 0 -> greedy text decoding
CFG_IMAGE=${CFG_IMAGE:-}                 # optional Options.img.cfg.guidance_scale_image
CFG_TEXT=${CFG_TEXT:-}                   # optional Options.img.cfg.guidance_scale_text
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}         # Options.max_seq_len ceiling
RUN_TAG=${RUN_TAG:-}

# --- output paths ----------------------------------------------------------
BASE_DIR=${BASE_DIR:-/home/yoonjeon.kim/dLLM-EVAL/outputs}
MODEL_NAME=${CKPT_NAME//\//__}           # GAIR/Anole-7b -> GAIR__Anole-7b
OUTPUT_SUFFIX=""
if [[ -n "${RUN_TAG}" ]]; then
    OUTPUT_SUFFIX="_${RUN_TAG}"
fi
OUTPUT_DIR="${BASE_DIR}/anole/${MODEL_NAME}${OUTPUT_SUFFIX}"
GEN_IMG_DIR="${OUTPUT_DIR}/gen_imgs"
mkdir -p "${OUTPUT_DIR}" "${GEN_IMG_DIR}"

# --- python interpreter ----------------------------------------------------
PYTHON=${PYTHON:-/home/yoonjeon.kim/d1-lavidao/.venv/bin/python}

echo "============================================================"
echo " Anole lmms-eval run"
echo "  CKPT_NAME       = ${CKPT_NAME}"
echo "  CKPT_PATH       = ${CKPT_PATH}"
echo "  TASKS           = ${TASKS}"
echo "  BATCH_SIZE      = ${BATCH_SIZE}  (single-pass batched generate)"
echo "  TEMPERATURE     = ${TEMPERATURE}"
echo "  LIMIT           = ${LIMIT:-<none>}"
echo "  CFG_IMAGE       = ${CFG_IMAGE:-<chameleon default>}"
echo "  CFG_TEXT        = ${CFG_TEXT:-<chameleon default>}"
echo "  MAX_SEQ_LEN     = ${MAX_SEQ_LEN}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "============================================================"

# --- assemble model_args ---------------------------------------------------
# IMPORTANT: do NOT add batch_size= here. The framework auto-injects it via
# create_from_arg_string() and would collide. Pass --batch_size separately.
MODEL_ARGS="pretrained=${CKPT_PATH}"
MODEL_ARGS="${MODEL_ARGS},gen_img_dir=${GEN_IMG_DIR}"
MODEL_ARGS="${MODEL_ARGS},temperature=${TEMPERATURE}"
MODEL_ARGS="${MODEL_ARGS},max_seq_len=${MAX_SEQ_LEN}"
if [[ -n "${CFG_IMAGE}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},cfg_image=${CFG_IMAGE}"
fi
if [[ -n "${CFG_TEXT}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},cfg_text=${CFG_TEXT}"
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT,,}" != "none" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

set -x
${PYTHON} -m lmms_eval \
    --model anole \
    --model_args "${MODEL_ARGS}" \
    --batch_size "${BATCH_SIZE}" \
    --tasks "${TASKS}" \
    --log_samples \
    --log_samples_suffix anole \
    --output_path "${OUTPUT_DIR}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
set +x

echo "Done. Results -> ${OUTPUT_DIR}"
echo "Generated images -> ${GEN_IMG_DIR}"
