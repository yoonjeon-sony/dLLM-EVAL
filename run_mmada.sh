#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=lmmseval-mmada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

# Example:
#   CHAT_MODE=image_gen TASKS="blink_jigsaw,vstar_bench" sbatch run_mmada.sh 0
CHAT_MODE=${CHAT_MODE:-image_gen} # text_gen,image_gen
USE_BBOX=${USE_BBOX:-False}       # accepted for parity, ignored by Mmada
CKPT_INDEX=$1
shift

declare -a CKPTS=(
  "tyfeld/MMaDA-Parallel-M"
  # add MMaDA fine-tunes here, e.g.
  # "/scratch2/yoonjeon.kim/sft_MMaDA-PM-thinkmorph_zebracot/checkpoint-8000"
)
if [[ -z "${CKPT_INDEX}" ]]; then
    echo "Error: CKPT_INDEX (first positional argument) is required. Valid range: 0-$((${#CKPTS[@]} - 1))" >&2
    exit 1
fi
if ! [[ "${CKPT_INDEX}" =~ ^[0-9]+$ ]] || (( CKPT_INDEX >= ${#CKPTS[@]} )); then
    echo "Error: CKPT_INDEX='${CKPT_INDEX}' is out of range. Valid range: 0-$((${#CKPTS[@]} - 1))" >&2
    exit 1
fi
CKPT="${CKPTS[${CKPT_INDEX}]}"
LIMIT=${LIMIT:-}
NUM_GPUS=${NUM_GPUS:-2}
TASKS=${TASKS:-}

SAVE_PARITY_CASES=${SAVE_PARITY_CASES:-0}
PARITY_CASES_ROOT=${PARITY_CASES_ROOT:-DEBUG/parity_text_gen}
PARITY_CASES_MAX_PER_TASK=${PARITY_CASES_MAX_PER_TASK:-4}

export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1

MODEL_NAME=$(basename "$(dirname "$CKPT")")-$(basename "$CKPT")
BASE_DIR="${BASE_DIR:-/scratch2/yoonjeon.kim/outputs}"

OUTPUT_DIR="${BASE_DIR}/mmada_${CHAT_MODE}_usebbox${USE_BBOX}/${MODEL_NAME}"

if [ "${NUM_GPUS}" -eq 1 ]; then
    LAUNCH_CMD="python"
    LAUNCH_ARGS="-m lmms_eval"
else
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK
    echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NUM_GPUS=${NUM_GPUS}"

    LAUNCH_CMD="accelerate launch --num_machines=1 --machine_rank=0 --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --num_processes=${NUM_GPUS}"
    LAUNCH_ARGS="-m lmms_eval"
fi

echo "Running MMaDA with TASKS=${TASKS} CKPT=${CKPT} CHAT_MODE=${CHAT_MODE}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT,,}" != "none" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi
PARITY_ARGS=()
if [[ "${SAVE_PARITY_CASES}" == "1" || "${SAVE_PARITY_CASES}" == "true" || "${SAVE_PARITY_CASES}" == "True" ]]; then
    PARITY_ARGS=(--save_parity_cases --parity_cases_root "${PARITY_CASES_ROOT}" --parity_cases_max_per_task "${PARITY_CASES_MAX_PER_TASK}")
fi

${LAUNCH_CMD} ${LAUNCH_ARGS} \
    --model mmada \
    --model_args pretrained=$CKPT,gen_img_dir=${OUTPUT_DIR}/gen_imgs,chat_mode=${CHAT_MODE} \
    --tasks "$TASKS" \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix mmada \
    --output_path ${OUTPUT_DIR} \
    --wandb_args "project=lmms-eval,job_type=eval,name=${EVAL_RUN:-mmada-debug}" \
    "${LIMIT_ARGS[@]}" \
    "${PARITY_ARGS[@]}" \
    "$@"

echo "Done evaluating checkpoint: ${CKPT}"
echo "Results saved to: ${OUTPUT_DIR}"
