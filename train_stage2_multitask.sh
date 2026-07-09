#!/usr/bin/env bash
set -eu
[ -n "${BASH_VERSION:-}" ] && set -o pipefail

# Stage 2: 5-task unified training from Stage-1 checkpoint.
# Default: single GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_ITER="${TOTAL_ITER:-250000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-docres_stage2_multitask_250k}"
STAGE1_CKPT="${STAGE1_CKPT:-./checkpoints/docres_stage1_dewarp_100k/100000.pkl}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-1}"
RESUME_MODEL_ONLY_ARG=""
if [ "${RESUME_MODEL_ONLY}" = "1" ]; then
  RESUME_MODEL_ONLY_ARG="--resume_model_only"
fi

python3 train.py \
  --train_stage multitask \
  --tboard \
  --resume "${STAGE1_CKPT}" \
  ${RESUME_MODEL_ONLY_ARG} \
  --batch_size "${BATCH_SIZE}" \
  --total_iter "${TOTAL_ITER}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "$@"
