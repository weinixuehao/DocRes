#!/usr/bin/env bash
set -eu
[ -n "${BASH_VERSION:-}" ] && set -o pipefail

# One-command single-process training:
# same model + same LR schedule, but first stage trains dewarping only.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOGDIR="${LOGDIR:-./checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-2}"
STAGE1_TOTAL_ITER="${STAGE1_TOTAL_ITER:-60000}"
STAGE2_TOTAL_ITER="${STAGE2_TOTAL_ITER:-290000}"
L_RATE="${L_RATE:-2e-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-docres_two_stage_350k}"
TOTAL_ITER=$((STAGE1_TOTAL_ITER + STAGE2_TOTAL_ITER))
RESUME_CKPT="${RESUME_CKPT:-}"

echo "[two-stage] single-process mode: stage1=${STAGE1_TOTAL_ITER}, stage2=${STAGE2_TOTAL_ITER}, total=${TOTAL_ITER}, bs=${BATCH_SIZE}"
if [ -n "${RESUME_CKPT}" ]; then
  python3 train.py \
    --tboard \
    --logdir "${LOGDIR}" \
    --batch_size "${BATCH_SIZE}" \
    --total_iter "${TOTAL_ITER}" \
    --stage1_iter "${STAGE1_TOTAL_ITER}" \
    --l_rate "${L_RATE}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --resume "${RESUME_CKPT}" \
    "$@"
else
  python3 train.py \
    --tboard \
    --logdir "${LOGDIR}" \
    --batch_size "${BATCH_SIZE}" \
    --total_iter "${TOTAL_ITER}" \
    --stage1_iter "${STAGE1_TOTAL_ITER}" \
    --l_rate "${L_RATE}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    "$@"
fi
