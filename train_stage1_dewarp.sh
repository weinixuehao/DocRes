#!/usr/bin/env bash
set -eu
[ -n "${BASH_VERSION:-}" ] && set -o pipefail

# Stage 1: dewarping pre-training (paper setting: 100k steps).
# Default: single GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_ITER="${TOTAL_ITER:-100000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-docres_stage1_dewarp_100k}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_ARGS=()
[ -n "${RESUME_CKPT}" ] && RESUME_ARGS=(--resume "${RESUME_CKPT}")

python3 train.py \
  --train_stage dewarp_pretrain \
  --tboard \
  --batch_size "${BATCH_SIZE}" \
  --total_iter "${TOTAL_ITER}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "${RESUME_ARGS[@]}" \
  "$@"
