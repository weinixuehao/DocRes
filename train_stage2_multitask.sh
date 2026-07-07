#!/usr/bin/env bash
set -eu
[ -n "${BASH_VERSION:-}" ] && set -o pipefail

# Stage 2: 5-task unified training from Stage-1 checkpoint.
# Default: single GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-26413}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_ITER="${TOTAL_ITER:-100000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-docres_stage2_multitask_100k}"
STAGE1_CKPT="${STAGE1_CKPT:-./checkpoints/docres_stage1_dewarp_50k/50000.pkl}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --train_stage multitask \
  --tboard \
  --resume "${STAGE1_CKPT}" \
  --batch_size "${BATCH_SIZE}" \
  --total_iter "${TOTAL_ITER}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "$@"
