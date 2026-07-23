#!/usr/bin/env bash
set -euo pipefail

# Standalone TextSLIP brain MRI training launcher.
#
# Required:
#   TEXTSLIP_PRETRAINED=/path/to/b16_400m.pt
#
# Common optional overrides:
#   TEXTSLIP_NGPUS=6
#   TEXTSLIP_LOG_DIR=./logs
#   TEXTSLIP_TRAIN_DATA='/path/to/dataset-{000001..000010}.tar::...'
#   TEXTSLIP_TRAIN_NUM_SAMPLES=7334713

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_NAME="${TEXTSLIP_CONFIG:-b16_400m_textslip_brain}"
NGPUS="${TEXTSLIP_NGPUS:-6}"
NNODES="${TEXTSLIP_NNODES:-1}"
NODE_RANK="${TEXTSLIP_NODE_RANK:-0}"
MASTER_ADDR="${TEXTSLIP_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${TEXTSLIP_MASTER_PORT:-29500}"
PRETRAINED="${TEXTSLIP_PRETRAINED:-}"
LOG_ROOT="${TEXTSLIP_LOG_DIR:-./logs}"
RUN_NAME="${TEXTSLIP_RUN_NAME:-textslip_brain_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${PRETRAINED}" ]]; then
  echo "ERROR: set TEXTSLIP_PRETRAINED=/path/to/pretrained_checkpoint.pt" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED}" >&2
  exit 1
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NGPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/training/main.py \
  "${CONFIG_NAME}" \
  "${LOG_ROOT}/${RUN_NAME}" \
  "${PRETRAINED}"
