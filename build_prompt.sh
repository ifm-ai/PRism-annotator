#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 INPUT_PATH OUTPUT_DIR MODEL_NAME [TEMPLATE_FILE]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="$1"
OUTPUT_DIR="$2"
MODEL_NAME="$3"
TEMPLATE_FILE="${4:-${SCRIPT_DIR}/prompt_template.txt}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TP="${TP:-2}"
MNS="${MNS:-128}"
MNBT="${MNBT:-65536}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-12000}"
MAX_FILE_RETRIES="${MAX_FILE_RETRIES:-2}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_prompt.py" \
  "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --template-file "${TEMPLATE_FILE}" \
  --model-name "${MODEL_NAME}" \
  --tp "${TP}" \
  --mns "${MNS}" \
  --mnbt "${MNBT}" \
  --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
  --max-file-retries "${MAX_FILE_RETRIES}"
