#!/usr/bin/env bash
#SBATCH --job-name=generate
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH --output=generate.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DEFAULT_PYTHON_BIN="python3"
if [[ -x "${PROJECT_ROOT}/emo-model/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${PROJECT_ROOT}/emo-model/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
if [[ ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-output_model/gpt2-medium_finetuned}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"
STOP_TOKEN="${STOP_TOKEN:-<|endoftext|>}"

PROMPTS=(
  "[CLS] [DA]General_ChatIntent[/DA] [TOPIC]general[/TOPIC] I think I should buy a new jacket for winter. [SEP]"
  "[CLS] [DA]General_ChatIntent[/DA] [TOPIC]general[/TOPIC] I just started a new job and I am a bit nervous. [SEP]"
  "[CLS] [DA]General_ChatIntent[/DA] [TOPIC]other[/TOPIC] Can you recommend something fun to do this weekend? [SEP]"
)

for prompt in "${PROMPTS[@]}"; do
  echo "=== Prompt ==="
  echo "${prompt}"
  "${PYTHON_BIN}" src/generate_text.py \
    --model_type gpt2 \
    --model_name_or_path "${MODEL_PATH}" \
    --prompt "${prompt}" \
    --num_return_sequences "${NUM_RETURN_SEQUENCES}" \
    --length "${MAX_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --stop_token "${STOP_TOKEN}"
done
