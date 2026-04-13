#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH -n 1
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --output=train.log

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

CONFIG_PATH="${CONFIG_PATH:-configs/train_model.json}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

CONFIG_EXPORTS="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as handle:
  cfg = json.load(handle)

required_keys = [
  "model_type",
  "model_name",
  "allowed_models",
  "train_file",
  "eval_file",
  "output_dir_template",
  "num_epochs",
  "train_batch_size",
  "eval_batch_size",
  "learning_rate",
  "logging_steps",
  "save_steps",
  "overwrite_output_dir",
  "line_by_line",
]

missing = [key for key in required_keys if key not in cfg]
if missing:
  raise SystemExit(
    "Missing keys in training config "
    f"'{config_path}': {', '.join(missing)}"
  )

allowed_models = cfg["allowed_models"]
if not isinstance(allowed_models, list) or not allowed_models:
  raise SystemExit(
    f"Key 'allowed_models' must be a non-empty list in '{config_path}'."
  )

if not all(isinstance(model, str) and model.strip() for model in allowed_models):
  raise SystemExit(
    f"All entries in 'allowed_models' must be non-empty strings in '{config_path}'."
  )

def to_shell_value(value):
  if isinstance(value, bool):
    return "true" if value else "false"
  return str(value)

mapping = {
  "CFG_MODEL_TYPE": "model_type",
  "CFG_MODEL_NAME": "model_name",
  "CFG_TRAIN_FILE": "train_file",
  "CFG_EVAL_FILE": "eval_file",
  "CFG_OUTPUT_DIR_TEMPLATE": "output_dir_template",
  "CFG_NUM_EPOCHS": "num_epochs",
  "CFG_TRAIN_BATCH_SIZE": "train_batch_size",
  "CFG_EVAL_BATCH_SIZE": "eval_batch_size",
  "CFG_LEARNING_RATE": "learning_rate",
  "CFG_LOGGING_STEPS": "logging_steps",
  "CFG_SAVE_STEPS": "save_steps",
  "CFG_OVERWRITE_OUTPUT_DIR": "overwrite_output_dir",
  "CFG_LINE_BY_LINE": "line_by_line",
}

for env_name, key in mapping.items():
  shell_value = shlex.quote(to_shell_value(cfg[key]))
  print(f"{env_name}={shell_value}")

print("CFG_ALLOWED_MODELS=" + shlex.quote(",".join(allowed_models)))
PY
)"

eval "${CONFIG_EXPORTS}"

MODEL_NAME="${MODEL_NAME:-${CFG_MODEL_NAME}}"
MODEL_TYPE="${MODEL_TYPE:-${CFG_MODEL_TYPE}}"
TRAIN_FILE="${TRAIN_FILE:-${CFG_TRAIN_FILE}}"
EVAL_FILE="${EVAL_FILE:-${CFG_EVAL_FILE}}"
NUM_EPOCHS="${NUM_EPOCHS:-${CFG_NUM_EPOCHS}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${CFG_TRAIN_BATCH_SIZE}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${CFG_EVAL_BATCH_SIZE}}"
LEARNING_RATE="${LEARNING_RATE:-${CFG_LEARNING_RATE}}"
LOGGING_STEPS="${LOGGING_STEPS:-${CFG_LOGGING_STEPS}}"
SAVE_STEPS="${SAVE_STEPS:-${CFG_SAVE_STEPS}}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-${CFG_OVERWRITE_OUTPUT_DIR}}"
LINE_BY_LINE="${LINE_BY_LINE:-${CFG_LINE_BY_LINE}}"

IFS=',' read -r -a ALLOWED_MODELS <<< "${CFG_ALLOWED_MODELS}"
MODEL_ALLOWED="false"
for allowed_model in "${ALLOWED_MODELS[@]}"; do
  if [[ "${MODEL_NAME}" == "${allowed_model}" ]]; then
    MODEL_ALLOWED="true"
    break
  fi
done

if [[ "${MODEL_ALLOWED}" != "true" ]]; then
  echo "MODEL_NAME is not in allowed_models: ${MODEL_NAME}" >&2
  echo "Allowed models: ${CFG_ALLOWED_MODELS}" >&2
  exit 1
fi

SAFE_MODEL_NAME="${MODEL_NAME//\//-}"
DEFAULT_OUTPUT_DIR="${CFG_OUTPUT_DIR_TEMPLATE//\{model_name\}/${SAFE_MODEL_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file not found: ${TRAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
  echo "Evaluation file not found: ${EVAL_FILE}" >&2
  exit 1
fi

TRAIN_ARGS=(
  --num_train_epochs="${NUM_EPOCHS}"
  --model_type="${MODEL_TYPE}"
  --model_name_or_path "${MODEL_NAME}"
  --do_train
  --do_eval
  --train_data_file "${TRAIN_FILE}"
  --eval_data_file "${EVAL_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --per_gpu_train_batch_size="${TRAIN_BATCH_SIZE}"
  --per_gpu_eval_batch_size="${EVAL_BATCH_SIZE}"
  --logging_steps="${LOGGING_STEPS}"
  --save_steps="${SAVE_STEPS}"
  --learning_rate="${LEARNING_RATE}"
)

if [[ "${LINE_BY_LINE}" == "true" ]]; then
  TRAIN_ARGS+=(--line_by_line)
fi

if [[ "${OVERWRITE_OUTPUT_DIR}" == "true" ]]; then
  TRAIN_ARGS+=(--overwrite_output_dir)
fi

"${PYTHON_BIN}" src/train_model.py "${TRAIN_ARGS[@]}"
