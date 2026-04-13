# Emotional GPT

Fine-tuning and text generation with GPT-2 and DialoGPT models using Hugging Face Transformers.

## Requirements

- Python 3.9+

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Workflow

### 1. Download raw CSV files

Source: https://github.com/CHANEL-JSALT-2020/datasets

Required files:

- `DAILYD_main.csv`
- `DAILYD_dialoginfo.csv`

### 2. Place raw files in the expected folder

```text
data/raw/DAILYD_main.csv
data/raw/DAILYD_dialoginfo.csv
```

### 3. Build train/dev files

```bash
python src/build_dataset.py
```

Default arguments in `src/build_dataset.py`:

- `--data-dir data/raw`
- `--main-file DAILYD_main.csv`
- `--info-file DAILYD_dialoginfo.csv`
- `--output-dir data/gpt-dialogues`
- `--dev-size 0.2`
- `--seed 42`

Equivalent explicit command:

```bash
python src/build_dataset.py \
  --data-dir data/raw \
  --main-file DAILYD_main.csv \
  --info-file DAILYD_dialoginfo.csv \
  --output-dir data/gpt-dialogues \
  --dev-size 0.2 \
  --seed 42
```

This script generates:

- `data/gpt-dialogues/train.txt`
- `data/gpt-dialogues/dev.txt`

## Train a Model

```bash
bash scripts/train_model.sh
```

Defaults are loaded from `configs/train_model.json`:

- `model_type=gpt2`
- `model_name=gpt2-medium`
- `allowed_models=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl, microsoft/DialoGPT-small, microsoft/DialoGPT-medium, microsoft/DialoGPT-large]`
- `train_file=data/gpt-dialogues/train.txt`
- `eval_file=data/gpt-dialogues/dev.txt`
- `output_dir_template=output_model/{model_name}_finetuned`
- `num_epochs=4.0`
- `train_batch_size=6`
- `eval_batch_size=6`
- `learning_rate=1e-5`
- `logging_steps=5000`
- `save_steps=5000`
- `overwrite_output_dir=true`
- `line_by_line=true`

Run with an explicit config path:

```bash
CONFIG_PATH=configs/train_model.json bash scripts/train_model.sh
```

Override values per run using environment variables:

```bash
MODEL_NAME=gpt2-large \
OUTPUT_DIR=output_model/gpt2-large_finetuned \
NUM_EPOCHS=3 \
TRAIN_BATCH_SIZE=4 \
EVAL_BATCH_SIZE=4 \
LEARNING_RATE=5e-5 \
bash scripts/train_model.sh
```

Supported `MODEL_NAME` values (validated against `allowed_models`):

- `distilgpt2`
- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`
- `microsoft/DialoGPT-small`
- `microsoft/DialoGPT-medium`
- `microsoft/DialoGPT-large`

## Generate Sample Outputs

```bash
bash scripts/generate_text_samples.sh
```

Defaults:

- `MODEL_PATH=output_model/gpt2-medium_finetuned`
- `NUM_RETURN_SEQUENCES=5`
- `MAX_LENGTH=128`
- `TEMPERATURE=0.8`

## Project Structure

```text
emotional_gpt/
├── configs/
│   └── train_model.json              # Training defaults and allowed models
├── data/
│   ├── README.md                     # Dataset preparation notes
│   └── gpt-dialogues/
├── scripts/
│   ├── train_model.sh                # Launches training with config/env overrides
│   └── generate_text_samples.sh      # Generates text from a fine-tuned checkpoint
└── src/
    ├── build_dataset.py              # Builds train/dev splits from raw CSV files
    ├── train_model.py                # Fine-tuning entrypoint
    └── generate_text.py              # Inference / text generation entrypoint
```

Local artifacts such as `emo-model/`, `output_model/`, `runs/`, and raw/generated dataset files are ignored by `.gitignore` and are not intended to be pushed to GitHub.
