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

 DAILYD resources source: [`CHANEL-JSALT-2020/datasets`](https://github.com/CHANEL-JSALT-2020/datasets) 

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

By default, the builder creates emotion-conditioned dialogue pairs:

```text
<bos><source_emotion>source utterance<sep><target_emotion>target utterance<|endoftext|>
```

DailyDialog numeric emotion labels are mapped as:

- `0`: `no emotion`
- `1`: `anger`
- `2`: `disgust`
- `3`: `fear`
- `4`: `happiness`
- `5`: `sadness`
- `6`: `surprise`

Default arguments in `src/build_dataset.py`:

- `--data-dir data/raw`
- `--main-file DAILYD_main.csv`
- `--info-file DAILYD_dialoginfo.csv`
- `--output-dir data/gpt-dialogues`
- `--format emotional-pairs`
- `--dev-size 0.2`
- `--seed 42`

Equivalent explicit command:

```bash
python src/build_dataset.py \
  --data-dir data/raw \
  --main-file DAILYD_main.csv \
  --info-file DAILYD_dialoginfo.csv \
  --output-dir data/gpt-dialogues \
  --format emotional-pairs \
  --dev-size 0.2 \
  --seed 42
```

To reproduce the old non-emotional format, pass `--format plain-pairs`.

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
- `output_dir_template=models/{model_name}`
- `num_epochs=4.0`
- `train_batch_size=6`
- `eval_batch_size=6`
- `gradient_accumulation_steps=1`
- `gradient_checkpointing=false`
- `learning_rate=1e-5`
- `logging_steps=5000`
- `save_steps=5000`
- `save_total_limit=1`
- `additional_special_tokens=[<sep>, <no emotion>, <anger>, <disgust>, <fear>, <happiness>, <sadness>, <surprise>]`
- `pad_token=<pad>`
- `bos_token=<bos>`
- `eos_token=<|endoftext|>`
- `overwrite_output_dir=true`
- `line_by_line=true`

Run with an explicit config path:

```bash
CONFIG_PATH=configs/train_model.json bash scripts/train_model.sh
```

Override values per run using environment variables:

```bash
MODEL_NAME=gpt2-large \
OUTPUT_DIR=models/gpt2-large \
NUM_EPOCHS=3 \
TRAIN_BATCH_SIZE=4 \
EVAL_BATCH_SIZE=4 \
LEARNING_RATE=5e-5 \
bash scripts/train_model.sh
```

Train the default emotional GPT-2 Medium model:

```bash
MODEL_NAME=gpt2-medium OUTPUT_DIR=models/gpt2-medium bash scripts/train_model.sh
```

Train GPT-2 Large with gradient accumulation for 24 GB GPUs:

```bash
MODEL_NAME=gpt2-large \
OUTPUT_DIR=models/gpt2-large \
TRAIN_BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=6 \
GRADIENT_CHECKPOINTING=true \
bash scripts/train_model.sh
```

Resume from the latest checkpoint in an output directory:

```bash
MODEL_NAME=gpt2-large \
OUTPUT_DIR=models/gpt2-large \
TRAIN_BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=6 \
GRADIENT_CHECKPOINTING=true \
SHOULD_CONTINUE=true \
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

- `MODEL_PATH=models/gpt2-medium`
- `NUM_RETURN_SEQUENCES=5`
- `MAX_LENGTH=128`
- `TEMPERATURE=0.8`

Generation prompts should include the source utterance emotion and the desired
response emotion:

```text
<bos><fear>I just started a new job and I am a bit nervous.<sep><no emotion>
```

## Hugging Face Model

The fine-tuned GPT-2 Medium model is available on Hugging Face:

- [`mario-rc/emotional-gpt2-medium`](https://huggingface.co/mario-rc/emotional-gpt2-medium)

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

Local artifacts such as `emo-model/`, `models/`, `output_model/`, `runs/`, and raw/generated dataset files are ignored by `.gitignore` and are not intended to be pushed to GitHub.
