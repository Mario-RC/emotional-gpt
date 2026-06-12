# Data layout

Generated training files expected by training scripts:

- `data/gpt-dialogues/train.txt`
- `data/gpt-dialogues/dev.txt`

By default, generated rows are emotion-conditioned dialogue pairs:

```text
<bos><source_emotion>source utterance<sep><target_emotion>target utterance<|endoftext|>
```

Source CSV files required to build them:

- `data/source/DAILYD_main.csv`
- `data/source/DAILYD_dialoginfo.csv`

Source: https://github.com/CHANEL-JSALT-2020/datasets

Build command:

```bash
python src/build_dataset.py
```

Notes:

- `src/build_dataset.py` reads from `data/source` by default.
- You can override paths with `--data-dir`, `--main-file`, `--info-file`, and `--output-dir`.
- Use `--format plain-pairs` only if you need the old non-emotional `utterance <eos> reply` format.
