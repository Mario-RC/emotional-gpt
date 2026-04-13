# Data layout

Generated training files expected by training scripts:

- `data/gpt-dialogues/train.txt`
- `data/gpt-dialogues/dev.txt`

Raw CSV files required to build them:

- `data/raw/DAILYD_main.csv`
- `data/raw/DAILYD_dialoginfo.csv`

Source: https://github.com/CHANEL-JSALT-2020/datasets

Build command:

```bash
python src/build_dataset.py
```

Notes:

- `src/build_dataset.py` reads from `data/raw` by default.
- You can override paths with `--data-dir`, `--main-file`, `--info-file`, and `--output-dir`.
