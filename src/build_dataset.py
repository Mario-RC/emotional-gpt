#!/usr/bin/env python3
"""Build train/dev text files from DailyDialog-style CSV inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_existing_file(data_dir: Path, preferred_name: str, fallback_name: str) -> Path:
    preferred_path = data_dir / preferred_name
    fallback_path = data_dir / fallback_name

    if preferred_path.exists():
        return preferred_path
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Missing file. Checked: '{preferred_path}' and '{fallback_path}'."
    )


def load_data(main_path: Path, info_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not main_path.exists():
        raise FileNotFoundError(f"Main file not found: {main_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Dialog info file not found: {info_path}")

    main_df = pd.read_csv(main_path)
    info_df = pd.read_csv(info_path)
    return main_df, info_df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "DIAL": "dialog_id",
        "DIALOG": "dialog_id",
        "DIALOG_ID": "dialog_id",
        "SID": "speaker",
        "SPEAKER": "speaker",
        "SEG": "text",
        "UTTERANCE": "text",
        "TEXT": "text",
    }

    out = df.copy()
    for src, dst in rename_candidates.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    required = {"dialog_id", "speaker", "text"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Missing required columns in main dataset: {missing}")

    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def merge_emotion_if_possible(main_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    out = main_df.copy()
    if "emotion" in info_df.columns and len(info_df) == len(out):
        out["emotion"] = info_df["emotion"].astype(str)
    return out


def build_pair_samples(df: pd.DataFrame) -> list[str]:
    samples: list[str] = []
    grouped = df.groupby("dialog_id", sort=False)

    for _, dialog in grouped:
        utterances = dialog["text"].tolist()
        for i in range(len(utterances) - 1):
            left = utterances[i].strip()
            right = utterances[i + 1].strip()
            if left and right:
                samples.append(f"{left} <eos> {right}")

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/dev files from DailyDialog CSV files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory with raw CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gpt-dialogues"), help="Output directory.")
    parser.add_argument("--main-file", type=str, default="DAILYD_main.csv", help="Main CSV filename.")
    parser.add_argument("--info-file", type=str, default="DAILYD_dialoginfo.csv", help="Dialog info CSV filename.")
    parser.add_argument("--dev-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_path = resolve_existing_file(args.data_dir, args.main_file, "DAILYD_main.csv")
    info_path = resolve_existing_file(args.data_dir, args.info_file, "DAILYD_dialoginfo.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_df, info_df = load_data(main_path, info_path)
    main_df = normalize_columns(main_df)
    main_df = merge_emotion_if_possible(main_df, info_df)

    samples = build_pair_samples(main_df)
    if not samples:
        raise ValueError("No training samples were generated from the input files.")

    train_samples, dev_samples = train_test_split(
        samples,
        test_size=args.dev_size,
        random_state=args.seed,
        shuffle=True,
    )

    train_path = args.output_dir / "train.txt"
    dev_path = args.output_dir / "dev.txt"

    train_path.write_text("\n".join(train_samples), encoding="utf-8")
    dev_path.write_text("\n".join(dev_samples), encoding="utf-8")

    print(f"Saved train: {train_path} ({len(train_samples)} rows)")
    print(f"Saved dev:   {dev_path} ({len(dev_samples)} rows)")


if __name__ == "__main__":
    main()
