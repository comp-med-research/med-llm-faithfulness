from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import datetime as _dt

import pandas as pd


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y_%m_%d_%H%M")


def _normalize_model_name(raw: str) -> str:
    name = (raw or "").lower()
    if "claude" in name or "anthropic" in name:
        return "claude"
    if "gemini" in name or "google" in name:
        return "gemini"
    if "gpt" in name or "chatgpt" in name or "openai" in name:
        return "chatgpt"
    return name or "chatgpt"


def _infer_model_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for token in parts[::-1]:
        nm = _normalize_model_name(token)
        if nm in {"claude", "gemini", "chatgpt"}:
            return nm
    return "chatgpt"


def _collect_input_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.csv")))
        elif p.suffix.lower() == ".csv" and p.exists():
            files.append(p)
    # De-duplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collate model CSV responses into one long-form CSV")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="List of CSV files and/or directories containing CSVs (recursively)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Output CSV path. If omitted, saves to results/collated/<timestamp>.csv",
    )
    return parser.parse_args()


def read_and_standardize(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, keep_default_na=False)
    expected_cols = {"id", "prediction"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} missing required columns: {sorted(missing)}")
    if "model" not in df.columns:
        inferred = _infer_model_from_path(file_path)
        df["model"] = inferred
    df["model"] = df["model"].map(_normalize_model_name)
    # Optional enrichment columns
    for opt in ["post_title", "post_text"]:
        if opt not in df.columns:
            df[opt] = ""
    if "meta" not in df.columns:
        df["meta"] = ""
    # Keep canonical order, including optional columns if present
    cols = ["id", "model", "prediction", "post_title", "post_text", "meta"]
    return df[cols]


def collate(files: List[Path]) -> Tuple[pd.DataFrame, int]:
    frames: List[pd.DataFrame] = []
    for fp in files:
        frames.append(read_and_standardize(fp))
    if not frames:
        return pd.DataFrame(columns=["id", "model", "prediction", "post_title", "post_text", "meta"]), 0
    combined = pd.concat(frames, ignore_index=True)
    # Drop exact duplicates on (id, model) keeping first
    before = len(combined)
    combined = combined.drop_duplicates(subset=["id", "model"], keep="first")
    after = len(combined)
    # Sort for readability: by id then model
    combined = combined.sort_values(by=["id", "model"]).reset_index(drop=True)
    return combined, before - after


def main() -> None:
    args = parse_args()
    input_files = _collect_input_files(args.inputs)
    if not input_files:
        raise SystemExit("No CSV files found in provided inputs")

    df, num_deduped = collate(input_files)

    out_dir = Path("results/collated") if args.out is None else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        (Path("results/collated") / f"{_timestamp()}.csv")
        if args.out is None or args.out.is_dir()
        else args.out
    )

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} (deduped {num_deduped})")


if __name__ == "__main__":
    main()


